import os
import argparse
import random
import math
import numpy as np
import torch
import time
from torch import nn
from torch.nn import functional as F
from torchvision.ops import focal_loss
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
#from scipy.ndimage import gaussian_filter
from dataset.medical_zero import MedTestDataset, MedTrainDataset_modified
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import precision_recall_curve
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, encode_text_with_prompt_ensemble
from prompt import REAL_NAME


#import warnings
#warnings.filterwarnings("ignore")

torch.autograd.set_detect_anomaly(True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -2:'Chest', -3:'Histopathology'}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return 0


def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Retina_RESC')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt_r/zero-shot/')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    setup_seed(args.seed)

    
    print(f"Using device: {device}")
    print(f"----- Running zero-shot training for class: {args.obj} -----")

    # create log dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # fixed feature extractor
    print("Instantiating model...")
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    for name, param in model.named_parameters():
        param.requires_grad = True


    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))


    # load dataset and loader
    print("Loading dataset...")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = MedTrainDataset_modified(args.data_path, args.obj, args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, **kwargs)
    print(f"Train dataset size: {len(train_dataset)}")

    test_dataset = MedTestDataset(args.data_path, args.obj, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    print(f"Test dataset size: {len(test_dataset)}")


    # losses
    loss_focal = focal_loss.sigmoid_focal_loss # MODIFIED: using torchvision focal loss instead of custom 
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    text_feature_list = [0]
    # text prompt
    print("Extracting text-features...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        for i in [1,2,3,-3,-2,-1]:
            text_feature = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[CLASS_INDEX_INV[i]], device)
            text_feature_list.append(text_feature)

    save_score = 0.0
    losses_dict = {
        "focal" : [],
        "dice" : [],
        "bce" : [],
        "total" : []
        }

    for epoch in range(args.epoch):
        start_time = time.time()
        print(f"epoch: {epoch}", end=" - ")

        focal_loss_list = []
        dice_loss_list = []
        bce_loss_list = []

        loss_list = []

        for i, (image, image_label, mask, seg_idx) in enumerate(train_loader):

            print(seg_idx)

            image = image.squeeze(0).to(device)
            image_label = image_label.squeeze(0).to(device)
            mask = mask.squeeze(0).to(device)


            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)

                seg_patch_tokens = torch.cat([torch.unsqueeze(s[:, 1:, :], 0) for s in seg_patch_tokens]) # shape: 4 x B x 289 x 768
                det_patch_tokens = torch.cat([torch.unsqueeze(d[:, 1:, :], 0) for d in det_patch_tokens])
                seg_patch_tokens = torch.permute(seg_patch_tokens, (1, 0, 2, 3)) # shape: B x 4 x 289 x 768
                det_patch_tokens = torch.permute(det_patch_tokens, (1, 0, 2, 3))

                # image level
                det_patch_tokens = det_patch_tokens / det_patch_tokens.norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * det_patch_tokens @ text_feature_list[seg_idx]) # shape: B x 4 x 289 x 2
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, :, 1]
                anomaly_score = torch.mean(anomaly_map, dim=-1) # shape: B x 4

                det_loss = []
                for l in range(anomaly_score.shape[1]):
                    det_loss.append(loss_bce(anomaly_score[:, l], image_label))

                det_loss = torch.sum(torch.stack(det_loss))

                if seg_idx > 0: ######################## TODO: non parallelizable... seg_idx is not equal for all the batch
                    # pixel level
                    loss_f = []
                    loss_d = []
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

                    seg_patch_tokens = seg_patch_tokens / seg_patch_tokens.norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * seg_patch_tokens @ text_feature_list[seg_idx]) # shape: B x 4 x 289 x 2
                    anomaly_map = torch.softmax(anomaly_map, dim=3)[:,:,:,1] # shape: B x 4 x 289

                    B, L, P = anomaly_map.shape
                    H = int(np.sqrt(P))

                    anomaly_map = F.interpolate(anomaly_map.view(B, L, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True) # shape: B x 4 x H x H
                    
                    for l in range(anomaly_map.shape[1]):
                        loss_f.append(loss_focal(anomaly_map[:, l, :, :], mask[:, 0, :, :], reduction='mean'))
                        loss_d.append(loss_dice(anomaly_map[:, l, :, :], mask[:, 0, :, :]))

                    # losses
                    loss_f = torch.sum(torch.stack(loss_f))
                    loss_d = torch.sum(torch.stack(loss_d))
                    seg_loss = loss_d + loss_f

                    
                    # total loss
                    loss = seg_loss + det_loss

                    # store losses
                    bce_loss_list.append(det_loss.item())
                    focal_loss_list.append(loss_f.item())
                    dice_loss_list.append(loss_d.item())

                    # backward pass
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    # total loss
                    loss = det_loss

                    # store losses
                    bce_loss_list.append(det_loss.item())
                    focal_loss_list.append(0)
                    dice_loss_list.append(0)

                    # backward pass
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()
                    det_optimizer.step()

            if (i%1==0) : ## ADDED
                print(f"batch: {i}/{len(train_loader)} - loss: {loss.item():.5f}")

                
        loss_list.append(loss.item())

        #train_dataset.shuffle_dataset()
        #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

        # logs

        epoch_loss = np.mean(loss_list)
        print(f"TOT LOSS: {epoch_loss:.5f} - elapsed time: {time.time()-start_time:.2f}s")

        losses_dict["focal"].append(np.mean(focal_loss_list))
        losses_dict["dice"].append(np.mean(dice_loss_list))
        losses_dict["bce"].append(np.mean(bce_loss_list))
        losses_dict["total"].append(epoch_loss)


        # test on every epoch
        if (epoch > -1) : ## ADDED
            score = test(args, model, test_loader, text_feature_list[CLASS_INDEX[args.obj]])
            if score >= save_score:
                save_score = score
                if args.save_model == 1:
                    ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                    torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                                'det_adapters': model.det_adapters.state_dict()}, 
                                ckp_path)        
        
    with open(f"{args.log_dir}/{args.obj}-losses.csv", "w") as file:
        k = losses_dict.keys()
        file.write(f"epoch, {', '.join(k)}\n")
        for e in range(len(losses_dict["total"])):
            file.write(f"{e}, {losses_dict['focal'][e]}, {losses_dict['dice'][e]}, {losses_dict['bce'][e]}, {losses_dict['total'][e]}\n")


def test(args, seg_model, test_loader, text_features):
    start_time = time.time()
    print("**Running test:", end=" ")

    gt_list = []
    gt_mask_list = []
    image_scores = []
    segment_scores = []
    
    for _, (image, y, mask) in enumerate(test_loader):
        image = image.to(device)

        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, ori_seg_patch_tokens, ori_det_patch_tokens = seg_model(image)

            ori_seg_patch_tokens = torch.cat([torch.unsqueeze(s[:, 1:, :], 0) for s in ori_seg_patch_tokens]) # shape: 4 x B x 289 x 768
            ori_det_patch_tokens = torch.cat([torch.unsqueeze(d[:, 1:, :], 0) for d in ori_det_patch_tokens])
            ori_seg_patch_tokens = torch.permute(ori_seg_patch_tokens, (1, 0, 2, 3)) # shape: B x 4 x 289 x 768
            ori_det_patch_tokens = torch.permute(ori_det_patch_tokens, (1, 0, 2, 3))

            # image
            ori_det_patch_tokens = ori_det_patch_tokens / ori_det_patch_tokens.norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * ori_det_patch_tokens @ text_features) # shape: B x 4 x 289 x 2
            anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, :, 1]
            anomaly_score = torch.sum(torch.mean(anomaly_map, dim=-1), dim=-1) # shape: B

            image_scores.append(anomaly_score)

            # pixel
            ori_seg_patch_tokens = ori_seg_patch_tokens / ori_seg_patch_tokens.norm(dim=-1, keepdim=True)
            anomaly_map = (100.0 * ori_seg_patch_tokens @ text_features) # shape: B x 4 x 289 x 2
            anomaly_map = torch.softmax(anomaly_map, dim=3)[:,:,:,1] # shape: B x 4 x 289

            B, L, P = anomaly_map.shape
            H = int(np.sqrt(P))

            anomaly_map = F.interpolate(anomaly_map.view(B, L, H, H),
                                        size=args.img_size, mode='bilinear', align_corners=True) # shape: B x 4 x H x H
            final_score_map = torch.sum(anomaly_map, dim=1)
            
            gt_mask_list.append(mask.squeeze(1))
            gt_list.append(y)
            segment_scores.append(final_score_map)

    gt_list = torch.cat(gt_list).cpu().detach().numpy()
    gt_mask_list = torch.cat(gt_mask_list)
    gt_mask_list = (gt_mask_list>0).int().cpu().detach().numpy()

    segment_scores = torch.cat(segment_scores).cpu().detach().numpy()
    image_scores = torch.cat(image_scores).cpu().detach().numpy()

    segment_scores = (segment_scores - segment_scores.min()) / (segment_scores.max() - segment_scores.min())
    image_scores = (image_scores - image_scores.min()) / (image_scores.max() - image_scores.min())

    img_roc_auc_det = roc_auc_score(gt_list, image_scores)
    print(f"imageAUC : {round(img_roc_auc_det,4)} - elapsed time: {time.time()-start_time:.2f}s")

    #if CLASS_INDEX[args.obj] > 0:
    #    seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
    #    print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')
    #    return seg_roc_auc + img_roc_auc_det
    #else:
    #    return img_roc_auc_det

    # save AUROC and scores
    with open(f"{args.log_dir}/{args.obj}-scores.csv", "w") as file:
        file.write(f"test: imageAUROC={round(img_roc_auc_det, 5)}\n")
        file.write("img_score_zero\n")
        for i in range(image_scores.shape[0]):
            file.write(f"{image_scores[i]}\n")

    return img_roc_auc_det

if __name__ == '__main__':
    main()


