import os
import time
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim_parallel, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#import warnings
#warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help="ViT-B-16-plus-240, ViT-L-14-336")
    parser.add_argument('--pretrain', type=str, default='openai', help="laion400m, openai")
    parser.add_argument('--obj', type=str, default='Liver')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--support_batch_size', type=int, default=1)
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt_r/few-shot/')
    parser.add_argument('--log_dir', type=str, default='./logs/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    parser.add_argument('--only_normal', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    print(f"Using device: {device}")
    print(f"OBJECT: {args.obj}")
    
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

    milestones = []
    gamma = 0.1
    seg_scheduler = torch.optim.lr_scheduler.MultiStepLR(seg_optimizer, milestones=milestones, gamma=gamma)
    det_scheduler = torch.optim.lr_scheduler.MultiStepLR(det_optimizer, milestones=milestones, gamma=gamma)

    # load test dataset
    print("Loading dataset...")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate, args.only_normal)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    print(f"Dataset size: {len(test_dataset)}")

    # few-shot image augmentation
    print("Applying augmentations...")
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)
    augment_abnorm_img, augment_abnorm_mask = [augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
                                               if not args.only_normal else torch.tensor([]), torch.tensor([])]


    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    print(f"Post-augmentation dataset size: {len(train_dataset)}")

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.support_batch_size, shuffle=True, **kwargs)
    print(f"Support (memory bank) dataset size: {len(support_dataset)}")

    # losses
    loss_focal = sigmoid_focal_loss
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    print("Extracting text-features...")
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device) # shape: 768 x 2 

    losses_dict = {
        "focal" : [],
        "dice" : [],
        "bce" : [],
        "total" : []
        }
    
    print("Starting training")
    for epoch in range(args.epoch):
        start_time = time.time()
        print(f"epoch: {epoch}", end=" - ")
        
        focal_loss_list = []
        dice_loss_list = []
        bce_loss_list = []
        tot_loss_list = []

        for i, (image, mask, label) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            label = label.to(device)

            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)

                seg_patch_tokens = torch.cat([torch.unsqueeze(s[:, 1:, :], 0) for s in seg_patch_tokens]) # shape: 4 x B x 289 x 768
                det_patch_tokens = torch.cat([torch.unsqueeze(d[:, 1:, :], 0) for d in det_patch_tokens])
                seg_patch_tokens = torch.permute(seg_patch_tokens, (1, 0, 2, 3)) # shape: B x 4 x 289 x 768
                det_patch_tokens = torch.permute(det_patch_tokens, (1, 0, 2, 3))

                # 1. IMAGE LEVEL LOSS
                det_patch_tokens = det_patch_tokens / det_patch_tokens.norm(dim=-1, keepdim=True) # shape: B x 4 x 289 x 768

                # multiply DET tokens with text features
                anomaly_map = 100 * det_patch_tokens @ text_features # shape: B x 4 x 289 x 2
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, :, 1] # shape: B x 4 x 289
                anomaly_score = torch.mean(anomaly_map, dim=-1) # shape: B x 4

                # det loss (bce)
                bce_loss = 0 
                for l in range(anomaly_score.shape[1]):
                    bce_loss += loss_bce(anomaly_score[:, l], label)


                if CLASS_INDEX[args.obj] > 0:
                # 2. PIXEL LEVEL LOSS
                    seg_patch_tokens = seg_patch_tokens / seg_patch_tokens.norm(dim=-1, keepdim=True)

                    # multiply SEG tokens with text features
                    anomaly_map = (100.0 * seg_patch_tokens @ text_features)
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, :, 1] # shape: B x 4 x 289

                    
                    # resize/reshape anomaly_map to image size
                    B, L, P = anomaly_map.shape
                    H = int(np.sqrt(P))
                    anomaly_map = F.interpolate(anomaly_map.view(B, L, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True) # shape: B* x 4 x H x H

                    # binarize mask (keeping into account nan values)
                    mask[mask>0.5], mask[mask<=0.5] = 1, 0

                    # seg loss (focal + dice)     
                    focal_loss = 0
                    dice_loss = 0
                    for l in range(anomaly_map.shape[1]):
                        focal_loss = focal_loss + loss_focal(anomaly_map[:, l, :, :], mask[:, 0, :, :], alpha=-1, gamma=0.5, reduction='mean')
                        dice_loss = dice_loss + loss_dice(anomaly_map[:, l, :, :], mask[:, 0, :, :])

                    # total loss
                    tot_loss = focal_loss + dice_loss + bce_loss
                    
                    # store losses
                    bce_loss_list.append(bce_loss.item())
                    focal_loss_list.append(focal_loss.item())
                    dice_loss_list.append(dice_loss.item())
                    tot_loss_list.append(tot_loss.item())

                    # backward pass
                    tot_loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    tot_loss.backward()
                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    # tot loss
                    tot_loss = bce_loss

                    # store losses
                    bce_loss_list.append(bce_loss.item())
                    focal_loss_list.append(np.nan)
                    dice_loss_list.append(np.nan)
                    tot_loss_list.append(tot_loss.item())

                    # backward pass
                    tot_loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    tot_loss.backward()
                    det_optimizer.step()

        # lr scheduler update
        seg_scheduler.step()
        det_scheduler.step()

        # logs
        losses_dict["focal"].append(np.mean(focal_loss_list))
        losses_dict["dice"].append(np.mean(dice_loss_list))
        losses_dict["bce"].append(np.mean(bce_loss_list))
        losses_dict["total"].append(np.mean(tot_loss_list))
        print(f"EPOCH LOSS: {losses_dict['total'][-1]:.8f} - LR {det_scheduler.get_last_lr()[0]:.6f} - FOCAL LOSS: {losses_dict['focal'][-1]:.8f} - DICE LOSS: {losses_dict['dice'][-1]:.8f} - BCE LOSS: {losses_dict['bce'][-1]:.8f} - elapsed time: {time.time()-start_time:.2f}s")


        # memory bank branch
        seg_features = []
        det_features = []
        for image, in support_loader:
            image = image.to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image) # lists (len=4) of shape: (B x 290 x 768)
                seg_patch_tokens = torch.stack(seg_patch_tokens) # shape: 4 x B x 290 x 768
                det_patch_tokens = torch.stack(det_patch_tokens)
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        seg_mem_features = torch.cat(seg_features, dim=1) # shape: 4 x Nimgs x 290 x 768
        det_mem_features = torch.cat(det_features, dim=1)
        seg_mem_features = torch.flatten(seg_mem_features, start_dim=1, end_dim=2) # shape: 4 x (290Nimgs) x 768
        det_mem_features = torch.flatten(det_mem_features, start_dim=1, end_dim=2)

        if epoch > -1:
            result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
            #if result > best_result:
            #    best_result = result
            #    if args.save_model == 1:
            #        ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
            #        torch.save({'seg_adapters': model.seg_adapters.state_dict(),
            #                    'det_adapters': model.det_adapters.state_dict()}, 
            #                    ckp_path)
    
    #with open(f"{args.log_dir}/{args.obj}-losses.csv", "w") as file:
    #    k = losses_dict.keys()
    #    file.write(f"epoch, {', '.join(k)}\n")
    #    for e in range(len(losses_dict["total"])):
    #        file.write(f"{e}, {losses_dict['focal'][e]}, {losses_dict['dice'][e]}, {losses_dict['bce'][e]}, {losses_dict['total'][e]}\n")

          


def test(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    start_time = time.time()

    print("Running test...", end=" ")

    gt_list = []
    gt_mask_list = []
    
    det_image_scores_zero = []
    det_image_scores_few = []
    
    seg_score_map_zero = []
    seg_score_map_few= []


    for i, (image, y, mask) in enumerate(test_loader):
        B = image.shape[0]
        image = image.to(device)
        
        # binarize mask 
        mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = torch.stack([s[:, 1:, :] for s in seg_patch_tokens]) # shape: 4 x B x 289 x 768
            det_patch_tokens = torch.stack([d[:, 1:, :] for d in det_patch_tokens])


            if CLASS_INDEX[args.obj] > 0:
                # FEW-SHOT / SEG HEAD
                anomaly_maps_few_shot = []
                
                for idx, patch in enumerate(seg_patch_tokens): # patch.shape: B x 289 x 768
                    # fix tensors shapes
                    mem = seg_mem_features[idx]  # shape: (290Nimgs) x 768

                    # cosine similarity --> distance
                    cos = cos_sim_parallel(mem, patch) # shape: B x 289 x (290Nimgs)
                    cos = torch.min((1 - cos), dim=2)[0] # shape: B x 289

                    # reshape/resize to image size
                    H = int(np.sqrt(cos.shape[1]))
                    anomaly_map_few_shot = cos.reshape(B, 1, H, H) # need 2nd dim for interpolation (4D input)
                    anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                                      size=args.img_size, mode='bilinear', align_corners=True)     # shape: B x 1 x size x size
                    anomaly_maps_few_shot.append(anomaly_map_few_shot)
                anomaly_maps_few_shot = torch.cat(anomaly_maps_few_shot, dim=1) # shape: B x 4 x size x size
                score_map_few = np.sum(anomaly_maps_few_shot.cpu().numpy(), axis=1) # shape: B x size x size
                seg_score_map_few.append(score_map_few)


                # ZERO-SHOT / SEG HEAD
                anomaly_maps = []
                for idx, patch in enumerate(seg_patch_tokens):
                    # normalization
                    patch /= patch.norm(dim=-1, keepdim=True) 
                    # multiply SEG tokens with text features
                    anomaly_map = (100.0 * patch @ text_features) # shape: B x 289 x 2
                    # reshape/resize to image size
                    B, L, P = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=args.img_size, mode='bilinear', align_corners=True) # shape: B x 2 x size x size
                    anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :] # shape B x size x size
                    anomaly_maps.append(anomaly_map)
                anomaly_maps = torch.stack(anomaly_maps) # shape: 4 x B x size x size
                anomaly_maps = torch.permute(anomaly_maps, (1, 0, 2, 3)) # shape: B x 4 x size x size
                score_map_zero = np.sum(anomaly_maps.cpu().numpy(), axis=1) # shape: B x size x size
                seg_score_map_zero.append(score_map_zero)
                

            else:
                # FEW-SHOT / DET HEAD
                score_few_det = np.zeros(B)
                for idx, patch in enumerate(det_patch_tokens): # patch.shape: B x 289 x 768
                    # fix tensors shapes
                    mem = det_mem_features[idx]  # shape: (290Nimgs) x 768

                    # cosine similarity --> distance
                    cos = cos_sim_parallel(mem, patch) # shape: B x 289 x (290Nimgs)
                    cos = torch.min((1 - cos), dim=2)[0] # shape: B x 289

                    score_few_det += torch.mean(cos, dim=1).cpu().numpy() # shape: B
                det_image_scores_few.append(score_few_det)

                # ZERO-SHOT / DET HEAD
                anomaly_score = np.zeros(B)
                for idx, patch in enumerate(det_patch_tokens): # patch.shape: B x 289 x 768
                    # normalization
                    patch /= patch.norm(dim=-1, keepdim=True)
                    # multiply DET tokens with text features
                    anomaly_map = (100.0 * patch @ text_features) # shape: B x 289 x 2
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1] # shape: B x 289
                    # average
                    anomaly_score += torch.mean(anomaly_map, dim=1).cpu().numpy()
                det_image_scores_zero.append(anomaly_score)

            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.append(y.cpu().detach().numpy())

    gt_list = np.concatenate(gt_list, axis=0)
    gt_mask_list = np.concatenate(gt_mask_list, axis=0)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 0:
        seg_score_map_zero = np.concatenate(seg_score_map_zero, axis=0)
        seg_score_map_few = np.concatenate(seg_score_map_few, axis=0)


        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        print(f'pAUC = {round(seg_roc_auc,4)}', end=" - ")

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'AUC = {round(roc_auc_im, 4)}', end=" - ")

        print(f"elapsed time: {time.time()-start_time:.2f}s")
        return seg_roc_auc + roc_auc_im

        # save AUROC and scores
        #with open(f"{args.log_dir}/{args.obj}-scores.csv", "w") as file:
        #    file.write(f"AUROC: {round(roc_auc_im, 5)}\n")
        #    file.write("img_score_zero, img_score_few\n")
        #    for i in range(seg_score_map_zero.shape[0]):
        #        file.write(f"{np.max(seg_score_map_zero[i])}, {np.max(seg_score_map_few[i])}\n")
        

    else:
        det_image_scores_zero = np.concatenate(det_image_scores_zero, axis=0)
        det_image_scores_few = np.concatenate(det_image_scores_few, axis=0)


        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'AUC = {round(img_roc_auc_det,4)}', end=" - ")
        
        #save AUROC and scores
        #with open(f"{args.log_dir}/{args.obj}-scores.csv", "w") as file:
        #    file.write(f"AUROC: {round(img_roc_auc_det, 5)}\n")
        #    file.write("img_score_zero, img_score_few\n")
        #    for i in range(image_scores.shape[0]):
        #        file.write(f"{det_image_scores_zero[i]}, {det_image_scores_few[i]}\n")
        
        print(f"elapsed time: {time.time()-start_time:.2f}s")
        return img_roc_auc_det



if __name__ == '__main__':
    main()


