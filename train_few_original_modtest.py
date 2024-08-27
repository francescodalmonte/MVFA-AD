import os
import time # added
import pickle # added
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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

import warnings
warnings.filterwarnings("ignore")

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
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--log_dir', type=str, default='./logs/few-shot/') # added
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=4)
    parser.add_argument('--iterate', type=int, default=0)
    args = parser.parse_args()

    setup_seed(args.seed)

    print(f"OBJECT: {args.obj}")

    # fixed feature extractor
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device, pretrained=args.pretrain, require_pretrained=True)
    clip_model.eval()

    model = CLIP_Inplanted(clip_model=clip_model, features=args.features_list).to(device)
    model.eval()

    # changed this part: in original paper they were setting .requires_grad = True for all parameters of the model
    for name, param in model.named_parameters():
        if "seg_adapters" in name or "det_adapters" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    seg_scheduler = torch.optim.lr_scheduler.MultiStepLR(seg_optimizer, [999], gamma=0.25)
    det_scheduler = torch.optim.lr_scheduler.MultiStepLR(det_optimizer, [999], gamma=0.25)


    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)
    
    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs) # modified: in original batch_size=1
    print('train_dataset len:', len(train_dataset))
    print('train_loader len:', len(train_loader))

    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss(alpha=0.25, gamma=2.)
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    logs = {"epoch": [],
            "tot_loss": [],
            "focal": [],
            "dice": [],
            "bce": [],
            "gradients": {name: [] for name, params in model.named_parameters() if(params.requires_grad) and ("bias" not in name)}
            }
    for epoch in range(args.epoch):
        print('epoch ', epoch, ':')

        loss_list = []
        focal_list = [] # added
        dice_list = [] # added
        bce_list = [] # added
        gradients = {name: [] for name, params in model.named_parameters() if(params.requires_grad) and ("bias" not in name)} # added
        for (image, gt, label) in train_loader:
            image = image.to(device)
            with torch.cuda.amp.autocast():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[:, 1:, :] for p in seg_patch_tokens] # modified: in original p[0, 1:, :]
                det_patch_tokens = [p[:, 1:, :] for p in det_patch_tokens] # modified: in original p[0, 1:, :]
                    
                # det loss
                det_loss = 0
                image_label = label.to(device)
                for layer in range(len(det_patch_tokens)):
                    det_patch_tokens[layer] = det_patch_tokens[layer] / det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features)#.unsqueeze(0) removed from original --> not working     
                    anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                    anomaly_score = torch.mean(anomaly_map, dim=-1)
                    det_loss += loss_bce(anomaly_score, image_label)

                if CLASS_INDEX[args.obj] > 0:
                    # pixel level
                    lfocal = 0 # added
                    ldice = 0 # added
                    mask = gt.squeeze(0).to(device)
                    mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                    for layer in range(len(seg_patch_tokens)):
                        seg_patch_tokens[layer] = seg_patch_tokens[layer] / seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                        anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features)#.unsqueeze(0) removed from original --> not working    
                        B, L, C = anomaly_map.shape
                        H = int(np.sqrt(L))
                        anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                    size=args.img_size, mode='bilinear', align_corners=True)
                        anomaly_map = torch.softmax(anomaly_map, dim=1)
                        lfocal += loss_focal(anomaly_map, mask)
                        ldice += loss_dice(anomaly_map[:, 1, :, :], mask)
                    seg_loss = ldice + lfocal
                    
                    loss = seg_loss + det_loss
                    loss.requires_grad_(True)
                    seg_optimizer.zero_grad()
                    det_optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # added

                    # Added
                    for name, params in model.named_parameters():
                        if(params.requires_grad) and ("bias" not in name):
                            if params.grad is not None:
                                gradients[name].append([params.grad.abs().mean().cpu().numpy(), params.grad.abs().max().cpu().numpy()])


                    seg_optimizer.step()
                    det_optimizer.step()

                else:
                    lfocal = np.nan # added
                    ldice = np.nan # added
                    loss = det_loss
                    loss.requires_grad_(True)
                    det_optimizer.zero_grad()
                    loss.backward()

                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # added

                    # Added
                    for name, params in model.named_parameters():
                        if(params.requires_grad) and ("bias" not in name):
                            if params.grad is not None:
                                gradients[name].append([params.grad.abs().mean().cpu().numpy(), params.grad.abs().max().cpu().numpy()])

                    det_optimizer.step()

                # save losses
                loss_list.append(loss.item())
                if lfocal==lfocal and ldice==ldice:
                    focal_list.append(lfocal.item()) # added
                    dice_list.append(ldice.item()) # added
                else:
                    focal_list.append(np.nan)
                    dice_list.append(np.nan)
                bce_list.append(det_loss.item()) # added
            
        # lr scheduler
        seg_scheduler.step()
        det_scheduler.step()


        # save logs
        logs["epoch"].append(epoch)
        logs["tot_loss"].append(np.mean(loss_list))
        logs["focal"].append(np.mean(focal_list))
        logs["dice"].append(np.mean(dice_list))
        logs["bce"].append(np.mean(bce_list))
        for name, params in model.named_parameters():
            if(params.requires_grad) and ("bias" not in name):
                gradients[name] = np.array(gradients[name])
                if len(gradients[name]) > 0:
                    logs["gradients"][name].append((gradients[name][:, 0].mean(), gradients[name][:, 1].mean()))
                
        print(f"LOSS: {np.mean(loss_list)} - LR: {seg_scheduler.get_last_lr()[0]:.6f}/{det_scheduler.get_last_lr()[0]:.6f} - FOCAL: {np.mean(focal_list)} - DICE: {np.mean(dice_list)} - BCE: {np.mean(bce_list)}") # modified



        seg_features = []
        det_features = []
        for image in support_loader:
            image = image[0].to(device)
            with torch.no_grad():
                _, seg_patch_tokens, det_patch_tokens = model(image)
                seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
                det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
                seg_features.append(seg_patch_tokens)
                det_features.append(det_patch_tokens)
        seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
        det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]
        

        #result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)
        if args.save_model == 1:
            if epoch % 1 == 0:
                ckp_path = os.path.join(args.save_path, f'{args.obj}-{epoch}.pth')
                torch.save({'seg_adapters': model.seg_adapters.state_dict(),
                            'det_adapters': model.det_adapters.state_dict()}, 
                            ckp_path)
    
    # save logs to file (added)
    with open(f"{args.log_dir}/{args.obj}-logs.pkl", "wb") as file:
        pickle.dump(logs, file)
          


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
                score_map_zero = np.sum(anomaly_maps.cpu().numpy(), axis=1)
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
                    anomaly_score += torch.mean(anomaly_map, dim=1).cpu().numpy() # shape: B
                det_image_scores_zero.append(anomaly_score)

            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.append(y.cpu().detach().numpy())

    gt_list = np.concatenate(gt_list, axis=0)
    gt_mask_list = np.concatenate(gt_mask_list, axis=0)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)


    if CLASS_INDEX[args.obj] > 999: # TODO change to 0
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


