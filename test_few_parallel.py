import os
import time
import argparse
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
#from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from CLIP.clip import create_model
from CLIP.tokenizer import tokenize
from CLIP.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble, cos_sim_parallel
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print(f"Running on: {device}")

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
    parser.add_argument('--ckpt_path', type=str, default='./ckpt/few-shot/')
    parser.add_argument('--ckpt_epoch', type=str, default=0)
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

    # for single epoch test
#    checkpoint = torch.load(os.path.join(f'{args.ckpt_path}', f'{args.obj}-{args.ckpt_epoch}.pth'))
#    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
#    model.det_adapters.load_state_dict(checkpoint["det_adapters"])

    for name, param in model.named_parameters():
        param.requires_grad = True

    # optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))



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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[args.obj], device)

    best_result = 0


    # for single epoch test
#    result = test(args, model, test_loader, text_features, seg_mem_features, det_mem_features)

    # for multiple epochs test
    for epoch in range(0, 150, 5):
        checkpoint = torch.load(os.path.join(f'{args.ckpt_path}', f'{args.obj}-{epoch}.pth'), map_location=device)
        model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
        model.det_adapters.load_state_dict(checkpoint["det_adapters"])

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
    

        print(f"epoch {epoch}", end=" - ")
        _ = test_parallel(args, model, test_loader, text_features, seg_mem_features, det_mem_features)    



def test_parallel(args, model, test_loader, text_features, seg_mem_features, det_mem_features):
    """
    Just like test() but with parallel processing
    """
    start_time = time.time()

    print(f"Running test (parallel)...", end=" ")

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
                                                                      size=args.img_size, mode='bilinear', align_corners=True) # shape: B x 1 x size x size
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
                    anomaly_score += torch.mean(anomaly_map, dim=1).cpu().numpy() # shape: B
                det_image_scores_zero.append(anomaly_score)

            
            gt_mask_list.append(mask.squeeze().cpu().detach().numpy())
            gt_list.append(y.cpu().detach().numpy())

    gt_list = np.concatenate(gt_list, axis=0)
    gt_mask_list = np.concatenate(gt_mask_list, axis=0)
    gt_mask_list = (gt_mask_list>0).astype(np.int_)
    print(f"Elapsed time (anomaly scores): {time.time() - start_time:.2f}s")


    if CLASS_INDEX[args.obj] > 0:
        start_time = time.time()

        seg_score_map_zero = np.concatenate(seg_score_map_zero)
        seg_score_map_few = np.concatenate(seg_score_map_few)

        seg_score_map_zero = (seg_score_map_zero - seg_score_map_zero.min()) / (seg_score_map_zero.max() - seg_score_map_zero.min())
        seg_score_map_few = (seg_score_map_few - seg_score_map_few.min()) / (seg_score_map_few.max() - seg_score_map_few.min())
    
        segment_scores = 0.5 * seg_score_map_zero + 0.5 * seg_score_map_few
        
        #seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), segment_scores.flatten())
        #print(f'{args.obj} pAUC : {round(seg_roc_auc,4)}')

        segment_scores_flatten = segment_scores.reshape(segment_scores.shape[0], -1)
        roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')

        print(f"Elapsed time (AUC/pAUC scores): {time.time() - start_time:.2f}s")

        return roc_auc_im #+ seg_roc_auc

    else:
        start_time = time.time()

        det_image_scores_zero = np.concatenate(det_image_scores_zero)
        det_image_scores_few = np.concatenate(det_image_scores_few)

        det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / (det_image_scores_zero.max() - det_image_scores_zero.min())
        det_image_scores_few = (det_image_scores_few - det_image_scores_few.min()) / (det_image_scores_few.max() - det_image_scores_few.min())
    
        image_scores = 0.5 * det_image_scores_zero + 0.5 * det_image_scores_few
        img_roc_auc_det = roc_auc_score(gt_list, image_scores)
        print(f'{args.obj} AUC : {round(img_roc_auc_det,4)}')


        print(f"Elapsed time (AUC scores): {time.time() - start_time:.2f}s")


        return img_roc_auc_det
    


if __name__ == '__main__':
    main()


