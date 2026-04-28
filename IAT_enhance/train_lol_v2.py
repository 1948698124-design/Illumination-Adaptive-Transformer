import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F

import os
import sys
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import vgg16

from data_loaders.lol import lowlight_loader
from model.IAT_main import IAT

from IQA_pytorch import SSIM
from utils import PSNR, adjust_learning_rate, validation, LossNetwork, visualization

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default=1)
parser.add_argument('--img_path', type=str, default="/data/unagi0/cui_data/light_dataset/LOL/Train/Low/")
parser.add_argument('--img_val_path', type=str, default="/data/unagi0/cui_data/light_dataset/LOL/Test/Low/")
parser.add_argument("--normalize", action="store_false", help="Default Normalize in LOL training.")
parser.add_argument('--model_type', type=str, default='s')
parser.add_argument('--no_gate', action='store_true', help='Disable illumination gate for ablation.')

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--pretrain_dir', type=str, default=None)

parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--display_iter', type=int, default=10)
parser.add_argument('--snapshots_folder', type=str, default="workdirs/snapshots_folder_lol")
parser.add_argument('--w_perceptual', type=float, default=0.04)
parser.add_argument('--w_rank', type=float, default=0.01)
parser.add_argument('--w_dark_tv', type=float, default=0.002)
parser.add_argument('--rank_margin', type=float, default=0.02)
parser.add_argument('--aux_start_epoch', type=int, default=20)

config = parser.parse_args()

print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

if not os.path.exists(config.snapshots_folder):
    os.makedirs(config.snapshots_folder)

# Model Setting
model = IAT(type=config.model_type, use_gate=not config.no_gate).cuda()
if config.pretrain_dir is not None:
    model.load_state_dict(torch.load(config.pretrain_dir))

# Data Setting
train_dataset = lowlight_loader(images_path=config.img_path, normalize=config.normalize)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True)
val_dataset = lowlight_loader(images_path=config.img_val_path, mode='test', normalize=config.normalize)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

# Loss & Optimizer Setting & Metric
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()

for param in vgg_model.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

# optimizer = torch.optim.Adam([{'params': model.global_net.parameters(),'lr':config.lr*0.1},
#             {'params': model.local_net.parameters(),'lr':config.lr}], lr=config.lr, weight_decay=config.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

device = next(model.parameters()).device
print('the device is:', device)

L1_loss = nn.L1Loss()
L1_smooth_loss = F.smooth_l1_loss

def rgb_to_luminance(img):
    if img.shape[1] >= 3:
        return 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]
    return img.mean(dim=1, keepdim=True)

def luminance_rank_loss(enhance_img, high_img, margin=0.02):
    y_pred = rgb_to_luminance(enhance_img)
    y_gt = rgb_to_luminance(high_img)
    pred_diff = y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:]
    gt_diff = y_gt[:, :, :, :-1] - y_gt[:, :, :, 1:]
    return F.relu(margin - pred_diff * gt_diff).mean()

def dark_region_tv_loss(enhance_img, low_img):
    dark_weight = (1.0 - rgb_to_luminance(low_img)).clamp(0.0, 1.0)
    grad_h = torch.abs(enhance_img[:, :, 1:, :] - enhance_img[:, :, :-1, :])
    grad_w = torch.abs(enhance_img[:, :, :, 1:] - enhance_img[:, :, :, :-1])
    weight_h = dark_weight[:, :, 1:, :]
    weight_w = dark_weight[:, :, :, 1:]
    return (grad_h * weight_h).mean() + (grad_w * weight_w).mean()

loss_network = LossNetwork(vgg_model)
loss_network.eval()

ssim = SSIM()
psnr = PSNR()
ssim_high = 0
psnr_high = 0

model.train()
print('######## Start IAT Training #########')
for epoch in range(config.num_epochs):
    # adjust_learning_rate(optimizer, epoch)
    print('the epoch is:', epoch)
    for iteration, imgs in enumerate(train_loader):
        low_img, high_img = imgs[0].cuda(), imgs[1].cuda()
        # Checking!
        #visualization(low_img, 'show/low', iteration)
        #visualization(high_img, 'show/high', iteration)
        optimizer.zero_grad()
        model.train()
        mul, add, enhance_img = model(low_img)

        rec_loss = L1_smooth_loss(enhance_img, high_img)
        perceptual_loss = loss_network(enhance_img, high_img)
        rank_loss = luminance_rank_loss(enhance_img, high_img, margin=config.rank_margin)
        dark_tv = dark_region_tv_loss(enhance_img, low_img)
        aux_scale = 1.0 if epoch >= config.aux_start_epoch else 0.0
        loss = rec_loss + config.w_perceptual * perceptual_loss + aux_scale * (
            config.w_rank * rank_loss + config.w_dark_tv * dark_tv
        )
        
        loss.backward()
        
        optimizer.step()
        scheduler.step()

        if ((iteration + 1) % config.display_iter) == 0:
            print(
                "Loss at iteration", iteration + 1, ":",
                "total=", round(loss.item(), 6),
                "rec=", round(rec_loss.item(), 6),
                "perc=", round(perceptual_loss.item(), 6),
                "rank=", round(rank_loss.item(), 6),
                "dark_tv=", round(dark_tv.item(), 6),
                "aux_scale=", aux_scale
            )

    # Evaluation Model
    model.eval()
    PSNR_mean, SSIM_mean = validation(model, val_loader)

    with open(config.snapshots_folder + '/log.txt', 'a+') as f:
        f.write('epoch' + str(epoch) + ':' + 'the SSIM is' + str(SSIM_mean) + 'the PSNR is' + str(PSNR_mean) + '\n')

    if SSIM_mean > ssim_high:
        ssim_high = SSIM_mean
        print('the highest SSIM value is:', str(ssim_high))
        torch.save(model.state_dict(), os.path.join(config.snapshots_folder, "best_Epoch" + '.pth'))

    f.close()
