#!/usr/bin/env python3
# Developed by Junyi Ma, Xieyuanli Chen, and Jun Zhang
# This file is covered by the LICENSE file in the root of the project OverlapTransformer:
# https://github.com/haomo-ai/OverlapTransformer/
# Brief: OverlapTransformer modules for KITTI sequences


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
import torch
import torch.nn as nn

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
# from tools.read_samples import read_one_need_from_seq
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_v2 import SwinTransformerV2
from swin_transformer_v2 import swin_transformer_v2_t, swin_transformer_v2_s

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, padding_mode='circular'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add skip connection
        out = self.relu(out)
        return out
    
class ResNetPanoramaCNNFeatureExtractor(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, num_blocks=[2, 2, 2, 2, 2]):
        super(ResNetPanoramaCNNFeatureExtractor, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.in_planes = 16
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        self.bn1 = norm_layer(16)
        self.relu = nn.ReLU(inplace=True)
        
        # Define ResNet blocks
        self.layer1 = self._make_layer(32, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(64, num_blocks[1], stride=1, norm_layer=norm_layer)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=1, norm_layer=norm_layer)
        self.layer4 = self._make_layer(128, num_blocks[3], stride=1, norm_layer=norm_layer)
        self.layer5 = self._make_layer(128, num_blocks[4], stride=1, norm_layer=norm_layer)

    def _make_layer(self, planes, num_blocks, stride, norm_layer):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, norm_layer))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        out = self.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        
        return out

    
class PatchEmbedding(nn.Module):
    def __init__(self, in_chan, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = embed_dim
        self.project = nn.Conv2d(in_chan, self.emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_size))

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 패치 수 계산 (이미지 크기가 가변적이므로 동적 계산)
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Conv2d로 패치 임베딩
        x = self.project(x)  # [B, emb_size, num_patches_h, num_patches_w]
        
        # 패치 수에 맞게 텐서 변형
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, emb_size]
        
        # cls_token을 반복하여 배치 크기에 맞춤
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, emb_size]
        
        # cls_token과 패치 임베딩 결합
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, emb_size]
        
        # 위치 임베딩 (num_patches에 맞춰 동적으로 생성)
        positions = torch.randn(num_patches + 1, self.emb_size, device=x.device)
        x += positions  # [B, num_patches + 1, emb_size]

        return x

class OverlapTransformerSwinT(nn.Module):
    def __init__(self, height=64, width=900, patch_size=7, channels=1, embed_dim=128, num_layers=1, num_heads=4):
        super(OverlapTransformerSwinT, self).__init__()
        self.transformer_encoder = SwinTransformerV2(input_resolution=(height, width),
                                                        window_size=patch_size,
                                                        in_channels=channels,
                                                        use_checkpoint=False,
                                                        sequential_self_attention=False,
                                                        embedding_channels=96,
                                                        depths=(2, 2, 6, 2),
                                                        number_of_heads=(3, 6, 12, 24))
        # self.transformer_encoder = swin_transformer_v2_t(in_channels=channels,
        #                                                         window_size=4,
        #                                                         input_resolution=(height, width),
        #                                                         sequential_self_attention=False,
        #                                                         use_checkpoint=False)
        
        self.net_vlad = NetVLADLoupe(feature_size=256, max_samples=225, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

    def forward(self, x_l):

        # 트랜스포머에 입력
        print(x_l.shape) # torch.Size([6, 1, 64, 900])
        out_l = self.transformer_encoder(x_l)  # [B, 패치 개수, 임베딩 차원]
        print(out_l.shape) # torch.Size([6, 225, 128])

        # 최종 출력 처리
        out_l = F.normalize(out_l, dim=2)
        out_l = out_l.unsqueeze(3)
        print(out_l.shape) # torch.Size([6, 225, 256, 1]) but should be torch.Size([6, 1024, 225, 1])
        out_l = out_l.permute(0, 2, 1, 3)
        print(out_l.shape) # torch.Size([6, 256, 225, 1]) but should be torch.Size([6, 1024, 225, 1])

        out_l = self.net_vlad(out_l)
        print(out_l.shape) # torch.Size([6, 256])
        out_l = F.normalize(out_l, dim=1)

        return out_l