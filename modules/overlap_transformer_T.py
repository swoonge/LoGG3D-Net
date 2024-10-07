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


class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, embed_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        # CNN 블록: 간단한 CNN 네트워크로 피처를 추출
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)  # 최종 임베딩 차원으로 변환
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 최종 임베딩 차원으로 변환
        self.conv5 = nn.Conv2d(256, embed_dim, kernel_size=3, stride=1, padding=1)  # 최종 임베딩 차원으로 변환
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN으로 피처 추출
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))  
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x)) # [B, embed_dim, H, W]
        return x
    
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

class OverlapTransformerViT(nn.Module):
    def __init__(self, height=64, width=900, patch_size=8, channels=1, embed_dim=256, num_layers=1, num_heads=4):
        super(OverlapTransformerViT, self).__init__()
        self.cnn_feature_extractor = CNNFeatureExtractor(in_channels=channels, embed_dim=embed_dim)
        self.patch_embedding = PatchEmbedding(in_chan=256, patch_size=patch_size, embed_dim=embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=1024, activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.net_vlad = NetVLADLoupe(feature_size=512, max_samples=225, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

    def forward(self, x_l):
        # CNN으로 피처 추출
        out_l = self.cnn_feature_extractor(x_l)  # [B, embed_dim, H, W]
        # print(out_l.shape) # torch.Size([6, 256, 64, 900])
        
        # 패치 임베딩
        out_l = self.patch_embedding(out_l)  # [B, 패치 개수, 임베딩 차원]
        # print(out_l.shape) # torch.Size([6, 225, 256])

        # 트랜스포머에 입력
        out_l_1 = self.transformer_encoder(out_l)  # [B, 패치 개수, 임베딩 차원]
        # print(out_l_1.shape) # torch.Size([6, 225, 256])
        
        out_l = torch.cat((out_l_1, out_l), dim=2)
        # print(out_l.shape) # torch.Size([6, 225, 256*2])

        # 최종 출력 처리
        out_l = F.normalize(out_l, dim=2)
        out_l = out_l.unsqueeze(3)
        # print(out_l.shape) # torch.Size([6, 225, 512, 1]) but should be torch.Size([6, 1024, 225, 1])
        out_l = out_l.permute(0, 2, 1, 3)
        # print(out_l.shape) # torch.Size([6, 512, 225, 1]) but should be torch.Size([6, 1024, 225, 1])

        out_l = self.net_vlad(out_l)
        # print(out_l.shape) # torch.Size([12, 256])
        out_l = F.normalize(out_l, dim=1)

        return out_l