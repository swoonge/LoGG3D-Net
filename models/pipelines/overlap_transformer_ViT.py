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

from aggregators.netvlad import NetVLADLoupe
import torch.nn.functional as F
# from tools.read_samples import read_one_need_from_seq
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SimpleViT(nn.Module):
    def __init__(self, *, image_height, image_width, patch_size, dim = 256, depth, heads, mlp_dim, channels = 1, dim_head = 64):
        super().__init__()
        patch_height, patch_width = pair(patch_size)
        print(patch_size, patch_height, patch_width, image_height, image_width)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)


    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        # print("x0", x.shape)
        # x += self.pos_embedding.to(device, dtype=x.dtype)

        # print("x1", x.shape)
        x = self.transformer(x)
        # print("x2", x.shape)

        return x
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class OverlapTransformerViT_torch(nn.Module):
    def __init__(self, height=64, width=1024, patch_size=8, channels=1, embed_dim=128, num_layers=1, num_heads=4):
        super(OverlapTransformerViT_torch, self).__init__()
        #     def __init__(self, *, image_height, image_width, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        self.transformer_encoder = SimpleViT(image_height=height,
                                            image_width=width,
                                            patch_size = 8,
                                            dim = 256,
                                            depth = 6,
                                            heads = 16,
                                            mlp_dim = 512,
                                            channels = channels,
                                            dim_head=64)

        self.net_vlad = NetVLADLoupe(feature_size=256, max_samples=1024, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

    def forward(self, x_l):
        # 트랜스포머에 입력
        # print("x_l", x_l.shape) # torch.Size([6, 1, 64, 1024])
        out_l = self.transformer_encoder(x_l)  # [B, 패치 개수, 임베딩 차원]
        # print(out_l.shape) # ([6, 1024, 256]) [B, 패치 개수, 임베딩 차원]

        # 최종 출력 처리
        out_l = F.normalize(out_l, dim=2)
        out_l = out_l.unsqueeze(3)
        # print(out_l.shape) # torch.Size([6, 225, 256, 1]) but should be torch.Size([6, 1024, 225, 1])
        out_l = out_l.permute(0, 2, 1, 3)
        # print(out_l.shape) # torch.Size([6, 256, 225, 1]) but should be torch.Size([6, 1024, 225, 1])

        out_l = self.net_vlad(out_l)
        # print(out_l.shape) # torch.Size([6, 256])
        out_l = F.normalize(out_l, dim=1)

        return out_l