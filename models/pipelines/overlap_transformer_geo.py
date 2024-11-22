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
import numpy as np

@torch.no_grad()
def compute_cosine_similarity_matrix(tensor, eps=1e-8):
    # tensor: (batch_size, num_queries, feature_dim)
    
    # L2 정규화 (각 쿼리 벡터를 단위 벡터로 변환)
    tensor_norm = tensor / (tensor.norm(dim=-1, keepdim=True) + eps)
    
    # 코사인 유사도 계산: (batch_size, num_queries, num_queries)
    cosine_similarity_matrix = torch.bmm(tensor_norm, tensor_norm.transpose(1, 2))
    
    return cosine_similarity_matrix

@torch.no_grad()
def compute_l2distance_matrix(tensor, eps=1e-8):
    # tensor: (batch_size, num_queries, feature_dim)
    
    # (batch_size, num_queries, 1, feature_dim)로 확장하여 L2 거리 계산
    tensor_1 = tensor.unsqueeze(2)
    tensor_2 = tensor.unsqueeze(1)
    
    # (batch_size, num_queries, num_queries)에서 L2 거리 계산
    l2_distance_matrix = torch.norm(tensor_1 - tensor_2, dim=-1)

    # 수치적 안정성을 위한 epsilon 추가
    l2_distance_matrix = l2_distance_matrix + eps
    
    return l2_distance_matrix

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(LearnablePositionalEncoding, self).__init__()
        # 포지션을 위한 학습 가능한 임베딩 레이어
        self.position_embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        # 입력 x의 형태: (batch_size, d_model, seq_len) -> [6, 256, 900]
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, d_model) -> [6, 900, 256]
        batch_size, seq_len, _ = x.size()  # [6, 900, 256]

        # 각도 기반 포지션 인덱스 생성
        position_ids = self.angle_position(x)  # [6, 900]

        # 포지셔널 임베딩 생성
        position_embeddings = self.position_embedding(position_ids)  # [6, 900, 256]

        # 입력 데이터와 포지셔널 임베딩 더하기
        x = x + position_embeddings  # [6, 900, 256]

        # 원래 형태로 복원 (batch_size, d_model, seq_len)
        return x.permute(0, 2, 1)  # [6, 256, 900]

    @torch.no_grad()
    def angle_position(self, tensor):
        # tensor: (batch_size, num_queries, feature_dim)
        batch_size, num_queries, feature_dim = tensor.shape
        
        # 코사인 유사도 계산: (batch_size, num_queries, num_queries)
        cosine_similarity_matrix = compute_l2distance_matrix(tensor)
        
        # 자기 자신과의 유사도를 무한대로 설정하여 제외
        eye_mask = torch.eye(num_queries, device=tensor.device, dtype=cosine_similarity_matrix.dtype).unsqueeze(0)
        cosine_similarity_matrix = cosine_similarity_matrix + eye_mask * float('inf')
        
        # 가장 유사한 (가장 가까운) 쿼리의 인덱스를 찾음
        most_similar_indices = torch.argmin(cosine_similarity_matrix, dim=-1)
        
        # 현재 인덱스와 most_similar_indices 간의 각도 차이 계산
        current_indices = torch.arange(num_queries, device=tensor.device).unsqueeze(0)  # (1, num_queries)
        angles = torch.abs(current_indices - most_similar_indices)  # (batch_size, num_queries)
        return angles  # [batch_size, num_queries]
    

class OverlapTransformer_geo(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer = True):
        super(OverlapTransformer_geo, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,1), stride=(1,1), bias=False)
        self.bn1 = norm_layer(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), bias=False)
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,1), stride=(2,1), bias=False)
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn7 = norm_layer(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn8 = norm_layer(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn9 = norm_layer(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn10 = norm_layer(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.bn11 = norm_layer(128)
        self.relu = nn.ReLU(inplace=True)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        self.learnable_pos_enc = LearnablePositionalEncoding(900, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.proj_a = nn.Linear(256, 256)

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        """TODO: How about adding some dense layers?"""
        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, x_l):

        out_l = self.relu(self.conv1(x_l)) # [batch, feature, H, W]
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))
        out_l = self.relu(self.conv8(out_l))
        out_l = self.relu(self.conv9(out_l))
        out_l = self.relu(self.conv10(out_l))
        out_l = self.relu(self.conv11(out_l)) # [batch, feature, H, W] [6, 128, 1, 900]

        out_l = out_l.permute(0,1,3,2) # [6, 128, 900, 1]
        out_l = self.relu(self.convLast1(out_l)) # [6, 256, 900, 1]

        """Using transformer needs to decide whether batch_size first"""
        if self.use_transformer:
            out_l = out_l.squeeze(3) # [6, 256, 900]
            out_l = self.learnable_pos_enc(out_l) # [6, 256, 900]
            out_l_1 = out_l
            out_l = out_l.permute(0, 2, 1) # [6, 900, 256]
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(0, 2, 1) # [6, 256, 900]
            out_l_1 = out_l_1.unsqueeze(3)
            out_l = out_l.unsqueeze(3)
            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        return out_l

from config.config import get_config
if __name__ == '__main__':
    # load config ================================================================
    config = get_config()
    # ============================================================================

    torch.backends.cudnn.benchmark = True
    feature_extracter=OverlapTransformer_geo(use_transformer=True, channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)
    feature_extracter.eval()
    random_tensor = torch.randn(6, 1, 64, 900).to(device)

    # print("model architecture: \n")
    # print(feature_extracter)

    gloabal_descriptor = feature_extracter(random_tensor)
    print(f"size of gloabal descriptor: {gloabal_descriptor.size()}")
