#!/usr/bin/env python3
import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import torch
import torch.nn as nn

from aggregators.netvlad import NetVLADLoupe
import torch.nn.functional as F
from backbones.GAT.gatv2_conv import GATv2Conv
from torch_geometric.data import Data, Batch

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from models.aggregators.SOP import *

class PatchEmbedding(nn.Module):
    """Divide image into patches and embed them."""
    def __init__(self, in_channels=1, patch_size=(8, 8), embed_dim=128):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection for patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Args:
            x: Input image of shape [batch, in_channels, H, W].
        Returns:
            patches: Patch embeddings of shape [batch, num_patches, embed_dim].
        """
        batch_size, _, H, W = x.size()

        # Project patches
        x = self.proj(x)  # [batch, embed_dim, H/patch_h, W/patch_w]
        patches = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        return patches
    

class TransformerEncoder(nn.Module):
    """Transformer Encoder block."""
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [num_patches, batch, embed_dim].
        Returns:
            x: Output tensor of the same shape as input.
        """
        # Multi-head Self-Attention
        attn_out, _ = self.msa(x, x, x)  # [num_patches, batch, embed_dim]
        x = x + self.dropout1(attn_out)
        x = self.layer_norm1(x)

        # Feed-forward Network
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.layer_norm2(x)

        return x

class VisionTransformer(nn.Module):
    """Vision Transformer for non-square images."""
    def __init__(self, in_channels=1, patch_size=(8, 8), embed_dim=128, depth=4, num_heads=4, ff_dim=512):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 896, embed_dim))  # Positional embeddings
        self.dropout = nn.Dropout(0.1)

        # Transformer Encoder layers
        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: Input image of shape [batch, in_channels, H, W].
        Returns:
            feature_map: Feature tensor of shape [batch, num_patches, embed_dim].
        """
        batch_size, _, H, W = x.size()
        patches = self.patch_embed(x)  # [batch, num_patches, embed_dim]

        # Add positional embeddings
        x = patches + self.pos_embedding[:, :patches.size(1), :]
        x = self.dropout(x)

        # Transformer Encoder
        x = x.permute(1, 0, 2)  # [num_patches, batch, embed_dim]
        for layer in self.transformer:
            x = layer(x)

        x = x.permute(1, 0, 2)  # [batch, num_patches, embed_dim]
        return x
    

class CNNPatchEmbedding(nn.Module):
    """Patch Embedding using CNN with adjustable depth."""
    def __init__(self, in_channels=1, embed_dim=128, base_channels=32, depth=3, patch_size=(8, 8)):
        """
        Args:
            in_channels (int): Number of input channels.
            embed_dim (int): Final embedding dimension.
            base_channels (int): Initial number of channels.
            depth (int): Number of CNN layers.
            patch_size (tuple): Size of the patch (height, width).
        """
        super(CNNPatchEmbedding, self).__init__()

        self.cnn_layers = nn.ModuleList()
        current_channels = in_channels
        stride = patch_size[0] // 2

        # First layer with kernel size 5
        self.cnn_layers.append(
            self._conv_block(current_channels, base_channels, kernel_size=5, stride=stride)
        )
        current_channels = base_channels

        # Dynamically add remaining layers with kernel size 3
        for i in range(1, depth - 1):
            out_channels = base_channels * (2 ** i)
            self.cnn_layers.append(
                self._conv_block(current_channels, out_channels, kernel_size=3, stride=stride)
            )
            current_channels = out_channels

        # Final layer to match embed_dim with kernel size 3
        self.cnn_layers.append(
            self._conv_block(current_channels, embed_dim, kernel_size=3, stride=1)
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, stride):
        """Helper function to create a CNN block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, in_channels, H, W].
        Returns:
            Feature tensor of shape [batch, embed_dim, H', W'].
        """
        for layer in self.cnn_layers:
            print(x.size())
            x = layer(x)
        print(x.size())  
        return x

    
class PatchCNNVisionTransformer(nn.Module):
    """Vision Transformer for non-square images."""
    def __init__(self, in_channels=1, patch_size=(8, 8), embed_dim=128, 
                 depth=4, num_heads=4, ff_dim=512, dropout=0.1):
        super(PatchCNNVisionTransformer, self).__init__()
        self.patch_embed = CNNPatchEmbedding(in_channels=in_channels, 
                                             embed_dim=embed_dim, 
                                             base_channels=32, 
                                             depth=4, 
                                             patch_size=patch_size)
        self.embed_dim = embed_dim
        self.depth = depth
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder layers
        self.transformer = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Positional embedding (initialized later dynamically)
        self.pos_embedding = None

    def _init_pos_embedding(self, num_patches):
        if self.pos_embedding is None or self.pos_embedding.size(1) != num_patches:
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)  # Standard initialization

    def forward(self, x):
        """
        Args:
            x: Input image of shape [batch, in_channels, H, W].
        Returns:
            feature_map: Feature tensor of shape [batch, num_patches, embed_dim].
        """
        patches = self.patch_embed(x)  # [batch, num_patches, embed_dim]
        batch_size, num_patches, _ = patches.size()

        # Initialize positional embeddings dynamically
        self._init_pos_embedding(num_patches)

        # Add positional embeddings
        x = patches + self.pos_embedding[:, :num_patches, :]
        x = self.dropout(x)

        # Transformer Encoder
        x = x.permute(1, 0, 2)  # [num_patches, batch, embed_dim]
        for layer in self.transformer:
            x = layer(x)

        x = x.permute(1, 0, 2)  # [batch, num_patches, embed_dim]
        return x


class PyramidCNN(nn.Module):
    """Feature extraction with adjusted patch dimensions based on input aspect ratio."""
    def __init__(self, in_channels=1, norm_layer="gn", output_shape=(16, 250),
                 channels=[32, 64, 128], kernel_sizes=[5, 3, 3], strides=[1, 2, 2]):
        super(PyramidCNN, self).__init__()

        self.norm_layer = norm_layer

        # Build convolutional blocks dynamically
        self.conv_blocks = self._make_layers(in_channels, channels, kernel_sizes, strides)

        # Adaptive pooling to reduce dimensions while preserving aspect ratio
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_shape)  # Target size based on aspect ratio
        self.relu = nn.ReLU(inplace=True)

    def _make_layers(self, in_channels, channels, kernel_sizes, strides):
        layers = []
        for i in range(len(channels)):
            layers.append(nn.Conv2d(in_channels, channels[i], kernel_size=kernel_sizes[i], 
                                    stride=strides[i], padding=kernel_sizes[i] // 2, bias=False,
                                    padding_mode='circular'))
            layers.append(self._get_norm_layer(channels[i], 4 * (2 ** i)))  # Dynamic group size
            in_channels = channels[i]
        return nn.Sequential(*layers)

    def _get_norm_layer(self, num_channels, num_groups):
        if self.norm_layer == "gn":
            return nn.GroupNorm(num_groups, num_channels)
        else:
            return nn.BatchNorm2d(num_channels)

    def forward(self, x):
        """
        Args:

            x: Input image of shape [batch, 1, 64, 900].
        Returns:
            patch_features: Features of shape [batch, num_patches, feature_dim].
        """
        # Apply convolutional blocks
        x = self.conv_blocks(x)

        # Adaptive pooling to match target patch count and aspect ratio
        x = self.adaptive_pool(x)  # [batch, 128, 16, 250]
        return x
   

class GraphNetwork(nn.Module):
    def __init__(self, gat_channels):
        super(GraphNetwork, self).__init__()
        self.num_layers = len(gat_channels)
        self.gat_layers = nn.ModuleList()

        for layer in range(self.num_layers - 1):
            self.gat_layers.append(GATv2Conv(gat_channels[layer], gat_channels[layer + 1], heads=1, residual=True, dropout=0.1))

    def forward(self, x, edge_index):
        for gat in self.gat_layers:
            x =  F.relu((gat(x, edge_index)))
        return x


class GATNet(nn.Module):
    def __init__(self, config, in_channels=1):
        super(GATNet, self).__init__()
        self.top_k_list = config.topk_list
        self.patch_radius = config.patch_radius
        self.feature_h, self.feature_w = config.cnn_output_shape  # >> 16, 225
        self.pooling_method = config.pooling_method
        self.config = config

        if config.feature_extractor_backbone == "CNN":
            self.feature_extractor = PyramidCNN(in_channels, config.cnn_norm_layer, config.cnn_output_shape,
                                        config.cnn_channels, config.cnn_kernel_sizes, config.cnn_strides)
        elif config.feature_extractor_backbone == "PatchCNNViT":
            self.feature_extractor = PatchCNNVisionTransformer(in_channels, patch_size=config.PatchCNNViT_patch_size, 
                                                               embed_dim=config.PatchCNNViT_embed_dim, depth=4, num_heads=4, 
                                                               ff_dim=256, dropout=0.1)

        self.gat_layers = GraphNetwork(config.gat_channels)

        if self.pooling_method == 'NetVLAD':
            self.net_vlad = NetVLADLoupe(feature_size=config.gat_channels[-1], max_samples=self.top_k_list[1], cluster_size=64,
                                         output_dim=256, gating=True, add_batch_norm=False,
                                         is_training=True)
        elif self.pooling_method == 'Attention':
            self.attn_linear = nn.Linear(config.gat_channels[-1], 1)  # For attention pooling
        elif self.pooling_method == 'DeepSet':
            self.mlp = nn.Sequential(
                nn.Linear(config.gat_channels[-1], 256),
                nn.ReLU(),
                nn.Linear(256, 256)
            )
        elif self.pooling_method == 'SelfAttention':
            self.query = nn.Linear(config.gat_channels[-1], config.gat_channels[-1])
            self.key = nn.Linear(config.gat_channels[-1], config.gat_channels[-1])
            self.value = nn.Linear(config.gat_channels[-1], config.gat_channels[-1])
        elif self.pooling_method == 'SOP':
            self.sop = SOP(signed_sqrt=False, do_fc=False, input_dim=256, is_tuple=False)

    def create_edges(self, topk_indices, patch_radius):
        if topk_indices.dim() == 1:
            topk_indices = topk_indices.unsqueeze(0)  # [1, top_k]
        
        # 2D 좌표 변환 [h, w]
        y_coords = topk_indices // self.feature_w  # [batch, top_k] 
        x_coords = topk_indices % self.feature_w   # [batch, top_k]
        
        # y, x 좌표 스택 [batch, top_k, 2]
        coords = torch.stack((y_coords, x_coords), dim=2).float()

        # 브로드캐스팅 방식으로 거리 계산 (불필요한 축 제거)
        delta = coords[:, :, None] - coords[:, None, :]  # [batch, top_k, top_k, 2]
        
        y_dist = delta[..., 0].abs()  # y 방향 거리
        
        # x 방향(가로)에서 순환성을 고려한 거리 계산
        x_dist = delta[..., 1].abs()
        x_dist = torch.minimum(x_dist, self.feature_w - x_dist)  # 순환 거리 계산
        
        # 유클리드 거리 계산
        distances = torch.sqrt((y_dist)**2 + (x_dist)**2)  # [batch, top_k, top_k]

        # 일정 거리 내의 패치 연결
        within_radius = distances <= patch_radius

        # 자기 자신 엣지 제거
        no_self_loop = torch.eye(topk_indices.size(1), device=topk_indices.device).bool()
        within_radius &= ~no_self_loop.unsqueeze(0)

        # 엣지 인덱스 생성
        src = topk_indices.repeat_interleave(topk_indices.size(1))
        dst = topk_indices.repeat(1, topk_indices.size(1)).flatten()

        # 유효한 엣지만 필터링
        valid_edges = within_radius.flatten()
        src = src[valid_edges]
        dst = dst[valid_edges]

        return torch.stack((src, dst), dim=0)
    
    def pooling(self, node_features):
        if self.pooling_method == 'Mean':
            pooled = torch.mean(node_features, dim=1)  # Global Average Pooling (GAP)
            return F.normalize(pooled, dim=1, eps=1e-8)
        elif self.pooling_method == 'Max':
            pooled = torch.max(node_features, dim=1)[0]  # Global Max Pooling (GMP)
            return F.normalize(pooled, dim=1, eps=1e-8)
        elif self.pooling_method == 'Attention':
            attn_weights = F.softmax(self.attn_linear(node_features), dim=1)  # [batch, num_nodes, 1]
            pooled = torch.sum(attn_weights * node_features, dim=1)  # Weighted sum
            return F.normalize(pooled, dim=1, eps=1e-8)
        elif self.pooling_method == 'DeepSet':
            pooled = self.mlp(torch.sum(node_features, dim=1))
            return F.normalize(pooled, dim=1, eps=1e-8)  # MLP 이후 정규화
        elif self.pooling_method == 'SelfAttention':
            q = self.query(node_features)  # [batch, num_nodes, feature_dim]
            k = self.key(node_features)
            v = self.value(node_features)
            attn_scores = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (q.size(-1) ** 0.5), dim=-1)
            global_descriptor = torch.bmm(attn_scores, v)
            pooled = torch.mean(global_descriptor, dim=1)
            return pooled  # SelfAttention은 정규화 없음
        elif self.pooling_method == 'NetVLAD':
            return F.normalize(self.net_vlad(node_features.permute(0, 2, 1).unsqueeze(3)), dim=1, eps=1e-8)
        elif self.pooling_method == 'SOP':
            return self.sop(node_features)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

    def visualize_edges(self, topk_indices, edge_index, feature, x):
        from torchvision.transforms.functional import resize
        from sklearn.decomposition import PCA
        
        x_resized = resize(x, [self.feature_h, self.feature_w]).squeeze().cpu().numpy()  # [h, w]
        topk_indices = topk_indices.cpu().numpy()  # [top_k]

        # 창 0: 원본 이미지 시각화
        plt.figure(figsize=(80, 5))
        plt.imshow(resize(x, [self.feature_h*4, self.feature_w*4]).squeeze().cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.colorbar(label='Pixel Intensity')
        plt.show(block=False)  # [h, w]

        # 창 1: x_resized의 픽셀 값 사용
        plt.figure(figsize=(80, 5))
        y_coords = np.arange(self.feature_h * self.feature_w) // self.feature_w
        x_coords = np.arange(self.feature_h * self.feature_w) % self.feature_w
        
        # topk_indices 점에 흑백 픽셀값 매핑
        colors = x_resized.flatten()
        plt.scatter(x_coords, y_coords, c=colors, cmap='gray', s=100)
        plt.scatter(x_coords[topk_indices], y_coords[topk_indices], facecolors='none', edgecolors='r', s=150, linewidth=1.5)

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            src_y, src_x = src // self.feature_w, src % self.feature_w
            dst_y, dst_x = dst // self.feature_w, dst % self.feature_w
            plt.plot([src_x, dst_x], [src_y, dst_y], color='green', linewidth=0.2, alpha=0.2)

        plt.gca().invert_yaxis()
        plt.title('Top-K Nodes with Grayscale Values')
        plt.colorbar(label='Pixel Intensity')
        plt.legend()
        plt.show(block=False)

        # 창 2: feature를 PCA로 축소해 RGB 색상 매핑
        feature = feature.detach().cpu().numpy()  # [num_patches, feature_dim]
        pca = PCA(n_components=3)
        feature_pca = pca.fit_transform(feature)
        feature_pca = np.clip(feature_pca, 0, 1)  # RGB 값으로 사용하기 위해 0~1로 클리핑

        plt.figure(figsize=(80, 5))
        
        # PCA 색상을 topk_indices에 매핑
        colors = feature_pca
        plt.scatter(x_coords, y_coords, c=colors, label='Top-K Nodes (PCA Colors)', s=100)

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            src_y, src_x = src // self.feature_w, src % self.feature_w
            dst_y, dst_x = dst // self.feature_w, dst % self.feature_w
            plt.plot([src_x, dst_x], [src_y, dst_y], color='green', linewidth=0.5, alpha=0.3)

        plt.gca().invert_yaxis()
        plt.title('Top-K Nodes with PCA-based RGB Colors')
        plt.legend()
        plt.show()

    def forward(self, x):
        # Patch-wise feature extraction
        cnn_features = self.feature_extractor(x)  # [batch, feature_dim, h, w] >> [batch, 128, 16, 225]
        # Reshape into patch-wise features
        batch_size, dim, _, _ = cnn_features.size() 
        patch_features = cnn_features.view(batch_size, dim, -1).permute(0, 2, 1)  # [batch, num_patches, feature_dim] [6, 904, 128]
        # print("patch_features", patch_features.size())

        # Top-K patch selection
        patch_scores_1 = torch.norm(patch_features, dim=2)  # [batch, num_patches] >> [6, 904]
        topk_indices_1 = torch.topk(patch_scores_1, self.top_k_list[0], dim=1).indices  # [batch, top_k] >> [6, top_k]
        # print("patch_scores_1", patch_scores_1.size(), " | topk_indices_1", topk_indices_1.size())

        # Create edges for GAT
        out_list = []
        edge_index_list = [
            self.create_edges(topk_indices_1[i], self.patch_radius).to(patch_features.device)
            for i in range(batch_size)
        ]
        # print("edge_index", edge_index[0].size())

        # GAT layer
        for i in range(batch_size):
            node_features = self.gat_layers(patch_features[i], edge_index_list[i])  # [num_queries, feature_dim] >> [700, 128]
            out_list.append(node_features[topk_indices_1[i]])
        gat_output = torch.stack(out_list, dim=0)  # [batch_size, topk, feature_dim] >> [6, 1000, 256]
        # print("gat_output", gat_output.size()) 

        if self.top_k_list[0] != self.top_k_list[1]:
            patch_scores_2 = torch.norm(gat_output, dim=2)  # [batch, num_patches]
            topk_indices_2 = torch.topk(patch_scores_2, self.top_k_list[1], dim=1).indices  # [batch, top_k] >> [13, 300]
            gat_output = gat_output[torch.arange(batch_size)[:, None], topk_indices_2]
        # gat_output = gat_output.permute(0, 2, 1).unsqueeze(3) # [batch, feature_dim, topk, 1] >> [6, 256, 500, 1]

        # self.visualize_edges(topk_indices_1[1], edge_index_list[1], patch_features[1], x[1])
        
        # Apply selected pooling method
        global_output = self.pooling(gat_output)  # [batch, feature_dim] or [batch, k*feature_dim]
        # print("global_output", vlad_output.size())

        return global_output

sys.path.append(os.path.join(os.path.dirname(os.path.join(os.path.dirname(os.path.join(os.path.dirname(__file__)))))))
from config.config import get_config
from utils.data_loaders.make_dataloader import make_data_loader
from time import time
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

def shift_image(image, shift):
    """Shift image along width direction."""
    return torch.roll(image, shifts=shift, dims=3)

def visualize_feature_maps(feature_map1, feature_map2):
    """Visualize feature maps."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes[0].imshow(feature_map1.cpu().detach().numpy(), cmap='viridis')
    axes[0].set_title('Feature Map 1')
    axes[1].imshow(feature_map2.cpu().detach().numpy(), cmap='viridis')
    axes[1].set_title('Feature Map 2')
    plt.show()

def test_overlap_gat_net():
    cfg = get_config()
    model = GATNet(cfg).to("cuda")
    # checkpoint = torch.load('/home/vision/Models/LoGG3D-Net/checkpoints/GATNet/2024-12-26_21-27-12_GATNet_NetVLAD_kitti08/epoch_59.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=True)
    batch = next(iter(train_loader)).unsqueeze(1).to("cuda")
    tt = time()
    output = model(batch)
    print("Inference Time:", time() - tt)
    print("Output Shape:", output.shape)

def test_pyramid_cnn():
    import matplotlib.pyplot as plt

    cfg = get_config()
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=True)
    single_image = next(iter(train_loader)).unsqueeze(1).to("cuda")[0:1]
    shifted_image = shift_image(single_image, shift=300)
    batch = torch.cat([single_image, shifted_image], dim=0)
    print("Batch Shape:", batch.size())

    pyramid_cnn = PyramidCNN(in_channels=1, norm_layer="gn", output_shape=(16, 250),
                             channels=[32, 64, 128], kernel_sizes=[5, 3, 3], strides=[1, 2, 2]).to("cuda")
    feature_maps = pyramid_cnn(batch)
    print("Feature Maps Shape:", feature_maps.size())
    reversed_shifted_feature_map = shift_image(feature_maps[1:2], shift=-300)
    visualize_feature_maps(feature_maps[0, 0], reversed_shifted_feature_map[0, 0])
    difference = torch.abs(feature_maps[0] - reversed_shifted_feature_map)
    print("Difference between feature maps:", difference.mean().item())

if __name__ == "__main__":
    test_overlap_gat_net()
    # test_pyramid_cnn()
