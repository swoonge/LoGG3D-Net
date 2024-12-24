#!/usr/bin/env python3
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
from backbones.GAT.gatv2_conv import GATv2Conv
from torch_geometric.data import Data, Batch

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

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
            self.gat_layers.append(GATv2Conv(gat_channels[layer], gat_channels[layer + 1], residual=True, dropout=0.1))

    def forward(self, x, edge_index):
        for gat in self.gat_layers:
            x =  F.relu((gat(x, edge_index)))
        return x


class OverlapGATNet(nn.Module):
    def __init__(self, config, in_channels=1):
        super(OverlapGATNet, self).__init__()
        self.top_k_list = config.topk_list
        self.patch_radius = config.patch_radius
        self.config = config

        if config.feature_extractor_backbone == "CNN":
            self.feature_extractor = PyramidCNN(in_channels, config.cnn_norm_layer, config.cnn_output_shape,
                                        config.cnn_channels, config.cnn_kernel_sizes, config.cnn_strides)
        elif config.feature_extractor_backbone == "PatchCNNViT":
            self.feature_extractor = PatchCNNVisionTransformer(in_channels, patch_size=config.PatchCNNViT_patch_size, 
                                                               embed_dim=config.PatchCNNViT_embed_dim, depth=4, num_heads=4, 
                                                               ff_dim=256, dropout=0.1)

        self.gat_layers = GraphNetwork(config.gat_channels)
        self.net_vlad = NetVLADLoupe(feature_size=config.gat_channels[-1], max_samples=self.top_k_list[1], cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

    def create_edges(self, topk_indices, patch_radius, height, width):
        if topk_indices.dim() == 1:
            topk_indices = topk_indices.unsqueeze(0)  # [1, top_k]
        
        # Top-K 인덱스를 2D 좌표로 변환
        y_coords = topk_indices // width  # [batch, top_k]
        x_coords = topk_indices % width   # [batch, top_k]

        # Top-K 패치 좌표 [batch, top_k, 2]
        coords = torch.stack((y_coords, x_coords), dim=2).float()

        # 패치 간 거리 계산 (유클리드 거리)
        distances = torch.cdist(coords, coords, p=2)  # [batch, top_k, top_k]

        # 거리 내에 있는 패치 (patch_radius 이하)
        within_radius = distances <= patch_radius  # [batch, top_k, top_k]

        # 자기 자신 엣지 제거 (거리가 0인 경우)
        no_self_loop = torch.eye(topk_indices.size(1), device=topk_indices.device).bool()
        within_radius &= ~no_self_loop.unsqueeze(0)  # [batch, top_k, top_k]

        # 엣지 인덱스 생성
        src = topk_indices.repeat_interleave(topk_indices.size(1))  # [batch, top_k * top_k]
        dst = topk_indices.repeat(1, topk_indices.size(1)).flatten()  # [batch, top_k * top_k]

        # 유효한 엣지만 필터링 (자기 자신 제외)
        valid_edges = within_radius.flatten()
        src = src[valid_edges]
        dst = dst[valid_edges]

        # [2, num_edges] 형태로 반환
        return torch.stack((src, dst), dim=0)

    def visualize_edges(self, topk_indices, edge_index, height, width, patch_h, patch_w, feature):
        plt.figure(figsize=(10, 10))
        all_coords = torch.arange(patch_h * patch_w).numpy()
        all_y = (all_coords // width)
        all_x = (all_coords % width)
        plt.scatter(all_x, all_y, color='gray', label='All Nodes')

        # print("topk_indices", topk_indices, " | edge_index", edge_index.size())
        y_coords = (topk_indices // width).cpu().numpy()
        x_coords = (topk_indices % width).cpu().numpy()
        plt.scatter(x_coords, y_coords, color='blue', label='Top-K Nodes')

        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            src_y, src_x = src // width, src % width
            dst_y, dst_x = dst // width, dst % width
            plt.plot([src_x, dst_x], [src_y, dst_y], color='green', alpha=0.5)
        
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title('Visualization of Edges and Top-K Nodes')
        plt.show()

    def forward(self, x):
        # Patch-wise feature extraction
        patch_features = self.feature_extractor(x)  # [batch, feature_dim, h, w] >> [batch, 128, 16, 225]
        # Reshape into patch-wise features
        batch_size, channels, _, _ = patch_features.size() 
        patch_features = patch_features.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch, num_patches, feature_dim]
        # print("patch_features", patch_features.size())

        # Top-K patch selection
        patch_scores_1 = torch.norm(patch_features, dim=2)  # [batch, num_patches] >> [6, 3600]
        topk_indices_1 = torch.topk(patch_scores_1, self.top_k_list[0], dim=1).indices  # [batch, top_k] >> [6, 1000]
        # print("patch_scores_1", patch_scores_1.size(), " | topk_indices_1", topk_indices_1.size())

        # Create edges for GAT
        patch_h, patch_w = self.config.cnn_output_shape  # >> 16, 225
        height = x.size(2) // patch_h # >> 4
        width = x.size(3) // patch_w # >> 4
        # print("patch_h", patch_h, " | patch_w", patch_w, " | batch_size", batch_size)
        # print("height", height, " | width", width)

        out_list = []
        edge_index_list = [
            self.create_edges(topk_indices_1[i], self.patch_radius, height, width).to(patch_features.device)
            for i in range(batch_size)
        ]
        # print("edge_index", edge_index[0].size())
        # self.visualize_edges(topk_indices_1[0], edge_index_list[0], height, width, patch_h, patch_w, patch_features[0])

        # GAT layer
        for i in range(batch_size):
            node_features = self.gat_layers(patch_features[i], edge_index_list[i])  # [num_queries, feature_dim] >> [700, 128]
            out_list.append(node_features[topk_indices_1[i]])
        gat_output = torch.stack(out_list, dim=0)  # [batch_size, topk, feature_dim] >> [6, 1000, 256]
        # print("gat_output", gat_output.size()) 

        patch_scores_2 = torch.norm(gat_output, dim=2)  # [batch, num_patches]
        topk_indices_2 = torch.topk(patch_scores_2, self.top_k_list[1], dim=1).indices  # [batch, top_k] >> [13, 300]
        # print("patch_scores_2", patch_scores_2.size(), " | topk_indices_2", topk_indices_2.size())
        
        gat_output = gat_output[torch.arange(batch_size)[:, None], topk_indices_2]
        gat_output = F.normalize(gat_output.permute(0, 2, 1).unsqueeze(3), dim=1) # [batch, feature_dim, topk, 1] >> [6, 256, 500, 1]
        # print("gat_output", gat_output.size())

        # Reshape GAT output for NetVLAD
        vlad_output = self.net_vlad(gat_output) # >> [6, 256]
        # print("vlad_output", vlad_output.size())

        return F.normalize(vlad_output, dim=1)

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
    model = OverlapGATNet(cfg).to("cuda")
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
