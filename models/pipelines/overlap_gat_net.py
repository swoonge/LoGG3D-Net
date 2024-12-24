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

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    
class CNNPatchEmbedding(nn.Module):
    """Patch Embedding using 3-layer CNN."""
    def __init__(self, in_channels=1, embed_dim=128):
        """
        Args:
            in_channels (int): Number of input channels.
            embed_dim (int): Final embedding dimension.
        """
        super(CNNPatchEmbedding, self).__init__()
        self.cnn_layers = nn.Sequential(
            # First CNN Layer: 1 -> 23
            nn.Conv2d(in_channels, 23, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(23),
            nn.ReLU(),

            # Second CNN Layer: 23 -> 64
            nn.Conv2d(23, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            ## if embed_dim == 128:
            # Third CNN Layer: 64 -> 128
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()

            ## if embed_dim == 256:
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(embed_dim),
            # nn.ReLU(),

            # nn.Conv2d(128, embed_dim, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(embed_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, in_channels, H, W].
        Returns:
            Feature tensor of shape [batch, 1, 1, embed_dim].
        """
        # Pass through CNN layers
        x = self.cnn_layers(x)  # Output shape: [batch, embed_dim, H', W']
        return x


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
    
class PatchCNNVisionTransformer(nn.Module):
    """Vision Transformer for non-square images."""
    def __init__(self, in_channels=1, patch_size=(8, 8), embed_dim=128, depth=4, num_heads=4, ff_dim=512):
        super(PatchCNNVisionTransformer, self).__init__()
        self.patch_embed = CNNPatchEmbedding(in_channels, patch_size, embed_dim)
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


class PyramidCNN(nn.Module):
    """Feature extraction with adjusted patch dimensions based on input aspect ratio."""
    def __init__(self, in_channels=1):
        super(PyramidCNN, self).__init__()

        # Conv layer 1
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(4, 32)

        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1_1 = nn.GroupNorm(4, 32)

        # Conv layer 2 with stride
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.gn2 = nn.GroupNorm(8, 64)

        # Increase kernel size to 5x5 for better feature extraction
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2_1 = nn.GroupNorm(8, 64)

        # Conv layer 3 with stride
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False)
        self.gn3 = nn.GroupNorm(16, 128)

        # Increase kernel size to 5x5 for global context learning
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn3_1 = nn.GroupNorm(16, 128)

        # Adaptive pooling to reduce dimensions while preserving aspect ratio
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 112))  # Target size based on aspect ratio

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input image of shape [batch, 1, 64, 900].
        Returns:
            patch_features: Features of shape [batch, num_patches, feature_dim].
        """
        # Conv block 1
        x = self.relu(self.gn1(self.conv1(x)))  # [batch, 32, H, W]
        x = self.relu(self.gn1_1(self.conv1_1(x)))  # [batch, 32, H, W]

        # Conv block 2
        x = self.relu(self.gn2(self.conv2(x)))  # [batch, 64, H/2, W/2]
        x = self.relu(self.gn2_1(self.conv2_1(x)))  # [batch, 64, H/2, W/2]

        # Conv block 3
        x = self.relu(self.gn3(self.conv3(x)))  # [batch, 128, H/4, W/4]
        x = self.relu(self.gn3_1(self.conv3_1(x)))  # [batch, 128, H/4, W/4]

        # print("x size: ", x.size())
        # Adaptive pooling to match target patch count and aspect ratio
        x = self.adaptive_pool(x)  # [batch, 128, 8, 112]

        # Reshape into patch-wise features
        batch_size, channels, height, width = x.size()  # [batch, 896, 128]
        # print("batch_size, channels, height, width: ", batch_size, channels, height, width)
        patch_features = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch, num_patches, feature_dim]

        return patch_features



class OverlapGATNet(nn.Module):
    def __init__(self, top_k_list=[500, 300], patch_radius=9, in_channels=1):
        super(OverlapGATNet, self).__init__()
        self.pyramid_cnn = PyramidCNN(in_channels)
        # self.pyramid_resnet_cnn = PatchCNNVisionTransformer(in_channels)
        # self.vit = VisionTransformer(in_channels, patch_size=(8, 8), embed_dim=128, depth=4, num_heads=4, ff_dim=256)
        # self.decoder = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gat_conv1 = GATv2Conv(128, 256, residual=True, dropout=0.1)
        self.gat_conv2 = GATv2Conv(256, 256, residual=True, dropout=0.1)
        # self.gat_conv3 = GATv2Conv(256, 256, residual=True, dropout=0.1)
        self.net_vlad = NetVLADLoupe(feature_size=256, max_samples=top_k_list[1], cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.top_k_list = top_k_list
        self.patch_radius = patch_radius

    def create_edges(self, topk_indices, patch_radius, height, width):
        """
        Args:
            topk_indices: Tensor of shape [batch, top_k] containing the indices of selected queries.
            patch_radius: Radius within which to connect patches.
            height: Number of patches along the height of the grid.
            width: Number of patches along the width of the grid.
        Returns:
            edge_index: Tensor of shape [2, edge_num] containing edge connections.
        """
        batch_size, top_k = topk_indices.size()
        edge_list = []

        # Precompute grid coordinates
        y_coords = topk_indices // width  # [batch, top_k]
        x_coords = topk_indices % width   # [batch, top_k]

        # Create relative index ranges for neighbors
        rel_indices = torch.arange(-patch_radius, patch_radius + 1, device=topk_indices.device)
        rel_y, rel_x = torch.meshgrid(rel_indices, rel_indices, indexing='ij')  # [2*radius+1, 2*radius+1]
        rel_y = rel_y.flatten()
        rel_x = rel_x.flatten()

        for b in range(batch_size):
            # Get coordinates for current batch
            y = y_coords[b]  # [top_k]
            x = x_coords[b]  # [top_k]

            # Broadcast coordinates for all neighbors
            y_neighbors = y[:, None] + rel_y[None, :]  # [top_k, num_neighbors]
            x_neighbors = x[:, None] + rel_x[None, :]  # [top_k, num_neighbors]

            # Mask out-of-bound neighbors
            valid_mask = (y_neighbors >= 0) & (y_neighbors < height) & (x_neighbors >= 0) & (x_neighbors < width)
            valid_y_neighbors = y_neighbors[valid_mask]  # Flatten valid y neighbors
            valid_x_neighbors = x_neighbors[valid_mask]  # Flatten valid x neighbors

            # Convert valid neighbors back to 1D indices
            neighbors = valid_y_neighbors * width + valid_x_neighbors  # [num_valid_neighbors]
            src = topk_indices[b].repeat_interleave(rel_y.numel())[valid_mask.flatten()]  # Match neighbors' shape
            dst = neighbors  # [num_valid_neighbors]

            # Add edges (src, dst)
            edges = torch.stack((src, dst), dim=0)  # [2, num_edges]
            edge_list.append(edges)

        # Combine all batch edges
        # edge_index = torch.stack(edge_list, dim=0)  # [2, total_edges]
        return edge_list

    def forward(self, x):
        import time
        # Patch-wise feature extraction
        # tt = time.time()
        patch_features = self.pyramid_cnn(x)  # [batch, num_patches, feature_dim] >> [13, 896, 128] # 0.0002 sec
        # patch_features = self.pyramid_resnet_cnn(x)  # [batch, num_patches, feature_dim] >> [13, 896, 128] # 0.0004 sec
        # patch_features = self.vit(x)  # [batch, num_patches, feature_dim] >> [13, 896, 128] # 0.0007 sec
        # print("patch_features size: ", patch_features.size())
        # print("pyramid_cnn time: ", time.time()-tt)

        # Top-K patch selection
        # tt = time.time()
        patch_scores_1 = torch.norm(patch_features, dim=2)  # [batch, num_patches]
        topk_indices_1 = torch.topk(patch_scores_1, self.top_k_list[0], dim=1).indices  # [batch, top_k] >> [13, 300]
        # print("topk time: ", time.time()-tt)

        # Create edges for GAT
        batch_size, _, _ = patch_features.size()
        patch_h = patch_w = 8
        height = x.size(2) // patch_h # >> 8
        width = x.size(3) // patch_w # >> 112

        # GAT processing
        out_list = []
        # tt = time.time()
        edge_index = [edge.to(patch_features.device) for edge in self.create_edges(topk_indices_1, self.patch_radius, height, width)] # [batch, 2, n]
        # print("create_edges time: ", time.time()-tt)
        # tt = time.time()
        for i in range(batch_size):
            node_features = patch_features[i]  # [num_queries, feature_dim] >> [700, 128]
            node_features = F.relu(self.gat_conv1(node_features, edge_index[i]))
            node_features = F.relu(self.gat_conv2(node_features, edge_index[i]))[topk_indices_1[i]]
            # node_features = F.relu(self.gat_conv3(node_features, edge_index[i]))[topk_indices_1[i]]
            out_list.append(node_features)
        gat_output = torch.stack(out_list, dim=0)  # [batch_size, topk, feature_dim]

        patch_scores_2 = torch.norm(gat_output, dim=2)  # [batch, num_patches]
        topk_indices_2 = torch.topk(patch_scores_2, self.top_k_list[1], dim=1).indices  # [batch, top_k] >> [13, 300]
        
        gat_output = gat_output[torch.arange(batch_size)[:, None], topk_indices_2]
        gat_output = gat_output.permute(0, 2, 1).unsqueeze(3) # [batch, feature_dim, topk, 1]
        # print("gat_output size: ", gat_output.size())
        # print("gat time: ", time.time()-tt)

        # Reshape GAT output for NetVLAD
        # tt = time.time()
        gat_output = F.normalize(gat_output, dim=1)
        vlad_output = self.net_vlad(gat_output)
        # print("netvlad time: ", time.time()-tt)

        return F.normalize(vlad_output, dim=1)

