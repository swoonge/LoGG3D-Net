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
from backbones.GAT.gatv2_conv import GATv2Conv


class ResidualBlock(nn.Module):
    """Residual Block with two convolutional layers."""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # Residual connection
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class PyramidCNN(nn.Module):
    """Feature extraction with increased channels and Residual Blocks."""
    def __init__(self, in_channels=1, patch_size=(8, 8)):
        super(PyramidCNN, self).__init__()
        self.patch_size = patch_size

        # First convolution to increase feature dimensions
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual Blocks with increasing channels
        self.res_block1 = ResidualBlock(32)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(128)

        # Downsample layers to increase spatial abstraction
        self.downsample1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_down1 = nn.BatchNorm2d(64)
        self.downsample2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_down2 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input image of shape [batch, 1, 64, 900].
        Returns:
            patch_features: Patch-wise features of shape [batch, num_patches, feature_dim].
        """
        batch_size, _, height, width = x.size()
        patch_h, patch_w = self.patch_size

        # Divide image into patches
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.contiguous().view(batch_size, 1, -1, patch_h, patch_w)  # [batch, 1, num_patches, 8, 8]

        patch_features = []
        for patch_idx in range(patches.size(2)):
            patch = patches[:, :, patch_idx, :, :]  # [batch, 1, 8, 8] -> valid for Conv2d

            # Pass through convolutional and residual layers
            patch = self.relu(self.bn1(self.conv1(patch)))  # [batch, 32, 8, 8]
            patch = self.res_block1(patch)                 # Residual Block 1
            patch = self.relu(self.bn_down1(self.downsample1(patch)))  # Downsample to [batch, 64, 4, 4]
            patch = self.res_block2(patch)                 # Residual Block 2
            patch = self.relu(self.bn_down2(self.downsample2(patch)))  # Downsample to [batch, 128, 2, 2]
            patch = self.res_block3(patch)                 # Residual Block 3

            # Adaptive pooling to reduce dimensions
            patch = F.adaptive_avg_pool2d(patch, (1, 1)).squeeze(-1).squeeze(-1)  # [batch, 128]
            patch_features.append(patch)

        patch_features = torch.stack(patch_features, dim=1)  # [batch, num_patches, feature_dim(128)]
        return patch_features


class OverlapGATNet(nn.Module):
    def __init__(self, top_k=300, patch_radius=5, in_channels=1):
        super(OverlapGATNet, self).__init__()
        self.pyramid_cnn = PyramidCNN(in_channels)
        self.decoder = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.gat_conv1 = GATv2Conv(128, 128, residual=True, dropout=0.1)
        self.gat_conv2 = GATv2Conv(128, 256, residual=True, dropout=0.1)
        self.net_vlad = NetVLADLoupe(feature_size=256, max_samples=top_k, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        self.top_k = top_k
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
        tt = time.time()
        patch_features = self.pyramid_cnn(x)  # [batch, num_patches, feature_dim] >> [13, 896, 128]
        print("pyramid_cnn time: ", time.time()-tt)

        # Top-K patch selection
        tt = time.time()
        patch_scores = torch.norm(patch_features, dim=2)  # [batch, num_patches]
        topk_indices = torch.topk(patch_scores, self.top_k, dim=1).indices  # [batch, top_k] >> [13, 300]
        print("topk time: ", time.time()-tt)

        # Create edges for GAT
        batch_size, num_patches, feature_dim = patch_features.size()
        patch_h, patch_w = self.pyramid_cnn.patch_size
        height = x.size(2) // patch_h # >> 8
        width = x.size(3) // patch_w # >> 112

        # GAT processing
        out_list = []
        tt = time.time()
        edge_index = [edge.to(patch_features.device) for edge in self.create_edges(topk_indices, self.patch_radius, height, width)] # [batch, 2, n]
        print("create_edges time: ", time.time()-tt)
        tt = time.time()
        for i in range(batch_size):
            node_features = patch_features[i]  # [num_queries, feature_dim] >> [896, 128]
            node_features = F.relu(self.gat_conv1(node_features, edge_index[i]))
            node_features = F.relu(self.gat_conv2(node_features, edge_index[i]))[topk_indices[i]]
            out_list.append(node_features)
        gat_output = torch.stack(out_list, dim=0)  # [batch_size, topk, feature_dim]
        gat_output = gat_output.permute(0, 2, 1).unsqueeze(3) # [batch, feature_dim, topk, 1]
        print("gat time: ", time.time()-tt)

        # Reshape GAT output for NetVLAD
        tt = time.time()
        gat_output = F.normalize(gat_output, dim=1)
        vlad_output = self.net_vlad(gat_output)
        print("netvlad time: ", time.time()-tt)

        return F.normalize(vlad_output, dim=1)



# if __name__ == '__main__':
#     # load config ================================================================
#     config_filename = '../config/config.yml'
#     config = yaml.safe_load(open(config_filename))
#     seqs_root = config["data_root"]["data_root_folder"]
#     # ============================================================================

#     combined_tensor = read_one_need_from_seq(seqs_root, "000000","00")
#     combined_tensor = torch.cat((combined_tensor,combined_tensor), dim=0)

#     feature_extracter=featureExtracter(use_transformer=True, channels=1)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     feature_extracter.to(device)
#     feature_extracter.eval()

#     print("model architecture: \n")
#     print(feature_extracter)

#     gloabal_descriptor = feature_extracter(combined_tensor)
#     print("size of gloabal descriptor: \n")
#     print(gloabal_descriptor.size())
