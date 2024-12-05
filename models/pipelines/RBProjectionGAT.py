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
from torch_geometric.nn import GATConv
from backbones.GAT.gatv2_conv import GATv2Conv

    
class ConvBlock(nn.Module):
    """A basic convolutional block with ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        return self.relu(x)
    
class RBProjectionGAT(nn.Module):
    def __init__(self, height=64, width=900, channels=5):
        super(RBProjectionGAT, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # Convolutional layers
        self.conv_blocks = nn.ModuleList([
            ConvBlock(channels, 16, kernel_size=(5, 1), stride=(1, 1)),
            ConvBlock(16, 32, kernel_size=(3, 1), stride=(2, 1)),
            ConvBlock(32, 64, kernel_size=(3, 1), stride=(2, 1)),
            ConvBlock(64, 64, kernel_size=(3, 1), stride=(2, 1)),
            ConvBlock(64, 128, kernel_size=(2, 1), stride=(2, 1)),
            ConvBlock(128, 128, kernel_size=(1, 1), stride=(2, 1)),
            ConvBlock(128, 128, kernel_size=(1, 1), stride=(2, 1)),
            ConvBlock(128, 128, kernel_size=(1, 1), stride=(2, 1)),
            ConvBlock(128, 128, kernel_size=(1, 1), stride=(2, 1)),
            ConvBlock(128, 128, kernel_size=(1, 1), stride=(2, 1)),
            ConvBlock(128, 128, kernel_size=(1, 1), stride=(2, 1))])

        self.convLast1 = ConvBlock(128, 256, kernel_size=(1, 1), stride=(1, 1))
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128 * 900, 256)

        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=300, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        
        # Graph Attention Network (GATv2Conv)
        self.gat_conv1 = GATv2Conv(256, 256, residual=True, dropout=0.1)
        self.gat_conv2 = GATv2Conv(256, 512, residual=True, dropout=0.1)
        self.gat_conv3 = GATv2Conv(512, 512, residual=True, dropout=0.1)

        self.edge_index = self.create_edges(width, width/30)
        self.edge_index_1 = self.create_edges(600, width/30)
        self.edge_index_2 = self.create_edges(300, width/30)
        
    def apply_convs(self, x):
        """Apply convolutional blocks based on input height."""
        num_layers_to_apply = 9
        if self.x_high >= 32:
            num_layers_to_apply += 1
        if self.x_high >= 64:
            num_layers_to_apply += 1

        for i in range(num_layers_to_apply):
            x = self.conv_blocks[i](x)
        return x

    def create_edges(self, num_nodes, range_size=30):
        edges = []
        for i in range(num_nodes):
            for j in range(int(-range_size/2), int(range_size/2 + 1)):  # Connect range_size nodes before and after
                neighbor = (i + j) % num_nodes  # Rotational handling
                if neighbor != i:
                    edges.append((i, neighbor))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def filter_edges(self, edge_index, topk_indices, num_nodes):
        """Filters edges based on the selected top-k nodes."""
        mask = (edge_index[0].unsqueeze(1) == topk_indices.unsqueeze(0)).any(dim=1)
        mask &= (edge_index[1].unsqueeze(1) == topk_indices.unsqueeze(0)).any(dim=1)
        filtered_edges = edge_index[:, mask]
        
        # Map old node indices to new ones
        index_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(topk_indices)}
        filtered_edges = torch.stack([
            torch.tensor([index_mapping[node.item()] for node in filtered_edges[0]]),
            torch.tensor([index_mapping[node.item()] for node in filtered_edges[1]])
        ])
        return filtered_edges

    def forward(self, x_l):
        ## Convolutional processing
        out_l = self.apply_convs(x_l)
        out_l = self.convLast1(out_l)  # [batch, 256, 1, width]
        batch_size, _, _, num_queries = out_l.size()
        out_l = out_l.permute(0, 3, 1, 2).squeeze(3)  # [batch, query, feature_dim]

        ## Graph Attention Network processing
        out_list = []
        
        for i in range(batch_size):
            node_features = out_l[i]  # [num_queries, feature_dim]
            edge_index = self.edge_index.to(out_l.device)
            edge_index_1 = self.edge_index_1.to(out_l.device)
            edge_index_2 = self.edge_index_2.to(out_l.device)
            
            # Step 1: GAT Conv1
            node_features = self.gat_conv1(node_features, edge_index)  # [num_queries, feature_dim]
            
            # Step 2: Top-k selection (900 -> 600)
            scores = torch.norm(node_features, dim=1)  # Compute importance scores
            topk_indices = torch.topk(scores, 600, dim=0).indices  # Select top 600 nodes
            node_features = node_features[topk_indices]  # Filter top-k nodes
            # edge_index_1 = self.filter_edges(edge_index, topk_indices, num_nodes=num_queries).to(out_l.device)  # Update edges

            # Step 3: GAT Conv2
            node_features = self.gat_conv2(node_features, edge_index_1)  # [600, feature_dim]

            # Step 4: Top-k selection (600 -> 300)
            scores = torch.norm(node_features, dim=1)  # Recompute importance scores
            topk_indices = torch.topk(scores, 300, dim=0).indices  # Select top 300 nodes
            node_features = node_features[topk_indices]  # Filter top-k nodes
            # edge_index_2 = self.filter_edges(edge_index_1, topk_indices, num_nodes=600).to(out_l.device)  # Update edges

            # Step 5: GAT Conv3
            node_features = self.gat_conv3(node_features, edge_index_2)  # [300, feature_dim]

            # Step 6: Append to output list
            out_list.append(node_features)

        out_l = torch.stack(out_list, dim=0)  # [batch_size, num_queries, feature_dim]
        out_l = out_l.permute(0, 2, 1).unsqueeze(3) # [batch, feature_dim(512), num_queries, 1]

        ## NetVLAD processing
        out_l = self.relu(self.convLast2(out_l)) # [batch, 1024, num_queries, 1]
        out_l = F.normalize(out_l, dim=1)
        out_l = self.net_vlad(out_l)
        out_l = F.normalize(out_l, dim=1)

        return out_l

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
