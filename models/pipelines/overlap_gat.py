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

"""
    Feature extracter of OverlapTransformer.
    Args:
        height: the height of the range image (64 for KITTI sequences). 
                 This is an interface for other types LIDAR.
        width: the width of the range image (900, alone the lines of OverlapNet).
                This is an interface for other types LIDAR.
        channels: 1 for depth only in our work. 
                This is an interface for multiple cues.
        norm_layer: None in our work for better model.
        use_transformer: Whether to use MHSA.
"""
class OverlapGAT(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer = True):
        super(OverlapGAT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.x_high = height

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,1), stride=(1,1), bias=False) # [batch, 16, 60, 900]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), bias=False) # [batch, 32, 29, 900]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), bias=False)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,1), stride=(2,1), bias=False)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*900, 256)

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        # Graph Attention Network
        self.gat_conv = GATConv(256, 256, heads=4, concat=False, dropout=0.2)

    def create_edges(self, num_nodes):
        # 이 메서드는 인스턴스 메서드로 클래스 내에서 정의되어 있어야 함
        edges = []
        for i in range(num_nodes):
            for j in range(-15, 15):  # Connect 90 nodes in total, 45 before and 45 after
                neighbor = (i + j) % num_nodes  # Rotational handling for the ends
                if neighbor != i:
                    edges.append((i, neighbor))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x_l):

        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))
        out_l = self.relu(self.conv8(out_l))
        out_l = self.relu(self.conv9(out_l))
        if self.x_high >= 32:
            out_l = self.relu(self.conv10(out_l))
        if self.x_high >= 64:
            out_l = self.relu(self.conv11(out_l))

        out_l_1 = out_l.permute(0,1,3,2)
        out_l_1 = self.relu(self.convLast1(out_l_1))

        # Prepare input for GAT
        batch_size, feature_dim, num_queries, _ = out_l_1.size()
        out_l = out_l_1.squeeze(3).permute(0, 2, 1)  # [batch, query, feature_dim]
        
        edge_index = self.create_edges(num_queries).to(out_l.device)
        # edge_index = edge_index  # edge_index를 동일한 디바이스로 이동
        out_list = []

        for i in range(batch_size):
            node_features = out_l[i]  # [num_queries, feature_dim]
            gat_out = self.gat_conv(node_features, edge_index)
            out_list.append(gat_out)

        out_l = torch.stack(out_list, dim=0)  # [batch_size, num_queries, feature_dim]
        out_l = out_l.permute(0, 2, 1).unsqueeze(3)  # [batch, feature_dim, query, H=1]

        out_l = torch.cat((out_l_1, out_l), dim=1)
        out_l = self.relu(self.convLast2(out_l))
        out_l = F.normalize(out_l, dim=1)
        out_l = self.net_vlad(out_l)
        out_l = F.normalize(out_l, dim=1)

        return out_l


class OverlapGATv2(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer=True):
        super(OverlapGATv2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.x_high = height
        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5, 1), stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128 * 900, 256)

        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        # Graph Attention Network (GATv2Conv)
        self.gat_conv1 = GATv2Conv(256, 256, residual=True, dropout=0.1)
        self.gat_conv2 = GATv2Conv(256, 512, residual=True, dropout=0.1)
        self.gat_conv3 = GATv2Conv(512, 512, residual=True, dropout=0.1)

    def create_edges(self, num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(-15, 15):  # Connect 90 nodes in total, 45 before and 45 after
                neighbor = (i + j) % num_nodes  # Rotational handling for the ends
                if neighbor != i:
                    edges.append((i, neighbor))
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x_l):
        out_l = self.relu(self.conv1(x_l))
        out_l = self.relu(self.conv2(out_l))
        out_l = self.relu(self.conv3(out_l))
        out_l = self.relu(self.conv4(out_l))
        out_l = self.relu(self.conv5(out_l))
        out_l = self.relu(self.conv6(out_l))
        out_l = self.relu(self.conv7(out_l))
        out_l = self.relu(self.conv8(out_l))
        out_l = self.relu(self.conv9(out_l))
        if self.x_high >= 32:
            out_l = self.relu(self.conv10(out_l))
        if self.x_high >= 64:
            out_l = self.relu(self.conv11(out_l)) # [batch, 128, 1, 900]

        out_l = self.relu(self.convLast1(out_l)) # [batch, 256, 1, 900]

        batch_size, _, _, num_queries = out_l.size()
        out_l = out_l.permute(0, 3, 1, 2).squeeze(3)# [batch, query, feature_dim]

        edge_index = self.create_edges(num_queries).to(out_l.device)
        out_list = []

        for i in range(batch_size):
            node_features = out_l[i]
            gat_out = self.gat_conv1(node_features, edge_index) # [num_queries, feature_dim(256)]
            gat_out = self.gat_conv2(gat_out, edge_index) # [num_queries, feature_dim(512)]
            gat_out = self.gat_conv3(gat_out, edge_index) # [num_queries, feature_dim(512)]
            out_list.append(gat_out)

        out_l = torch.stack(out_list, dim=0) # [batch, num_queries, feature_dim(512)]
        out_l = out_l.permute(0, 2, 1).unsqueeze(3) # [batch, feature_dim(512), num_queries, 1]

        # out_l = torch.cat((out_l_1, out_l), dim=1)
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
