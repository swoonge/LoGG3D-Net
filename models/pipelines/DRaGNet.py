# Dynamic Range Graph Network
import os, sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
from backbones.GAT.gatv2_conv import GATv2Conv as GATConv
from aggregators.netvlad import NetVLADLoupe


# Depth-aware Backbone Network
class DepthAwareBackbone(nn.Module):
    def __init__(self):
        super(DepthAwareBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((16, 150))  # 고정 크기 출력 (16, 150)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.adaptive_pool(x)  # 출력 크기 (16, 150)
        return x

# Sparse Representation
class SparseRepresentation(nn.Module):
    def __init__(self, threshold=0.5, topk=None):
        super(SparseRepresentation, self).__init__()
        self.threshold = threshold
        self.topk = topk

    def forward(self, x):
        if self.topk:
            # Top-k 방식으로 상위 k개 피처 선택
            values, indices = torch.topk(x.view(x.size(0), -1), self.topk, dim=-1)
            mask = torch.zeros_like(x.view(x.size(0), -1)).scatter_(-1, indices, 1).view_as(x)
        else:
            # Threshold 방식으로 피처 선택
            mask = (x > self.threshold).float()
        sparse_x = x * mask
        return sparse_x

# Graph Construction (3D-based)
class GraphConstruction3D:
    def __init__(self, k=10):
        self.k = k

    def spherical_to_cartesian(self, depth_map, theta, phi):
        r = depth_map  # 깊이 값
        x = r * torch.cos(phi) * torch.sin(theta)
        y = r * torch.sin(phi)
        z = r * torch.cos(phi) * torch.cos(theta)
        return torch.stack((x, y, z), dim=-1)  # (batch_size, num_nodes, 3)

    def construct_graph(self, nodes, depth_map):
        batch_size, num_nodes, _ = nodes.shape
        h, w = depth_map.shape[-2:]  # 깊이 이미지 크기
        theta = torch.linspace(-torch.pi, torch.pi, w).to(depth_map.device)
        phi = torch.linspace(-torch.pi / 2, torch.pi / 2, h).to(depth_map.device)

        # 3D 좌표 계산
        xyz = self.spherical_to_cartesian(depth_map, theta.unsqueeze(0), phi.unsqueeze(1))
        feature_xyz = xyz.view(batch_size, -1, 3)  # (batch_size, num_nodes, 3)

        # 3D 거리 계산
        distances = torch.cdist(feature_xyz, feature_xyz, p=2)
        edges = distances.topk(self.k, largest=False).indices
        return feature_xyz, edges

# Graph Neural Network
class GraphNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphNetwork, self).__init__()
        # self.gat1 = GATConv(in_channels, 128, heads=4, concat=True)
        # self.gat2 = GATConv(128 * 4, out_channels, heads=1, concat=False)

        self.gat1 = GATConv(in_channels, 128, residual=True, dropout=0.1)
        self.gat2 = GATConv(128, 256, residual=True, dropout=0.1)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        return x

# Global Descriptor (NetVLAD)
class NetVLADDescriptor(nn.Module):
    def __init__(self, feature_size, cluster_size, output_dim):
        super(NetVLADDescriptor, self).__init__()
        self.net_vlad = NetVLADLoupe(feature_size=feature_size, max_samples=150, cluster_size=cluster_size,
                                     output_dim=output_dim, gating=True, add_batch_norm=False,
                                     is_training=True)

    def forward(self, x):
        x = self.net_vlad(x)
        return x

# Full Model (DRaGNet3D)
class DRaGNet3D(nn.Module):
    def __init__(self):
        super(DRaGNet3D, self).__init__()
        self.backbone = DepthAwareBackbone()
        self.sparse_representation = SparseRepresentation(topk=1000)
        self.graph_network = GraphNetwork(in_channels=64, out_channels=256)
        self.global_descriptor = NetVLADDescriptor(feature_size=256, cluster_size=64, output_dim=256)
        self.graph_constructor = GraphConstruction3D(k=10)

    def forward(self, x, depth_map):
        x = self.backbone(x)  # Backbone을 통한 피처 추출 >> [6, 64, 16, 150])
        print("x shape:", x.shape)
        x = self.sparse_representation(x)  # Sparse Representation 적용 >> [6, 64, 16, 150]
        print("x shape:", x.shape)
        nodes = x.view(x.size(0), -1, x.size(2))  # (batch, nodes, features) >> [6, 9600, 16]
        print("nodes shape:", nodes.shape)
        nodes_3d, edge_index = self.graph_constructor.construct_graph(nodes, depth_map)  # 3D 그래프 생성
        print("nodes_3d shape:", nodes_3d.shape, "edge_index shape:", edge_index.shape)
        x = self.graph_network(nodes_3d, edge_index)  # GNN으로 피처 강화
        x = self.global_descriptor(x)  # NetVLAD로 글로벌 디스크립터 생성
        return x

# 모델 초기화 및 테스트
if __name__ == "__main__":
    model = DRaGNet3D()
    input_tensor = torch.randn(6, 1, 64, 900)  # 예제 입력 (batch_size, channels, height, width)
    depth_map = torch.randn(6, 64, 900)  # 예제 깊이 맵
    output = model(input_tensor, depth_map)
    print("Output Shape:", output.shape)  # 예상 출력: torch.Size([1, 256])
