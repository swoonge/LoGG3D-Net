import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.aggregators.SOP import *
# from models.backbones.spvnas.model_zoo import spvcnn
from models.pipelines.pipeline_utils import *
from models.backbones.kpconv.utils.config import *
from models.backbones.kpconv.models.architectures import KPFCNN
from aggregators.netvlad import NetVLADLoupe
from config.kpconv_config import KittiConfig

class LOGG3D_kpfcnn(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, use_transformer = True):
        super(LOGG3D_kpfcnn, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.x_high = height

        self.use_transformer = use_transformer

        KPFCNN_cfg = KittiConfig()
        self.kpfcnn = KPFCNN(KPFCNN_cfg)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, activation='relu', batch_first=True, dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(256, 512, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(512)

        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        self.net_vlad = NetVLADLoupe(feature_size=512, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        """TODO: How about adding some dense layers?"""
        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    
    def depth_image_to_pointcloud(self, depth_image):
        """
        Convert depth image [batch, depth_value, H, W] to point cloud [batch, H * W, 3]
        Args:
            depth_image (torch.Tensor): Depth image tensor of shape [batch, depth_value, H, W]
        Returns:
            torch.Tensor: Point cloud tensor of shape [batch, H * W, 3]
        """
        batch, _, H, W = depth_image.shape
        
        # Create grid for pixel indices
        pixel_x = torch.arange(W, device=depth_image.device).view(1, 1, 1, W).expand(batch, 1, H, W)
        pixel_y = torch.arange(H, device=depth_image.device).view(1, 1, H, 1).expand(batch, 1, H, W)
        
        # Compute radius (arc length along horizontal direction)
        radius = (2 * torch.pi * depth_image) * pixel_x / W
        
        # Compute height (Cartesian z-coordinate)
        height = depth_image * (pixel_y / (H - 1))  # Scale pixel_y to height range
        
        # Combine depth, radius, and height
        point_cloud = torch.stack([depth_image, radius, height], dim=-1)  # [batch, H, W, 3]
        
        # Reshape to [batch, H * W, 3]
        point_cloud = point_cloud.view(batch, H * W, 3)
        
        return point_cloud

    def forward(self, x_l):
        x_l = self.depth_image_to_pointcloud(x_l)
        x_l = self.kpfcnn(x_l)


        # """Using transformer needs to decide whether batch_size first"""
        # if self.use_transformer:
        #     out_l = out_l_1.squeeze(3)
        #     out_l = out_l.permute(0, 2, 1)
        #     out_l = self.transformer_encoder(out_l)
        #     out_l = out_l.permute(0, 2, 1)
        #     out_l = out_l.unsqueeze(3)
        #     out_l = torch.cat((out_l_1, out_l), dim=1)
        #     out_l = self.relu(self.convLast2(out_l))
        #     out_l = F.normalize(out_l, dim=1)
        #     out_l = self.net_vlad(out_l)
        #     out_l = F.normalize(out_l, dim=1)

        # else:
        #     out_l = torch.cat((out_l_1, out_l_1), dim=1)
        #     out_l = F.normalize(out_l, dim=1)
        #     out_l = self.net_vlad(out_l)
        #     out_l = F.normalize(out_l, dim=1)

        return x_l
