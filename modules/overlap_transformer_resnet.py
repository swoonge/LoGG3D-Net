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

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
# from tools.read_samples import read_one_need_from_seq
import yaml

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

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, padding_mode='circular'):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, padding_mode=padding_mode, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add skip connection
        out = self.relu(out)
        return out
    
class ResNetPanoramaCNN(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, num_blocks=[3, 3, 3], use_transformer=True):
        super(ResNetPanoramaCNN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer
        self.in_planes = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='circular', bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Define ResNet blocks
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, norm_layer=norm_layer)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=1, norm_layer=norm_layer)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=1, norm_layer=norm_layer)
        
        # Final convolution to adjust to desired output size (128, 1, 300)
        # self.final_conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='circular', bias=False)
        # self.bn_final = norm_layer(128)

    def _make_layer(self, planes, num_blocks, stride, norm_layer):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, norm_layer))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        out = self.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        out = self.layer1(out)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.pool2(out)
        out = self.layer3(out)

        # Final convolution for output adjustment
        # out = self.relu(self.bn_final(self.final_conv(out)))
        
        return out
    
class ResNetPanoramaCNNViT(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, num_blocks=[3, 3, 3], use_transformer=True):
        super(ResNetPanoramaCNNViT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer
        self.in_planes = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, padding_mode='circular', bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Define ResNet blocks
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, norm_layer=norm_layer)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, norm_layer=norm_layer)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, norm_layer=norm_layer)

    def _make_layer(self, planes, num_blocks, stride, norm_layer):
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, norm_layer))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        out = self.relu(self.bn1(self.conv1(x)))
        
        # ResNet layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Final convolution for output adjustment
        # out = self.relu(self.bn_final(self.final_conv(out)))
        
        return out
    
class ReduceHeightNet(nn.Module):
    def __init__(self, norm_layer=None):
        super(ReduceHeightNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # ([6, 256, 16, 225])
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(5, 1), stride=(1, 1), bias=False) 
        self.bn1 = norm_layer(256)  # [6, 256, 12, 225]

        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), bias=False) 
        self.bn2 = norm_layer(256)  # [6, 256, 5, 225]
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(2, 1), bias=False) # ([6, 256, 1, 225])
        self.bn3 = norm_layer(256)  # [6, 256, 2, 225]

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(2, 1), stride=(2, 1), bias=False) # ([6, 256, 1, 225])
        self.bn4 = norm_layer(256)  # [6, 256, 1, 225]

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), stride=(2, 1), bias=False) # ([6, 256, 1, 225])
        self.bn5 = norm_layer(256)  # [6, 256, 1, 225]

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = self.relu(self.bn5(self.conv5(out)))
        
        return out  # 최종 출력 크기: torch.Size([6, 256, 1, 225])

class OverlapTransformer_resnet(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, use_transformer = True, mode = 'original'): # late_1dconv
        super(OverlapTransformer_resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer
        self.mode = mode

        self.relu = nn.ReLU(inplace=True)
        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        if self.mode == 'original':
            self.cnn_backborn = ResNetPanoramaCNN()
            self.conv_backborn_last = ReduceHeightNet()
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=True, dropout=0.1)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=225, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        elif self.mode == 'CViT':
            self.cnn_backborn = ResNetPanoramaCNNViT()
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=True, dropout=0.1)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=904, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        # self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        # self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*300, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """

        """TODO: How about adding some dense layers?"""
        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, x_l):
        out_l = self.cnn_backborn(x_l)
        # print(out_l.shape) # torch.Size([6, 256, 16, 225]) [batch, dim, height, width]
        if self.mode == 'original':
            out_l = self.conv_backborn_last(out_l)
            # print(out_l.shape) # torch.Size([6, 256, 1, 225]) [batch, dim, height, width]
            
            out_l_1 = out_l.permute(0,1,3,2)
            # print(out_l_1.shape) # torch.Size([6, 256, 225, 1]) [batch, dim, width, height]

            """Using transformer needs to decide whether batch_size first"""
            out_l = out_l_1.squeeze(3) # [6, 256, 225] [batch, dim, width]
            out_l = out_l.permute(0, 2, 1) # [225, 6, 256] [width, batch, dim]
            # print(out_l.shape) # torch.Size([225, 6, 256])
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(0, 2, 1)
            # print(out_l.shape) # torch.Size([6, 256, 225])
            out_l = out_l.unsqueeze(3) # [6, 256, 225, 1]
            out_l = torch.cat((out_l_1, out_l), dim=1) # [6, 512, 225, 1]
            out_l = self.relu(self.convLast2(out_l))
            
            out_l = F.normalize(out_l, dim=1)
            # print(out_l.shape) # torch.Size([6, 1024, 225, 1])
            out_l = self.net_vlad(out_l) 
            # print(out_l.shape) # torch.Size([6, 256])
            out_l = F.normalize(out_l, dim=1)

        elif self.mode == 'CViT':
            # print(out_l.shape) # torch.Size([6, 256, 16, 225]) [batch, dim, height, width]
            out_l_1 = out_l.view(out_l.shape[0], 256, 8 * 113) # [6, 256, 904] [batch, dim, height * width]

            """Using transformer needs to decide whether batch_size first"""
            out_l = out_l_1.permute(2, 0, 1) # [904, 6, 256] [height * width, batch, dim]
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0) # [6, 256, 904] [batch, dim, height * width]
            out_l = torch.cat((out_l_1, out_l), dim=1) # [6, 512, height * width]
            out_l = out_l.unsqueeze(3) # [6, 512, 904, 1] [6, 256, height * width, 1]
            out_l = self.relu(self.convLast2(out_l)) # [6, 1024, 904, 1]
            
            out_l = F.normalize(out_l, dim=1) # [6, 1024, 904, 1]
            out_l = self.net_vlad(out_l) # [6, 256]
            out_l = F.normalize(out_l, dim=1) # [6, 256]

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
