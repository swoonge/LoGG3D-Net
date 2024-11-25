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
from backbones.vit.ViT import ViT_feature_extractor

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
class OverlapViT(nn.Module):
    def __init__(self, height=64, width=900, channels=1, norm_layer=None, ):
        super(OverlapViT, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Replacing the original Conv2D-based feature extractor with ViT (Vision Transformer)
        self.vit = ViT_feature_extractor(image_size = (64, 904),
                                        patch_size = (8,8),
                                        dim = 256,
                                        depth = 6,
                                        heads = 4,
                                        mlp_dim = 1024,
                                        dropout = 0.1,
                                        emb_dropout = 0.1
                                    )
        # Adjusting the number of transformer encoder layers to reduce model size
        # self.vit.encoder.layers = self.vit.encoder.layers[:6]  # Retain only the first 6 layers
        # self.vit.head = nn.Linear(self.vit.head.in_features, 256)
        # self.vit.conv_proj = nn.Conv2d(1, self.vit.conv_proj.out_channels, kernel_size=(1, 1), stride=(1, 1))
        
        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, activation='relu', batch_first=True, dropout=0.)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.convLast2 = nn.Conv2d(512, 512, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(512)

        self.linear = nn.Linear(256 * 100, 256)

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        self.net_vlad = NetVLADLoupe(feature_size=256, max_samples=100, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)
        
    def circle_pad_to_multiple_of_8(self, img):
        height, width = img.shape[2:]
        print("height: ", height)
        print("width: ", width)
        target_width = ((width + 7) // 8) * 8  # 8의 배수로 올림
        pad_width = target_width - width

        if pad_width == 0:
            return img  # 이미 8의 배수인 경우 그대로 반환

        # 오른쪽에 패딩 추가
        pad = (0, pad_width, 0, 0)  # (left, right, top, bottom)
        padded_img = F.pad(img, pad, mode='constant', value=0)

        return padded_img

    def forward(self, x_l):
        # Using ViT as feature extractor
        
        x_l = self.circle_pad_to_multiple_of_8(x_l) # >> [13, 1, 64, 904]
        print(x_l.size())
        out_l = self.vit(x_l) # >> [13, 256]
        
        print(out_l.size()) 
        out_l_1 = out_l.unsqueeze(3)  # Adding an extra dimension for consistency
        

        # Transformer processing
        out_l = out_l_1.squeeze(3)
        out_l = out_l.permute(0, 2, 1)
        print(out_l_1.size())

        # Selecting top-k queries (e.g., top 100)
        k = 100
        out_l, _ = torch.topk(out_l, k, dim=1)
        
        out_l = self.transformer_encoder(out_l)
        out_l = out_l.permute(0, 2, 1)
        out_l = out_l.unsqueeze(3)
        out_l = torch.cat((out_l_1[:, :, :k, :], out_l), dim=1)
        out_l = self.relu(self.convLast2(out_l))
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
