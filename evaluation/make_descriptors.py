# Developed by Soowoong Park in KIST RobotVisionLab 
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils.utils import *
import torch
from utils.data_loaders.make_dataloader import *
from models.pipelines.pipeline_utils import *
from tools.utils.evaluator import Evaluator
from tools.utils.evaluator_diff_ch import Evaluator as Evaluator_diff_ch
import pickle

@torch.no_grad()
def __main__(test_models_list, linspace, diff_ch = None):
    for test_models in test_models_list:

        model_path = test_models[0]
        save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        for checkpoint_name in test_models[1]:
            checkpoint_path = os.path.join('../checkpoints', model_path, checkpoint_name)
            if diff_ch == None:
                evaluator = Evaluator(checkpoint_path, linspace)
            else:
                evaluator = Evaluator_diff_ch(checkpoint_path, linspace, diff_ch)

            evaluator.make_and_save_descriptors(64)

            del evaluator

if __name__ == '__main__':
    test_models_list = [
        # ['OverlapTransformer_geo/2024-10-24_16-07-32',['epoch_best_82.pth', 'epoch_best_187.pth']], # kitti08 geo
        # ['OverlapTransformer/2024-10-12_08-41-37', ['epoch_best_66.pth', 'epoch_best_46.pth', 'epoch_best_97.pth']], # triplet lazy False 00
        # ['LoGG3D/kitti_10cm_loo', ['3n24h_Kitti_v10_q29_10s0.pth',
        #                            '3n24h_Kitti_v10_q29_10s2.pth',
        #                            '3n24h_Kitti_v10_q29_10s5.pth',
        #                            '3n24h_Kitti_v10_q29_10s6.pth',
        #                            '3n24h_Kitti_v10_q29_10s7.pth',
        #                            '3n24h_Kitti_v10_q29_10s8.pth']], # triplet lazy False 00
        # ['LoGG3D/kitti_10cm_loo', ['3n24h_Kitti_v10_q29_10s0.pth']], # triplet lazy False 00
        ['OverlapTransformer/2024-10-11_00-46-31', ['epoch_best_71.pth']],
        ['OverlapTransformer/2024-10-10_15-48-19', ['epoch_best_34.pth']]
    ]
    linspace = [0.001, 1.0, 1000]
    __main__(test_models_list, linspace, diff_ch=[64,16])