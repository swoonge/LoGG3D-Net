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
            results_dict = {}
            checkpoint_path = os.path.join('../checkpoints', model_path, checkpoint_name)
            if diff_ch == None:
                evaluator = Evaluator(checkpoint_path, linspace)
            else:
                evaluator = Evaluator_diff_ch(checkpoint_path, linspace, diff_ch)

            matching_results, metrics_list = evaluator.run()

            max_f1_score_idx = evaluator.analyze_metrics(metrics_list)
            results_dict['matching_results'] = matching_results[max_f1_score_idx]
            results_dict['metrics_list'] = metrics_list
            results_dict['max_f1_score_idx'] = max_f1_score_idx
            results_dict['linspace'] = linspace

            save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
            with open(os.path.join(save_folder_path, 'evaluation_results_' + checkpoint_name.split('.')[0] + '_64ch.pkl'), 'wb') as file:
                pickle.dump(results_dict, file)

            del evaluator, matching_results, metrics_list, results_dict

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
        # ['OverlapTransformer/2024-10-11_00-46-31', ['epoch_best_71.pth']],
        # ['OverlapTransformer/2024-10-10_15-48-19', ['epoch_best_34.pth']],
        # ['CVTNet/2024-11-07_16-15-36', ['epoch_best_42.pth', 'epoch_69.pth', 'epoch_141.pth']],
        # ['CVTNet/pretrained_NCLT',['pretrained_model.pth.tar']]
        # ['OverlapTransformer/2024-10-10_14-06-46',['epoch_best_27.pth', 'epoch_best_43.pth']],
        # ['CVTNet/2024-11-15_20-06-45',['epoch_best_21.pth', 'epoch_best_46.pth', 'epoch_24.pth']], # kitti00
        # ['CVTNet/2024-11-16_07-14-07',['epoch_best_11.pth', 'epoch_best_31.pth', 'epoch_26.pth', 'epoch_36.pth']], # kitti08
        # ['CVTNet/2024-11-15_20-02-02',['epoch_best_28.pth', 'epoch_best_36.pth', 'epoch_best_42.pth', 'epoch_30.pth']], # NCLT
        # ['OverlapTransformer/2024-11-15_10-49-15',['epoch_best_32.pth', 'epoch_best_47.pth']],
        # ['OverlapTransformer/2024-11-15_10-50-28',['epoch_best_32.pth', 'epoch_best_26.pth']],
        ['OverlapTransformer_nclt/2024-11-18_11-05-57',['epoch_46.pth', 'epoch_best_48.pth']],
    ]
    linspace = [0.001, 1.0, 1000]
    diff_ch = None #[64,16]
    __main__(test_models_list, linspace, diff_ch=diff_ch)