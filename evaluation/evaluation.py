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
import pickle

@torch.no_grad()
def __main__(test_models_list):
    for test_models in test_models_list:
        model_path = test_models[0]
        save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        for checkpoint_name in test_models[1]:
            results_dict = {}
            checkpoint_path = os.path.join('../checkpoints', model_path, checkpoint_name)
            evaluator = Evaluator(checkpoint_path)

            matching_results, metrics_list = evaluator.run()

            max_f1_score_idx = evaluator.analyze_metrics(metrics_list)
            results_dict['matching_results'] = matching_results[max_f1_score_idx]
            results_dict['metrics_list'] = metrics_list
            results_dict['max_f1_score_idx'] = max_f1_score_idx
            results_dict['linspace'] = [evaluator.args.cd_thresh_min, evaluator.args.cd_thresh_max, evaluator.args.num_thresholds]

            save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
            with open(os.path.join(save_folder_path, 'evaluation_results_' + checkpoint_name.split('.')[0] + '_64ch.pkl'), 'wb') as file:
                pickle.dump(results_dict, file)

            del evaluator, matching_results, metrics_list, results_dict

if __name__ == '__main__':
    test_models_list = [
        ['OverlapTransformer/2024-11-22_19-16-43_OT_default_kitti00',['epoch_48.pth', 'epoch_best_30.pth', 'epoch_best_36.pth']],
        ['OverlapGAT/2024-11-25_13-23-14_OTGAT_default_kitti00test', ['epoch_best_1.pth']]
    ]
    __main__(test_models_list)