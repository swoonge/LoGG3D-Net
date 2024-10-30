# Developed by Xieyuanli Chen and Thomas Läbe
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
from tools.utils.evaluator import *
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

            save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
            with open(os.path.join(save_folder_path, 'evaluation_results_' + checkpoint_name.split('.')[0] + '.pkl'), 'wb') as file:
                pickle.dump(results_dict, file)

if __name__ == '__main__':
    test_models_list = [
        # ['OverlapTransformer_geo/2024-10-24_16-07-32',['epoch_best_82.pth', 'epoch_best_187.pth']], # kitti08 geo
        ['OverlapTransformer/2024-10-12_08-41-37', ['epoch_best_66.pth', 'epoch_best_46.pth', 'epoch_best_97.pth']], # triplet lazy False 00
    ]
    __main__(test_models_list)