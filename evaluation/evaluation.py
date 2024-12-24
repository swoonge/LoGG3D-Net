# Developed by Soowoong Park in KIST RobotVisionLab 
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import re
from tools.utils.utils import *
import torch
from utils.data_loaders.make_dataloader import *
from models.pipelines.pipeline_utils import *
from tools.utils.evaluator import Evaluator
from tools.utils.evaluator_diff_ch import Evaluator_diff_ch
import pickle

@torch.no_grad()
def __main__(test_models_list, test_all=False, test_for_val_set=False, multi_ch=None):
    for test_models in test_models_list:
        metrics_dict = {}
        model_path = test_models[0]
        save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

        if test_all:
            check_points = [f for f in os.listdir(os.path.join('../checkpoints', model_path)) if f.endswith('.pth')]
            check_points = sorted(check_points, key=lambda x: int(re.search(r'\d+', x).group()))[10:]
        else:
            check_points = test_models[1]

        for checkpoint_name in check_points:
            results_dict = {}
            checkpoint_path = os.path.join('../checkpoints', model_path, checkpoint_name)
            if multi_ch is None:
                evaluator = Evaluator(checkpoint_path, test_dataset_forced=None, test_seq_forced=None, test_for_val_set=test_for_val_set) # '2012-06-15'
            else:
                evaluator = Evaluator_diff_ch(checkpoint_path, test_dataset_forced=None, test_seq_forced=None, test_for_val_set=test_for_val_set, multi_ch=multi_ch)
            matching_results, metrics_list = evaluator.run()

            max_f1_score_idx, top_metrics = evaluator.analyze_metrics(metrics_list)
            results_dict['matching_results'] = matching_results[max_f1_score_idx]
            results_dict['metrics_list'] = metrics_list
            results_dict['max_f1_score_idx'] = max_f1_score_idx
            results_dict['linspace'] = [evaluator.args.cd_thresh_min, evaluator.args.cd_thresh_max, evaluator.args.num_thresholds]

            save_folder_path = os.path.join(os.path.dirname(__file__), 'results', model_path)
            with open(os.path.join(save_folder_path, 'evaluation_results_' + checkpoint_name.split('.')[0] + '_64ch.pkl'), 'wb') as file:
                pickle.dump(results_dict, file)

            metrics_dict[checkpoint_name] = top_metrics

            del evaluator, matching_results, metrics_list, results_dict

        print('\n\n' + '*'*50 + f'\n* [{model_path}] evaluation results')
        for key in metrics_dict.keys():
            print(f'*\t[checkpoint: {key}]\tF1-Score: {metrics_dict[key][0]:.3f}, Recall: {metrics_dict[key][1]:.3f}, Acc: {metrics_dict[key][2]:.3f}, at thresholds: {metrics_dict[key][3]:.3f}')
        print('*'*50 +'\n\n')

if __name__ == '__main__':
    test_models_list = [
        # # LOGG3D
        # ['LoGG3D/kitti_10cm_loo', ['3n24h_Kitti_v10_q29_10s0.pth']],
        # ['LOGG3D/2024-11-22_11-00-00_LOGG3D_default_kitti05', ['epoch_30.pth']],
        # ['LoGG3D/kitti_10cm_loo', ['3n24h_Kitti_v10_q29_10s8.pth']],
        ['LOGG3D/2024-12-19_18-00-54_LOGG3D_gm02', ['epoch_30.pth', 'epoch_best_41.pth']],
        ['LOGG3D/2024-12-19_18-02-15_LOGG3D_gm03', ['epoch_30.pth', 'epoch_best_41.pth']],
        # ['LOGG3D/2024-11-25_11-00-00_LOGG3D_default_nclt', ['epoch_best_42.pth']],
        # ['LOGG3D/2024-11-25_11-00-00_LOGG3D_default_nclt', ['epoch_best_42.pth']],

        # # OverlapTransformer
        # ['OverlapTransformer/2024-11-22_19-16-43_OT_default_kitti00',['epoch_best_36.pth']],
        # ['OverlapTransformer/2024-11-22_19-23-31_OT_default_kitti05',['epoch_30.pth', 'epoch_best_40.pth']],
        # ['OverlapTransformer/2024-11-22_19-24-19_OT_default_kitti08',['epoch_best_30.pth', 'epoch_best_36.pth']],
        # ['OverlapTransformer/2024-11-28_10-19-41_OT_default_kitti08_random_rotation', ['epoch_best_41.pth']],
        # ['OverlapTransformer/2024-11-22_12-13-26_OT_default_gm02', ['epoch_30.pth', 'epoch_best_49.pth']],
        # ['OverlapTransformer/2024-11-22_12-14-03_OT_default_gm03', ['epoch_30.pth', 'epoch_best_49.pth']],
        # ['OverlapTransformer/2024-11-25_11-52-28_OT_default_nclt', ['epoch_47.pth']],
        # ['OverlapTransformer/original', ['pretrained_overlap_transformer.pth.tar']],
        # ['OverlapTransformer/2024-12-04_11-41-56_OT_topk_reduceDim_kitti00', ['epoch_46.pth']],
        # ['OverlapTransformer/2024-12-04_15-59-48_OT_reduceDim_kitti00', ['epoch_38.pth']],

        # # OverlapNetTransformer
        # ['OverlapNetTransformer/2024-11-22_11-33-49_OLT_default_kitti00', ['epoch_30.pth']],
        # ['OverlapNetTransformer/2024-11-22_11-34-32_OLT_default_kitti05', ['epoch_30.pth']],
        # ['OverlapNetTransformer/2024-11-22_11-35-33_OLT_default_kitti08', ['epoch_30.pth']],
        # ['OverlapNetTransformer/2024-11-22_12-08-15_OLT_default_gm02', ['epoch_30.pth', 'epoch_best_41.pth']],
        # ['OverlapNetTransformer/2024-11-22_12-07-08_OLT_default_gm03', ['epoch_30.pth', 'epoch_best_41.pth']],
        # ['OverlapNetTransformer/2024-11-25_11-06-28_OLT_default_nclt', ['epoch_best_42.pth']],

        # CVTNet
        # ['CVTNet/2024-11-26_20-08-56_CVTNet_default_kitti00', ['epoch_32.pth', ]],
        # ['CVTNet/2024-11-26_11-11-57_CVTNet_default_kitti05', ['epoch_best_27.pth']],
        # ['CVTNet/2024-11-26_11-13-05_CVTNet_default_kitti08', ['epoch_40.pth']],
        # ['CVTNet/2024-11-26_12-08-27_CVTNet_default_kitti08_random_rotation', ['epoch_46.pth']],
        # ['CVTNet/2024-11-25_11-30-07_CVTNet_default_gm02', ['epoch_best_34.pth']],
        # ['CVTNet/2024-11-25_11-30-59_CVTNet_default_gm03', ['epoch_49.pth']],
        # ['CVTNet/2024-11-25_11-11-42_CVTNet_default_nclt', ['epoch_47.pth']],
        # ['CVTNet/pretrained_NCLT', ['pretrained_model.pth.tar']],

        # # OverlapGAT
        # ['OverlapGAT/2024-11-25_13-23-14_OTGAT_default_kitti00test', ['epoch_46.pth']],

        # OverlapGATv2
        # ['OverlapGATv2/2024-11-27_13-03-36_OTGATv2_default_kitti00', ['epoch_best_13.pth']],
        # ['OverlapGATv2/2024-11-29_11-13-27_OTGATv2_default_kitti08', ['epoch_best_29.pth']],
        # ['OverlapGATv2_5/2024-12-02_20-11-27_OTGATv2_5_default_kitti00', ['epoch_49.pth', 'epoch_41.pth', 'epoch_best_39.pth']],
        # ['OverlapGATv3/2024-12-03_21-35-45_OverlapGATv3_default_kitti00', ['epoch_35.pth']],

        # OverlapGATNet_cnn
        # ['OverlapGATNet/2024-12-05_17-31-57_OTGATNet_cnn_kitti00', ['epoch_best_23.pth']],
        # ['OverlapGATNet/2024-12-06_15-35-00_OTGATNet_cnn_kitti08', ['epoch_best_46.pth']],
        # ['OverlapGATNet/2024-12-07_17-15-46_OTGATNet_cnn_gm02', ['epoch_best_33.pth']],
        
        # ['OverlapGATNet/2024-12-13_10-49-52_OTGATNet_cnn_gm02', ['epoch_best_22.pth', 'epoch_best_25.pth', 'epoch_best_36.pth', 'epoch_57.pth']], # 22, 25가 best
        # ['OverlapGATNet/2024-12-13_10-49-31_OTGATNet_cnn_gm03', ['epoch_37.pth']],
        # ['OverlapGATNet/2024-12-13_10-50-08_OTGATNet_cnn_nclt', ['epoch_59.pth', 'epoch_best_102.pth']],
        # ['OverlapGATNet/2024-12-09_21-42-47_OTGATNet_cnn_kitti00', ['epoch_best_37.pth']],
        # ['OverlapGATNet/2024-12-09_21-43-03_OTGATNet_cnn_kitti05', ['epoch_46.pth']],
        # ['OverlapGATNet/2024-12-09_21-43-19_OTGATNet_cnn_kitti08', ['epoch_29.pth']],
    ]

    __main__(test_models_list, test_all=True, test_for_val_set=True)
    # __main__(test_models_list, test_all=False, test_for_val_set=False)
    # __main__(test_models_list, test_all=False, test_for_val_set=False, multi_ch=[64, 16])