# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
import glob
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils.utils import *
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import yaml
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score, precision_recall_curve
from scipy.spatial.distance import pdist, squareform
from utils.data_loaders.make_dataloader import *
from models.pipelines.pipeline_utils import *

# # load config ================================================================
# config_filename = '../config/config.yml'
# config = yaml.safe_load(open(config_filename))
# test_weights = config["demo1_config"]["test_weights"]
# # ============================================================================

class Evaluator_kitti:
    def __init__(self, args, seq) -> None:
        self.args = args
        self.sequence = f"{seq:02d}"
        self.dataset_path = os.path.join(args.kitti_dir, 'sequences', self.sequence)
        self.preprocessed_data_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data')

        self.pose_threshold = [3.0, 20.0]  # 실제 pose 거리 임계값 3m, 20m
        self.thresholds = np.linspace(args.cd_thresh_min, args.cd_thresh_max, int(args.num_thresholds))
        self.thresholds_num = len(self.thresholds)
        # self.threshold = args.OverlapTransformer_thresholds
        self.descriptors = []
        self.data = []

        self.load_kitti_poses_and_timestamps()
        # self.load_descriptors()

    def load_kitti_poses_and_timestamps(self):
        self.timestamps = np.array(load_timestamps(self.dataset_path + '/times.txt'))

        preprocessed_pose_path = os.path.join(self.preprocessed_data_path, self.args.eval_dataset, "poses_" + self.sequence + ".npy")
        if os.path.exists(preprocessed_pose_path):
            print("Loading poses from: ", preprocessed_pose_path)
            self.poses = np.load(preprocessed_pose_path)
        else:
            # load calibrations
            calib_file = os.path.join(self.dataset_path, 'calib.txt')
            T_cam_velo = load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            T_velo_cam = np.linalg.inv(T_cam_velo)

            # load poses
            poses_file = os.path.join(self.dataset_path, 'poses.txt')
            poses = load_poses(poses_file)
            pose0_inv = np.linalg.inv(poses[0])

            poses_new = []
            for pose in poses:
                poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
            self.poses = np.array(poses_new)

            if self.args.eval_save_descriptors:
                dir_path = os.path.dirname(preprocessed_pose_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                np.save(preprocessed_pose_path, poses)

    def load_descriptors(self):
        preprocessed_descriptor_path = os.path.join(self.preprocessed_data_path, self.args.eval_dataset, "descriptors_" + self.sequence + ".npy")
        if os.path.exists(preprocessed_descriptor_path):
            print("Loading descriptors ...")
            self.descriptors = np.load(preprocessed_descriptor_path)
            return True
        else:
            print("No descriptors found. should run the model first.")
            return False
        
    def save_descriptors(self):
        preprocessed_descriptor_path = os.path.join(self.preprocessed_data_path, self.args.eval_dataset, "descriptors_" + self.sequence + ".npy")
        if isinstance(self.descriptors, np.ndarray):
            if self.args.eval_save_descriptors:
                np.save(preprocessed_descriptor_path, self.descriptors)
        else:
            print("descriptors are not in np.ndarray format")

    def put_descriptor(self, descriptor):
        if len(self.descriptors) < len(self.poses):
            self.descriptors.append(descriptor[0])
        if len(self.descriptors) >= len(self.poses):
            if not isinstance(self.descriptors, np.ndarray):
                self.descriptors = np.array(self.descriptors)

    def evaluate(self):
        if len(self.descriptors) < len(self.poses):
            print("Not enough data to evaluate")
            print("Descriptors: ", len(self.descriptors))
            print("Poses: ", len(self.poses))
            return
        matching_results = self.find_matching_poses()

        metrics_list = []
        for th_idx in range(self.thresholds_num):
            metrics_list.append(self.calculate_metrics(matching_results[th_idx], top_k=5))
        return metrics_list
    
    def calculate_pose_distance(self, pose1, pose2):
        translation1 = pose1[:3, 3]
        translation2 = pose2[:3, 3]
        return np.linalg.norm(translation1 - translation2)

    def calculate_pose_distances(self):
        n = len(self.poses)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distances[i, j] = self.calculate_pose_distance(self.poses[i], self.poses[j])
                distances[j, i] = distances[i, j]
        return distances

    def find_matching_poses(self):
        # 미리 pose와 descriptor들 끼리의 거리 계산
        print("Calculating poses and descriptors distances ...")
        pose_distances = self.calculate_pose_distances()
        descriptor_distances = squareform(pdist(self.descriptors, 'euclidean'))
        
        print("Finding matching poses ...")

        start_time = self.timestamps[0]
        matching_results_list = [[] for _ in range(self.thresholds_num)]

        for i in tqdm(range(0, len(self.poses))):
            query_time = self.timestamps[i] 
            if (query_time - start_time - self.args.skip_time) < 0: # Build retrieval database using entries 30s prior to current query. 
                continue        

            # 0초 ~ 현재 시간까지의 pose 중에서 30초 이전의 pose까지만 사용
            tt = next(x[0] for x in enumerate(self.timestamps)
                if x[1] > (query_time - self.args.skip_time))
            
            revisit = []
            match_candidates = [[] for _ in range(self.thresholds_num)]
            
            for j in range(0, tt+1):
                for th_idx in range(self.thresholds_num):
                    if descriptor_distances[i, j] < self.thresholds[th_idx]:
                        match_candidates[th_idx].append((j, pose_distances[i, j], descriptor_distances[i, j]))
                if pose_distances[i, j] < self.pose_threshold[0]:
                    revisit.append(j)
            for th_idx in range(self.thresholds_num):
                match_candidates[th_idx].sort(key=lambda x: x[2])
                match_candidates[th_idx] = np.array(match_candidates[th_idx])
            
            for th_idx in range(self.thresholds_num):
                matches = []
                if match_candidates[th_idx].shape[0] > 0: # Positive Prediction 
                    for candidate in match_candidates[th_idx]:
                        #  matching된 j     gt에 있는 j
                        if candidate[1] <= self.pose_threshold[0]:
                            # True Positive (TP): 매칭에 성공하고, 그 pose가 실제 매칭되어야 하는 경우
                            matches.append((i, candidate[0], candidate[1], candidate[2], "tp"))
                        elif candidate[1] > self.pose_threshold[1]:
                            # False Positive (FP): 매칭에 성공했으나, 그 pose가 실제로 매칭되어야 하는 것이 아닌 경우
                            matches.append((i, candidate[0], candidate[1], candidate[2], "fp"))
                    # match_candidates가 존재하기에 모두 처리했는데도 매칭된 모두가 3~20m 사이에 있어 tp, fp 모두 안된 경우. 이 경우는 거의 없다.
                    if not matches:
                        if revisit:
                            # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                            matches.append((i, -1, -1, -1, "fn"))
                        else:
                            # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                            matches.append((i, -1, -1, -1, "tn"))
                else: # Negative Prediction
                    if revisit:
                        # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                        matches.append((i, -1, -1, -1, "fn"))
                    else:
                        # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                        matches.append((i, -1, -1, -1, "tn"))
                
                matching_results_list[th_idx].append(matches)
        
        return matching_results_list

    def calculate_metrics(self, matching_results, top_k=5):
        tp = 0  # True Positives
        tn = 0  # True Negatives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        total_attempts = len(matching_results)
        
        topk_tp = 0  # Top-K Recall 계산을 위한 변수
        
        for matches in matching_results:
            first_match = matches[0]  # 첫 번째 매칭 (top-1) 결과
            
            if first_match[4] == "tp":
                tp += 1
            elif first_match[4] == "tn":
                tn += 1
            elif first_match[4] == "fp":
                fp += 1
            elif first_match[4] == "fn":
                fn += 1
            
            # Top-K Recall 계산 (상위 K개의 매칭에서 적어도 하나가 True Positive일 경우 성공으로 간주)
            if any(match[4] == "tp" for match in matches[:top_k]):
                topk_tp += 1
        
        # 메트릭 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total_attempts if total_attempts > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        topk_recall = topk_tp / (topk_tp + fn) if (topk_tp + fn) > 0 else 0
        
        return {
            "True Positives": tp,
            "True Negatives": tn,
            "False Positives": fp,
            "False Negatives": fn,
            "Precision": precision,
            "Recall (TPR)": recall,
            "F1-Score": f1_score,
            "Accuracy": accuracy,
            "False Positive Rate (FPR)": fpr,
            "Top-{} Recall".format(top_k): topk_recall
        }

@torch.no_grad()
def __main__(args, model, device, file_name):

    evaluator = Evaluator_kitti(args, args.kitti_data_split['test'][0])

    load_descriptors_flag = False
    if load_descriptors_flag == False:
        test_loader = make_data_loader(args,
                                    args.test_phase,
                                    args.eval_batch_size,
                                    num_workers=args.test_num_workers,
                                    shuffle=False)
        
        print("===== Generating Descriptors =====")
        print("= Dataset: ", args.eval_dataset)
        print("= Data Size: ", len(test_loader.dataset))

        test_loader_progress_bar = tqdm(test_loader)

        for i, batch in enumerate(test_loader_progress_bar, 0):
            if i >= len(test_loader.dataset):
                break
            if args.pipeline == 'LOGG3D':
                lidar_pc = batch[0][0]
                input_st = make_sparse_tensor(lidar_pc, args.voxel_size).to(device=device)   
                output_desc, output_feats = model(input_st) 
                output_feats = output_feats[0]
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                evaluator.put_descriptor(global_descriptor)

            elif args.pipeline.split('_')[0] == 'OverlapTransformer':
                input_t = torch.tensor(batch[0][0]).type(torch.FloatTensor).to(device=device)
                input_t = input_t.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device=device)
                output_desc = model(input_t)
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                evaluator.put_descriptor(global_descriptor)            

        # evaluator.save_descriptors()
        print("=================================")
        
    print("Evaluating ...")
    metrics = evaluator.evaluate()

    f1_scores = [result["F1-Score"] for result in metrics]
    max_f1_index = f1_scores.index(max(f1_scores))
    best_metrics = metrics[max_f1_index]

    print("===== Evaluation Results =====")
    print("Best F1-Score: {:.3f} at thresholds {:.3f}".format(max(f1_scores), evaluator.thresholds[max_f1_index]))
    print("Matching Metrics :")
    for key, value in best_metrics.items():
        print(f"{key}: {value:.3f}")
    print("=================================")

    import pickle

    # results_OT_trained, results_LOGG3D_trained, results_OT_trained, results_OTsp_trained_181827
    save_folder_path = os.path.join(os.path.dirname(__file__), 'results', args.save_file_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    with open(save_folder_path + file_name + '.pkl', 'wb') as file:
        pickle.dump(metrics, file)


if __name__ == '__main__':
    from config.eval_config_new import get_config_eval
    from models.pipeline_factory import get_pipeline

    ## get config and device info
    args = get_config_eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## get all epoch results
    dir_path = os.path.dirname(args.checkpoint_name) if args.checkpoint_name[-4:] == '.pth' else args.checkpoint_name

    # file_list = os.listdir(dir_path)
    file_list = ['epoch_best_22.pth', 'epoch_best_24.pth', 'epoch_best_44.pth', 'epoch_best_48.pth']
    
    for file in file_list:
        file_name =  os.path.splitext(os.path.basename(file))[0]
        ## load model
        model = get_pipeline(args).to(device)

        ## load checkpoint
        print('Loading checkpoint from: ', os.path.join(dir_path, file))
        checkpoint = torch.load(os.path.join(dir_path, file))
        model.load_state_dict(checkpoint['model_state_dict']) # state_dict, model_state_dict
        # print('model training info: ', checkpoint['optimizer_state_dict'])
        model.eval()
        
        __main__(args, model, device, file_name)
