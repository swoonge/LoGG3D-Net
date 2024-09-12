# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.
import sys
import os
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
from tools.utils.utils import *
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import pdist, squareform
from utils.data_loaders.make_dataloader import *
from models.pipelines.pipeline_utils import *
from abc import ABC, abstractmethod
from time import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_file_name', type=str, default='results', help='save file name')
parser.add_argument('--vis', type=bool, default=False, help='visualize the results')

class Evaluator(ABC):
    def __init__(self, args) -> None:
        self.args = args
        self.ready_to_evaluate = False

        self.revisit_criterias = [args.revisit_criteria, args.not_revisit_criteria]
        self.thresholds = np.linspace(args.cd_thresh_min, args.cd_thresh_max, int(args.num_thresholds))
        self.thresholds_num = int(args.num_thresholds)

        self.load_data()

    def load_data(self):
        self.timestamps =  self.load_timestamps()
        self.poses = self.load_poses()
        self.descriptors = self.load_descriptors() if self.args.eval_save_descriptors else np.empty((0, 256))
        self.raw_datas = np.empty((0, 256))

    @abstractmethod
    def load_timestamps(self):
        return np.array([])

    @abstractmethod
    def load_poses(self):
        return np.array([])

    @abstractmethod
    def load_descriptors(self):
        return np.array([])
    
    def put_descriptor(self, descriptor):
        self.descriptors = np.append(self.descriptors, descriptor, axis = 0)
        self.ready_to_evaluate = True if len(self.descriptors) >= len(self.poses) else False
        if self.args.eval_save_descriptors and self.ready_to_evaluate:
            self.save_descriptors()

    def save_descriptors(self):
        preprocessed_descriptor_path = os.path.join(self.preprocessed_data_path, self.args.eval_dataset, "descriptors_" + self.sequence + ".npy")
        if isinstance(self.descriptors, np.ndarray):
            if self.args.eval_save_descriptors:
                np.save(preprocessed_descriptor_path, self.descriptors)
        else:
            print("descriptors are not in np.ndarray format")

    def put_raw_data(self, raw_data):
        self.raw_datas = np.append(self.raw_datas, raw_data, axis = 0)
    
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
    
    def matching(self, top_k=1):
        pose_distances = self.calculate_pose_distances()
        descriptor_distances = squareform(pdist(self.descriptors, 'euclidean'))
        
        poses_progress_bar = tqdm(range(len(self.poses)), desc="Matching...", leave=True)

        start_time = self.timestamps[0]
        matching_results_list = [[] for _ in range(self.thresholds_num)]

        # Matching Loop for each pose
        for i in poses_progress_bar:
            query_time = self.timestamps[i] 
            if (query_time - start_time - self.args.skip_time) < 0:
                continue        

            # 0초 ~ 현재 시간까지의 pose 중에서 30초 이전의 pose까지만 사용
            tt = next(x[0] for x in enumerate(self.timestamps) if x[1] > (query_time - self.args.skip_time))
            
            is_revisit = False
            match_candidates = [[] for _ in range(self.thresholds_num)]

            # iterate 0 to tt poses
            for j in range(0, tt+1):
                for th_idx in range(self.thresholds_num):
                    if descriptor_distances[i, j] < self.thresholds[th_idx]:
                        match_candidates[th_idx].append((j, pose_distances[i, j], descriptor_distances[i, j]))
                if pose_distances[i, j] < self.revisit_criterias[0]:
                    is_revisit = True
            for th_idx in range(self.thresholds_num):
                match_candidates[th_idx].sort(key=lambda x: x[2])
                match_candidates[th_idx] = np.array(match_candidates[th_idx])
            
            for th_idx in range(self.thresholds_num):
                matches = []
                if match_candidates[th_idx].shape[0] > 0: # Positive Prediction 
                    for candidate in match_candidates[th_idx]:
                        #  matching된 j     gt에 있는 j
                        if candidate[1] <= self.revisit_criterias[0]:
                            # True Positive (TP): 매칭에 성공하고, 그 pose가 실제 매칭되어야 하는 경우
                            matches.append([i, candidate[0], candidate[1], candidate[2], "tp"])
                        elif candidate[1] > self.revisit_criterias[1]:
                            # False Positive (FP): 매칭에 성공했으나, 그 pose가 실제로 매칭되어야 하는 것이 아닌 경우
                            matches.append([i, candidate[0], candidate[1], candidate[2], "fp"])
                    # match_candidates가 존재하기에 모두 처리했는데도 매칭된 모두가 3~20m 사이에 있어 tp, fp 모두 안된 경우. 이 경우는 거의 없다.
                    if not matches:
                        if is_revisit:
                            # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                            matches.append([i, -1, -1, -1, "fn"])
                        else:
                            # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                            matches.append([i, -1, -1, -1, "tn"])
                else: # Negative Prediction
                    if is_revisit:
                        # False Negative (FN): 매칭에 실패했으나, 실제로 매칭해야 하는 것이 있는 경우
                        matches.append([i, -1, -1, -1, "fn"])
                    else:
                        # True Negative (TN): 매칭에 실패하고, 실제로도 매칭되는 것이 없는 경우
                        matches.append([i, -1, -1, -1, "tn"])
                
                if len(matches) > top_k:
                    matches = matches[:top_k]
                matching_results_list[th_idx].append(matches)
        
        return matching_results_list

    def metric(self, matching_results, top_k=1):
        tp = 0  # True Positives
        tn = 0  # True Negatives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        for matches in matching_results:
            if any(match[4] == "tp" for match in matches[:top_k]):
                tp += 1
            elif any(match[4] == "tn" for match in matches[:top_k]):
                tn += 1
            elif matches[0][4] == "fp":
                fp += 1
            elif matches[0][4] == "fn":
                fn += 1
        
        # 메트릭 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(matching_results) if len(matching_results) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
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
        }

    def evaluate(self, top_k=1):
        matching_results = self.matching(top_k=top_k)

        metrics_list = []
        for th_idx in range(self.thresholds_num):
            metrics_list.append(self.metric(matching_results[th_idx], top_k=top_k))
        return metrics_list

class Evaluator_kitti(Evaluator):
    def __init__(self, args, use_processed_data = False) -> None:
        '''
        args: config arguments
        note:
        - Required arguments in args:
            - args.kitti_dir: KITTI dataset directory
            - args.cd_thresh_min: minimum threshold for the Chamfer Distance
            - args.cd_thresh_max: maximum threshold for the Chamfer Distance
            - args.num_thresholds: number of thresholds for the Chamfer Distance
            - args.eval_dataset: evaluation dataset name
            - args.eval_save_descriptors: save descriptors or not
            - args.OverlapTransformer_thresholds: thresholds for the OverlapTransformer
        '''
        self.args = args
        self.sequence = f"{args.kitti_data_split['test'][0]:02d}"
        self.dataset_path = os.path.join(args.kitti_dir, 'sequences', self.sequence)
        self.preprocessed_data_path = os.path.join(os.path.dirname(__file__), 'preprocessed_data')
        self.use_processed_data = use_processed_data

        self.ready_to_evaluate = False
        self.revisit_criterias = [args.revisit_criteria, args.not_revisit_criteria]
        self.thresholds = np.linspace(args.cd_thresh_min, args.cd_thresh_max, int(args.num_thresholds))
        self.thresholds_num = int(args.num_thresholds)
        
        self.load_data()

    def load_timestamps(self):
        return np.array(load_timestamps(self.dataset_path + '/times.txt'))

    def load_poses(self):
        preprocessed_pose_path = os.path.join(self.preprocessed_data_path, self.args.eval_dataset, "poses_" + self.sequence + ".npy")
        if self.args.eval_save_descriptors and os.path.exists(preprocessed_pose_path):
            print("Loading poses from: ", preprocessed_pose_path)
            return np.load(preprocessed_pose_path)
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
            poses = np.array(poses_new)

            if self.args.eval_save_descriptors:
                dir_path = os.path.dirname(preprocessed_pose_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                np.save(preprocessed_pose_path, poses)
        return poses

    def load_descriptors(self):
        preprocessed_descriptor_path = os.path.join(self.preprocessed_data_path, self.args.eval_dataset, "descriptors_" + self.sequence + ".npy")
        if os.path.exists(preprocessed_descriptor_path) and self.args.eval_save_descriptors:
            print("Loading descriptors from: ", preprocessed_descriptor_path)
            self.ready_to_evaluate = True
            return np.load(preprocessed_descriptor_path)
        else:
            print("No descriptors found. should run the model first.")
            return np.empty((0, 256))

@torch.no_grad()
def __main__(args, model, device):

    evaluator = Evaluator_kitti(args, args.kitti_data_split['test'][0])

    if evaluator.ready_to_evaluate == False:
        test_loader = make_data_loader(args,
                                    args.test_phase,
                                    args.eval_batch_size,
                                    num_workers=args.test_num_workers,
                                    shuffle=False)
        
        test_loader_progress_bar = tqdm(test_loader)

        print("===== Generating Descriptors =====")
        print("= Dataset: ", args.eval_dataset)
        print("= Data Size: ", len(test_loader.dataset))

        for i, batch in enumerate(test_loader_progress_bar, 0):
            if i >= len(test_loader.dataset):
                break
            if args.eval_pipeline == 'LOGG3D':
                lidar_pc = batch[0][0]
                input_st = make_sparse_tensor(lidar_pc, args.voxel_size).to(device=device)   
                output_desc, output_feats = model(input_st) 
                output_feats = output_feats[0]
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                evaluator.put_descriptor(global_descriptor)

            elif args.eval_pipeline == 'OverlapTransformer':
                input_t = torch.tensor(batch[0][0]).type(torch.FloatTensor).to(device=device)
                input_t = input_t.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device=device)
                output_desc = model(input_t)
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                evaluator.put_descriptor(global_descriptor)

        evaluator.save_descriptors()
        print("=================================")
    
    metrics_topk = []
    for k in [1, 5, 20]:
        print('=====evaluating top', k, '=====')
        metrics = evaluator.evaluate(top_k=k)

        f1_scores = [result["F1-Score"] for result in metrics]
        max_f1_index = f1_scores.index(max(f1_scores))
        best_metrics = metrics[max_f1_index]

        print("===== Evaluation Results =====")
        print("Best F1-Score: {:.3f} at thresholds {:.3f}".format(max(f1_scores), evaluator.thresholds[max_f1_index]))
        print("Matching Metrics :")
        for key, value in best_metrics.items():
            print(f"{key}: {value:.3f}")
        print("=================================")

        if args.vis:
            plt.plot(evaluator.thresholds, f1_scores, marker='o')
            plt.title("F1-Score Values")
            plt.xlabel("Thresholds")  # x축 레이블 설정
            plt.ylabel("F1-Score")
            plt.grid(True)
            plt.show()

        metrics_topk.append(metrics)
    metrics_topk = np.array(metrics_topk)

    import pickle
    # results 리스트를 파일로 저장
    with open('/home/vision/GD_model/LoGG3D-Net/evaluation/results/' + args.save_file_name + '.pkl', 'wb') as file:
        pickle.dump(metrics_topk, file)


if __name__ == '__main__':
    from config.eval_config_new import get_config_eval
    from models.pipeline_factory import get_pipeline

    ## get config and device info
    args = get_config_eval()

    print(args.vis)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## load model
    model = get_pipeline(args.eval_pipeline)
    model.to(device)

    ## load checkpoint
    print('Loading checkpoint from: ', args.checkpoint_name)
    checkpoint = torch.load(args.checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict']) # state_dict, model_state_dict
    model.eval()
    
    __main__(args, model, device)
