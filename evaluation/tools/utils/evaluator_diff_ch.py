# Developed by Soowoong Park in KIST RobotVisionLab 
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.

import os, sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import numpy as np
from .utils import load_poses
from scipy.spatial.distance import pdist, squareform, cdist
from utils.data_loaders.kitti.kitti_rangeimage_dataset import load_timestamps
from utils.data_loaders.make_dataloader import *
from tools.utils.utils import *
from models.pipelines.pipeline_utils import *
from models.pipeline_factory import get_pipeline
from tqdm import tqdm
import pandas as pd
from config.eval_config import *

torch.backends.cudnn.benchmark = True

def calculate_pose_distances_with_pdist(poses):
    # 각 pose에서 translation 벡터 (x, y, z)를 추출
    translations = np.array([pose[:3, 3] for pose in poses])
    
    # pdist로 모든 쌍별 유클리드 거리 계산
    distances = squareform(pdist(translations, metric='euclidean'))
    
    return distances

# @torch.no_grad()
class Evaluator:
    def __init__(self, checkpoint_path, thresholds_linspace = [0.0, 1.0, 1000], multi_ch = [64,16]) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path)
        try:
            self.args = self.checkpoint['config']
        except:
            self.args = get_config_eval()
        if not hasattr(self.args, 'skip_time'):
            self.args.skip_time = 30  # Default value for skip_time
        self.multi_ch = multi_ch

        self.args.kitti_dir = '/media/vision/SSD1/Datasets/kitti/dataset/'
        self.args.gm_dir = '/media/vision/SSD1/Datasets/gm_datasets/'

        self.model = get_pipeline(self.args).to(self.device)
        try:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        except:
            self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()

        if 'Kitti' in self.args.dataset :
            self.pose_threshold = [3.0, 20.0]
            self.sequence = f"{self.args.kitti_data_split['test'][0]:02d}" # for trained
            # self.sequence = f"{int(checkpoint_path.split('.')[-2][-1]):02d}" # for Logg3D pretrained
            # self.args.kitti_data_split['test'] = [int(checkpoint_path.split('.')[-2][-1])] # for Logg3D pretrained
            self.args.kitti_eval_seq = int(checkpoint_path.split('.')[-2][-1]) # for trained
            self.dataset_path = os.path.join(self.args.kitti_dir, 'sequences', self.sequence)
            if self.args.dataset == 'KittiRangeImageTupleDataset': self.args.dataset = 'KittiRangeImageDataset' 
            if self.args.dataset == 'KittiCVTTupleDataset': self.args.dataset = 'KittiCVTDataset' 
        elif 'GM' in self.args.dataset:
            self.pose_threshold = [1.5, 10.0]
            self.sequence = f"{self.args.gm_data_split['test'][0]:02d}"
            self.dataset_path = os.path.join(self.args.gm_dir, self.sequence)
            self.args.dataset = GMRangeImageDataset


        
        self.thresholds = np.linspace(thresholds_linspace[0], thresholds_linspace[1], thresholds_linspace[2])#[300:600]
        self.thresholds_num = len(self.thresholds)
        self.descriptors = []

        self.args.target_channel = multi_ch[0]
        self.data_loader_base = make_data_loader(self.args,
                            self.args.test_phase, # 'test'
                            self.args.batch_size, # 
                            num_workers=self.args.train_num_workers,
                            shuffle=False)

        self.args.target_channel = multi_ch[1]
        self.data_loader_query = make_data_loader(self.args,
                            self.args.test_phase, # 'test'
                            self.args.batch_size, # 
                            num_workers=self.args.train_num_workers,
                            shuffle=False)
        
        print('*' * 50)
        print(f"* Evaluator for {self.args.dataset}_{self.sequence} dataset")
        print(f"* pose_threshold: {self.pose_threshold}")
        print(f"* checkpoint: {checkpoint_path}")
        print(f"* model: {self.args.pipeline}")
        print('*' * 50)
    
    def run(self):
        print('*' * 50)
        print('* evaluator run ...')
        # print("* processing for gt matching ...")
        timestamps = np.array(load_timestamps(self.dataset_path + '/times.txt'))
        poses = load_poses(os.path.join(self.dataset_path, 'poses.txt'))
        pose_distances_matrix = calculate_pose_distances_with_pdist(poses)

        descriptors_file_path_database = "preprocessed_descriptors/" + self.checkpoint_path.split('/')[-2] + '_' + self.checkpoint_path.split('/')[-1].split('.')[0] + f'_{self.multi_ch[0]}'+ ".npz"
        if os.path.exists(descriptors_file_path_database):
            print('* load descriptors at ', descriptors_file_path_database)
            descriptors_database = np.load(descriptors_file_path_database)["descriptors"]
        else: 
            descriptors_database = self._make_descriptors(self.data_loader_base)
            np.savez_compressed(descriptors_file_path_database, descriptors=descriptors_database)

        descriptors_file_path_query = "preprocessed_descriptors/" + self.checkpoint_path.split('/')[-2] + '_' + self.checkpoint_path.split('/')[-1].split('.')[0] + f'_{self.multi_ch[1]}'+ ".npz"
        if os.path.exists(descriptors_file_path_query):
            print('* load descriptors at ', descriptors_file_path_query)
            descriptors_query = np.load(descriptors_file_path_query)["descriptors"]
        else: 
            descriptors_query = self._make_descriptors(self.data_loader_query)
            np.savez_compressed(descriptors_file_path_query, descriptors=descriptors_query)

        descriptor_distances_matrix = cdist(descriptors_database, descriptors_query, metric='cosine')
        top_matchings = self._find_matching_poses(timestamps, descriptor_distances_matrix, pose_distances_matrix)
        
        metrics_list = self._calculate_metrics(top_matchings)
        return top_matchings, metrics_list     


    @torch.no_grad()
    def _make_descriptors(self, data_loader):
        descriptors_list = []
        test_loader_progress_bar = tqdm(data_loader, desc="* Make global descriptors", leave=True)
        for i, batch in enumerate(test_loader_progress_bar, 0):
            if i >= len(test_loader_progress_bar):
                break
            if self.args.pipeline == 'LOGG3D':
                lidar_pc = batch[0][0]
                input_st = make_sparse_tensor(lidar_pc, self.args.voxel_size).to(device=self.device)
                output_desc, output_feats = self.model(input_st)
                output_feats = output_feats[0]
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                descriptors_list.append(global_descriptor[0])

            elif self.args.pipeline.split('_')[0] == 'OverlapTransformer':
                input_t = torch.tensor(batch[0][0]).type(torch.FloatTensor).to(device=self.device)
                input_t = input_t.unsqueeze(0).unsqueeze(0).type(torch.FloatTensor).to(device=self.device)
                output_desc = self.model(input_t)
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                descriptors_list.append(global_descriptor[0])

        return np.array(descriptors_list)
    
    def make_and_save_descriptors(self, ch):
        descriptors = self._make_descriptors()
        descriptors_file_path = "preprocessed_descriptors/" + self.checkpoint_path.split('/')[-2] + '_' + self.checkpoint_path.split('/')[-1].split('.')[0] + f'_{ch}'+ ".npz"
        np.savez_compressed(descriptors_file_path, descriptors=descriptors)

    def _find_matching_poses(self, timestamps, descriptor_distances_matrix, pose_distances_matrix):
        start_time = timestamps[0]
        self.revist = [0 for _ in range(pose_distances_matrix.shape[0])]
        top_matchings = []

        for i in tqdm(range(0, pose_distances_matrix.shape[0]), desc="* Matching descriptors", leave=True):
            query_time = timestamps[i]

            # 처음 skip_time 이전의 pose는 사용하지 않음
            if (query_time - start_time - self.args.skip_time) < 0:
                continue

            # 0초 ~ 현재 시간까지의 pose 중에서 30초 이전의 pose까지만 사용
            tt = next(x[0] for x in enumerate(timestamps)
                if x[1] > (query_time - self.args.skip_time))
            
            seen_descriptors_dist = descriptor_distances_matrix[i,:tt+1]
            seen_poses_dist = pose_distances_matrix[i,:tt+1]

            nearList = seen_poses_dist[seen_poses_dist <= 3]
            if(len(nearList) > 0):
                self.revist[i] = 1

            # seen_descriptors에서 상위 20개의 인덱스 추출
            top_indices = np.argsort(seen_descriptors_dist)[:20]
            # top_indices의 각 인덱스에 대해 (인덱스, pose 거리, descriptor 거리) 튜플을 생성하여 top_matchings에 추가
            top_matchings.append([(i, t_idx, seen_poses_dist[t_idx], seen_descriptors_dist[t_idx]) for t_idx in top_indices])

        return top_matchings

    def _calculate_metrics(self, top_matchings):
        matching_results = [[] for _ in range(self.thresholds_num)]
        total_attempts = len(top_matchings)
        for th_idx, th in enumerate(self.thresholds):
            tp = 0  # True Positives
            tn = 0  # True Negatives
            fp = 0  # False Positives
            fn = 0  # False Negatives
            for matches in top_matchings: # top_matchings -> i-tt0개
                # if th_idx == 0:
                #     tqdm.write(f"* id: {matches[0][0]}, n_id: {matches[0][1]}, is_rev: {self.revist[matches[0][0]]}, min_dist: {matches[0][3]:.3f}, p_dist: {matches[0][2]:.3f}")
                # Positive
                if matches[0][3] < th:
                    # True
                    if matches[0][2] <= self.pose_threshold[0]:
                        tp += 1
                    # False
                    elif matches[0][2] > self.pose_threshold[1]:
                        fp += 1
                # Negative
                else:
                    # True
                    if self.revist[matches[0][0]] == 0:
                        tn += 1
                    # False
                    else:
                        fn += 1

            if th_idx == 82:
                print(f"tn: {tn}, fp: {fp}, tp: {tp}, fn: {fn}")

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / total_attempts if total_attempts > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            top5_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            top20_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            matching_results[th_idx] = {"True Positives": tp,
                                        "True Negatives": tn,
                                        "False Positives": fp,
                                        "False Negatives": fn,
                                        "Precision": precision,
                                        "Recall (TPR)": recall,
                                        "F1-Score": f1_score,
                                        "Accuracy": accuracy,
                                        "False Positive Rate (FPR)": fpr,
                                        "Top5_Recall": top5_recall,
                                        "Top20_Recall": top20_recall}
        
        return matching_results

    def analyze_metrics(self, metrics_list):
        df = pd.DataFrame(metrics_list) 
        df.insert(0, 'Thresholds', self.thresholds[:len(df)])
        max_f1_score = df["F1-Score"].max()
        max_f1_score_idx = df["F1-Score"].idxmax()
        corresponding_threshold = df["Thresholds"][max_f1_score_idx]
        corresponding_recall = df["Recall (TPR)"][max_f1_score_idx]
        F1_TN = df["True Negatives"][max_f1_score_idx]
        F1_FN = df["False Negatives"][max_f1_score_idx]
        F1_FP = df["False Positives"][max_f1_score_idx]
        F1_TP = df["True Positives"][max_f1_score_idx]
        print(f"* Best F1-Score:\t {max_f1_score:.3f}, \tRecall: {corresponding_recall:.3f}, \tat thresholds: {corresponding_threshold:.3f}")
        print(f"* num_revist: {sum(self.revist)}")
        print(f"* TP: {F1_TP}, TN: {F1_TN}, FP: {F1_FP}, FN: {F1_FN}")
        # print(f"* Best F1-Score:\t {max_f1_score}, at that metrics:\n*\t", df[df["F1-Score"] == max_f1_score])
        return max_f1_score_idx