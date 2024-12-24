# Developed by Soowoong Park in KIST RobotVisionLab 
# This file is covered by the LICENSE file in the root of this project.
# Brief: This script generate the overlap and orientation combined mapping file.

import os, sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils.data_loaders.make_dataloader import *
from tools.utils.utils import *
from models.pipelines.pipeline_utils import *
from models.pipeline_factory import get_pipeline
from tqdm import tqdm
import pandas as pd
from config.config_eval import *

torch.backends.cudnn.benchmark = True

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_translations(poses):
    """
    주어진 포즈 배열에서 변환된 위치(translations)를 추출하고 3D로 시각화합니다.

    Args:
        poses (list of np.array): 각 pose가 4x4 변환 행렬인 리스트
    """
    # 각 pose에서 translation(위치) 부분 추출
    translations = np.array([pose[:3, 3] for pose in poses])

    # 3D 산점도 그리기
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # x, y, z 좌표 추출
    x = translations[:, 0]
    y = translations[:, 1]
    z = translations[:, 2]

    # 3D 산점도 그리기
    ax.scatter(x, y, z, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Pose Translations')

    # 축 스케일을 동일하게 설정
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_box_aspect([1, 1, 1])  # 동일한 스케일 비율 유지
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()


def calculate_pose_distances_with_pdist(poses):
    # 각 pose에서 translation 벡터 (x, y, z)를 추출
    translations = np.array([pose[:3, 3] for pose in poses])
    
    # pdist로 모든 쌍별 유클리드 거리 계산
    distances = squareform(pdist(translations, metric='euclidean'))
    
    return distances

# @torch.no_grad()
class Evaluator:
    def __init__(self, checkpoint_path, test_dataset_forced=None, test_seq_forced=None, test_for_val_set=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dataset_forced = test_dataset_forced
        self.test_for_val_set = test_for_val_set
        self.checkpoint_path = checkpoint_path
        self.checkpoint = torch.load(checkpoint_path)
        try:
            self.args = self.checkpoint['config']
        except Exception as e:
            self.args = get_config()
        # self.args = get_config() >> for pretrained model

        self.args.kitti_dir = '/media/vision/SSD1/Datasets/kitti/dataset'
        self.args.gm_dir = '/media/vision/SSD1/Datasets/gm_datasets'
        self.args.mulran_dir = '/media/vision/SSD1/Datasets/MulRan'
        self.args.nclt_dir = '/media/vision/SSD1/Datasets/NCLT'

        self.model = get_pipeline(self.args).to(self.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        try:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        except Exception as e:
            self.model.load_state_dict(self.checkpoint['state_dict'])
        
        self.model.eval()

        if self.test_dataset_forced is not None:
            self.args.dataset = self.test_dataset_forced

        if 'Kitti' in self.args.dataset :
            self.pose_threshold = [3.0, 20.0]
            if self.test_for_val_set:
                self.args.kitti_data_split['test'] = self.args.kitti_data_split['val']
            # self.sequence = f"{int(checkpoint_path.split('.')[-2][-1]):02d}" # for Logg3D pretrained
            # self.args.kitti_data_split['test'] = [int(checkpoint_path.split('.')[-2][-1])] # for Logg3D pretrained
            # self.args.kitti_eval_seq = int(checkpoint_path.split('.')[-2][-1]) # for trained
            # self.args.kitti_eval_seq = 0 # for pretrained model
            if test_seq_forced is not None:
                self.args.kitti_data_split['test'][0] = test_seq_forced
            self.sequence = f"{self.args.kitti_data_split['test'][0]:02d}" # for trained
            if "Overlap" in self.args.pipeline or "CVT" in self.args.pipeline:
                self.args.dataset = 'KittiDepthImageDataset'
            elif "LOGG3D" in self.args.pipeline: # >> 확인 필요
                self.args.dataset = 'KittiDataset'
        elif 'GM' in self.args.dataset:
            self.pose_threshold = [1.5, 10.0]
            if self.test_for_val_set:
                self.args.gm_data_split['test'] = self.args.gm_data_split['val']
            self.sequence = f"{self.args.gm_data_split['test'][0]:02d}"
            if "Overlap" in self.args.pipeline or "CVT" in self.args.pipeline:
                self.args.dataset = 'GMDepthImageDataset'
            elif "LOGG3D" in self.args.pipeline: # >> 확인 필요
                self.args.dataset = 'GMDataset'
        elif 'NCLT' in self.args.dataset:
            self.pose_threshold = [3.0, 20.0]
            if self.test_for_val_set:
                self.args.nclt_data_split['test'][0] = "2012-01-15" if self.args.nclt_data_split['test'][0] == self.args.nclt_data_split['val'][0] else self.args.nclt_data_split['val'][0]
            self.sequence = self.args.nclt_data_split['test'][0]
            if test_seq_forced is not None:
                self.args.nclt_data_split['test'][0] = test_seq_forced
                self.sequence = self.args.nclt_data_split['test'][0]
            if "Overlap" in self.args.pipeline or "CVT" in self.args.pipeline:
                self.args.dataset = 'NCLTDepthImageDataset'
            elif "LOGG3D" in self.args.pipeline: # >> 확인 필요
                self.args.dataset = 'NCLTSparseTupleDataset'
        
        self.thresholds = np.linspace(self.args.cd_thresh_min, self.args.cd_thresh_max, self.args.num_thresholds)#[300:600]
        self.thresholds_num = len(self.thresholds)
        self.descriptors = []

        self.data_loader = make_data_loader(self.args,
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
        timestamps = self.data_loader.dataset.timestamps_dict[self.sequence]
        poses = self.data_loader.dataset.poses_dict[self.sequence]
        
        # translations = np.array([pose[:3, 3] for pose in poses])
        # plot_translations(poses)

        pose_distances_matrix = calculate_pose_distances_with_pdist(poses)
        # try:
        descriptors_file_path = "preprocessed_descriptors/" + self.checkpoint_path.split('/')[-2] + '_' + self.checkpoint_path.split('/')[-1].split('.')[0] +"_"+ str(self.data_loader.dataset.drive_ids[0])
        # except:
        #     descriptors_file_path = "preprocessed_descriptors/" + self.checkpoint_path.split('/')[-2] + '_' + self.checkpoint_path.split('/')[-1].split('.')[0] + "_64"

        # if os.path.exists(descriptors_file_path+'.npy') and self.test_dataset_forced is None:
        #     print(f"* Load preprocessed descriptors from {descriptors_file_path}")
        #     descriptors = np.load(descriptors_file_path + '.npy')
        # else: 
        #     descriptors = self._make_descriptors()
        #     if not os.path.exists("preprocessed_descriptors"):
        #         os.makedirs("preprocessed_descriptors")
        #     np.save(descriptors_file_path, descriptors)

        # must make descriptors test
        descriptors = self._make_descriptors()

        descriptor_distances_matrix = squareform(pdist(descriptors, 'euclidean')) #euclidean
        top_matchings = self._find_matching_poses(timestamps, descriptor_distances_matrix, pose_distances_matrix)
        
        metrics_list = self._calculate_metrics(top_matchings)
        return top_matchings, metrics_list     


    @torch.no_grad()
    def _make_descriptors(self):
        descriptors_list = []
        test_loader_progress_bar = tqdm(self.data_loader, desc="* Make global descriptors", leave=True)
        for i, batch in enumerate(test_loader_progress_bar, 0):
            if i >= len(test_loader_progress_bar):
                break
            if self.args.pipeline == 'LOGG3D':
                input_st = make_sparse_tensor(batch[0][0], self.args.voxel_size).to(device=self.device)
                output_desc, output_feats = self.model(input_st)
                output_feats = output_feats[0]
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                descriptors_list.append(global_descriptor[0])

            elif 'Overlap' in self.args.pipeline.split('_')[0]:
                input_t = torch.tensor(batch[0][0]).type(torch.FloatTensor).to(device=self.device)
                if input_t.ndim == 3:
                    input_t = input_t.unsqueeze(0)[:, 0, :, :].unsqueeze(1)
                else:
                    input_t = input_t.unsqueeze(0).unsqueeze(0)

                output_desc = self.model(input_t)
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                descriptors_list.append(global_descriptor[0])

            elif 'CVT' in self.args.pipeline:
                input_t = torch.tensor(batch[0][0]).type(torch.FloatTensor).to(device=self.device)
                input_t = input_t.unsqueeze(0)
                output_desc = self.model(input_t)
                global_descriptor = output_desc.cpu().detach().numpy()
                global_descriptor = np.reshape(global_descriptor, (1, -1))
                descriptors_list.append(global_descriptor[0])
                
        return np.array(descriptors_list)
        
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
        corresponding_accuracy = df["Accuracy"][max_f1_score_idx]
        F1_TN = df["True Negatives"][max_f1_score_idx]
        F1_FN = df["False Negatives"][max_f1_score_idx]
        F1_FP = df["False Positives"][max_f1_score_idx]
        F1_TP = df["True Positives"][max_f1_score_idx]
        print(f"* Best F1-Score:\t {max_f1_score:.3f}, \tRecall: {corresponding_recall:.3f}, \tAcc: {corresponding_accuracy:.3f},\tat thresholds: {corresponding_threshold:.3f}")
        print(f"* num_revist: {sum(self.revist)}")
        print(f"* TP: {F1_TP}, TN: {F1_TN}, FP: {F1_FP}, FN: {F1_FN}")
        # print(f"* Best F1-Score:\t {max_f1_score}, at that metrics:\n*\t", df[df["F1-Score"] == max_f1_score])
        return max_f1_score_idx, [max_f1_score, corresponding_recall, corresponding_accuracy, corresponding_threshold]