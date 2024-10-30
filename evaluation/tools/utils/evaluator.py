import os, sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import numpy as np
from .utils import load_calib, load_poses
from scipy.spatial.distance import pdist, squareform
from utils.data_loaders.kitti.kitti_rangeimage_dataset import load_timestamps
from utils.data_loaders.make_dataloader import *
from tools.utils.utils import *
from models.pipelines.pipeline_utils import *
# from models.pipelines.pipeline_utils import *
from models.pipeline_factory import get_pipeline
from tqdm import tqdm
import pandas as pd

def calculate_pose_distance(pose1, pose2):
    translation1 = pose1[:3, 3]
    translation2 = pose2[:3, 3]
    return np.linalg.norm(translation1 - translation2)

def calculate_pose_distances(poses):
    n = len(poses)
    n_progress_bar = tqdm(range(n), desc="* Calculate pose matching", leave=True)
    distances = np.zeros((n, n))
    for i in n_progress_bar:
        for j in range(i + 1, n):
            distances[i, j] = calculate_pose_distance(poses[i], poses[j])
            distances[j, i] = distances[i, j]
    return distances

# @torch.no_grad()
class Evaluator:
    def __init__(self, checkpoint_path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.checkpoint = torch.load(checkpoint_path)
        self.args = self.checkpoint['config']

        self.args.kitti_dir = '/media/vision/Data0/DataSets/kitti/dataset/'
        self.args.gm_dir = '/media/vision/Data0/DataSets/gm_datasets/'

        self.model = get_pipeline(self.args).to(self.device)
        try:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
        except:
            self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()

        if 'Kitti' in self.args.dataset :
            self.pose_threshold = [3.0, 20.0]
            self.sequence = f"{self.args.kitti_data_split['test'][0]:02d}"
            self.dataset_path = os.path.join(self.args.kitti_dir, 'sequences', self.sequence)
        elif 'GM' in self.args.dataset:
            self.pose_threshold = [1.5, 10.0]
            self.sequence = f"{self.args.gm_data_split['test'][0]:02d}"
            self.dataset_path = os.path.join(self.args.gm_dir, self.sequence)
        
        self.thresholds = np.linspace(0.001, 1.0, 500)
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
        print(f"* train_loss_function: {self.args.train_loss_function}")
        print(f"* lazy_loss: {self.args.lazy_loss}")
        print(f"* Optimizer: {self.args.optimizer}")
        print(f"* base_learning_rate: {self.args.base_learning_rate}")
        print(f"* scheduler: {self.args.scheduler}")
        print(f"* dataset: {self.args.dataset}")
        print(f"* use_random_rotation: {self.args.use_random_rotation}")
        print(f"* use_random_occlusion: {self.args.use_random_occlusion}")
        print(f"* use_random_scale: {self.args.use_random_scale}")
        print('*' * 50)
    
    def run(self):
        print('*' * 50)
        print('* evaluator run ...')
        # print("* processing for gt matching ...")
        timestamps = np.array(load_timestamps(self.dataset_path + '/times.txt'))
        poses = load_poses(os.path.join(self.dataset_path, 'poses.txt'))
        pose_distances_matrix = calculate_pose_distances(poses)
        # print("* processing for descriptors ...")
        descriptors = self._make_descriptors()
        descriptor_distances_matrix = squareform(pdist(descriptors, 'euclidean'))
        # print("* evaluate ...")
        matching_results = self._find_matching_poses(timestamps, pose_distances_matrix, descriptor_distances_matrix)
        
        metrics_list = []
        for th_idx in range(self.thresholds_num):
            metrics_list.append(self._calculate_metrics(matching_results[th_idx]))
        return matching_results, metrics_list

    @torch.no_grad()
    def _make_descriptors(self):
        descriptors_list = []
        test_loader_progress_bar = tqdm(self.data_loader, desc="* Make global descriptors", leave=True)
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
        

    def _find_matching_poses(self, timestamps, pose_distances_matrix, descriptor_distances_matrix):
        start_time = timestamps[0]
        matching_results_list = [[] for _ in range(self.thresholds_num)]

        for i in tqdm(range(0, pose_distances_matrix.shape[0], 1), desc="* Matching descriptors", leave=True):
            query_time = timestamps[i] 
            if (query_time - start_time - 30) < 0: # Build retrieval database using entries 30s prior to current query. 
                continue        

            # 0초 ~ 현재 시간까지의 pose 중에서 30초 이전의 pose까지만 사용
            tt = next(x[0] for x in enumerate(timestamps)
                if x[1] > (query_time - 30))
            
            revisit = []
            match_candidates = [[] for _ in range(self.thresholds_num)]
            
            for j in range(0, tt+1):
                for th_idx in range(self.thresholds_num):
                    if descriptor_distances_matrix[i, j] < self.thresholds[th_idx]:
                        match_candidates[th_idx].append((j, pose_distances_matrix[i, j], descriptor_distances_matrix[i, j]))
                if pose_distances_matrix[i, j] < self.pose_threshold[0]:
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

    def _calculate_metrics(self, matching_results):
        tp = 0  # True Positives
        tn = 0  # True Negatives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        total_attempts = len(matching_results)
        
        top5_tp = 0  # Top-K Recall 계산을 위한 변수
        top20_tp = 0  # Top-K Recall 계산을 위한 변수
        
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
            if any(match[4] == "tp" for match in matches[:5]):
                top5_tp += 1
            if any(match[4] == "tp" for match in matches[:20]):
                top20_tp += 1
        
        # 메트릭 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / total_attempts if total_attempts > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        top5_recall = top5_tp / (top5_tp + fn) if (top5_tp + fn) > 0 else 0
        top20_recall = top20_tp / (top20_tp + fn) if (top20_tp + fn) > 0 else 0
        
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
            "Top5_Recall": top5_recall,
            "Top20_Recall": top20_recall
        }

    def analyze_metrics(self, metrics_list):
        df = pd.DataFrame(metrics_list) 
        df.insert(0, 'Thresholds', self.thresholds[:len(df)])
        max_f1_score = df["F1-Score"].max()
        max_f1_score_idx = df["F1-Score"].idxmax()
        corresponding_threshold = df["Thresholds"][max_f1_score_idx]
        corresponding_recall = df["Recall (TPR)"][max_f1_score_idx]
        print(f"* Best F1-Score:\t {max_f1_score}, \tRecall: {corresponding_recall}, \tat thresholds: {corresponding_threshold}")
        # print(f"* Best F1-Score:\t {max_f1_score}, at that metrics:\n*\t", df[df["F1-Score"] == max_f1_score])
        return max_f1_score_idx