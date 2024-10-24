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
from utils.data_loaders.kitti.kitti_rangeimage_dataset import load_timestamps

# # load config ================================================================
# config_filename = '../config/config.yml'
# config = yaml.safe_load(open(config_filename))
# test_weights = config["demo1_config"]["test_weights"]
# # ============================================================================

class Evaluator:
    def __init__(self, args, seq, threshold) -> None:
        self.args = args
        self.sequence = f"{seq:02d}"
        
        self.pose_threshold = [3.0, 20.0]  # 실제 pose 거리 임계값 3m, 20m
        self.thresholds = [threshold]
        self.thresholds_num = 1
        self.descriptors = []
        self.data = []

        self.dataset_path = os.path.join(args.kitti_dir, 'sequences', self.sequence)
        self.load_kitti_poses_and_timestamps()

    def load_kitti_poses_and_timestamps(self):
        self.timestamps = np.array(load_timestamps(self.dataset_path + '/times.txt'))

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

        return matching_results[0]
    
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
            if (query_time - start_time - 30) < 0: # Build retrieval database using entries 30s prior to current query. 
                continue        

            # 0초 ~ 현재 시간까지의 pose 중에서 30초 이전의 pose까지만 사용
            tt = next(x[0] for x in enumerate(self.timestamps)
                if x[1] > (query_time - 30))
            
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

@torch.no_grad()
def __main__(args, model, device, model_name, train_name, file_name, threshold):
    print("===test for kitti sequence {}===".format(args.kitti_data_split['test'][0]))
    evaluator = Evaluator(args, args.kitti_data_split['test'][0], threshold)

    load_descriptors_flag = False
    if load_descriptors_flag == False:
        test_loader = make_data_loader(args,
                                    args.test_phase, # 'test'
                                    args.batch_size, # 
                                    num_workers=args.train_num_workers,
                                    shuffle=False)
        
        print("===== Generating Descriptors =====")
        print("= Dataset: ", args.dataset)
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

        print("=================================")
        
    print("Evaluating ...")
    matching_results = evaluator.evaluate()

    import pickle
    
    save_folder_path = os.path.join(os.path.dirname(__file__), 'matching_results', model_name, train_name)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    with open(os.path.join(save_folder_path, file_name + '.pkl'), 'wb') as file:
        pickle.dump(matching_results, file)
    print("Results saved at: ", os.path.join(save_folder_path, file_name + '.pkl'))


if __name__ == '__main__':
    from models.pipeline_factory import get_pipeline

    test_models_list = [
        ['OverlapTransformer/2024-10-10_14-06-46', 'epoch_82', 0.457343],
        ['OverlapTransformer/2024-10-11_00-46-31', 'epoch_best_71', 0.492569],
        ['OverlapTransformer/2024-10-14_04-48-23', 'epoch_best_76', 0.486164],
        ['OverlapTransformer/2024-10-14_04-48-04', 'epoch_best_71', 0.500575],
        ['OverlapTransformer/2024-10-14_04-31-11', 'epoch_best_44', 0.434926],
        ['OverlapTransformer_resnet/2024-10-11_09-33-09', 'epoch_44', 0.383687],
        ['OverlapTransformer_geo/2024-10-11_09-30-16', 'epoch_84', 0.474956],
        ['OverlapTransformer/2024-10-10_14-07-02', 'epoch_best_92', 0.418914],
        ['OverlapTransformer/2024-10-10_15-48-19', 'epoch_best_34', 0.386890],
        ['OverlapTransformer_resnet/2024-10-11_09-33-26', 'epoch_best_67', 0.367675],
        ['OverlapTransformer_geo/2024-10-11_09-30-40', 'epoch_91', 0.508581],
        ['OverlapTransformer/2024-10-10_14-06-02', 'epoch_best_42', 0.412509],
        ['OverlapTransformer/2024-10-10_15-48-11', 'epoch_best_64', 0.473355],
        ['OverlapTransformer_resnet/2024-10-12_07-18-40', 'epoch_best_24', 0.329246],
        ['OverlapTransformer_geo/2024-10-12_07-52-41', 'epoch_best_70', 0.490968],
    ]
    for model_name in test_models_list:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True # cuDNN의 성능을 최적화하기 위한 설정. 데이터 크기가 일정할 때 효율적

        model_path = model_name[0]
        file = model_name[1] + '.pth'
        threshold = model_name[2]
        # file_list = [os.path.basename(f) for f in glob.glob(os.path.join('../training/checkpoints', model_path, '*.pth'))]
        
        # for file in file_list:
        file_name =  os.path.splitext(os.path.basename(file))[0]
        model_name = model_path.split('/')[0]
        train_name = model_path.split('/')[1]

        ## load checkpoint
        print('Loading checkpoint from: ', os.path.join('../training/checkpoints', model_path, file))
        checkpoint = torch.load(os.path.join('../training/checkpoints', model_path, file))
        args = checkpoint['config']
        args.kitti_dir = '/media/vision/Data0/DataSets/kitti/dataset/'
        if len(args.kitti_data_split['test']) == 0:
            args.kitti_data_split['test'] = [5]

        ## load model
        model = get_pipeline(args).to(device)
        model.load_state_dict(checkpoint['model_state_dict']) # state_dict, model_state_dict
        model.eval()

        __main__(args, model, device, model_name, train_name, file_name, threshold)