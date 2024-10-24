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
import pickle

class plotter:
    def __init__(self, dataset_path, seq, data, img_file_name) -> None:
        self.dataset_path = os.path.join(dataset_path, '{:02d}'.format(seq))
        self.matchings = data
        self.img_file_name = img_file_name

        # self.timestamps = np.array(load_timestamps(self.dataset_path + '/times.txt'))

        # load poses
        poses_file = os.path.join(self.dataset_path, 'poses.txt')
        self.poses = load_poses(poses_file)
    
    def plot_total_matching(self, vis=True):
        # 전체 지도 생성 -> x, y 좌표 추출
        x_map, y_map = zip(*[pose[:2, 3] for pose in self.poses])

        # 전체 화면 창 설정
        plt.figure(figsize=(12, 12))

        # 지도 플롯
        plt.plot(x_map, y_map, color='gray', linestyle='-', marker='.', label='Map', alpha=0.5)
        plt.title("Total Matching")
        plt.xlabel('X')
        plt.ylabel('Y')

        # 매칭 플롯
        for match in self.matchings:
            if not match:
                continue
            first_match = match[0]
            x_current, y_current = x_map[int(first_match[0])], y_map[int(first_match[0])]
            x_matching, y_matching = x_map[int(first_match[1])], y_map[int(first_match[1])]

            if first_match[4] == "fp":
                plt.plot([x_current, x_matching], [y_current, y_matching], 'r-', label='False Positive', linewidth=0.5, markevery=[0, -1], marker='x', markersize=3, markeredgewidth=0.2)
            elif first_match[4] == "tp":
                plt.plot([x_current, x_matching], [y_current, y_matching], 'b-', label='True Positive', linewidth=0.5)
            elif first_match[4] == "fn":
                plt.plot(x_current, y_current, 'coral', marker='^', label='False Negative', markersize=1)

        # 범례 설정 (중복 제거)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # 이미지 저장 (저장 경로와 파일명을 지정해야 합니다)
        plt.savefig("matching_vis_{}.png".format(self.img_file_name), dpi=900)
        plt.show() if vis else plt.close()

    

@torch.no_grad()
def __main__(key, value, kitti_path, seq, save_file_name):
    print('='*50 ,'\n=\tplot for {}'.format(key))
    print('=\tseq: ', seq)

    kitti_plot = plotter(kitti_path, seq, value, save_file_name)
    kitti_plot.plot_total_matching(vis=True)

    print('='*50)


if __name__ == '__main__':
    gm_path = '/media/vision/Data0/DataSets/gm_datasets/'
    
    path_list = [
        # ['OverlapTransformer/2024-10-12_08-41-37', 0, 373], # kitti00 geo
        # ['OverlapTransformer/2024-10-12_19-23-47', 1, 398], # kitti05 geo
        # ['OverlapTransformer/2024-10-14_16-40-13', 2, 239], # kitti08 geo
        ['OverlapTransformer/2024-10-14_07-42-01', 3, 215], # kitti08 geo
    ]

    # metric_results = {}
    data = 0

    for path in path_list:
        file_path = 'matching_results/'+path[0]
        pickle_files = [f for f in os.listdir(file_path) if f.endswith('.pkl')]  # .pkl 확장자 파일들
        pickle_files.sort()
        for idx, p in enumerate(pickle_files):
            with open(os.path.join(file_path, p), 'rb') as file:
                print(file_path, p)
                data = pickle.load(file)
                # metric_results['{}'.format(path[0])] = data
        save_file_name = path[0].split('/')[1]
        __main__(path[0], data[int(path[2])], gm_path, path[1], save_file_name)