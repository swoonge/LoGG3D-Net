# matching_results.pkl 파일을 읽어서 matching 결과를 시각화하는 코드
# matching_results.pkl 파일은 evaluation/tools/utils/matching_plotter.py 에서 생성됨
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
from utils.data_loaders.make_dataloader import *
from models.pipelines.pipeline_utils import *
from utils.data_loaders.kitti.kitti_rangeimage_dataset import load_timestamps
import pickle
from evaluation.tools.utils.matching_plotter import plotter

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', default='kitti', type=str, help='dataset name', choices=['kitti', 'gm'])
argparser.add_argument('--kitti_path', default='/media/vision/Data0/DataSets/kitti/dataset/', type=str, help='kitti dataset path')
argparser.add_argument('--gm_path', default='/media/vision/Data0/DataSets/gm_datasets/', type=str, help='gm dataset path')
argparser.add_argument('--matching_data_folder', default='./matching_results/', type=str, help='matching data folder')

@torch.no_grad()
def __main__(args, model_path_list):
    Plotter = plotter()
    
    for path in model_path_list:
        seq = path[1]
        file_path = args.matching_data_folder + path[0]
        # pickle_files = [f for f in os.listdir(file_path) if f.endswith('.pkl') and 'matching' in f]  # .pkl 확장자 파일들 중 'matching' 단어가 포함된 파일들
        pickle_files = [f for f in os.listdir(file_path) if f.endswith('.pkl')]  # .pkl 확장자 파일들
        pickle_files.sort()

        if len(pickle_files) == 0:
            print('No matching data found in {}'.format(file_path))
            continue
        elif len(pickle_files) > 1:
            print('*' * 50)
            print('Multiple matching data found in {}'.format(file_path))
            print('Available matching data files:')
            for i, file in enumerate(pickle_files):
                print(f'{i}: {file}')
            selected_index = int(input('>> Select the index of the file to use: '))
            selected_file = pickle_files[selected_index]
            print(f'Selected file: {selected_file}')
            print('*' * 50)
        else:
            selected_file = pickle_files[0]

        with open(os.path.join(file_path, selected_file), 'rb') as file:
            matching_data = pickle.load(file) # 이래도 돼?
        save_file_name = os.path.join(file_path, path[0].split('/')[1] + "_matching_plot.png") ## 저장파일 이름 확정해야 함.

        Plotter.data_setting(args.dataset, args.kitti_path, seq, matching_data, save_file_name)
        Plotter.plot_total_matching(vis=False)


if __name__ == '__main__':
    args = argparser.parse_args()
    
    model_path_list = [
        # [model_file_name, seq]
        ['OverlapTransformer/2024-10-10_14-06-46', 0],
        # ['OverlapTransformer/2024-10-11_00-46-31', 0],
        # ['OverlapTransformer/2024-10-14_04-48-23', 0],
        # ['OverlapTransformer/2024-10-14_04-48-04', 0],
        # ['OverlapTransformer/2024-10-14_04-31-11', 0],
        # ['OverlapTransformer_resnet/2024-10-11_09-33-09', 0],
        # ['OverlapTransformer_geo/2024-10-11_09-30-16', 0],
        # ['OverlapTransformer/2024-10-10_14-07-02', 5],
        # ['OverlapTransformer/2024-10-10_15-48-19', 5],
        # ['OverlapTransformer_resnet/2024-10-11_09-33-26', 5],
        # ['OverlapTransformer_geo/2024-10-11_09-30-40', 5],
        # ['OverlapTransformer/2024-10-10_14-06-02', 8],
        # ['OverlapTransformer/2024-10-10_15-48-11', 8],
        # ['OverlapTransformer_resnet/2024-10-12_07-18-40', 8],
        # ['OverlapTransformer_geo/2024-10-12_07-52-41', 8],
    ]
    __main__(args, model_path_list)