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
from tools.utils.matching_plotter import plotter

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', default='kitti', type=str, help='dataset name', choices=['kitti', 'gm'])
argparser.add_argument('--kitti_path', default='/media/vision/SSD1/Datasets/kitti/dataset/', type=str, help='kitti dataset path')
argparser.add_argument('--gm_path', default='/media/vision/SSD1/Datasets/gm_datasets/', type=str, help='gm dataset path')
argparser.add_argument('--matching_data_folder', default='./results/', type=str, help='matching data folder')
argparser.add_argument('--all_pkl', default=True, type=bool, help='whether to use all pkl files in the folder')

@torch.no_grad()
def __main__(args, model_path_list):
    Plotter = plotter()
    
    for path in model_path_list:
        args.dataset = path[1]
        args.seq = f"{path[2]:02d}"
        file_path = args.matching_data_folder + path[0]

        pickle_files = [f for f in os.listdir(file_path) if f.endswith('.pkl')] if args.all_pkl else model_path_list[3]
        pickle_files.sort()

        if len(pickle_files) == 0:
            print('No matching data found in {}'.format(file_path))
            continue
        elif len(pickle_files) > 1:
            print('*' * 50)
            print('* Multiple matching data found in {}'.format(file_path))
            print('* Available matching data files:')
            for i, file in enumerate(pickle_files):
                print(f'{i}: {file}')
            selected_index = int(input('* >> Select the index of the file to use: '))
            selected_file = pickle_files[selected_index]
            print(f'* Selected file: {selected_file}')
            print('*' * 50)
        else:
            selected_file = pickle_files[0]

        with open(os.path.join(file_path, selected_file), 'rb') as file:
            matching_data = pickle.load(file)['matching_results'] # 이래도 돼?
        save_file_name = os.path.join(file_path, path[0].split('/')[1] + "_matching_plot.png") ## 저장파일 이름 확정해야 함.

        Plotter.data_setting(args, matching_data, save_file_name)
        Plotter.plot_total_matching(vis=False)


if __name__ == '__main__':
    args = argparser.parse_args()
    
    model_path_list = [
        # [model_file_name, dataset, seq, [checkpoint_file_name]]
        # ['OverlapTransformer_geo/2024-10-24_16-07-32', 'kitti', 8, []], # kitti08 geo
        ['OverlapTransformer/2024-10-12_08-41-37', 'gm', 0, []], # triplet lazy False 00
        
    ]
    __main__(args, model_path_list)