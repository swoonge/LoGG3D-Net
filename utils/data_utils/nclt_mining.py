import os
import numpy as np
from gen_ri_bev import *
from tqdm import tqdm
import struct
import open3d as o3d
import json

import scipy.interpolate
from scipy.spatial.transform import Rotation
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from time import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gen_ri_bev', action='store_true', help="Generate RI BEV images")
parser.add_argument('--no-gen_ri_bev', dest='gen_ri_bev', action='store_false', help="Do not generate RI BEV images")
parser.add_argument('--tuple_mining', action='store_true', help="Enable tuple mining")
parser.add_argument('--no-tuple_mining', dest='tuple_mining', action='store_false', help="Disable tuple mining")
parser.set_defaults(gen_ri_bev=False, tuple_mining=False)

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8

def p_dist(pose1, pose2, threshold=3):
    dist = np.linalg.norm(pose1 - pose2)
    if abs(dist) <= threshold:
        return True
    else:
        return False

def t_dist(t1, t2, threshold=10):
    if abs(t1-t2) > threshold:
        return True
    else:
        return False
    
def convert_timestamps_to_seconds(timestamps):
    """
    주어진 넘파이 배열의 타임스탬프를 첫 번째 요소를 기준으로 초 단위로 변환합니다.
    
    Parameters:
        timestamps (np.ndarray): 유닉스 타임스탬프가 포함된 넘파이 배열 (마이크로초 단위)
    
    Returns:
        np.ndarray: 변환된 초 단위 타임스탬프 배열
    """
    # 첫 번째 타임스탬프를 기준으로 0초부터 시작하도록 변환
    start_time = timestamps[0]
    seconds_array = (timestamps - start_time) / 1_000_000  # 마이크로초를 초 단위로 변환
    return seconds_array

class NCLTPreprocessor:
    def __init__(self, base_dir, range_thresh, height_thresh, fov_up, fov_down, proj_H, proj_W, max_range, min_range, min_height, max_height):
        self.base_dir = base_dir
        self.range_thresh = range_thresh
        self.height_thresh = height_thresh
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.max_range = max_range
        self.min_range = min_range
        self.min_height = min_height
        self.max_height = max_height

    def get_velodyne_fn(self, drive_id, fname):
        return os.path.join(self.base_dir, 'velodyne_data', f'{drive_id}', 'velodyne_sync', fname)
    
    def data2xyzi(self, data, flip=True):
        xyzil = data.view(velodatatype)
        xyz = np.hstack(
            [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
        xyz = xyz * 0.005 - 100.0

        if flip:
            R = np.eye(3)
            R[2, 2] = -1
            xyz = np.matmul(xyz, R)
        return xyz, xyzil['i']

    def get_velo(self, velofile):
        return self.data2xyzi(np.fromfile(velofile))
    
    def timestamps_files_and_gt(self, root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        root_path = Path(root_path)
        assert root_path.exists(), f"{root_path} does not exist."
        ground_truth_dir = root_path / "ground_truth"
        assert ground_truth_dir.exists(), f"{ground_truth_dir} does not exist."
        velodyne_dir = root_path / "velodyne_data" / sequence_id / "velodyne_sync"

        velodyne_files = np.array(sorted(os.listdir(str(velodyne_dir))), dtype=np.str_)
        timestamps = np.array([file.split(".")[0] for file in velodyne_files], dtype=np.int64)
        ground_truth_file = ground_truth_dir / f"groundtruth_{sequence_id}.csv"

        gt = None
        if ground_truth_file.exists():
            ground_truth = self.__ground_truth(str(ground_truth_file))

            # Ground truth timestamps and LiDARs don't match, interpolate
            gt_t = ground_truth[:, 0]
            t_min = np.min(gt_t)
            t_max = np.max(gt_t)
            inter = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)

            # Limit the sequence to timestamps for which a ground truth exists
            filter_ = (timestamps > t_min) * (timestamps < t_max)
            timestamps = timestamps[filter_]
            velodyne_files = velodyne_files[filter_]

            gt = inter(timestamps)
            gt_tr = gt[:, :3]
            gt_euler = gt[:, 3:][:, [2, 1, 0]]
            gt_rot = Rotation.from_euler("ZYX", gt_euler).as_matrix()

            gt = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(gt.shape[0], axis=0)
            gt[:, :3, :3] = gt_rot
            gt[:, :3, 3] = gt_tr

            gt = np.einsum("nij,jk->nik", gt, np.array([[1.0, 0.0, 0.0, 0.0],
                                                        [0.0, -1.0, 0.0, 0.0],
                                                        [0.0, 0.0, -1.0, 0.0],
                                                        [0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
            gt = np.einsum("ij,njk->nik", np.array([[1.0, 0.0, 0.0, 0.0],
                                                    [0.0, -1.0, 0.0, 0.0],
                                                    [0.0, 0.0, -1.0, 0.0],
                                                    [0.0, 0.0, 0.0, 1.0]], dtype=np.float32), gt)
            
            # 첫 번째 타임스탬프를 기준으로 0초부터 시작하도록 변환
            start_time = timestamps[0]
            timestamps = (timestamps - start_time) / 1_000_000  # 마이크로초를 초 단위로 변환

        return timestamps, velodyne_files, gt
    
    def __ground_truth(self, gt_file: str):
        gt = pd.read_csv(gt_file, sep=",", low_memory=False).values
        return gt
        
    def get_ri_bev_tensor(self, drive_id, velodyne_file):
        fname = self.get_velodyne_fn(drive_id, velodyne_file)
        current_vertex = self.get_velo(fname)[0]
        ri_bev = np.zeros((len(self.range_thresh) + len(self.height_thresh), self.proj_H, self.proj_W))

        for i in range(len(self.range_thresh) - 1):
            nearer_bound = self.range_thresh[i]
            farer_bound = self.range_thresh[i + 1]
            lower_bound = self.height_thresh[i]
            upper_bound = self.height_thresh[i + 1]

            proj_range, _, _ = range_projection(current_vertex,
                                                fov_up=self.fov_up,
                                                fov_down=self.fov_down,
                                                proj_H=self.proj_H,
                                                proj_W=self.proj_W,
                                                max_range=self.max_range,
                                                cut_range=True,
                                                lower_bound=nearer_bound,
                                                upper_bound=farer_bound)
            proj_bev = bev_projection(current_vertex,
                                      proj_H=self.proj_H,
                                      proj_W=self.proj_W,
                                      max_range=self.max_range,
                                      cut_height=True,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)

            ri_bev[int(i + 1), :, :] = proj_range
            ri_bev[int(i + 1 + len(self.range_thresh)), :, :] = proj_bev

        ri_bev[0, :, :], _, _ = range_projection(current_vertex,
                                                 fov_up=self.fov_up,
                                                 fov_down=self.fov_down,
                                                 proj_H=self.proj_H,
                                                 proj_W=self.proj_W,
                                                 max_range=self.max_range,
                                                 cut_range=True,
                                                 lower_bound=self.min_range,
                                                 upper_bound=self.max_range)
        ri_bev[len(self.range_thresh), :, :] = bev_projection(current_vertex,
                                                              proj_H=self.proj_H,
                                                              proj_W=self.proj_W,
                                                              max_range=self.max_range,
                                                              cut_height=True,
                                                              lower_bound=self.min_height,
                                                              upper_bound=self.max_height)
        return ri_bev

    def gen_ri_bev_and_save_all(self, drive_ids):
        print('*' * 50)
        for drive_id in drive_ids:
            output_dir = os.path.join(self.base_dir, 'velodyne_data', f'{drive_id}', 'ri_bev')
            os.makedirs(output_dir, exist_ok=True)

            _, velodyne_files, _ = self.timestamps_files_and_gt(self.base_dir, drive_id)

            for velodyne_file in tqdm(velodyne_files, desc="Processing Point Clouds"):
                ri_bev_np = self.get_ri_bev_tensor(drive_id, velodyne_file)
                # self.visualize_pointcloud(drive_id, velodyne_file)
                file_name = velodyne_file.split('.')[0]
                output_fname = os.path.join(output_dir, f'{file_name}.npz')
                np.savez_compressed(output_fname, ri_bev=ri_bev_np)
                tqdm.write(f'* Saved: {output_fname}')
            print('*' * 50)

    def gen_positive_dict_and_save_all(self, drive_ids, output_dir, d_thresh, t_thresh):
        print('*' * 50)
        self.get_positive_dict_matrix(self.base_dir, drive_ids, output_dir, d_thresh[0], t_thresh)
        print('*' * 50)
        self.get_positive_dict_matrix(self.base_dir, drive_ids, output_dir, d_thresh[1], t_thresh)
        print('*' * 50)

    def get_positive_dict_matrix(self, basedir, sequences, output_dir, d_thresh, t_thresh):
        positive_dict = {}
        print('d_thresh: ', d_thresh)
        print('output_dir: ', output_dir)
        print('')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        iter_i = 1
        for sequence in sequences:
            scan_timestamps, _, scan_positions = self.timestamps_files_and_gt(basedir, sequence)
            # scan_positions = scan_positions[:, :3, 3]  # Extract (n, 3) positions

            sequence_id = sequence
            if sequence not in positive_dict:
                positive_dict[sequence_id] = {}

            translations = np.array([pose[:3, 3] for pose in scan_positions])
    
            # pdist로 모든 쌍별 유클리드 거리 계산
            dists = squareform(pdist(translations, metric='euclidean'))
            
            # 모든 시간 간의 차이를 계산
            time_diffs = np.abs(scan_timestamps[:, np.newaxis] - scan_timestamps[np.newaxis, :])

            for t1 in tqdm(range(len(scan_timestamps)), desc=f"[{iter_i}/{len(sequences)}]Processing sequence {sequence}"):
                # 거리와 시간 차이 조건에 맞는 인덱스 추출
                valid_indices = np.where((dists[t1] <= d_thresh) & (time_diffs[t1] >= t_thresh))[0]
                if valid_indices.size > 0:
                    positive_dict[sequence_id][t1] = valid_indices.tolist()
                else:
                    positive_dict[sequence_id][t1] = []

            iter_i += 1

        save_file_name = '{}/positive_sequence_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh)
        with open(save_file_name, 'w') as f:
            json.dump(positive_dict, f)
        print('Saved: ', save_file_name)

        return positive_dict

    def visualize_pointcloud(self, drive_id, velodyne_file):
        """
        Visualizes the point cloud for a specific timestamp using open3d.
        """
        fname = self.get_velodyne_fn(drive_id, velodyne_file)
        current_vertex = self.get_velo(fname)[0]
        
        # Create an open3d point cloud object
        pcd = o3d.geometry.PointCloud()
        
        # Set the point cloud data
        pcd.points = o3d.utility.Vector3dVector(current_vertex[:, :3])
        
        # Optionally set colors if intensity data is available (scaled to [0, 1] range)
        if current_vertex.shape[1] > 3:
            intensities = current_vertex[:, 3] / 255.0  # Normalize intensity values
            pcd.colors = o3d.utility.Vector3dVector(np.repeat(intensities[:, None], 3, axis=1))
        
        # Visualize
        o3d.visualization.draw_geometries([pcd])

# Example usage
if __name__ == '__main__':
    args = parser.parse_args()
    base_dir = '/media/vision/SSD1/Datasets/NCLT'
    
    gen_ri_bev_drive_ids = ['2012-02-12', '2012-02-18', '2012-02-19', '2012-03-17', '2012-03-25', '2012-03-31', 
                            '2012-04-29', '2012-05-11', '2012-05-26', '2012-06-15', '2012-08-04', '2012-08-20', 
                            '2012-09-28', '2012-10-28', '2012-11-04', '2012-11-16', '2012-11-17', '2012-12-01', 
                            '2013-01-10', '2013-02-23', '2013-04-05']
    tuple_mining_drive_ids = ['2012-01-08', '2012-01-15', '2012-01-22', '2012-02-02', '2012-02-04', '2012-02-05', 
                            '2012-02-12', '2012-02-18', '2012-02-19', '2012-03-17', '2012-03-25', '2012-03-31', 
                            '2012-04-29', '2012-05-11', '2012-05-26', '2012-06-15', '2012-08-04', '2012-08-20', 
                            '2012-09-28', '2012-10-28', '2012-11-04', '2012-11-16', '2012-11-17', '2012-12-01', 
                            '2013-01-10', '2013-02-23', '2013-04-05']

    # gen ri_bev iamge parameter
    range_thresh = [0, 15, 30, 45, 60]
    height_thresh = [-4, 0, 4, 8, 12]
    fov_up = 30.67
    fov_down = -10.67
    proj_H = 32
    proj_W = 900
    min_range = min(range_thresh)
    max_range = max(range_thresh)
    min_height = min(height_thresh)
    max_height = max(height_thresh)

    # tuple mining parameter
    nclt_tuples_output_dir = os.path.join(os.path.dirname(__file__), '../../config/nclt_tuples/')
    t_thresh = 0

    preprocessor = NCLTPreprocessor(base_dir, range_thresh, height_thresh, fov_up, fov_down, proj_H, proj_W, max_range, min_range, min_height, max_height)
    
    # gen ri_bev image
    if args.gen_ri_bev:
        preprocessor.gen_ri_bev_and_save_all(gen_ri_bev_drive_ids)

    # tuple mining
    if args.tuple_mining:
        preprocessor.gen_positive_dict_and_save_all(tuple_mining_drive_ids, nclt_tuples_output_dir, [3, 20], t_thresh)
