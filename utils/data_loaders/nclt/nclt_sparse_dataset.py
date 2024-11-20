import os
import sys
import glob
import numpy as np
import logging
import json
import pandas as pd
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from utils.o3d_tools import *
from utils.misc_utils import Timer, hashM
from utils.data_loaders.pointcloud_dataset import *
from utils.data_utils.utils import *
from typing import Tuple, Optional
import scipy
from config.train_config import *

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8

class NCLTSparseTupleDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from NCLT dataset.
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.nclt_dir
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = config.train_loss_function == 'quadruplet'
        self.gp_rem = config.gp_rem
        self.voxel_size = config.voxel_size
        self.num_points = config.num_points
        self.phase = phase
        
        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing NCLTSparseTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.nclt_data_split[phase]
        tuple_dir = os.path.join(os.path.dirname(__file__), '../../../config/nclt_tuples/')
        self.dict_close = json.load(open(tuple_dir + config.nclt_3m_json, "r"))
        self.dict_far = json.load(open(tuple_dir + config.nclt_20m_json, "r"))
        self.nclt_seq_lens = config.nclt_seq_lens
        self.id_file_dicts = {}
        self.gt = {}
        
        for sequence_id in sequences:
            timestamps, scan_files, gt = self.timestamps_files_and_gt(root, sequence_id)
            id_file_dict = {}
            for query_id, scan_file in enumerate(scan_files):
                # positives = self.get_positives(sequence_id, query_id)
                # negatives = self.get_negatives(sequence_id, query_id)
                self.files.append((sequence_id, query_id, scan_file, timestamps[query_id]))
                id_file_dict[query_id] = scan_file
                
            self.id_file_dicts[sequence_id] = id_file_dict
            self.gt[sequence_id] = gt
    
    def timestamps_files_and_gt(self, root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        지정된 경로에서 LiDAR 데이터 파일 및 Ground Truth 데이터를 불러와 타임스탬프와 파일 목록, 그리고 보간된 변환 행렬을 반환하는 함수입니다.

        Parameters:
        - root_path (str): 데이터셋의 루트 경로.
        - sequence_id (str): 특정 시퀀스를 나타내는 ID.

        Returns:
        - timestamps (np.ndarray): 초 단위로 변환된 LiDAR 타임스탬프 배열.
        - velodyne_files (np.ndarray): 정렬된 LiDAR 데이터 파일 이름 배열.
        - gt (Optional[np.ndarray]): Ground Truth 변환 행렬. 각 타임스탬프에 대한 4x4 변환 행렬을 포함하며, 
        타임스탬프에 해당하는 ground truth 파일이 없을 경우 None을 반환.

        설명:
        1. `root_path`에 있는 `ground_truth` 및 `velodyne_data/sequence_id/velodyne_sync` 디렉토리를 확인하여
        필요한 파일들이 존재하는지 확인합니다.
        2. `velodyne_sync` 디렉토리에서 파일 이름을 가져와 정렬 후 타임스탬프를 추출합니다.
        3. Ground Truth 파일이 존재하는 경우, 각 LiDAR 타임스탬프에 맞는 변환 행렬을 보간하여 생성합니다.
        변환 행렬은 translation 및 ZYX 순서의 Euler 각도를 기반으로 계산되며, OpenCV 좌표계와 맞추기 위해 추가 변환이 적용됩니다.
        4. 타임스탬프는 첫 번째 타임스탬프를 기준으로 0초부터 시작하도록 조정되며, 마이크로초를 초 단위로 변환합니다.

        """
        root_path = Path(root_path)
        assert root_path.exists(), f"{root_path} does not exist."  # root_path 경로가 존재하는지 확인
        ground_truth_dir = root_path / "ground_truth"
        assert ground_truth_dir.exists(), f"{ground_truth_dir} does not exist."  # ground_truth 디렉토리가 존재하는지 확인
        velodyne_dir = root_path / "velodyne_data" / sequence_id / "velodyne_sync"

        # velodyne_sync 디렉토리에서 파일을 정렬하여 로드
        velodyne_files = np.array(sorted(os.listdir(str(velodyne_dir))), dtype=np.str_)
        # 파일명에서 확장자를 제거하고, 정수형으로 변환하여 timestamps 생성
        timestamps = np.array([file.split(".")[0] for file in velodyne_files], dtype=np.int64)
        ground_truth_file = ground_truth_dir / f"groundtruth_{sequence_id}.csv"

        gt = None
        if ground_truth_file.exists():  # groundtruth 파일이 존재하는 경우
            ground_truth = self.__ground_truth(str(ground_truth_file))  # groundtruth 파일을 로드

            # Ground truth와 LiDAR 타임스탬프가 일치하지 않는 경우 보간
            gt_t = ground_truth[:, 0]  # ground truth의 타임스탬프
            t_min = np.min(gt_t)
            t_max = np.max(gt_t)
            # nearest 방식으로 보간을 수행
            inter = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)

            # Ground truth가 존재하는 타임스탬프 범위 내로 제한
            filter_ = (timestamps > t_min) * (timestamps < t_max)
            timestamps = timestamps[filter_]
            velodyne_files = velodyne_files[filter_]

            # 보간된 ground truth를 gt 변수에 저장
            gt = inter(timestamps)
            gt_tr = gt[:, :3]  # 변환 행렬의 translation 부분
            gt_euler = gt[:, 3:][:, [2, 1, 0]]  # ZYX 순서로 Euler 각도 변환
            gt_rot = scipy.spatial.transform.Rotation.from_euler("ZYX", gt_euler).as_matrix()  # 회전 행렬로 변환

            # 4x4 변환 행렬 형태로 변환하여 gt에 저장
            gt = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(gt.shape[0], axis=0)
            gt[:, :3, :3] = gt_rot
            gt[:, :3, 3] = gt_tr

            # 좌표계를 변환하여 OpenCV 좌표계와 맞춤
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

        # timestamps, velodyne 파일 목록, gt 변환 행렬 반환
        return timestamps, velodyne_files, gt
    
    def __ground_truth(self, gt_file: str):
        gt = pd.read_csv(gt_file, sep=",", low_memory=False).values
        return gt

    def get_velodyne_fn(self, drive_id, fname):
        return os.path.join(self.root, 'velodyne_data', f'{drive_id}', 'velodyne_sync', fname)
    
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

    def get_pointcloud_sparse_tensor(self, sequence_id, scan_file):
        fname = self.get_velodyne_fn(sequence_id, scan_file)
        xyz, intensity = self.get_velo(fname)
        xyzr = np.hstack([xyz, intensity.reshape(-1, 1)])

        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        if self.random_occlusion:
            xyzr = self.occlude_scan(xyzr)
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            xyzr = scale * xyzr

        pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = xyzr.astype(np.float32)

        _, inds = sparse_quantize(pc_,
                                  return_index=True,
                                  return_inverse=False)

        if 'train' in self.phase:
            if len(inds) > self.num_points:
                inds = np.random.choice(inds, self.num_points, replace=False)

        pc = pc_[inds]
        feat = feat_[inds]
        sparse_pc = SparseTensor(feat, pc)

        return sparse_pc


    def get_positives(self, sequence, index):
        assert sequence in self.dict_close.keys(), f"Sequence {sequence} not in close JSON file."
        close_files = self.dict_close[sequence]
        if str(int(index)) in close_files:
            positives = close_files[str(int(index))]
        else:
            positives = []
        if index in positives:
            positives.remove(index)
        return positives

    def get_negatives(self, sequence, index):
        assert sequence in self.dict_far.keys(), f"Sequence {sequence} not in far JSON file."
        far_files = self.dict_far[sequence]
        all_ids = set(np.arange(self.nclt_seq_lens[sequence]))
        neg_set_inv = far_files[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives
    
    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.nclt_seq_lens[drive_id])
        neighbour_ids = sel_positive_ids
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(drive_id, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        assert len(
            possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]

    def __getitem__(self, idx):
        sequence_id, query_id = self.files[idx][:2]
        positive_ids = self.get_positives(sequence_id, query_id)
        negative_ids = self.get_negatives(sequence_id, query_id)
        # positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]
        timestamp = self.files[idx][3]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives = [], []

        query_tensor = self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][query_id])
        for pos_id in selected_positive_ids:
            positives.append(self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][pos_id]))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][neg_id]))

        meta_info = {'sequence': sequence_id, 'query_id': query_id, 'timestamp': timestamp}

        if not self.quadruplet:
            return query_tensor, positives, negatives, meta_info
        else:  # For Quadruplet Loss
            other_negative_id = self.get_other_negative(sequence_id, query_id, selected_positive_ids, selected_negative_ids)
            other_negative_tensor = self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][other_negative_id])
            return query_tensor, positives, negatives, other_negative_tensor, meta_info
        
class NCLTPointSparseTupleDataset(NCLTSparseTupleDataset):
    r"""
    Generate single pointcloud frame from NCLT dataset.
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.nclt_dir
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = config.train_loss_function == 'quadruplet'
        self.gp_rem = config.gp_rem
        self.voxel_size = config.voxel_size
        self.num_points = config.num_points
        self.phase = phase
        
        NCLTSparseTupleDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing NCLTPointSparseTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.nclt_data_split[phase]
        tuple_dir = os.path.join(os.path.dirname(__file__), '../../../config/nclt_tuples/')
        self.dict_close = json.load(open(tuple_dir + config.nclt_3m_json, "r"))
        self.dict_far = json.load(open(tuple_dir + config.nclt_20m_json, "r"))
        self.nclt_seq_lens = config.nclt_seq_lens
        self.id_file_dicts = {}
        
        for sequence_id in sequences:
            timestamps, scan_files, _ = self.timestamps_files_and_gt(root, sequence_id)
            id_file_dict = {}
            for query_id, scan_file in enumerate(scan_files):
                # positives = self.get_positives(sequence_id, query_id)
                # negatives = self.get_negatives(sequence_id, query_id)
                self.files.append((sequence_id, query_id, scan_file, timestamps[query_id]))
                id_file_dict[query_id] = scan_file
            self.id_file_dicts[sequence_id] = id_file_dict
    
    def timestamps_files_and_gt(self, root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        지정된 경로에서 LiDAR 데이터 파일 및 Ground Truth 데이터를 불러와 타임스탬프와 파일 목록, 그리고 보간된 변환 행렬을 반환하는 함수입니다.

        Parameters:
        - root_path (str): 데이터셋의 루트 경로.
        - sequence_id (str): 특정 시퀀스를 나타내는 ID.

        Returns:
        - timestamps (np.ndarray): 초 단위로 변환된 LiDAR 타임스탬프 배열.
        - velodyne_files (np.ndarray): 정렬된 LiDAR 데이터 파일 이름 배열.
        - gt (Optional[np.ndarray]): Ground Truth 변환 행렬. 각 타임스탬프에 대한 4x4 변환 행렬을 포함하며, 
        타임스탬프에 해당하는 ground truth 파일이 없을 경우 None을 반환.

        설명:
        1. `root_path`에 있는 `ground_truth` 및 `velodyne_data/sequence_id/velodyne_sync` 디렉토리를 확인하여
        필요한 파일들이 존재하는지 확인합니다.
        2. `velodyne_sync` 디렉토리에서 파일 이름을 가져와 정렬 후 타임스탬프를 추출합니다.
        3. Ground Truth 파일이 존재하는 경우, 각 LiDAR 타임스탬프에 맞는 변환 행렬을 보간하여 생성합니다.
        변환 행렬은 translation 및 ZYX 순서의 Euler 각도를 기반으로 계산되며, OpenCV 좌표계와 맞추기 위해 추가 변환이 적용됩니다.
        4. 타임스탬프는 첫 번째 타임스탬프를 기준으로 0초부터 시작하도록 조정되며, 마이크로초를 초 단위로 변환합니다.

        """
        root_path = Path(root_path)
        assert root_path.exists(), f"{root_path} does not exist."  # root_path 경로가 존재하는지 확인
        ground_truth_dir = root_path / "ground_truth"
        assert ground_truth_dir.exists(), f"{ground_truth_dir} does not exist."  # ground_truth 디렉토리가 존재하는지 확인
        velodyne_dir = root_path / "velodyne_data" / sequence_id / "velodyne_sync"

        # velodyne_sync 디렉토리에서 파일을 정렬하여 로드
        velodyne_files = np.array(sorted(os.listdir(str(velodyne_dir))), dtype=np.str_)
        # 파일명에서 확장자를 제거하고, 정수형으로 변환하여 timestamps 생성
        timestamps = np.array([file.split(".")[0] for file in velodyne_files], dtype=np.int64)
        ground_truth_file = ground_truth_dir / f"groundtruth_{sequence_id}.csv"

        gt = None
        if ground_truth_file.exists():  # groundtruth 파일이 존재하는 경우
            ground_truth = self.__ground_truth(str(ground_truth_file))  # groundtruth 파일을 로드

            # Ground truth와 LiDAR 타임스탬프가 일치하지 않는 경우 보간
            gt_t = ground_truth[:, 0]  # ground truth의 타임스탬프
            t_min = np.min(gt_t)
            t_max = np.max(gt_t)
            # nearest 방식으로 보간을 수행
            inter = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)

            # Ground truth가 존재하는 타임스탬프 범위 내로 제한
            filter_ = (timestamps > t_min) * (timestamps < t_max)
            timestamps = timestamps[filter_]
            velodyne_files = velodyne_files[filter_]

            # 보간된 ground truth를 gt 변수에 저장
            gt = inter(timestamps)
            gt_tr = gt[:, :3]  # 변환 행렬의 translation 부분
            gt_euler = gt[:, 3:][:, [2, 1, 0]]  # ZYX 순서로 Euler 각도 변환
            gt_rot = scipy.spatial.transform.Rotation.from_euler("ZYX", gt_euler).as_matrix()  # 회전 행렬로 변환

            # 4x4 변환 행렬 형태로 변환하여 gt에 저장
            gt = np.eye(4, dtype=np.float32).reshape(1, 4, 4).repeat(gt.shape[0], axis=0)
            gt[:, :3, :3] = gt_rot
            gt[:, :3, 3] = gt_tr

            # 좌표계를 변환하여 OpenCV 좌표계와 맞춤
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

        # timestamps, velodyne 파일 목록, gt 변환 행렬 반환
        return timestamps, velodyne_files, gt
    
    def __ground_truth(self, gt_file: str):
        gt = pd.read_csv(gt_file, sep=",", low_memory=False).values
        return gt

    def get_velodyne_fn(self, drive_id, fname):
        return os.path.join(self.root, 'velodyne_data', f'{drive_id}', 'velodyne_sync', fname)
    
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

    def get_positives(self, sequence, index):
        assert sequence in self.dict_close.keys(), f"Sequence {sequence} not in close JSON file."
        close_files = self.dict_close[sequence]
        if str(int(index)) in close_files:
            positives = close_files[str(int(index))]
        else:
            positives = []
        if index in positives:
            positives.remove(index)
        return positives

    def get_negatives(self, sequence, index):
        assert sequence in self.dict_far.keys(), f"Sequence {sequence} not in far JSON file."
        far_files = self.dict_far[sequence]
        all_ids = set(np.arange(self.nclt_seq_lens[sequence]))
        neg_set_inv = far_files[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives
    
    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.nclt_seq_lens[drive_id])
        neighbour_ids = sel_positive_ids.copy()
        for neg in sel_negative_ids:
            neg_postives_files = self.get_positives(drive_id, neg)
            for pos in neg_postives_files:
                neighbour_ids.append(pos)
        possible_negs = list(set(all_ids) - set(neighbour_ids))
        assert len(
            possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]
    
    def get_sparse_pcd(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, self.id_file_dicts[drive_id][pc_id])
        xyz, intensity = self.get_velo(fname)
        xyzr = np.hstack([xyz, intensity.reshape(-1, 1)])
        if self.gp_rem:
            use_ransac = True
            if use_ransac:
                not_ground_mask = np.ones(len(xyzr), bool)
                raw_pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
                _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
                not_ground_mask[inliers] = 0
                xyzr = xyzr[not_ground_mask]
            else:
                xyzr = xyzr[xyzr[:, 2] > -1.5]

        xyzr_copy = copy.deepcopy(xyzr)
        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = xyzr.astype(np.float32)
        _, inds = sparse_quantize(pc_,
                                  return_index=True,
                                  return_inverse=False)
        if len(inds) > self.num_points:
            inds = np.random.choice(inds, self.num_points, replace=False)

        st = SparseTensor(feat_[inds], pc_[inds])
        pcd = make_open3d_point_cloud(xyzr_copy[inds][:, :3], color=None)
        return st, pcd
    
    def get_delta_pose(self, transforms):
        w_T_p1 = transforms[0]
        w_T_p2 = transforms[1]

        p1_T_w = np.linalg.inv(w_T_p1)
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
        return p1_T_p2
    
    def get_point_tuples(self, drive_id, query_id, pos_id):
        q_st, q_pcd = self.get_sparse_pcd(drive_id, query_id)
        p_st, p_pcd = self.get_sparse_pcd(drive_id, pos_id)
        p_pcd_temp = copy.deepcopy(p_pcd)

        matching_search_voxel_size = min(self.voxel_size*1.5, 0.1)
        all_odometry = [self.gt[drive_id][query_id], self.gt[drive_id][pos_id]]
        delta_T = self.get_delta_pose(all_odometry)
        p_pcd.transform(delta_T)
        reg = o3d.pipelines.registration.registration_icp(
            p_pcd, q_pcd, 0.2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        p_pcd.transform(reg.transformation)

        pos_pairs = get_matching_indices(
            q_pcd, p_pcd, matching_search_voxel_size)
        if not pos_pairs.ndim == 2:
            print('No pos_pairs for ', query_id, 'in drive id: ', drive_id)

        return q_st, p_st, pos_pairs

    def __getitem__(self, idx):
        sequence_id, query_id = self.files[idx][:2]
        positive_ids = self.get_positives(sequence_id, query_id)
        negative_ids = self.get_negatives(sequence_id, query_id)
        # positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]
        timestamp = self.files[idx][3]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives = [], []

        query_st, p_st, pos_pairs = self.get_point_tuples(
            sequence_id, query_id, selected_positive_ids[0])
        positives.append(p_st)

        for pos_id in selected_positive_ids[1:]:
            positives.append(self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][pos_id]))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][neg_id]))

        meta_info = {'drive': sequence_id, 'query_id': query_id, 'positive_ids': selected_positive_ids, 'negative_ids': selected_negative_ids, 'pos_pairs': pos_pairs}

        if not self.quadruplet:
            return {
                'query': query_st,
                'positives': positives,
                'negatives': negatives,
                'meta_info': meta_info
            }
        else:  # For Quadruplet Loss
            other_negative_id = self.get_other_negative(sequence_id, query_id, selected_positive_ids, selected_negative_ids)
            other_negative_tensor = self.get_pointcloud_sparse_tensor(sequence_id, self.id_file_dicts[sequence_id][other_negative_id])
            return {
                'query': query_st,
                'positives': positives,
                'negatives': negatives,
                'other_neg': other_negative_tensor,
                'meta_info': meta_info,
            }


import logging
import json

# 설정 로깅 설정
logging.basicConfig(level=logging.INFO)

# Config 객체 생성
config = get_config()

# 데이터셋 테스트
def test_nclt_datasets():
    # NCLTDataset 테스트
    # train_dataset = NCLTDataset(phase='train', config=config)
    # print("NCLTDataset 테스트 - 첫 번째 샘플:")
    # xyz_th, meta_info = train_dataset[0]
    # print("XYZ tensor:", xyz_th)
    # print("Metadata:", meta_info)

    # NCLTTupleDataset 테스트
    tuple_dataset = NCLTPointSparseTupleDataset(phase='train', config=config)
    print("\nNCLTTupleDataset 테스트 - 첫 번째 샘플:")

    query_tensor = tuple_dataset[0]['query']
    positives = tuple_dataset[0]['positives']
    negatives = tuple_dataset[0]['negatives']
    other_negative = tuple_dataset[0]['other_neg']
    meta_info = tuple_dataset[0]['meta_info']
    print("Query tensor:", query_tensor)
    print("Positives:", len(positives), positives[0])
    print("Negatives:", len(negatives), negatives[0])
    print("Negatives:", other_negative)
    print("Metadata:", meta_info)



# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_nclt_datasets()
