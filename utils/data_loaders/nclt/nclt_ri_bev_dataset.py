import os
import sys
import glob
import numpy as np
import logging
import json
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.misc_utils import Timer
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *
from utils.data_utils.utils_NCLT import *
from typing import Tuple, Optional
import scipy
from config.train_config import *

class NCLTRiBevDataset(PointCloudDataset):
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
        self.timer = Timer()

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing NCLTDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        self.sequences = config.nclt_data_split[phase]

        for sequence_id in self.sequences:
            # scan_files = self.get_all_scan_ids(sequence_id)
            self.timestamps, self.ri_bev_files, self.poses = self.timestamps_files_and_gt(root, sequence_id)
            print(f"Loaded {len(self.ri_bev_files)} ri-bev files from sequence {sequence_id}")
            for query_id, ri_bev_file in enumerate(self.ri_bev_files):
                self.files.append((sequence_id, query_id, ri_bev_file, self.timestamps[query_id]))
    
    def timestamps_files_and_gt(self, root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        지정된 경로에서 LiDAR 데이터 파일 및 Ground Truth 데이터를 불러와 타임스탬프와 파일 목록, 그리고 보간된 변환 행렬을 반환하는 함수입니다.
        단, 이 함수에서는 pointcloud가 아닌 ri-bev 데이터에 대해 처리합니다.

        Parameters:
        - root_path (str): 데이터셋의 루트 경로.
        - sequence_id (str): 특정 시퀀스를 나타내는 ID.

        Returns:
        - timestamps (np.ndarray): 초 단위로 변환된 LiDAR 타임스탬프 배열.
        - ri_bev (np.ndarray): 정렬된 ri_bev image 데이터 파일 이름 배열.
        - gt (Optional[np.ndarray]): Ground Truth 변환 행렬. 각 타임스탬프에 대한 4x4 변환 행렬을 포함하며, 
        타임스탬프에 해당하는 ground truth 파일이 없을 경우 None을 반환.

        설명:
        1. `root_path`에 있는 `ground_truth` 및 `velodyne_data/sequence_id/ri_bev` 디렉토리를 확인하여
        필요한 파일들이 존재하는지 확인합니다.
        2. `ri_bev` 디렉토리에서 파일 이름을 가져와 정렬 후 타임스탬프를 추출합니다.
        3. Ground Truth 파일이 존재하는 경우, 각 LiDAR 타임스탬프에 맞는 변환 행렬을 보간하여 생성합니다.
        변환 행렬은 translation 및 ZYX 순서의 Euler 각도를 기반으로 계산되며, OpenCV 좌표계와 맞추기 위해 추가 변환이 적용됩니다.
        4. 타임스탬프는 첫 번째 타임스탬프를 기준으로 0초부터 시작하도록 조정되며, 마이크로초를 초 단위로 변환합니다.

        """
        root_path = Path(root_path)
        assert root_path.exists(), f"{root_path} does not exist."  # root_path 경로가 존재하는지 확인
        ground_truth_dir = root_path / "ground_truth"
        assert ground_truth_dir.exists(), f"{ground_truth_dir} does not exist."  # ground_truth 디렉토리가 존재하는지 확인
        ri_bev_dir = root_path / "velodyne_data" / sequence_id / "ri_bev"

        # ri_bev 디렉토리에서 파일을 정렬하여 로드
        ri_bev_files = np.array(sorted(os.listdir(str(ri_bev_dir))), dtype=np.str_)
        # 파일명에서 확장자를 제거하고, 정수형으로 변환하여 timestamps 생성
        timestamps = np.array([file.split(".")[0] for file in ri_bev_files], dtype=np.int64)
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
            ri_bev_files = ri_bev_files[filter_]

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
        return timestamps, ri_bev_files, gt
    
    def __ground_truth(self, gt_file: str):
        gt = pd.read_csv(gt_file, sep=",", low_memory=False).values
        return gt

    def get_ri_bev_fn(self, drive_id, fname):
        return os.path.join(self.root, 'velodyne_data', f'{drive_id}', 'ri_bev', fname)

    def get_ri_bev_np(self, sequence_id, ri_bev_file):
        fname = self.get_ri_bev_fn(sequence_id, ri_bev_file)
        with np.load(fname) as data:
            ri_bev_image = data['ri_bev']
        return ri_bev_image

    def __getitem__(self, idx):
        sequence_id, query_id, ri_bev_file, timestamp = self.files[idx]

        ri_bev_image = self.get_ri_bev_np(sequence_id, ri_bev_file)
        meta_info = {'sequence': sequence_id, 'timestamp': timestamp}

        return (ri_bev_image, meta_info)

import random

class NCLTRiBevTupleDataset(NCLTRiBevDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss.
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
        
        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing NCLTTupleDataset")
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

        query_tensor = self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][query_id])
        for pos_id in selected_positive_ids:
            positives.append(self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][pos_id]))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][neg_id]))

        meta_info = {'sequence': sequence_id, 'query_id': query_id, 'timestamp': timestamp}

        if not self.quadruplet:
            return query_tensor, positives, negatives, meta_info
        else:  # For Quadruplet Loss
            other_negative_id = self.get_other_negative(sequence_id, query_id, selected_positive_ids, selected_negative_ids)
            other_negative_tensor = self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][other_negative_id])
            return query_tensor, positives, negatives, other_negative_tensor, meta_info
        
class NCLTRiBevRandomChannelTupleDataset(NCLTRiBevTupleDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss.
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

        self.target_channel = config.target_channel if hasattr(config, 'target_channel') else None
        
        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing NCLTTupleDataset")
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

    # 이걸 q, p, n, on 이미지에 어떻게 섞어서 적용할지 생각해보기
    def reduce_channel(self, range_image, target_channel):
        """
        채널 수를 감소시켜 target_channel에 맞추되, 선택한 채널만 남기고 나머지 채널은 검은색(0)으로 설정합니다.
        
        Args:
            range_image (numpy.ndarray): 원본 range image (C, W) 형태의 배열.
            target_channel (int): 유지하고자 하는 채널 수.
        
        Returns:
            numpy.ndarray: 채널을 줄인 후 크기를 유지한 range image (C, W) 형태의 배열.
        """
        # 현재 채널 수와 너비
        current_channel, _ = range_image.shape
        
        # 간격 계산
        step = current_channel // target_channel
        
        ## 입력과 같은 크기의 이미지 반환
        # 초기화된 0 배열 (검은색 채널로 설정)
        reduced_range_image = np.zeros_like(range_image)
        # 선택한 채널만 복사
        reduced_range_image[::step, :] = range_image[::step, :]

        # ## 줄어든 체널에 따라 이미지 크기 조정 -> 크기가 줄어들어서 사용하지 않음
        # # 간격에 따라 채널 선택
        # reduced_range_image = range_image[::step, :]
        # # 만약 초과로 선택되었을 경우 초과 채널을 잘라냄
        # reduced_range_image = reduced_range_image[:target_channel, :]
        
        return reduced_range_image

    def __getitem__(self, idx):
        sequence_id, query_id = self.files[idx][:2]
        positive_ids = self.get_positives(sequence_id, query_id)
        negative_ids = self.get_negatives(sequence_id, query_id)
        # positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]
        timestamp = self.files[idx][3]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives = [], []

        query_tensor = self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][query_id])
        for pos_id in selected_positive_ids:
            positives.append(self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][pos_id]))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][neg_id]))

        meta_info = {'sequence': sequence_id, 'query_id': query_id, 'timestamp': timestamp}

        if not self.quadruplet:
            return query_tensor, positives, negatives, meta_info
        else:  # For Quadruplet Loss
            other_negative_id = self.get_other_negative(sequence_id, query_id, selected_positive_ids, selected_negative_ids)
            other_negative_tensor = self.get_ri_bev_np(sequence_id, self.id_file_dicts[sequence_id][other_negative_id])
            return query_tensor, positives, negatives, other_negative_tensor, meta_info

import logging
import json
from matplotlib import pyplot as plt

# 설정 로깅 설정
logging.basicConfig(level=logging.INFO)

# Config 객체 생성
config = get_config()

# 데이터셋 테스트
def test_nclt_datasets():
    # # NCLTDataset 테스트
    # train_dataset = NCLTRiBevDataset(phase='train', config=config)
    # print("NCLTDataset 테스트 - 첫 번째 샘플:")
    # ri_bev, meta_info = train_dataset[0]
    # print("XYZ tensor:", ri_bev)
    # print("Metadata:", meta_info)

    # num_images = ri_bev.shape[0]
    # fig, axes = plt.subplots(num_images, 1, figsize=(15, 15))

    # for i in range(num_images):
    #     ax = axes[i] if num_images > 1 else axes
    #     ax.imshow(ri_bev[i], cmap='gray')
    #     ax.set_title(f'Layer {i}')
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.show()

    # NCLTTupleDataset 테스트
    tuple_dataset = NCLTRiBevTupleDataset(phase='train', config=config)
    print("\nNCLTTupleDataset 테스트 - 첫 번째 샘플:")
    query_tensor, positives, negatives, other_negative, meta_info = tuple_dataset[0]
    print("Query tensor:", query_tensor)
    print("Positives:", positives)
    print("Negatives:", negatives)
    print("Negatives:", other_negative)
    print("Metadata:", meta_info)

    ri_bevs = [query_tensor] + positives + negatives + [other_negative]

    num_images = ri_bevs[0].shape[0]
    num_columns = len(ri_bevs)
    fig, axes = plt.subplots(num_images, num_columns, figsize=(15, 30))

    for i in range(num_images):
        for j in range(num_columns):
            ax = axes[i, j] if num_images > 1 else axes[j]
            ax.imshow(ri_bevs[j][i], cmap='gray')
            ax.set_title(f'Layer {i}, Image {j}')
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_nclt_datasets()

