import os
import sys
import numpy as np
import json
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *
from config.config import *
from utils.data_utils.utils import *

class NCLTDepthImageDataset(PointCloudDataset):
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

        if 'OverlapNet' in config.pipeline:
            self.image_folder = 'range_images'
        elif 'CVT' in config.pipeline:
            self.image_folder = 'ri_bev'

        PointCloudDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.logger.info(f"Initializing Dataset with {self.image_folder} folder")

        self.id_file_dicts = {}
        self.timestamps_dict = {}
        self.poses_dict = {}
        self.files = []

        drive_ids = config.nclt_data_split[phase]
        for drive_id in drive_ids:
            files, poses, timestamps = load_nclt_files_poses_timestamps(root, drive_id)
            id_file_dict = {} 
            for query_id, file in enumerate(files):
                self.files.append((drive_id, query_id))
                id_file_dict[query_id] = file.split('.')[0]+'.npy'
            self.id_file_dicts[drive_id] = id_file_dict
            self.timestamps_dict[drive_id] = timestamps
            self.poses_dict[drive_id] = poses
    
    def get_npy_fn(self, drive_id, file):
        return os.path.join(self.root, 'velodyne_data', f'{drive_id}', self.image_folder, file)
    
    def get_npy_file(self, drive_id, pc_id):
        return np.load(self.get_npy_fn(drive_id, self.id_file_dicts[drive_id][pc_id]))
    
    def random_rotate_images(self, depth_images):
        angle = random.randint(0, self.rotation_range)

        if self.image_folder == 'range_images':
            _, width = depth_images.shape
            shift_pixels = int((angle / 360) * width)
            return np.roll(depth_images, shift_pixels, axis=-1)

        elif self.image_folder == 'ri_bev':
            # 배치 전체에 대해 동일한 회전 각도 랜덤 선택
            _, _, width = depth_images.shape
            shift_pixels = int((angle / 360) * width)
        
            return np.array([np.roll(depth_image, shift_pixels, axis=-1) for depth_image in depth_images])
    
    def __getitem__(self, idx):
        drive_id, query_id  = self.files[idx]

        xyz0_th = self.get_npy_file(drive_id, query_id)
        meta_info = {'drive_id': drive_id, 'query_id': query_id}

        return (xyz0_th,
                meta_info)

class NCLTDepthImageTupleDataset(NCLTDepthImageDataset):
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

        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True
        
        NCLTDepthImageDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        tuple_dir = os.path.join(os.path.dirname(__file__), '../../../config/nclt_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.nclt_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.nclt_20m_json, "r"))
        self.nclt_seq_lens = config.nclt_seq_lens

    def get_positives(self, drive_id, query_id):
        assert drive_id in self.dict_3m.keys(), f"Error: Sequence {drive_id} not in json."
        sq_1 = self.dict_3m[drive_id]
        if str(int(query_id)) in sq_1:
            positives = sq_1[str(int(query_id))]
        else:
            positives = []
        if query_id in positives:
            positives.remove(query_id)
        return positives

    def get_negatives(self, drive_id, query_id):
        assert drive_id in self.dict_20m.keys(), f"Error: Sequence {drive_id} not in json."
        drive_id_2 = self.dict_20m[drive_id]
        all_ids = set(np.arange(self.nclt_seq_lens[drive_id]))
        neg_set_inv = drive_id_2[str(int(query_id))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if query_id in negatives:
            negatives.remove(query_id)
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
        assert len(possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]
    
    def get_positives(self, drive_id, query_id):
        assert drive_id in self.dict_3m.keys(), f"Error: Sequence {drive_id} not in json."
        sq_1 = self.dict_3m[drive_id]
        if str(int(query_id)) in sq_1:
            positives = sq_1[str(int(query_id))]
        else:
            positives = []
        if query_id in positives:
            positives.remove(query_id)
        return positives

    def get_negatives(self, drive_id, query_id):
        assert drive_id in self.dict_20m.keys(), f"Error: Sequence {drive_id} not in json."
        drive_id_2 = self.dict_20m[drive_id]
        all_ids = set(np.arange(self.nclt_seq_lens[drive_id]))
        neg_set_inv = drive_id_2[str(int(query_id))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if query_id in negatives:
            negatives.remove(query_id)
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
        assert len(possible_negs) > 0, f"No other negatives for drive {drive_id} id {query_id}"
        other_neg_id = random.sample(possible_negs, 1)
        return other_neg_id[0]

    def __getitem__(self, idx):
        drive_id, query_id = self.files[idx]
        positive_ids = self.get_positives(drive_id, query_id)
        negative_ids = self.get_negatives(drive_id, query_id)

        if len(positive_ids) < self.positives_per_query:
            positive_ids = (positive_ids * (self.positives_per_query // len(positive_ids) + 1))[:self.positives_per_query]
        if len(negative_ids) < self.negatives_per_query:
            negative_ids = (negative_ids * (self.negatives_per_query // len(negative_ids) + 1))[:self.negatives_per_query]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_tensor = self.get_npy_file(drive_id, query_id)
        for pos_id in selected_positive_ids:
            positives.append(self.get_npy_file(drive_id, pos_id))
            # print("Positive:", pos_id, self.get_npy_file(drive_id, pos_id).shape)
        for neg_id in selected_negative_ids:
            negatives.append(self.get_npy_file(drive_id, neg_id))
            # print("Negative:", neg_id, self.get_npy_file(drive_id, neg_id).shape)
        
        if self.random_rotation:
            query = self.random_rotate_images(query)
            for i in range(len(positives)):
                positives[i] = self.random_rotate_images(positives[i])
            for i in range(len(negatives)):
                negatives[i] = self.random_rotate_images(negatives[i])

        meta_info = {'drive': drive_id, 'query_id': query_id, 'positive_ids': selected_positive_ids, 'negative_ids': selected_negative_ids}

        if not self.quadruplet:
            return (query_tensor, positives, negatives, meta_info)
        else:  # For Quadruplet Loss
            other_negative_id = self.get_other_negative(drive_id, query_id, selected_positive_ids, selected_negative_ids)
            other_negative_tensor = self.get_npy_file(drive_id, other_negative_id)
            if self.random_rotation:
                other_negative_tensor = self.random_rotate_images(other_negative_tensor)
            return (query_tensor, positives, negatives, other_negative_tensor, meta_info)

# Config 객체 생성
config = get_config()

# 데이터셋 테스트
def test_nclt_datasets():
    # # NCLTDataset 테스트
    config.pipline = 'OverlapTransformer'
    dataset = NCLTDepthImageDataset(phase='test', config=config)
    print("NCLTDepthImageDataset 테스트 - 첫 번째 샘플:")
    query_tensor, meta_info = dataset[0]
    print("XYZ tensor:", query_tensor.shape)
    print("Metadata:", meta_info)
    del dataset

    config.pipline = 'CVTNet'
    dataset = NCLTDepthImageDataset(phase='test', config=config)
    print("NCLTDepthImageDataset 테스트 - 첫 번째 샘플:")
    query_tensor, meta_info = dataset[0]
    print("XYZ tensor:", query_tensor.shape)
    print("Metadata:", meta_info)
    del dataset

    config.pipline = 'OverlapTransformer'
    tuple_dataset = NCLTDepthImageTupleDataset(phase='train', config=config)
    print("NCLTDepthImageTupleDataset 테스트 - 두 번째 샘플:")
    query_tensor, positives, negatives, other_negative, meta_info = tuple_dataset[0]
    print("Query tensor:", query_tensor.shape)
    print("Positives:", len(positives), positives[0].shape)
    print("Negatives:", len(negatives), negatives[0].shape)
    print("Other Negatives:", other_negative.shape)
    print("Metadata:", meta_info)
    del tuple_dataset

    config.pipline = 'CVTNet'
    tuple_dataset = NCLTDepthImageTupleDataset(phase='train', config=config)
    print("NCLTDepthImageTupleDataset 테스트 - 두 번째 샘플:")
    query_tensor, positives, negatives, other_negative, meta_info = tuple_dataset[0]
    print("Query tensor:", query_tensor.shape)
    print("Positives:", len(positives), positives[0].shape)
    print("Negatives:", len(negatives), negatives[0].shape)
    print("Other Negatives:", other_negative.shape)
    print("Metadata:", meta_info)
    del tuple_dataset

# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_nclt_datasets()

