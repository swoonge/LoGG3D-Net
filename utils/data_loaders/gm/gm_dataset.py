import os, sys, random, logging, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np

from utils.data_loaders.pointcloud_dataset import *
from utils.data_utils.utils import *

class GMDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from gm odometry dataset. 
    """
    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = config.gm_dir
        self.gp_rem = config.gp_rem
        self.pnv_prep = config.pnv_preprocessing

        PointCloudDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.logger.info("Initializing GMDataset")
        if self.gp_rem:
            self.logger.info("Dataloader initialized with Ground Plane removal.")

        self.id_file_dicts = {}
        self.files = []

        drive_ids = [str(drive_id).zfill(2) if isinstance(drive_id, int) else drive_id for drive_id in config.gm_data_split[phase]]
        for drive_id in drive_ids:
            files = load_kitti_files(self.root, drive_id, is_sorted=True) # kitti와 파일 구성이 동일
            id_file_dict = {}
            for query_id, file in enumerate(files):
                self.files.append((drive_id, query_id))
                id_file_dict[query_id] = file
            self.id_file_dicts[drive_id] = id_file_dict

    def get_velodyne_fn(self, drive, file):
        fname = os.path.join(self.root, 'sequences', drive, 'velodyne', file)
        return fname

    def get_pointcloud_np(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, self.id_file_dicts[drive_id][pc_id])
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)

        if self.gp_rem:
            not_ground_mask = np.ones(len(xyzr), bool)
            raw_pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
            _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
            not_ground_mask[inliers] = 0
            xyzr = xyzr[not_ground_mask]

        if self.pnv_prep:
            xyzr = self.pnv_preprocessing(xyzr)
        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        if self.random_occlusion:
            xyzr = self.occlude_scan(xyzr)
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            xyzr = scale * xyzr

        return xyzr

    def __getitem__(self, idx):
        drive_id, query_id = self.files[idx]

        query = self.get_pointcloud_np(drive_id, query_id)
        meta_info = {'drive_id': drive_id, 'query_id': query_id}

        return (query,
                meta_info)


class GMTupleDataset(GMDataset):
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
        
        GMDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        self.logger.info("Initializing GMTupleDataset")

        tuple_dir = os.path.join(os.path.dirname(__file__), '../../../config/gm_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.gm_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.gm_20m_json, "r"))
        self.gm_seq_lens = config.gm_seq_lens

        self.files = []
        drive_ids = [str(drive_id).zfill(2) if isinstance(drive_id, int) else drive_id for drive_id in config.gm_data_split[phase]]
        for drive_id in drive_ids:
            files = load_kitti_files(self.root, drive_id, is_sorted=True)
            for query_id, file in enumerate(files):
                positives = self.get_positives(drive_id, query_id)
                negatives = self.get_negatives(drive_id, query_id)
                self.files.append((drive_id, query_id, positives, negatives))

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
        all_ids = set(np.arange(self.gm_seq_lens[str(int(drive_id))]))
        neg_set_inv = drive_id_2[str(int(query_id))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if query_id in negatives:
            negatives.remove(query_id)
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.gm_seq_lens[str(int(drive_id))])
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
        drive_id, query_id, positive_ids, negative_ids = self.files[idx]

        if len(positive_ids) < self.positives_per_query:
            positive_ids = (positive_ids * (self.positives_per_query // len(positive_ids) + 1))[:self.positives_per_query]
        if len(negative_ids) < self.negatives_per_query:
            negative_ids = (negative_ids * (self.negatives_per_query // len(negative_ids) + 1))[:self.negatives_per_query]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query = self.get_pointcloud_np(drive_id, query_id)
        for pos_id in selected_positive_ids:
            positives.append(self.get_pointcloud_np(drive_id, pos_id))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_pointcloud_np(drive_id, neg_id))

        meta_info = {'drive': drive_id, 'query_id': query_id, 'positive_ids': selected_positive_ids, 'negative_ids': selected_negative_ids}
        if not self.quadruplet:
            return (query,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(drive_id, query_id, selected_positive_ids, selected_negative_ids)
            other_neg_th = self.get_pointcloud_np(drive_id, other_neg_id)
            meta_info['other_neg_id'] = other_neg_id
            return (query,
                    positives,
                    negatives,
                    other_neg_th,
                    meta_info)

#####################################################################################
# TEST
#####################################################################################

from config.config import *

# Config 객체 생성
config = get_config()

# 데이터셋 테스트
def test_nclt_datasets():
    
    gm_dataset = GMDataset(phase='test', config=config)
    print("GMDataset 테스트 - 첫 번째 샘플:")
    query_tensor, meta_info = gm_dataset[0]
    print("XYZ tensor:", query_tensor.shape)
    print("Metadata:", meta_info)

    gm_tuple_dataset = GMTupleDataset(phase='train', config=config)
    print("GMTupleDataset 테스트 - 두 번째 샘플:")
    query_tensor, positives, negatives, other_negative, meta_info = gm_tuple_dataset[0]
    print("Query tensor:", query_tensor.shape)
    print("Positives:", len(positives), positives[0].shape)
    print("Negatives:", len(negatives), negatives[0].shape)
    print("Negatives:", other_negative.shape)
    print("Metadata:", meta_info)

# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_nclt_datasets()
