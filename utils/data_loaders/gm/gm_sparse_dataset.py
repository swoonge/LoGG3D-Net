import os, sys, random
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np

from torchsparse.utils.quantize import sparse_quantize
from torchsparse import SparseTensor
from utils.misc_utils import hashM
from utils.o3d_tools import *
from utils.data_loaders.gm.gm_dataset import GMTupleDataset
from utils.data_utils.utils import *

class GMSparseTupleDataset(GMTupleDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss. 
    Convert all to sparse tensors
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        GMTupleDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.logger.info("GMSparseTupleDataset")
        self.voxel_size = config.voxel_size
        self.num_points = config.num_points
        self.phase = phase

    def get_pointcloud_sparse_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, self.id_file_dicts[drive_id][pc_id])
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        if self.gp_rem:
            use_ransac = False
            if use_ransac:
                not_ground_mask = np.ones(len(xyzr), bool)
                raw_pcd = make_open3d_point_cloud(xyzr[:, :3], color=None)
                _, inliers = raw_pcd.segment_plane(0.2, 3, 250)
                not_ground_mask[inliers] = 0
                xyzr = xyzr[not_ground_mask]

            else:
                xyzr = xyzr[xyzr[:, 2] > -1.5]

        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        if self.random_occlusion:
            xyzr = self.occlude_scan(xyzr)
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            xyzr = scale * xyzr

        pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)
        feat_ = xyzr

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

    def __getitem__(self, idx):
        drive_id, query_id, positive_ids, negative_ids = self.files[idx]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query = self.get_pointcloud_sparse_tensor(drive_id, query_id)
        for pos_id in selected_positive_ids:
            positives.append(self.get_pointcloud_sparse_tensor(drive_id, pos_id))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_pointcloud_sparse_tensor(drive_id, neg_id))

        meta_info = {'drive': drive_id, 'query_id': query_id, 'positive_ids': selected_positive_ids, 'negative_ids': selected_negative_ids}

        if not self.quadruplet:
            return {
                'query': query,
                'positives': positives,
                'negatives': negatives,
                'meta_info': meta_info
            }
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(drive_id, query_id, selected_positive_ids, selected_negative_ids)
            other_neg_th = self.get_pointcloud_sparse_tensor(drive_id, other_neg_id)
            meta_info['other_neg_id'] = other_neg_id
            return {
                'query': query,
                'positives': positives,
                'negatives': negatives,
                'other_neg': other_neg_th,
                'meta_info': meta_info,
            }


class GMPointSparseTupleDataset(GMSparseTupleDataset):
    r"""
    Generate tuples (anchor, positives, negatives) using distance
    Optional other_neg for quadruplet loss. 
    Convert all to sparse tensors
    Return additional Positive Point Pairs (for point-wise loss)
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        GMSparseTupleDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.logger.info("GMPointSparseTupleDataset")

        self.poses_dict = {}

        self.drive_ids = [str(drive_id).zfill(2) if isinstance(drive_id, int) else drive_id for drive_id in config.gm_data_split[phase]]  # 드라이브 ID 리스트 생성
        for drive_id in self.drive_ids:
            self.poses_dict[drive_id] = load_gm_poses(self.root, drive_id)

    def get_delta_pose(self, transforms):
        w_T_p1 = transforms[0]
        w_T_p2 = transforms[1]

        p1_T_w = np.linalg.inv(w_T_p1)
        p1_T_p2 = np.matmul(p1_T_w, w_T_p2)
        return p1_T_p2

    def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
        """
        Generate random negative pairs
        """
        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)
        if N_neg < 1:
            N_neg = positive_pairs.shape[0] * 2
        pos_keys = hashM(positive_pairs, hash_seed)

        neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
            np.int64)
        neg_keys = hashM(neg_pairs, hash_seed)
        mask = np.isin(neg_keys, pos_keys, assume_unique=False)
        return neg_pairs[np.logical_not(mask)]

    def get_sparse_pcd(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, self.id_file_dicts[drive_id][pc_id])
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
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
        feat_ = xyzr
        _, inds = sparse_quantize(pc_, return_index=True, return_inverse=False)
        if len(inds) > self.num_points:
            inds = np.random.choice(inds, self.num_points, replace=False)

        st = SparseTensor(feat_[inds], pc_[inds])
        pcd = make_open3d_point_cloud(xyzr_copy[inds][:, :3], color=None)
        return st, pcd

    def get_point_tuples(self, drive_id, query_id, pos_id):
        q_st, q_pcd = self.get_sparse_pcd(drive_id, query_id)
        p_st, p_pcd = self.get_sparse_pcd(drive_id, pos_id)
        p_pcd_temp = copy.deepcopy(p_pcd)

        matching_search_voxel_size = min(self.voxel_size*1.5, 0.1)

        delta_T = self.get_delta_pose([self.poses_dict[drive_id][query_id], self.poses_dict[drive_id][pos_id]])
        p_pcd.transform(delta_T)
        reg = o3d.pipelines.registration.registration_icp(p_pcd, q_pcd, 0.2, np.eye(4),
                                                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        p_pcd.transform(reg.transformation)
        pos_pairs = get_matching_indices(q_pcd, p_pcd, matching_search_voxel_size)
        if not pos_pairs.ndim == 2:
            print('No pos_pairs for ', query_id, 'in drive id: ', drive_id)
        return q_st, p_st, pos_pairs

    def __getitem__(self, idx):
        drive_id, query_id = self.files[idx][0], self.files[idx][1]
        positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

        if len(positive_ids) < self.positives_per_query:
            positive_ids = (positive_ids * (self.positives_per_query // len(positive_ids) + 1))[:self.positives_per_query]
        if len(negative_ids) < self.negatives_per_query:
            negative_ids = (negative_ids * (self.negatives_per_query // len(negative_ids) + 1))[:self.negatives_per_query]

        try:
            selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        except ValueError:
            self.logger.info(f"Drive id:{drive_id}, query id:{query_id}, ppq:{self.positives_per_query}, num_pos:{len(positive_ids)}")
            # Temporary fix to handle len(positive_ids) == 0
            if len(positive_ids) > 0:
                selected_positive_ids = random.choices(positive_ids, k=self.positives_per_query)
            else:
                selected_positive_ids = [query_id for np_id in range(self.positives_per_query)]

        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_st, p_st, pos_pairs = self.get_point_tuples(drive_id, query_id, selected_positive_ids[0])
        positives.append(p_st)

        for sp_id in selected_positive_ids[1:]:
            positives.append(self.get_pointcloud_sparse_tensor(drive_id, sp_id))
        for sn_id in selected_negative_ids:
            negatives.append(self.get_pointcloud_sparse_tensor(drive_id, sn_id))

        meta_info = {'drive': drive_id, 'query_id': query_id, 'positive_ids': selected_positive_ids, 'negative_ids': selected_negative_ids, 'pos_pairs': pos_pairs}

        if not self.quadruplet:
            return {
                'query': query_st,
                'positives': positives,
                'negatives': negatives,
                'meta_info': meta_info
            }
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(drive_id, query_id, selected_positive_ids, selected_negative_ids)
            other_neg_st = self.get_pointcloud_sparse_tensor(drive_id, other_neg_id)
            meta_info['other_neg_id'] = other_neg_id
            return {
                'query': query_st,
                'positives': positives,
                'negatives': negatives,
                'other_neg': other_neg_st,
                'meta_info': meta_info,
            }



#####################################################################################
# TEST
#####################################################################################

from config.config import *

# Config 객체 생성
config = get_config()

# 데이터셋 테스트
def test_datasets():
    # # KittiSparseTupleDataset 테스트
    kitti_sparse_tuple_dataset = GMSparseTupleDataset(phase='train', config=config)
    print("KittiSparseTupleDataset 테스트 - 첫 번째 샘플:")
    query_tensor = kitti_sparse_tuple_dataset[0]['query']
    positives = kitti_sparse_tuple_dataset[0]['positives']
    negatives = kitti_sparse_tuple_dataset[0]['negatives']
    other_negative = kitti_sparse_tuple_dataset[0]['other_neg']
    meta_info = kitti_sparse_tuple_dataset[0]['meta_info']
    print("Query tensor:", query_tensor, query_tensor.F.shape, query_tensor.C.shape)
    print("Positives:", len(positives), positives[0])
    print("Negatives:", len(negatives), negatives[0])
    print("Negatives:", other_negative)
    print("Metadata:", meta_info)

    # visualize_pc(query_tensor)

    from tqdm import tqdm

    kitti_point_sparse_tuple_dataset = GMPointSparseTupleDataset(phase='train', config=config)
    print("KittiPointSparseTupleDataset 테스트 - 첫 번째 샘플:")
    query_tensor = kitti_point_sparse_tuple_dataset[1000]['query']
    positives = kitti_point_sparse_tuple_dataset[1000]['positives']
    negatives = kitti_point_sparse_tuple_dataset[1000]['negatives']
    other_negative = kitti_point_sparse_tuple_dataset[1000]['other_neg']
    meta_info = kitti_point_sparse_tuple_dataset[1000]['meta_info']
    print("Query tensor:", query_tensor)
    print("Positives:", len(positives), positives[0])
    print("Negatives:", len(negatives), negatives[0])
    print("Negatives:", other_negative)
    print("Metadata:", meta_info)
    kitti_point_sparse_tuple_dataset_progress = tqdm(kitti_point_sparse_tuple_dataset, total=len(kitti_point_sparse_tuple_dataset))

# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_datasets()