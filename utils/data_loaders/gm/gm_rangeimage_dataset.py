import os
import sys
import glob
import random
import numpy as np
import logging
import json
from utils.data_utils.range_projection import range_projection

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.misc_utils import Timer
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *

class GMRangeImageDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from gm odometry dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.gm_dir
        self.gp_rem = config.gp_rem
        self.pnv_prep = config.pnv_preprocessing
        self.timer = Timer()

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing GMDataset")
        logging.info(f"Loading the subset {phase} from {root}")
        if self.gp_rem:
            logging.info("Dataloader initialized with Ground Plane removal.")

        sequences = config.gm_data_split[phase]
        for drive_id in sequences:
            drive_id = int(drive_id)
            inames = self.get_all_scan_ids(drive_id, is_sorted=True)
            for start_time in inames:
                self.files.append((drive_id, start_time))

    def get_all_scan_ids(self, drive_id, is_sorted=False):
        fnames = glob.glob(
            self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        if is_sorted:
            return sorted(inames)
        return inames

    def get_velodyne_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname
    
    def get_rangeimage_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/depth_map/%06d.png' % (drive, t)
        return fname

    def get_pointcloud_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
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

    def get_rangeimage_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        range_image, _, _, _ = range_projection(xyzr, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=80)
        # range_image = self.fill_zero_rows(range_image)
        return range_image

    def fill_zero_rows(self, depth_image):
        # depth_image는 [64, 900] 형태라고 가정
        depth_image = depth_image.copy()  # 원본 배열을 변경하지 않도록 복사

        # 0이 있는 행의 인덱스를 찾음
        zero_rows = np.where(np.all(depth_image == 0, axis=1))[0]

        for row in zero_rows:
            # 가장 가까운 행을 찾기 위해 위쪽, 아래쪽 모두 탐색
            upper_row = row - 1
            lower_row = row + 1

            while upper_row >= 0 or lower_row < depth_image.shape[0]:
                # 위쪽에 가장 가까운 0이 아닌 값을 찾음
                if upper_row >= 0 and not np.all(depth_image[upper_row, :] == 0):
                    depth_image[row, :] = depth_image[upper_row, :]
                    break
                
                # 아래쪽에 가장 가까운 0이 아닌 값을 찾음
                if lower_row < depth_image.shape[0] and not np.all(depth_image[lower_row, :] == 0):
                    depth_image[row, :] = depth_image[lower_row, :]
                    break
                
                # 위쪽과 아래쪽으로 계속해서 확장
                upper_row -= 1
                lower_row += 1

        return depth_image

    def __getitem__(self, idx):
        drive_id = self.files[idx][0]
        t0 = self.files[idx][1]

        xyz0_th = self.get_rangeimage_tensor(drive_id, t0)
        meta_info = {'drive': drive_id, 't0': t0}

        return (xyz0_th,
                meta_info)


class GMRangeImageTupleDataset(GMRangeImageDataset):
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
        self.root = root = config.gm_dir
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        self.gp_rem = config.gp_rem
        self.pnv_prep = config.pnv_preprocessing
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing GMTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.gm_data_split[phase]
        tuple_dir = os.path.join(os.path.dirname(
            __file__), '../../../config/gm_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.gm_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.gm_20m_json, "r"))
        self.gm_seq_lens = config.gm_seq_lens
        for drive_id in sequences:
            drive_id = int(drive_id)
            fnames = glob.glob(
                root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(
                fnames) > 0, f"Make sure that the path {root} has data {drive_id}"
            inames = sorted([int(os.path.split(fname)[-1][:-4])
                            for fname in fnames])

            for query_id in inames:
                positives = self.get_positives(drive_id, query_id)
                negatives = self.get_negatives(drive_id, query_id)
                self.files.append((drive_id, query_id, positives, negatives))

    def get_positives(self, sq, index):
        sq = str(int(sq))
        assert sq in self.dict_3m.keys(), f"Error: Sequence {sq} not in json."
        sq_1 = self.dict_3m[sq]
        if str(int(index)) in sq_1:
            positives = sq_1[str(int(index))]
        else:
            positives = []
        if index in positives:
            positives.remove(index)
        return positives

    def get_negatives(self, sq, index):
        sq = str(int(sq))
        assert sq in self.dict_20m.keys(), f"Error: Sequence {sq} not in json."
        sq_2 = self.dict_20m[sq]
        all_ids = set(np.arange(self.gm_seq_lens[sq]))
        neg_set_inv = sq_2[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.gm_seq_lens[str(drive_id)])
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
        drive_id, query_id = self.files[idx][0], self.files[idx][1]
        positive_ids, negative_ids = self.files[idx][2], self.files[idx][3]

        if len(positive_ids) < self.positives_per_query:
            positive_ids = positive_ids + positive_ids
        if len(negative_ids) < self.negatives_per_query:
            negative_ids = negative_ids + negative_ids

        sel_positive_ids = random.sample(
            positive_ids, self.positives_per_query)
        sel_negative_ids = random.sample(
            negative_ids, self.negatives_per_query)
        positives, negatives, other_neg = [], [], None

        query_th = self.get_rangeimage_tensor(drive_id, query_id)
        for sp_id in sel_positive_ids:
            positives.append(self.get_rangeimage_tensor(drive_id, sp_id))
        for sn_id in sel_negative_ids:
            negatives.append(self.get_rangeimage_tensor(drive_id, sn_id))

        meta_info = {'drive': drive_id, 'query_id': query_id}

        if not self.quadruplet:
            return (query_th,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(
                drive_id, query_id, sel_positive_ids, sel_negative_ids)
            other_neg_th = self.get_rangeimage_tensor(drive_id, other_neg_id)
            return (query_th,
                    positives,
                    negatives,
                    other_neg_th,
                    meta_info)


#####################################################################################
# Load poses
#####################################################################################

def transfrom_cam2velo(Tcam):
    R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                  -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
                  ]).reshape(3, 3)
    t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
    cam2velo = np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    return Tcam @ cam2velo


def load_poses_from_txt(file_name):
    """
    Modified function from: https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py
    """
    f = open(file_name, 'r')
    s = f.readlines()
    f.close()
    transforms = {}
    positions = []
    for cnt, line in enumerate(s):
        P = np.eye(4)
        line_split = [float(i) for i in line.split(" ") if i != ""]
        withIdx = len(line_split) == 13
        for row in range(3):
            for col in range(4):
                P[row, col] = line_split[row*4 + col + withIdx]
        if withIdx:
            frame_idx = line_split[0]
        else:
            frame_idx = cnt
        transforms[frame_idx] = P
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def parse_and_compute(value):
    # 먼저 숫자와 첫 번째 지수를 분리 (예: 4.799437e+04)
    base_part, remaining = value.split('e', 1)
    
    # 첫 번째 지수에서 두 번째 지수를 다시 분리 (예: +04e-03)
    first_exponent, second_exponent = remaining.split('e')

    # 첫 번째 부분을 처리 (숫자 부분 + 첫 번째 지수 계산)
    base_value = float(f"{base_part}e{first_exponent}")
    
    # 두 번째 지수를 float으로 변환
    second_exponent_value = float(f"1e{second_exponent}")
    
    # 최종 값 계산 (숫자 값에 두 번째 지수를 곱함)
    return base_value * second_exponent_value

def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    times_list = np.asarray([parse_and_compute(t.strip()) for t in stimes_list])
    file1.close()
    return times_list
