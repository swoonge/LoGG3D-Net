import os
import sys
import glob
import random
import numpy as np
import logging
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.misc_utils import Timer
from utils.o3d_tools import *
from utils.data_loaders.pointcloud_dataset import *
from data_utils.gen_ri_bev import *
from utils.tictoc import Timer_for_general

class KittiCVTDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from KITTI odometry dataset. 
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = root = config.kitti_dir

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing KittiDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        self.sequences = config.kitti_data_split[phase]
        for drive_id in self.sequences:
            drive_id = int(drive_id)
            self.timestamps, self.ri_bev_files, self.poses = self.timestamps_files_and_gt(drive_id, is_sorted=True)
            print(f"Loaded {len(self.ri_bev_files)} ri-bev files from sequence {drive_id}")
            for query_id, ri_bev_file in enumerate(self.ri_bev_files):
                self.files.append((drive_id, query_id, ri_bev_file))

    def timestamps_files_and_gt(self, drive_id, is_sorted=False):
        fnames = glob.glob(
            self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
        assert len(
            fnames) > 0, f"Make sure that the path {self.root} has drive id: {drive_id}"
        inames = [int(os.path.split(fname)[-1][:-4]) for fname in fnames]
        if is_sorted:
            inames = sorted(inames)
        
        timestamps = np.array(load_timestamps(self.root + '/sequences/%02d/times.txt' % drive_id))
        _, poses = load_poses_from_txt(os.path.join(self.root + '/sequences/%02d/poses.txt' % drive_id))
        
        return timestamps, inames, poses
    
    def get_npz_fn(self, drive, query_id):
        fname = self.root + '/sequences/%02d/ri_bev/%06d.npz' % (drive, query_id)
        return fname
    
    def get_ri_bev_tensor_at_file(self, drive_id, pc_id):
        return np.load(self.get_npz_fn(drive_id, pc_id))['ri_bev']

    def __getitem__(self, idx):
        drive_id, query_id, ri_bev_file  = self.files[idx]

        xyz0_th = self.get_ri_bev_tensor_at_file(drive_id, query_id)
        meta_info = {'drive': drive_id, 'query_id': query_id}

        return (xyz0_th,
                meta_info)


class KittiCVTTupleDataset(KittiCVTDataset):
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
        self.root = root = config.kitti_dir
        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False

        self.timer = Timer_for_general()

        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        PointCloudDataset.__init__(
            self, phase, random_rotation, random_occlusion, random_scale, config)

        logging.info("Initializing KittiTupleDataset")
        logging.info(f"Loading the subset {phase} from {root}")

        sequences = config.kitti_data_split[phase]
        tuple_dir = os.path.join(os.path.dirname(
            __file__), '../../../config/kitti_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.kitti_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.kitti_20m_json, "r"))
        self.kitti_seq_lens = config.kitti_seq_lens
        self.poses = {}
        self.timestamps = {}
        
        for drive_id in sequences:
            drive_id = int(drive_id)
            timestamps, self.ri_bev_files, poses = self.timestamps_files_and_gt(drive_id, is_sorted=True)
            self.timestamps[drive_id] = timestamps
            self.poses[drive_id] = poses
            for query_id, ri_bev_file in enumerate(self.ri_bev_files):
                positives = self.get_positives(drive_id, query_id)
                negatives = self.get_negatives(drive_id, query_id)
                self.files.append((drive_id, query_id, positives, negatives, timestamps[query_id]))

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
        all_ids = set(np.arange(self.kitti_seq_lens[sq]))
        neg_set_inv = sq_2[str(int(index))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if index in negatives:
            negatives.remove(index)
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.kitti_seq_lens[str(drive_id)])
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
        # self.timer.tic()
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

        query_th = self.get_ri_bev_tensor_at_file(drive_id, query_id)
        for sp_id in sel_positive_ids:
            positives.append(self.get_ri_bev_tensor_at_file(drive_id, sp_id))
        for sn_id in sel_negative_ids:
            negatives.append(self.get_ri_bev_tensor_at_file(drive_id, sn_id))

        meta_info = {'drive': drive_id, 'query_id': query_id}

        if not self.quadruplet:
            # print(self.timer.toc())
            return (query_th,
                    positives,
                    negatives,
                    meta_info)
        else:  # For Quadruplet Loss
            other_neg_id = self.get_other_negative(
                drive_id, query_id, sel_positive_ids, sel_negative_ids)
            other_neg_th = self.get_ri_bev_tensor_at_file(drive_id, other_neg_id)
            # print(self.timer.toc())
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
        transforms[frame_idx] = transfrom_cam2velo(P)
        positions.append([P[0, 3], P[2, 3], P[1, 3]])
    return transforms, np.asarray(positions)


#####################################################################################
# Load timestamps
#####################################################################################


def load_timestamps(file_name):
    # file_name = data_dir + '/times.txt'
    file1 = open(file_name, 'r+')
    stimes_list = file1.readlines()
    s_exp_list = np.asarray([float(t[-4:-1]) for t in stimes_list])
    times_list = np.asarray([float(t[:-2]) for t in stimes_list])
    times_listn = [times_list[t] * (10**(s_exp_list[t]))
                   for t in range(len(times_list))]
    file1.close()
    return times_listn
