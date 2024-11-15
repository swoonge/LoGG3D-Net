# Based on: https://github.com/kxhit/pointnetvlad/blob/master/submap_generation/KITTI/gen_gt.py

import sys
import os
import json
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.train_config import get_config
from scipy.spatial.distance import pdist, squareform
from time import time

cfg = get_config()


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


def get_positive_dict(basedir, sequences, output_dir, d_thresh, t_thresh):
    positive_dict = {}
    print('d_thresh: ', d_thresh)
    print('output_dir: ', output_dir)
    print('')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for sequence in sequences:
        print(sequence)
        _, scan_positions = load_poses_from_txt(
            basedir + '/sequences/' + sequence + '/poses.txt')
        scan_timestamps = load_timestamps(
            basedir + '/sequences/' + sequence + '/times.txt')

        sequence_id = str(int(sequence))
        if sequence not in positive_dict:
            positive_dict[sequence_id] = {}

        for t1 in tqdm(range(len(scan_timestamps))):
            for t2 in range(len(scan_timestamps)):
                if p_dist(scan_positions[t1], scan_positions[t2], d_thresh) & t_dist(scan_timestamps[t1], scan_timestamps[t2], t_thresh):
                    if t1 not in positive_dict[sequence_id]:
                        positive_dict[sequence_id][t1] = []
                    positive_dict[sequence_id][t1].append(t2)

    save_file_name = '{}/positive_sequence_D-{}_T-{}.json'.format(
        output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict

def get_positive_dict_matrix(basedir, sequences, output_dir, d_thresh, t_thresh):
    positive_dict = {}
    print('d_thresh: ', d_thresh)
    print('output_dir: ', output_dir)
    print('')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    iter_i = 1
    for sequence in sequences:
        _, translations = load_poses_from_txt(
            basedir + '/sequences/' + sequence + '/poses.txt')
        scan_timestamps = np.array(load_timestamps(
            basedir + '/sequences/' + sequence + '/times.txt'))

        sequence_id = sequence
        if sequence not in positive_dict:
            positive_dict[str(int(sequence_id))] = {}

        # pdist로 모든 쌍별 유클리드 거리 계산
        dists = squareform(pdist(translations, metric='euclidean'))
        
        # 모든 시간 간의 차이를 계산
        time_diffs = np.abs(scan_timestamps[:, np.newaxis] - scan_timestamps[np.newaxis, :])

        for t1 in tqdm(range(len(scan_timestamps)), desc=f"[{iter_i}/{len(sequences)}]Processing sequence {sequence}"):
            # 거리와 시간 차이 조건에 맞는 인덱스 추출
            valid_indices = np.where((dists[t1] <= d_thresh) & (time_diffs[t1] >= t_thresh) & (dists[t1] > 0))[0]
            if valid_indices.size > 0:
                positive_dict[str(int(sequence_id))][t1] = valid_indices.tolist()

        iter_i += 1

    save_file_name = '{}/positive_sequence_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh)
    with open(save_file_name, 'w') as f:
        json.dump(positive_dict, f)
    print('Saved: ', save_file_name)

    return positive_dict


#####################################################################################
if __name__ == "__main__":
    basedir = cfg.kitti_dir
    sequences = ['00', '01', '02', '03', '04',
                 '05', '06', '07', '08', '09', '10']
    output_dir = os.path.join(os.path.dirname(
        __file__), '../../config/kitti_tuples/')

    t_thresh = 0
    start_time = time()
    get_positive_dict_matrix(basedir, sequences, output_dir, 3, t_thresh)
    print(f"time consume: {time() - start_time}")
    start_time = time()
    get_positive_dict_matrix(basedir, sequences, output_dir, 20, t_thresh)
    print(f"time consume: {time() - start_time}")