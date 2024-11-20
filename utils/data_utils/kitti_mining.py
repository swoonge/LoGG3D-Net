import os, json, logging, time
import numpy as np
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.distance import pdist, squareform
import argparse

from utils import *

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='* [%(asctime)s] %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

parser = argparse.ArgumentParser()
parser.add_argument('--gen_ri_bev', action='store_true', help="Generate RI BEV images")
parser.add_argument('--no-gen_ri_bev', dest='gen_ri_bev', action='store_false', help="Do not generate RI BEV images")
parser.add_argument('--gen_ri', action='store_true', help="Generate Range Images")
parser.add_argument('--no-gen_ri', dest='gen_ri', action='store_false', help="Do not generate Range Images")
parser.add_argument('--tuple_mining', action='store_true', help="Enable tuple mining")
parser.add_argument('--no-tuple_mining', dest='tuple_mining', action='store_false', help="Disable tuple mining")
parser.add_argument('--gen_all', action='store_true', help="Generate all outputs")
parser.set_defaults(gen_ri_bev=False, gen_ri=False, tuple_mining=False, gen_all=False)

class Kitti_processor:
    def __init__(self, base_dir, all_drive_ids, range_thresh, height_thresh, fov_up, fov_down, proj_H, proj_W, max_range, min_range, min_height, max_height):
        self.logger = logging.getLogger()
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

        self.drive_ids = all_drive_ids
        self.fnames = {}
        self.timestamps = {}
        self.poses = {}

        for drive_id in tqdm(self.drive_ids, desc="* Load info for NCLT dataset"):
            self.fnames[drive_id] = load_kitti_files(self.base_dir, drive_id, True)
            self.poses[drive_id] = load_kitti_poses(self.base_dir, drive_id)
            self.timestamps[drive_id] = load_kitti_timestamps(self.base_dir, drive_id)

    def get_velodyne_fn(self, drive, t):
        return os.path.join(self.base_dir, 'sequences', drive, 'velodyne', t)

    def get_pointcloud_np(self, fpath):
        return np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
        
    def get_ri_bev_np(self, drive_id, velodyne_file):
        fpath = self.get_velodyne_fn(drive_id, velodyne_file)
        current_vertex = self.get_pointcloud_np(fpath)
        current_vertex = current_vertex[:, :3]
        ri_bev = np.zeros((len(self.range_thresh) + len(self.height_thresh), self.proj_H, self.proj_W))

        for i in range(len(self.range_thresh) - 1):
            nearer_bound = self.range_thresh[i]
            farer_bound = self.range_thresh[i + 1]
            lower_bound = self.height_thresh[i]
            upper_bound = self.height_thresh[i + 1]

            proj_range, _, _ = multi_layer_range_projection(current_vertex,
                                                fov_up=self.fov_up,
                                                fov_down=self.fov_down,
                                                proj_H=self.proj_H,
                                                proj_W=self.proj_W,
                                                max_range=self.max_range,
                                                cut_range=True,
                                                lower_bound=nearer_bound,
                                                upper_bound=farer_bound)
            proj_bev = multi_layer_bev_projection(current_vertex,
                                      proj_H=self.proj_H,
                                      proj_W=self.proj_W,
                                      max_range=self.max_range,
                                      cut_height=True,
                                      lower_bound=lower_bound,
                                      upper_bound=upper_bound)

            ri_bev[int(i + 1), :, :] = proj_range
            ri_bev[int(i + 1 + len(self.range_thresh)), :, :] = proj_bev

        ri_bev[0, :, :], _, _ = multi_layer_range_projection(current_vertex,
                                                 fov_up=self.fov_up,
                                                 fov_down=self.fov_down,
                                                 proj_H=self.proj_H,
                                                 proj_W=self.proj_W,
                                                 max_range=self.max_range,
                                                 cut_range=True,
                                                 lower_bound=self.min_range,
                                                 upper_bound=self.max_range)
        ri_bev[len(self.range_thresh), :, :] = multi_layer_bev_projection(current_vertex,
                                                              proj_H=self.proj_H,
                                                              proj_W=self.proj_W,
                                                              max_range=self.max_range,
                                                              cut_height=True,
                                                              lower_bound=self.min_height,
                                                              upper_bound=self.max_height)
        return ri_bev

    def gen_ri_bev_and_save_all(self, drive_ids):
        print('*' * 100)
        self.logger.info('Generating RI BEV images')
        time.sleep(1)
        for drive_id in drive_ids:
            self.logger.info(f'Generating RI BEV images for the following drives: {drive_id}')
            output_dir = os.path.join(self.base_dir, 'sequences', drive_id, 'ri_bev')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            fnames = self.fnames[drive_id]

            for fname in tqdm(fnames, desc="* Processing Point Clouds at Drive {}".format(drive_id)):
                ri_bev_np = self.get_ri_bev_np(drive_id, fname)
                # self.visualize_pointcloud(drive_id, velodyne_file)
                file_name = fname.split('.')[0]
                output_fname = os.path.join(output_dir, f'{file_name}.npy')
                np.save(output_fname, ri_bev_np)
                tqdm.write(f'* Saved: {output_fname}')
                
        self.logger.info('Finished generating RI BEV images')
        print('*' * 100)

    def get_ri_np(self, drive_id, velodyne_file):
        fpath = self.get_velodyne_fn(drive_id, velodyne_file)
        current_vertex = self.get_pointcloud_np(fpath)
        range_image, _, _, _ = range_projection(current_vertex, fov_up=self.fov_up, fov_down=self.fov_down, proj_H=self.proj_H, proj_W=self.proj_W, max_range=self.max_range)
        return range_image

    def gen_ri_and_save_all(self, drive_ids):
        print('*' * 100)
        self.logger.info('Generating Range Images')
        time.sleep(1)
        for drive_id in drive_ids:
            self.logger.info(f'Generating Range Images for the following drives: {drive_id}')
            output_dir = os.path.join(self.base_dir, 'sequences', drive_id, 'range_images')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            fnames = self.fnames[drive_id]

            for fname in tqdm(fnames, desc="* Processing Point Clouds at Drive {}".format(drive_id)):
                ri_np = self.get_ri_np(drive_id, fname)
                file_name = fname.split('.')[0]
                output_fname = os.path.join(output_dir, f'{file_name}.npy')
                np.save(output_fname, ri_np)
                tqdm.write(f'* Saved: {output_dir}')
                
        self.logger.info('Finished generating RI BEV images')
        print('*' * 100)

    def gen_positive_dict_and_save_all(self, drive_ids, output_dir, d_threshs, t_thresh):
        for d_thresh in d_threshs:
            print('*' * 100)
            self.logger.info('Generating positive tuples for d_thresh: {} and t_thresh: {}'.format(d_thresh, t_thresh))
            self.get_positive_dict_matrix(drive_ids, output_dir, d_thresh, t_thresh)
        self.logger.info('Finished generating positive tuples')
        print('*' * 100)

    def get_positive_dict_matrix(self, drive_ids, output_dir, d_thresh, t_thresh):
        positive_dict = {}

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for iter_i, drive_id in enumerate(drive_ids):
            poses = self.poses[drive_id]
            timestamps = self.timestamps[drive_id]

            if drive_id not in positive_dict:
                positive_dict[str(int(drive_id)).zfill(2)] = {}

            translations = np.array([pose[:3, 3] for pose in poses]) # Extract (n, 3) positions
    
            # pdist로 모든 쌍별 유클리드 거리 계산
            p_dists = squareform(pdist(translations, metric='euclidean'))
            
            # 모든 시간 간의 차이를 계산
            time_diffs = np.abs(timestamps[:, np.newaxis] - timestamps[np.newaxis, :])

            for query_id in tqdm(range(len(poses)), desc=f"* [{iter_i}/{len(drive_ids)}]Processing drive {drive_id}"):
                # 거리와 시간 차이 조건에 맞는 인덱스 추출
                valid_indices = np.where((p_dists[query_id] <= d_thresh) & (time_diffs[query_id] >= t_thresh))[0]
                if valid_indices.size > 0:
                    positive_dict[str(int(drive_id)).zfill(2)][query_id] = valid_indices.tolist()
                else:
                    positive_dict[str(int(drive_id)).zfill(2)][query_id] = []

        save_file_name = '{}/positive_sequence_D-{}_T-{}.json'.format(output_dir, d_thresh, t_thresh)
        with open(save_file_name, 'w') as f:
            json.dump(positive_dict, f)
        print('* Saved: ', save_file_name)

        return positive_dict

    def visualize_pointcloud(self, drive_id, velodyne_file):
        """
        Visualizes the point cloud for a specific timestamp using open3d.
        """
        fname = self.get_velodyne_fn(drive_id, velodyne_file)
        current_vertex = self.get_pointcloud_np(fname)[0]
        
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
    base_dir = '/media/vision/SSD1/Datasets/kitti/dataset'

    all_drive_ids = [str(i).zfill(2) for i in range(0, 11)]
    gen_ri_bev_drive_ids = [str(i).zfill(2) for i in range(0, 11)]
    gen_ri_drive_ids = [str(i).zfill(2) for i in range(0, 11)]
    tuple_mining_drive_ids = [str(i).zfill(2) for i in range(0, 11)]

    # gen ri_bev iamge parameter
    range_thresh = [0, 15, 30, 45, 80]  # Example thresholds
    height_thresh = [-3, -1.5, 0, 1.5, 5]    # Example thresholds
    fov_up = 3.0
    fov_down = -25.0
    proj_H = 64
    proj_W = 900
    min_range = min(range_thresh)
    max_range = max(range_thresh)
    min_height = min(height_thresh)
    max_height = max(height_thresh)

    # tuple mining parameter
    nclt_tuples_output_dir = os.path.join(os.path.dirname(__file__), '../../config/kitti_tuples/')
    t_thresh = 0

    preprocessor = Kitti_processor(base_dir, all_drive_ids, range_thresh, height_thresh, fov_up, fov_down, proj_H, proj_W, max_range, min_range, min_height, max_height)
    
    if args.gen_all:
        args.gen_ri_bev = args.gen_ri = args.tuple_mining = True

    # gen ri_bev image
    if args.gen_ri_bev:
        preprocessor.gen_ri_bev_and_save_all(gen_ri_bev_drive_ids)
    
    # gen ri image
    if args.gen_ri:
        preprocessor.gen_ri_and_save_all(gen_ri_drive_ids)

    # tuple mining
    if args.tuple_mining:
        preprocessor.gen_positive_dict_and_save_all(tuple_mining_drive_ids, nclt_tuples_output_dir, [3, 20], t_thresh)
