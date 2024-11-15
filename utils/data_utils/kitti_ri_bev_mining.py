import os
import numpy as np
from gen_ri_bev import *
from tqdm import tqdm

def range_projection(current_vertex, fov_up=10.67, fov_down=-30.67, proj_H=32, proj_W=900, max_range=80, cut_range=True,
                     lower_bound=0.1, upper_bound=6):

    fov_up = fov_up / 180.0 * np.pi
    fov_down = fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)

    if cut_range:
        current_vertex = current_vertex[
        (depth > lower_bound) & (depth < upper_bound)]
        depth = depth[(depth > lower_bound) & (depth < upper_bound)]
    else:
        current_vertex = current_vertex[(depth > 0) & (depth < max_range)]
        depth = depth[(depth > 0) & (depth < max_range)]

    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov

    proj_x *= proj_W
    proj_y *= proj_H

    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                            dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1,
                        dtype=np.int32)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices

    return proj_range, proj_vertex, proj_idx

def bev_projection(current_vertex, proj_H=32, proj_W=900, max_range=80, cut_height=True, lower_bound=10, upper_bound=20):

    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    scan_z_tmp = current_vertex[:, 2]
    if cut_height:
        current_vertex = current_vertex[(depth > 0) & (depth < max_range) & (scan_z_tmp > lower_bound) & (
                scan_z_tmp < upper_bound)]
        depth = depth[(depth > 0) & (depth < max_range) & (scan_z_tmp > lower_bound) & (scan_z_tmp < upper_bound)]
    else:
        current_vertex = current_vertex[(depth > 0) & (depth < max_range)]
        depth = depth[(depth > 0) & (depth < max_range)]

    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]

    if scan_z.shape[0] == 0:
        return np.full((proj_H, proj_W), 0, dtype=np.float32)

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    scan_r = depth * np.cos(pitch)

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = scan_r / max_range

    proj_x = proj_x * proj_W
    proj_y = proj_y * proj_H

    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)

    order = np.argsort(scan_z)
    scan_z = scan_z[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    kitti_lidar_height = 2

    proj_bev = np.full((proj_H, proj_W), 0,
                        dtype=np.float32)

    proj_bev[proj_y, proj_x] = scan_z + abs(kitti_lidar_height)

    return proj_bev

class KittiPreprocessor:
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

    def get_velodyne_fn(self, drive_id, pc_id):
        """
        Returns the file name for the velodyne point cloud data given a drive and point cloud ID.
        """
        return os.path.join(self.base_dir, f'{drive_id:02d}', 'velodyne', f'{pc_id:06d}.bin')

    def get_ri_bev_tensor(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, pc_id)
        current_vertex = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
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

    def process_and_save_all(self, drive_ids):
        """
        Processes all the point clouds and saves them as .npz files.
        """
        for drive_id in drive_ids:
            velodyne_dir = os.path.join(self.base_dir, f'{drive_id:02d}', 'velodyne')
            output_dir = os.path.join(self.base_dir, f'{drive_id:02d}', 'ri_bev')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            pc_ids = [int(os.path.splitext(f)[0]) for f in os.listdir(velodyne_dir) if f.endswith('.bin')]
            pc_ids.sort()
            print('*'*50)
            print(f'* process for seq {drive_id}')
            pc_ids_progress_bar = tqdm(pc_ids, desc="* ", leave=True)
            for pc_id in pc_ids_progress_bar:
                ri_bev_tensor = self.get_ri_bev_tensor(drive_id, pc_id)
                output_fname = os.path.join(output_dir, f'{pc_id:06d}.npz')
                np.savez_compressed(output_fname, ri_bev=ri_bev_tensor)
                tqdm.write(f'* Saved: {output_fname}')
            print('*'*50)


# Example usage
if __name__ == '__main__':
    base_dir = '/media/vision/SSD1/Datasets/kitti/dataset/sequences'
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


    preprocessor = KittiPreprocessor(base_dir, range_thresh, height_thresh, fov_up, fov_down, proj_H, proj_W, max_range, min_range, min_height, max_height)
    drive_ids = list(range(11))  # Sequences 00 to 10

    preprocessor.process_and_save_all(drive_ids)
