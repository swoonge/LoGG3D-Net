import os, sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import scipy

def load_kitti_poses(data_path, seq):
  """
  Load KITTI poses for a given sequence.
  Args:
    data_path (str): The base directory path where the KITTI dataset is stored.
    seq (int or str): The sequence number to load, formatted as a zero-padded string.
  Returns:
    np.ndarray: An array of poses in the LiDAR coordinate frame.
  Notes:
    - The function reads the ground truth poses from 'poses.txt' and the calibration data from 'calib.txt'.
    - If the calibration data contains 'Tr', the poses are transformed to the LiDAR coordinate frame.
    - If the ground truth poses file is not found, an error message is printed.
  """
  seq = str(seq).zfill(2)
  poses_path = os.path.join(data_path, 'sequences', seq, 'poses.txt')
  calib_path = os.path.join(data_path, 'sequences', seq, 'calib.txt')

  poses = []
  calib = {}
  try:
    with open(poses_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
        T_w_cam0 = T_w_cam0.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        poses.append(T_w_cam0)
    calib = load_kitti_calib(calib_path)

  except FileNotFoundError:
    print('Ground truth poses are not avaialble.')

  poses = np.array(poses)
  if 'Tr' in calib: # calib['Tr'] is the transformation matrix from LiDAR to camera
    Tr = np.linalg.inv(calib['Tr'])
    poses_lidar = [np.dot(Tr, pose) for pose in poses]
    poses = np.array(poses_lidar)
    
  return poses

def load_kitti_calib(filepath):
    """
    Read in a calibration file and parse into a dictionary.

    Args:
      filepath (str): Path to the calibration file.

    Returns:
      dict: A dictionary containing the parsed calibration data. The keys in the dictionary are P1, P2, P3, and Tr. 
          Tr is the lidar to cam transformation matrix.
    """
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                mtx = np.array([float(x) for x in value.split()]).reshape(3, 4)
                data[key] = np.vstack((mtx, [0, 0, 0, 1]))
            except ValueError:
                pass

    return data

def load_nclt_files_poses_timestamps(root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
        ground_truth = pd.read_csv(str(ground_truth_file), sep=",", low_memory=False).values

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
    return velodyne_files, gt, timestamps

def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=80):
  """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points
  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > 0) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_range, proj_vertex, proj_intensity, proj_idx

def multi_layer_range_projection(current_vertex, fov_up=10.67, fov_down=-30.67, proj_H=32, proj_W=900, max_range=80, cut_range=True,
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

def multi_layer_bev_projection(current_vertex, proj_H=32, proj_W=900, max_range=80, cut_height=True, lower_bound=10, upper_bound=20):

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

## 사용처가 없으면 삭제 예정
def load_vertex(scan_path):
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  current_points = current_vertex[:, 0:3]
  current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
  current_vertex[:, :-1] = current_points
  return current_vertex

## 사용처가 없으면 삭제 예정
def load_files(folder):
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths

def test():
  kitti_base_path = '/media/vision/SSD1/Datasets/kitti/dataset'
  poses= load_kitti_poses(kitti_base_path, 0)

  import matplotlib.pyplot as plt

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3], label='Trajectory from load_poses_from_txt')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.set_title('3D Trajectory Comparison')
  ax.legend()

  plt.show()

if __name__ == '__main__':
  test()