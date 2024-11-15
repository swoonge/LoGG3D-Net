# !/usr/bin/python
#
# Example code to read a velodyne_sync/[utime].bin file
# Plots the point cloud using matplotlib. Also converts
# to a CSV if desired.
#
# To call:
#
#   python read_vel_sync.py velodyne.bin [out.csv]
#
import sys, os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import scipy

def timestamps_files_and_gt(self, root_path: str, sequence_id: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
    return timestamps, velodyne_files, gt


