import os, sys, random, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np

from utils.data_loaders.pointcloud_dataset import *
from utils.data_utils.utils import *

class KittiDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from KITTI odometry dataset. 
    """
    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = config.kitti_dir  # KITTI 데이터셋의 루트 디렉토리 설정
        self.gp_rem = config.gp_rem  # Ground Plane 제거 여부 설정
        self.pnv_prep = config.pnv_preprocessing  # PointNet++ 전처리 여부 설정

        PointCloudDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)  # 부모 클래스 초기화

        self.logger.info("Initializing KittiDataset")  # 데이터셋 초기화 로그 출력
        if self.gp_rem:
            self.logger.info("Dataloader initialized with Ground Plane removal.")  # Ground Plane 제거 활성화 로그 출력

        self.id_file_dicts = {}  # 드라이브 ID와 파일 매핑을 저장하는 딕셔너리 초기화
        self.files = []

        drive_ids = [str(drive_id).zfill(2) if isinstance(drive_id, int) else drive_id for drive_id in config.kitti_data_split[phase]]  # 드라이브 ID 리스트 생성
        for drive_id in drive_ids:
            files = load_kitti_files(self.root, drive_id, is_sorted=True)  # 드라이브 ID에 해당하는 파일 리스트 로드
            id_file_dict = {}  # 쿼리 ID와 파일 매핑을 저장하는 딕셔너리 초기화
            for query_id, file in enumerate(files):
                self.files.append((drive_id, query_id))  # 드라이브 ID와 쿼리 ID 튜플을 파일 리스트에 추가
                id_file_dict[query_id] = file  # 쿼리 ID와 파일 매핑을 딕셔너리에 추가
            self.id_file_dicts[drive_id] = id_file_dict  # 드라이브 ID와 파일 매핑 딕셔너리를 전체 딕셔너리에 추가

    def get_velodyne_fn(self, drive, file):
        '''
        주어진 드라이브와 파일 이름을 사용하여 Velodyne 파일의 전체 경로를 반환합니다.
        
        매개변수:
        - drive (str): 드라이브 번호 또는 이름. ("00", ...)
        - file (str): 확장자 명을 포함한 파일 이름. (*.bin)
        
        반환값:
        - str: Velodyne 파일의 전체 경로.
        '''
        fname = os.path.join(self.root, 'sequences', drive, 'velodyne', file)
        return fname

    def get_pointcloud_np(self, drive_id, pc_id):
        """
        주어진 drive_id와 pc_id에 해당하는 포인트 클라우드를 numpy 배열로 반환합니다.

        매개변수:
        drive_id (int): 드라이브 ID.
        pc_id (int): 포인트 클라우드 ID.

        반환값:
        numpy.ndarray: 포인트 클라우드 데이터가 포함된 numpy 배열.

        동작:
        - Velodyne 파일에서 포인트 클라우드 데이터를 읽어옵니다.
        - Ground plane 제거가 활성화된 경우, 평면 분할을 통해 지면 포인트를 제거합니다.
        - PNV 전처리가 활성화된 경우, 전처리를 수행합니다.
        - 랜덤 회전이 활성화된 경우, 포인트 클라우드를 랜덤하게 회전시킵니다.
        - 랜덤 가림이 활성화된 경우, 포인트 클라우드를 일부 가립니다.
        - 랜덤 스케일링이 활성화된 경우, 포인트 클라우드를 랜덤하게 스케일링합니다.
        """
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


class KittiTupleDataset(KittiDataset):
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
        
        KittiDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        self.logger.info("Initializing KittiTupleDataset")

        tuple_dir = os.path.join(os.path.dirname(__file__), '../../../config/kitti_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.kitti_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.kitti_20m_json, "r"))
        self.kitti_seq_lens = config.kitti_seq_lens

        self.files = []
        drive_ids = [str(drive_id).zfill(2) if isinstance(drive_id, int) else drive_id for drive_id in config.kitti_data_split[phase]]
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
        all_ids = set(np.arange(self.kitti_seq_lens[str(int(drive_id))]))
        neg_set_inv = drive_id_2[str(int(query_id))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if query_id in negatives:
            negatives.remove(query_id)
        return negatives

    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.kitti_seq_lens[str(int(drive_id))])
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
    # # NCLTDataset 테스트
    kitti_dataset = KittiDataset(phase='test', config=config)
    print("NCLTDataset 테스트 - 첫 번째 샘플:")
    query_tensor, meta_info = kitti_dataset[0]
    print("XYZ tensor:", query_tensor.shape)
    print("Metadata:", meta_info)

    kitti_tuple_dataset = KittiTupleDataset(phase='train', config=config)
    print("\KittiTupleDataset 테스트 - 두 번째 샘플:")
    query_tensor, positives, negatives, other_negative, meta_info = kitti_tuple_dataset[0]
    print("Query tensor:", query_tensor.shape)
    print("Positives:", len(positives), positives[0].shape)
    print("Negatives:", len(negatives), negatives[0].shape)
    print("Negatives:", other_negative.shape)
    print("Metadata:", meta_info)

# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_nclt_datasets()