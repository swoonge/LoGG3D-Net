import os, sys, random, json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import numpy as np

from utils.data_loaders.pointcloud_dataset import *
from utils.data_utils.utils import *

from utils.o3d_tools import *

velodatatype = np.dtype({
    'x': ('<u2', 0),
    'y': ('<u2', 2),
    'z': ('<u2', 4),
    'i': ('u1', 6),
    'l': ('u1', 7)})
velodatasize = 8

class NCLTDataset(PointCloudDataset):
    r"""
    Generate single pointcloud frame from NCLT dataset.
    """

    def __init__(self,
                 phase,
                 random_rotation=False,
                 random_occlusion=False,
                 random_scale=False,
                 config=None):

        self.root = config.nclt_dir
        self.gp_rem = config.gp_rem  # Ground Plane 제거 여부 설정
        self.pnv_prep = config.pnv_preprocessing  # PointNet++ 전처리 여부 설정

        PointCloudDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.logger.info("Initializing NCLTDataset")  # 데이터셋 초기화 로그 출력
        if self.gp_rem:
            self.logger.info("Dataloader initialized with Ground Plane removal.")  # Ground Plane 제거 활성화 로그 출력

        self.id_file_dicts = {}
        self.id_timestamps_dict = {}
        self.files = []

        drive_ids = config.nclt_data_split[phase]  # 드라이브 ID 리스트 생성
        for drive_id in drive_ids:
            files, _, timestamps = load_nclt_files_poses_timestamps(self.root, drive_id)  # 드라이브 ID에 해당하는 파일 리스트 로드
            id_file_dict = {}  # 쿼리 ID와 파일 매핑을 저장하는 딕셔너리 초기화
            for query_id, file in enumerate(files):
                self.files.append((drive_id, query_id))  # 드라이브 ID와 쿼리 ID 튜플을 파일 리스트에 추가
                id_file_dict[query_id] = file  # 쿼리 ID와 파일 매핑을 딕셔너리에 추가
            self.id_file_dicts[drive_id] = id_file_dict  # 드라이브 ID와 파일 매핑 딕셔너리를 전체 딕셔너리에 추가
            self.id_timestamps_dict[drive_id] = timestamps

    def get_velodyne_fn(self, drive_id, fname):
        return os.path.join(self.root, 'velodyne_data', f'{drive_id}', 'velodyne_sync', fname)
    
    def data2xyzi(self, data, flip=True):
        xyzil = data.view(velodatatype)
        xyz = np.hstack(
            [xyzil[axis].reshape([-1, 1]) for axis in ['x', 'y', 'z']])
        xyz = xyz * 0.005 - 100.0

        if flip:
            R = np.eye(3)
            R[2, 2] = -1
            xyz = np.matmul(xyz, R)
        return xyz, xyzil['i']

    def get_velo(self, velofile):
        return self.data2xyzi(np.fromfile(velofile))

    def get_pointcloud_np(self, drive_id, pc_id):
        fname = self.get_velodyne_fn(drive_id, self.id_file_dicts[drive_id][pc_id])
        xyz, intensity = self.get_velo(fname)
        xyzr = np.hstack([xyz, intensity.reshape(-1, 1)])

        if self.random_rotation:
            xyzr = self.random_rotate(xyzr)
        if self.random_occlusion:
            xyzr = self.occlude_scan(xyzr)
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            xyzr = scale * xyzr

        return xyzr

    def __getitem__(self, idx):
        drive_id, query_id = self.files[idx]

        xyz_th = self.get_pointcloud_np(drive_id, query_id)
        meta_info = {'sequence': drive_id, 'query_id': query_id}

        return (xyz_th, meta_info)


class NCLTTupleDataset(NCLTDataset):
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
        
        NCLTDataset.__init__(self, phase, random_rotation, random_occlusion, random_scale, config)

        self.positives_per_query = config.positives_per_query
        self.negatives_per_query = config.negatives_per_query
        self.quadruplet = False
        if config.train_loss_function == 'quadruplet':
            self.quadruplet = True

        logging.info("Initializing NCLTTupleDataset")

        tuple_dir = os.path.join(os.path.dirname(__file__), '../../../config/nclt_tuples/')
        self.dict_3m = json.load(open(tuple_dir + config.nclt_3m_json, "r"))
        self.dict_20m = json.load(open(tuple_dir + config.nclt_20m_json, "r"))
        self.nclt_seq_lens = config.nclt_seq_lens

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
        all_ids = set(np.arange(self.nclt_seq_lens[drive_id]))
        neg_set_inv = drive_id_2[str(int(query_id))]
        neg_set = all_ids.difference(neg_set_inv)
        negatives = list(neg_set)
        if query_id in negatives:
            negatives.remove(query_id)
        return negatives
    
    def get_other_negative(self, drive_id, query_id, sel_positive_ids, sel_negative_ids):
        # Dissimillar to all pointclouds in triplet tuple.
        all_ids = range(self.nclt_seq_lens[drive_id])
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
        drive_id, query_id = self.files[idx]
        positive_ids = self.get_positives(drive_id, query_id)
        negative_ids = self.get_negatives(drive_id, query_id)

        if len(positive_ids) < self.positives_per_query:
            positive_ids = (positive_ids * (self.positives_per_query // len(positive_ids) + 1))[:self.positives_per_query]
        if len(negative_ids) < self.negatives_per_query:
            negative_ids = (negative_ids * (self.negatives_per_query // len(negative_ids) + 1))[:self.negatives_per_query]

        selected_positive_ids = random.sample(positive_ids, self.positives_per_query)
        selected_negative_ids = random.sample(negative_ids, self.negatives_per_query)
        positives, negatives = [], []

        query_tensor = self.get_pointcloud_np(drive_id, query_id)
        for pos_id in selected_positive_ids:
            positives.append(self.get_pointcloud_np(drive_id, pos_id))
        for neg_id in selected_negative_ids:
            negatives.append(self.get_pointcloud_np(drive_id, neg_id))

        meta_info = {'drive': drive_id, 'query_id': query_id, 'positive_ids': selected_positive_ids, 'negative_ids': selected_negative_ids}

        if not self.quadruplet:
            return query_tensor, positives, negatives, meta_info
        else:  # For Quadruplet Loss
            other_negative_id = self.get_other_negative(drive_id, query_id, selected_positive_ids, selected_negative_ids)
            other_negative_tensor = self.get_pointcloud_np(drive_id, other_negative_id)
            return query_tensor, positives, negatives, other_negative_tensor, meta_info


#####################################################################################
# TEST
#####################################################################################

from config.config import *

# Config 객체 생성
config = get_config()

# 데이터셋 테스트
def test_nclt_datasets():
    # # NCLTDataset 테스트
    dataset = NCLTDataset(phase='test', config=config)
    print("NCLTDataset 테스트 - 첫 번째 샘플:")
    query_tensor, meta_info = dataset[0]
    print("XYZ tensor:", query_tensor.shape)
    print("Metadata:", meta_info)

    tuple_dataset = NCLTTupleDataset(phase='train', config=config)
    print("NCLTTupleDataset 테스트 - 두 번째 샘플:")
    query_tensor, positives, negatives, other_negative, meta_info = tuple_dataset[0]
    print("Query tensor:", query_tensor.shape)
    print("Positives:", len(positives), positives[0].shape)
    print("Negatives:", len(negatives), negatives[0].shape)
    print("Negatives:", other_negative.shape)
    print("Metadata:", meta_info)

# 이 파일을 직접 실행했을 때만 테스트 함수 실행
if __name__ == "__main__":
    test_nclt_datasets()