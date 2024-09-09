import os
import sys
import torch
import logging
from torchpack import distributed as dist
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.misc_utils import log_config
from evaluation.evaluate import *
from utils.data_loaders.make_dataloader import *
from config.train_config import get_config
from models.pipeline_factory import get_pipeline
from training import train_utils
import open3d as o3d
from torchsparse import PointTensor, SparseTensor

cfg = get_config()

visualize = False

def visualize_pointcoud(data):
    # 좌표와 특징값 추출
    # pcs = data.C.numpy().astype(np.float32)  # 좌표를 float32로 변환
    coords = data[:,:3]  # 좌표
    colors = data[:,3]  # 강도

    # Open3D의 PointCloud 객체 생성
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords)

    # 네 번째 특징값을 색상으로 사용 (정규화하여 0~1 사이 값으로 변환)
    # 네 번째 컬럼을 사용하여 색상 지정
    colors = np.clip(colors, 0, 0.4)  # 0~1 사이로 클리핑
    colors = np.stack([colors, colors, colors], axis=1)  # Grayscale로 색상 생성 (RGB 동일 값)

    # PointCloud 객체에 색상 설정
    point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))  # 색상도 float32로 변환
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 100,origin=[0, 0, 0])

    # Open3D 시각화
    o3d.visualization.draw_geometries([point_cloud, axes],
                                    window_name='SparseTensor Visualization',
                                    width=800, height=600,
                                    left=50, top=50,
                                    point_show_normal=False)

def main():
    # Get data loader
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=False)
    
    for i, batch in enumerate(train_loader, 0):
        data = batch[0] # SparseTensor 타입 
        info = batch[1] # keys -> drive, query_id, pos_pairs
        print(batch.shape)

        # print("----------info----------")
        # print("drive: ", info['drive'])
        # print("query_id: ", info['query_id'])
        # print("pos_pairs num: ", info['pos_pairs'].shape)

        # print("----------data----------")
        # # LoGG3D 네트워크의 입련은 SparseTensor 타입이고, spvcnn으로 특징을 추출한다. spvcnn의 출려는 torch.Tensor 타입이다.
        # # data는 x, y, z, count로 구성된 SparseTensor 타입이다. count는 0, 1, 2, 3, 4, 5이며 포인트 클라우드의 종류를 구분한다.
        # _, counts = torch.unique(data.C[:, -1], return_counts=True) # 따라서 unique를 통해 counts를 구분해 두고,
        # pcs = torch.split(data.C, list(counts)) # spvcnn의 출력은 각 포인트의 피처인데, 이를 각 포인트 클라우드별로 나누기 위해 counts를 사용한다.
        # print("counts: ", counts)
        # # 특징 텐서의 모양 출력
        # print("Features Shape:", data.F.shape)
        # # 좌표 텐서의 모양 출력
        # print("Coordinates Shape:", data.C.shape)

        # for pc in pcs:
        #     visualize_pointcoud(pc.numpy())

        # if visualize == True:
        #     # 좌표와 특징값 추출
        #     pcs = data.C.numpy().astype(np.float32)  # 좌표를 float32로 변환
        #     coords = pcs[:, :3]  # 좌표
        #     intensity = pcs[:, 3]  # 강도

        #     # Open3D의 PointCloud 객체 생성
        #     point_cloud = o3d.geometry.PointCloud()
        #     point_cloud.points = o3d.utility.Vector3dVector(coords)

        #     # 네 번째 특징값을 색상으로 사용 (정규화하여 0~1 사이 값으로 변환)
        #     # 네 번째 컬럼을 사용하여 색상 지정
        #     colors = (intensity - intensity.min()) / (intensity.max() - intensity.min())  # 정규화
        #     colors = np.clip(colors, 0, 1)  # 0~1 사이로 클리핑
        #     colors = np.stack([colors, colors, colors], axis=1)  # Grayscale로 색상 생성 (RGB 동일 값)

        #     # PointCloud 객체에 색상 설정
        #     point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))  # 색상도 float32로 변환

        #     # Open3D 시각화
        #     o3d.visualization.draw_geometries([point_cloud],
        #                                     window_name='SparseTensor Visualization',
        #                                     width=800, height=600,
        #                                     left=50, top=50,
        #                                     point_show_normal=False)

        if i == 0:
            break
        # if cfg.train_pipeline == 'LOGG3D':
        #     batch_st = batch[0].to('cuda:%d' % dist.local_rank())
        #     if not batch[1]['pos_pairs'].ndim == 2:
        #         continue
        #     output = model(batch_st)
        #     scene_loss = loss_function(output[0], cfg)
        #     running_scene_loss += scene_loss.item()
        #     if cfg.point_loss_weight > 0:
        #         point_loss = point_loss_function(
        #             output[1][0], output[1][1], batch[1]['pos_pairs'], cfg)
        #         running_point_loss += point_loss.item()
        #         loss = cfg.scene_loss_weight * scene_loss + cfg.point_loss_weight * point_loss
        #     else:
        #         loss = scene_loss

if __name__ == '__main__':
    main()