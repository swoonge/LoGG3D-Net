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
import matplotlib.pyplot as plt

cfg = get_config()

visualize = False

def visualize_pointclouds(pointclouds):
    """
    6개의 포인트 클라우드를 받아서 6분할된 하나의 창에 시각화
    pointclouds: list of numpy arrays, each of shape (N, 4) where N is the number of points and 4 corresponds to (x, y, z, intensity)
    """
    fig = plt.figure(figsize=(12, 8))  # 큰 창 크기 설정

    # 각 포인트 클라우드를 반복하면서 subplot에 그리기
    for i, data in enumerate(pointclouds):
        # 좌표와 강도값 추출
        coords = data[:, :3]  # (x, y, z) 좌표
        intensity = data[:, 3]  # 강도

        # Open3D의 PointCloud 객체 생성
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(coords)

        # 강도를 색상으로 변환 (0 ~ 1로 정규화 후 grayscale로 적용)
        colors = np.clip(colors, 0, 1)  # 0 ~ 1 사이로 클리핑
        colors = np.stack([colors, colors, colors], axis=1)  # RGB 동일한 값으로 설정

        # PointCloud 객체에 색상 설정
        point_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

        # Subplot으로 시각화
        ax = fig.add_subplot(2, 3, i + 1)  # 2행 3열 구조에서 i+1번째 위치에 표시
        vis = o3d.visualization.Visualizer()  # Open3D 시각화 객체 생성
        vis.create_window(window_name=f'PointCloud {i+1}', width=400, height=300, visible=False)
        vis.add_geometry(point_cloud)
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

        # 렌더된 이미지 캡처 후 닫기
        img = vis.capture_screen_float_buffer(do_render=True)
        vis.destroy_window()
        ax.imshow(np.asarray(img))
        ax.axis('off')  # 축 제거
        ax.set_title(f'PointCloud {i+1}')

    plt.tight_layout()
    plt.show()

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

        print("----------info----------")
        print("drive: ", info['drive'])
        print("query_id: ", info['query_id'])
        print("pos_pairs num: ", info['pos_pairs'].shape)

        print("----------data----------")
        # LoGG3D 네트워크의 입련은 SparseTensor 타입이고, spvcnn으로 특징을 추출한다. spvcnn의 출려는 torch.Tensor 타입이다.
        # data는 x, y, z, count로 구성된 SparseTensor 타입이다. count는 0, 1, 2, 3, 4, 5이며 포인트 클라우드의 종류를 구분한다.
        _, counts = torch.unique(data.C[:, -1], return_counts=True) # 따라서 unique를 통해 counts를 구분해 두고,
        pcs = torch.split(data.C, list(counts)) # spvcnn의 출력은 각 포인트의 피처인데, 이를 각 포인트 클라우드별로 나누기 위해 counts를 사용한다.
        print("counts: ", counts)
        # 특징 텐서의 모양 출력
        print("Features Shape:", data.F.shape)
        # 좌표 텐서의 모양 출력
        print("Coordinates Shape:", data.C.shape)

        pcs_list = []
        for pc in pcs:
            pcs_list.append(pc.numpy())
        
        visualize_pointclouds(pcs_list)

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