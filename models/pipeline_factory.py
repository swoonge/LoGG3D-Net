import os
import sys
sys.path.append(os.path.dirname(__file__))
from pipelines.PointNetVLAD import *
from pipelines.LOGG3D import *
from pipelines.overlap_transformer import *
from pipelines.overlap_net_transformer import *
from pipelines.overlap_transformer_resnet import *
from pipelines.overlap_transformer_ViT import *
from pipelines.overlap_transformer_geo import *
from pipelines.cvtnet import CVTNet
from pipelines.overlap_gat import OverlapGAT, OverlapGATv2, OverlapGATv22, OverlapGATv2_5, OverlapGATv3
from pipelines.overlap_vit import OverlapViT
from pipelines.overlap_gat_net import GATNet

def get_pipeline(cfg):
    if cfg.pipeline == 'LOGG3D':
        pipeline = LOGG3D(feature_dim=16)
    elif cfg.pipeline == 'PointNetVLAD':
        pipeline = PointNetVLAD(global_feat=True, feature_transform=True, max_pool=False, output_dim=256, num_points=4096)
    elif cfg.pipeline == 'OverlapTransformer':
        pipeline = OverlapTransformer(channels=1, use_transformer=True)
    elif cfg.pipeline == 'OverlapNetTransformer':
        if "Kitti" in cfg.dataset:
            print("OverlapNetTransformer for kitti(height=64)")
            pipeline = OverlapNetTransformer(channels=1, height=64, use_transformer=True)
        elif "GM" in cfg.dataset or "NCLT" in cfg.dataset:
            pipeline = OverlapNetTransformer(channels=1, height=32, use_transformer=True)
    elif cfg.pipeline == 'OverlapTransformer_geo':
        pipeline = OverlapTransformer_geo(channels=1, use_transformer=True)
    elif cfg.pipeline == 'CVTNet':
        pipeline = CVTNet(channels=5, use_transformer=True)
    elif cfg.pipeline == 'OverlapViT':
        pipeline = OverlapViT(channels=1)
    elif cfg.pipeline == 'GATNet':
        pipeline = GATNet(config=cfg)
    return pipeline

# if __name__ == '__main__':
#     sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#     from config.eval_config import get_config
#     cfg = get_config()
#     model = get_pipeline(cfg.train_pipeline).cuda()
#     # print(model)

#     from utils.data_loaders.make_dataloader import *
#     train_loader = make_data_loader(cfg,
#                                     cfg.train_phase,
#                                     cfg.batch_size,
#                                     num_workers=cfg.train_num_workers,
#                                     shuffle=True)
#     iterator = train_loader.__iter__()
#     l = len(train_loader.dataset)
#     for i in range(l):
#         input_batch = next(iterator)
#         input_st = input_batch[0].cuda()
#         output = model(input_st)
#         print('')
