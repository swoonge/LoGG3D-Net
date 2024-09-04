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

cfg = get_config()

def main():
    # Get data loader
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=False)
    
    for i, batch in enumerate(train_loader, 0):
        # print(batch[0])
        # print(batch[1].keys())
        # for k in batch[1].keys():
        #     print(k, batch[1][k])

        if i == 5:
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