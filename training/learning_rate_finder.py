import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import torch

from utils.model_utils.lr_finder import LRFinder

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

import torch
from utils.data_loaders.make_dataloader import *
from config.config import get_config
from models.pipeline_factory import get_pipeline
from training import train_utils

def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger = logging.getLogger()
    
    # 모델 생성, 손실 함수, 옵티마이저, Learning Rate Finder 설정
    model = get_pipeline(cfg).to(device)
    n_params = sum([param.nelement() for param in model.parameters()])
    loss_function = train_utils.get_loss_function(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-7) # 초기 학습률은 매우 작게 설정
    lr_finder = LRFinder(model, optimizer, loss_function, device=device, cfg=cfg)

    logger.info('pipeline: ' + cfg.pipeline)
    logger.info('Number of model parameters: {}'.format(n_params))
    logger.info(optimizer)
    logger.info(lr_finder)

    # get data loader
    finder_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=False,
                                    )

    # 학습률을 증가시키며 100번의 iteration 동안 손실 변화 확인
    lr_finder.range_test(finder_loader, end_lr=1e-2, num_iter=1000) # 100

    # 학습률 제안 (손실이 가장 크게 감소한 지점을 기준으로 추정)
    best_loss = lr_finder.best_loss
    logger.info(f"best_loss: {best_loss}")

    # optimal_lr = lr_finder.lr_suggestion()
    # logger.info(f"optimal_lr: {optimal_lr}")

    # 학습률 대 손실 그래프 출력
    lr_finder.plot()

if __name__ == "__main__":
    cfg = get_config()
    main(cfg)