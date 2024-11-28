## 네트워크 모델에 대한 성능 측정을 위한 코드
## 1. 모델의 파라미터 수 계산
## 2. 모델의 처리 시간 측정
## 3. 메모리 사용량 측정

import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

from tools.utils.utils import Timer_for_torch as Timer
from models.pipelines.pipeline_utils import *

from tqdm import tqdm

import torch
from utils.misc_utils import log_config
from utils.data_loaders.make_dataloader import *
from config.config import get_config
from models.pipeline_factory import get_pipeline
torch.backends.cudnn.benchmark = True # cuDNN의 성능을 최적화하기 위한 설정. 데이터 크기가 일정할 때 효율적

cfg = get_config()

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger = logging.getLogger()
    log_config(cfg, logger)

    # Get model
    model = get_pipeline(cfg).to(device)
    model.eval()
    n_params = sum([param.nelement() for param in model.parameters()])

    if 'LOGG3D' in cfg.pipeline:
        cfg.dataset = 'KittiDataset'
    if "Overlap" in cfg.pipeline:
        cfg.dataset = 'KittiDepthImageDataset'
    
    loader = make_data_loader(cfg,
                                cfg.test_phase,
                                cfg.batch_size,
                                num_workers=cfg.train_num_workers,
                                shuffle=False,
                                )
    loader_progress_bar = tqdm(loader, desc=f'[testing with kitti ' + str(cfg.kitti_data_split['val'][0]).zfill(2) + '] seq')
     
    processing_timer = Timer()

    with torch.no_grad():
        for i, batch in enumerate(loader_progress_bar, 0):
            if i >= len(loader):
                break

            if cfg.pipeline == 'LOGG3D':
                data = make_sparse_tensor(batch[0][0], cfg.voxel_size).to(device)
                processing_timer.tic()
                output = model(data)
            elif cfg.pipeline == 'PointNetVLAD':
                data = batch[0].to(device)
                processing_timer.tic()
                output = model(data)
            elif "Overlap" in cfg.pipeline or "CVT" in cfg.pipeline:
                data = torch.tensor(batch[0][0])
                data = data.type(torch.FloatTensor).unsqueeze(0).to(device) if 'CVT' in cfg.pipeline else data.unsqueeze(0).unsqueeze(0).to(device)
                processing_timer.tic()
                output = model.forward_inference(data) if 'GAT' in cfg.pipeline else model(data)
            processing_timer.toc()
   
        
        logger.info('[Model info]')
        logger.info('    Training pipeline: ' + cfg.pipeline)
        logger.info('    Number of model parameters: {}'.format(n_params))
        logger.info(f"    Average processing time: {processing_timer.average_time()*1000} msec")
        logger.info(f"    Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()