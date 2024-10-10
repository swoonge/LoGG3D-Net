import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

from tqdm import tqdm

import torch
from utils.misc_utils import log_config
from utils.data_loaders.make_dataloader import *
from config.train_config import get_config
from models.pipeline_factory import get_pipeline
import time
torch.backends.cudnn.benchmark = True # cuDNN의 성능을 최적화하기 위한 설정. 데이터 크기가 일정할 때 효율적

cfg = get_config()

class Timer:
    def __init__(self):
        self.start_times = []
        self.total_time = 0.0
        self.num_measurements = 0

    def tic(self):
        # CUDA 동기화 (GPU에서 정확한 시간 측정을 위해)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times.append(time.time())

    def toc(self):
        # CUDA 동기화 (측정 끝에도 정확한 시간 기록을 위해)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if not self.start_times:
            raise ValueError("tic() must be called before toc()")

        # 시간 계산
        start_time = self.start_times.pop()
        elapsed_time = time.time() - start_time
        self.total_time += elapsed_time
        self.num_measurements += 1
        return elapsed_time

    def average_time(self):
        if self.num_measurements == 0:
            return 0.0
        return self.total_time / self.num_measurements

    def reset(self):
        self.start_times = []
        self.total_time = 0.0
        self.num_measurements = 0

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
    
    loader = make_data_loader(cfg,
                                cfg.val_phase,
                                cfg.batch_size,
                                num_workers=cfg.train_num_workers,
                                shuffle=False,
                                )
    loader_progress_bar = tqdm(loader, desc=f'[testing with kitti] 0' + str(cfg.kitti_data_split['val'][0]) + ' seq')
     
    processing_timer = Timer()

    with torch.no_grad():
        for i, batch in enumerate(loader_progress_bar, 0):
            if i >= len(loader):
                break

            data = batch[0].to(device)
            data = data.unsqueeze(0).unsqueeze(0)

            if cfg.pipeline == 'LOGG3D':
                processing_timer.tic()
                output = model(data[0])
            elif cfg.pipeline == 'PointNetVLAD':
                processing_timer.tic()
                output = model(data)
            elif cfg.pipeline.split('_')[0] == 'OverlapTransformer':
                processing_timer.tic()
                output = model(data)
            processing_timer.toc()
   
        
        logger.info('[Model info]')
        logger.info('    Training pipeline: ' + cfg.pipeline)
        logger.info('    Number of model parameters: {}'.format(n_params))
        logger.info(f"    Average processing time: {processing_timer.average_time()*1000} msec")
        logger.info(f"    Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")


if __name__ == "__main__":
    main()