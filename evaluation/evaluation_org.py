import os
import sys
import torch
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.misc_utils import log_config
from eval_sequence import *

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")


def evaluate_checkpoint(model, save_path, cfg):
    checkpoint = torch.load(save_path)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model = model.cuda()
    model.eval()

    return evaluate_sequence_reg(model, cfg)


if __name__ == "__main__":
    from models.pipeline_factory import get_pipeline
    from config.eval_config_org import get_config_eval

    cfg = get_config_eval()

    # Get model
    cfg.pipeline = 'LOGG3D'
    model = get_pipeline(cfg)

    # save_path = os.path.join(os.path.dirname(__file__), '../', cfg.checkpoint_name)
    save_path = '/home/vision/Models/LoGG3D-Net/checkpoints/LoGG3D/kitti_10cm_loo/3n24h_Kitti_v10_q29_10s0.pth'
    print('Loading checkpoint from: ', save_path)
    logging.info('\n' + ' '.join([sys.executable] + sys.argv))
    log_config(cfg, logging)

    eval_F1_max = evaluate_checkpoint(model, save_path, cfg)
    logging.info(
        '\n' + '******************* Evaluation Complete *******************')
    logging.info('Checkpoint Name: ' + str(cfg.checkpoint_name))
    if 'Kitti' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.kitti_eval_seq))
    elif 'MulRan' in cfg.eval_dataset:
        logging.info('Evaluated Sequence: ' + str(cfg.mulran_eval_seq))
    logging.info('F1 Max: ' + str(eval_F1_max))