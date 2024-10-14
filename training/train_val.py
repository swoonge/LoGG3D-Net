import os, sys, logging
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.misc_utils import log_config
from utils.data_loaders.make_dataloader import *
from config.train_config_server import get_config
from models.pipeline_factory import get_pipeline
from training import train_utils

cfg = get_config()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True # cuDNN의 성능을 최적화하기 위한 설정. 데이터 크기가 일정할 때 효율적

    model_save_path = os.path.join(os.path.join(os.path.dirname(__file__), 'checkpoints'), cfg.pipeline, f"{datetime.now(tz=None).strftime('%Y-%m-%d_%H-%M-%S')}")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    writer_save_path = os.path.join(model_save_path)
    if not os.path.exists(writer_save_path):
        os.makedirs(writer_save_path)
    writer = SummaryWriter(log_dir=writer_save_path)

    logger = logging.getLogger()
    logging.info('\n' + ' '.join([sys.executable] + sys.argv))
    logging.info('Slurm Job ID: ' + cfg.job_id)
    logging.info('Training pipeline: ' + cfg.pipeline)
    logging.info('SummartWriter Path: ' + writer_save_path)
    log_config(cfg, logging)
    cfg.experiment_name = f"{cfg.experiment_name}/{datetime.now(tz=None).strftime('%y-%m-%d_%H-%M-%S')}_{cfg.job_id}"
    logging.info("Experiment Name: " + cfg.experiment_name)

    # Get model
    model = get_pipeline(cfg).to(device)
    n_params = sum([param.nelement() for param in model.parameters()])
    logging.info('Number of model parameters: {}'.format(n_params))

    # Get train utils
    loss_function = train_utils.get_loss_function(cfg)
    point_loss_function = train_utils.get_point_loss_function(cfg)
    optimizer = train_utils.get_optimizer(cfg, model.parameters())
    scheduler = train_utils.get_scheduler(cfg, optimizer)

    if cfg.resume_training:
        resume_filename = cfg.resume_checkpoint
        logger.info("Resuming Model From ", os.path.join(model_save_path, resume_filename))
        checkpoint = torch.load(os.path.join(model_save_path, resume_filename))
        starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        starting_epoch = 0

    # Get data loader
    train_loader = make_data_loader(cfg,
                                    cfg.train_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=True,
                                    )
    
    val_loader = make_data_loader(cfg,
                                    cfg.val_phase,
                                    cfg.batch_size,
                                    num_workers=cfg.train_num_workers,
                                    shuffle=False,
                                    )
     
    best_val_loss = 1000000

    for epoch in range(starting_epoch, cfg.max_epoch):
        
        lr = optimizer.param_groups[0]['lr']
        logging.info('**** EPOCH %03d ****' % (epoch) + ' LR: %03f' % (lr))
        running_loss = 0.0
        running_scene_loss = 0.0
        running_point_loss = 0.0
        metric_epoch_loss = 0.0
        val_loss = 0.0

        model.train()

        train_loader_progress_bar = tqdm(train_loader, desc="Training", leave=True)

        for i, batch in enumerate(train_loader_progress_bar, 0):
            if i >= len(train_loader):
                break
            if cfg.pipeline == 'LOGG3D':
                batch_st = batch[0].to(device)
                if not batch[1]['pos_pairs'].ndim == 2:
                    continue
                output = model(batch_st)
                scene_loss = loss_function(output[0], cfg)
                running_scene_loss += scene_loss.item()
                if cfg.point_loss_weight > 0:
                    point_loss = point_loss_function(
                        output[1][0], output[1][1], batch[1]['pos_pairs'], cfg)
                    running_point_loss += point_loss.item()
                    loss = cfg.scene_loss_weight * scene_loss + cfg.point_loss_weight * point_loss
                else:
                    loss = scene_loss

            elif cfg.pipeline == 'PointNetVLAD':
                batch_t = batch.to(device)
                output = model(batch_t.unsqueeze(1))
                scene_loss = loss_function(output, cfg)
                running_scene_loss += scene_loss.item()
                loss = scene_loss

            elif cfg.pipeline.split('_')[0] == 'OverlapTransformer':
                if cfg.train_loss_function == 'quadruplet' and not batch.shape[0] == 6:
                    print("Batch size is not 6")
                    continue
                
                current_batch = torch.unsqueeze(batch, dim=1).type(torch.FloatTensor).to(device) # [6,1,64,900]
                output = model(current_batch)

                ## loss
                scene_loss = loss_function(output, cfg)
                running_scene_loss += scene_loss.item()
                metric_epoch_loss += scene_loss.item()
                loss = scene_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i % cfg.loss_log_step) == (cfg.loss_log_step - 1):
                avg_loss = running_loss / cfg.loss_log_step
                avg_scene_loss = running_scene_loss / cfg.loss_log_step
                avg_point_loss = running_point_loss / cfg.loss_log_step

                lr = optimizer.param_groups[0]['lr']
                tqdm.write('[' + str(i) + '/' + str(len(train_loader)) +'] avg running loss: ' + str(avg_loss)[:7] + ' LR: %03f' % (lr) + 
                                ' avg_scene_loss: ' + str(avg_scene_loss)[:7] + ' avg_point_loss: ' + str(avg_point_loss)[:7])
                writer.add_scalar('training loss', avg_loss, epoch * len(train_loader) + i)
                writer.add_scalar('training point loss', avg_point_loss, epoch * len(train_loader) + i)
                writer.add_scalar('training scene loss', avg_scene_loss, epoch * len(train_loader) + i)
                running_loss, running_scene_loss, running_point_loss = 0.0, 0.0, 0.0
        metric_epoch_loss = metric_epoch_loss / len(train_loader)
        scheduler.step(metric_epoch_loss)

        print("Validation")
        logger.info('**** Validation %03d ****' % (epoch))
        model.eval()
        with torch.no_grad():
            val_loader_progress_bar = tqdm(val_loader, desc="Validation", leave=True)
            for i, batch in enumerate(val_loader_progress_bar):
                if i >= len(val_loader_progress_bar):
                    break
                if cfg.pipeline == 'LOGG3D':
                    batch_st = batch[0].to(device)
                    if not batch[1]['pos_pairs'].ndim == 2:
                        continue
                    output = model(batch_st)
                    scene_loss = loss_function(output[0], cfg)
                    running_scene_loss += scene_loss.item()
                    if cfg.point_loss_weight > 0:
                        point_loss = point_loss_function(
                            output[1][0], output[1][1], batch[1]['pos_pairs'], cfg)
                        running_point_loss += point_loss.item()
                        loss = cfg.scene_loss_weight * scene_loss + cfg.point_loss_weight * point_loss
                    else:
                        loss = scene_loss

                elif cfg.pipeline == 'PointNetVLAD':
                    batch_t = batch.to(device)
                    output = model(batch_t.unsqueeze(1))
                    scene_loss = loss_function(output, cfg)
                    running_scene_loss += scene_loss.item()
                    loss = scene_loss

                elif cfg.pipeline.split('_')[0] == 'OverlapTransformer':
                    if cfg.train_loss_function == 'quadruplet' and not batch.shape[0] == 6:
                        print("Batch size is not 6")
                        continue
                    
                    current_batch = torch.unsqueeze(batch, dim=1).type(torch.FloatTensor).to(device) # [6,1,64,900]
                    output = model(current_batch)

                    ## loss
                    scene_loss = loss_function(output, cfg)
                    running_scene_loss += scene_loss.item() 
                    loss = scene_loss  
                        
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = str(model_save_path) + '/' + "epoch_best_" + str(epoch) + ".pth"
        else:
            save_path = str(model_save_path) + '/' + "epoch_" + str(epoch) + ".pth"
        logger.info("Saving to: " + str(save_path))
        if isinstance(model, torch.nn.DataParallel):
            model_to_save = model.module
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_to_save = model.module
        else:
            model_to_save = model
        torch.save({
            'config': cfg,
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, save_path)
        
        lr = optimizer.param_groups[0]['lr']
        logger.info('val loss: ' + str(val_loss)[:7])
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('val loss', val_loss, epoch)

    logging.info("Finished training.")

if __name__ == "__main__":
    main()