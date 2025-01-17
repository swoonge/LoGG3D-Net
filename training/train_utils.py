import os
import sys
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from loss.global_loss import *
from loss.local_consistency_loss import *

def get_loss_function(cfg):
    if cfg.train_loss_function == 'triplet':
        loss_function = triplet_loss
    elif cfg.train_loss_function == 'quadruplet':
        loss_function = quadruplet_loss
    else:
        raise NotImplementedError(cfg.train_loss_function)
    return loss_function

def get_point_loss_function(cfg):
    if cfg.point_loss_function == 'contrastive':
        point_loss_function = point_contrastive_loss
    elif cfg.point_loss_function == 'infonce':
        point_loss_function = point_infonce_loss
    else:
        raise NotImplementedError(cfg.point_loss_function)
    return point_loss_function   

def get_optimizer(cfg, model_parameters):
    if cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model_parameters, cfg.base_learning_rate, momentum=cfg.momentum)
    elif cfg.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_parameters, cfg.base_learning_rate)  
    else:
        raise NotImplementedError(cfg.optimizer)
    return optimizer 

def get_scheduler(cfg, optimizer):
    # See: https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
    if cfg.scheduler == 'lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1)
    elif cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    elif cfg.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.1)
    elif cfg.scheduler == 'multistep2':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,20,30,40,45,50,55,60,65,70,75,80,85,90,95], gamma=0.8)  
    elif cfg.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # default = 5, sp = 10
    elif cfg.scheduler == 'step2':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8) # default = 5, sp = 10
    elif cfg.scheduler == 'ot_multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 30, 50, 70, 90], gamma=0.2)  # default=1e-4
    elif cfg.scheduler == 'geo_multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 30, 50, 70, 90], gamma=0.2)  # default=1e-4
    else:
        raise NotImplementedError(cfg.scheduler)
    return scheduler
