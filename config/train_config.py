import argparse
arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# Training
trainer_arg = add_argument_group('Train')
# LOGG3D, OverlapTransformer, OverlapTransformer_resnet, OverlapTransformer_ViT, OverlapTransformer_geo, CVTNet
trainer_arg.add_argument('--pipeline', type=str, default='CVTNet')
trainer_arg.add_argument('--OverlapTransformer_resnet_mode', type=str, default='original') 
trainer_arg.add_argument('--resume_training', type=str2bool, default=False)
trainer_arg.add_argument('--resume_checkpoint', type=str, default='') # 2024-09-03_18-32-21_LOGG3D_Default_0

# Batch setting
trainer_arg.add_argument('--batch_size', type=int, default=1) # Batch size is limited to 1.
trainer_arg.add_argument('--train_num_workers', type=int,
                         default=8)  # per gpu in dist. try 8

# Contrastive
trainer_arg.add_argument('--train_loss_function',
                         type=str, default='quadruplet')  # quadruplet, triplet
trainer_arg.add_argument('--lazy_loss', type=str2bool, default=False)
trainer_arg.add_argument('--ignore_zero_loss', type=str2bool, default=False)
trainer_arg.add_argument('--positives_per_query', type=int, default=2)  # 2
trainer_arg.add_argument('--negatives_per_query', type=int, default=2)  # 2-18
trainer_arg.add_argument('--loss_margin_1', type=float, default=0.5)  # 0.5
trainer_arg.add_argument('--loss_margin_2', type=float, default=0.3)  # 0.3

# Point Contrastive
trainer_arg.add_argument('--point_loss_function', type=str,
                         default='contrastive')  # infonce, contrastive
trainer_arg.add_argument('--point_neg_margin', type=float, default=2.0)  # 1.4
trainer_arg.add_argument('--point_pos_margin', type=float, default=0.1)  # 0.1
trainer_arg.add_argument('--point_neg_weight', type=float, default=1.0)
trainer_arg.add_argument('--point_loss_weight', type=float, default=1.0)  # 0.1
trainer_arg.add_argument('--scene_loss_weight', type=float, default=1.0)  # 0.1

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='adam')  # 'sgd','adam'
opt_arg.add_argument('--max_epoch', type=int, default=200)  # 20
opt_arg.add_argument('--base_learning_rate', type=float, default=1e-5) # 1e-3 ##############
opt_arg.add_argument('--momentum', type=float, default=0.8)  # 0.9
#cosine #multistep(LoGG3D) #step(ot),. step2, ReduceLROnPlateau
opt_arg.add_argument('--scheduler', type=str,
                     default='ReduceLROnPlateau') 

# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiPointSparseTupleDataset(LoGG3D) #MulRanPointSparseTupleDataset # KittiRangeImageTupleDataset(ot) # GMRangeImageTupleDataset # KittiCVTTupleDataset
data_arg.add_argument('--dataset', type=str,
                      default='KittiCVTTupleDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument('--num_points', type=int, default=35000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=True, help="Remove ground plane.")
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

data_arg.add_argument('--kitti_dir', type=str, default='/media/vision/Data0/DataSets/kitti/dataset/',
                      help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_3m_json', type=str,
                      default='positive_sequence_D-3_T-0.json')
data_arg.add_argument('--kitti_20m_json', type=str,
                      default='positive_sequence_D-20_T-0.json')
data_arg.add_argument('--kitti_seq_lens', type=dict, default={
    "0": 4541, "1": 1101, "2": 4661, "3": 801, "4": 271, "5": 2761,
    "6": 1101, "7": 1101, "8": 4071, "9": 1591, "10": 1201})
data_arg.add_argument('--kitti_data_split', type=dict, default={
    'train': [1, 3, 4, 5, 6, 7, 8, 9, 10],
    'val': [2],
    'test': [0]
})

# RI and BEV generation
ri_bev_arg = add_argument_group('RI_BE_VGeneration')
ri_bev_arg.add_argument('--fov_up', type=float, default=25.0, help="Upper bound of vertical fov")
ri_bev_arg.add_argument('--fov_down', type=float, default=3.0, help="Lower bound of vertical fov")
ri_bev_arg.add_argument('--proj_H', type=int, default=64, help="Height of RIVs and BEVs")
ri_bev_arg.add_argument('--proj_W', type=int, default=900, help="Width of RIVs and BEVs")
ri_bev_arg.add_argument('--range_th', type=list, default=[0, 15, 30, 45, 80], help="Range thresholds to generate multi-layer inputs")
ri_bev_arg.add_argument('--height_th', type=list, default=[-3, -1.5, 0, 1.5, 5], help="Height thresholds to generate multi-layer inputs")
ri_bev_arg.add_argument('--train_seqs', type=list, default=["00"], help="Kitti sequences for training")

data_arg.add_argument('--mulran_dir', type=str,
                      default='/media/vision/Data0/DataSets/MulRan/', help="Path to the MulRan dataset")
data_arg.add_argument("--mulran_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--mulran_3m_json', type=str,
                      default='positive_sequence_D-3_T-0.json')
data_arg.add_argument('--mulran_20m_json', type=str,
                      default='positive_sequence_D-20_T-0.json')
data_arg.add_argument('--mulran_seq_lens', type=dict, default={
    "DCC/DCC_01": 5542, "DCC/DCC_02": 7561, "DCC/DCC_03": 7479,
    "KAIST/KAIST_01": 8226, "KAIST/KAIST_02": 8941, "KAIST/KAIST_03": 8629,
    "Sejong/Sejong_01": 28779, "Sejong/Sejong_02": 27494, "Sejong/Sejong_03": 27215,
    "Riverside/Riverside_01": 5537, "Riverside/Riverside_02": 8157, "Riverside/Riverside_03": 10476})
data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
})

data_arg.add_argument('--gm_dir', type=str,
                      default='/media/vision/Data0/DataSets/gm_datasets', help="Path to the gm dataset")
data_arg.add_argument("--gm_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--gm_3m_json', type=str,
                      default='positive_sequence_D-2_T-0.json')
data_arg.add_argument('--gm_20m_json', type=str,
                      default='positive_sequence_D-10_T-0.json')
data_arg.add_argument('--gm_seq_lens', type=dict, default={
    "0": 4150, "1": 6283, "2": 5340,"3": 5410})
data_arg.add_argument('--gm_data_split', type=dict, default={
    'train': [0, 1, 2],
    'val': [3],
    'test': [3]
})

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--train_pickles', type=dict, default={
    'new_dataset': "/path/to/new_dataset/training_both_5_50.pickle",
})
data_arg.add_argument('--gp_vals', type=dict, default={
    'apollo': 1.6, 'kitti':1.5, 'mulran':0.9
})
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=1.0)
data_arg.add_argument('--max_scale', type=float, default=1.0)

# Misc
misc_arg = add_argument_group('Misc')
# misc_arg.add_argument('--experiment_name', type=str, default='LOGG3D_Default_val')  ##############
misc_arg.add_argument('--experiment_name', type=str, default='CVTNet')  ##############
misc_arg.add_argument('--job_id', type=str, default='0')
misc_arg.add_argument('--save_model_after_epoch', type=str2bool, default=True)
misc_arg.add_argument('--eval_model_after_epoch', type=str2bool, default=False)
misc_arg.add_argument('--out_dir', type=str, default='logs')
misc_arg.add_argument('--loss_log_step', type=int, default=1000)
misc_arg.add_argument('--checkpoint_epoch_step', type=int, default=1)


def get_config():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config()
    dconfig = vars(cfg)
    print(dconfig)
