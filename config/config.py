import argparse, os
arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

# Set default values from environment variables
# export PIPELINE='OverlapTransformer'
# unset PIPELINE
pipeline_default = os.getenv('PIPELINE', 'OverlapGATNet')
                                        # LOGG3D
                                        # OverlapTransformer
                                        # OverlapNetTransformer
                                        # OverlapTransformer_geo
                                        # CVTNet
                                        # OverlapGAT
                                        # OverlapGATv2
                                        # OverlapGATv2_5
                                        # OverlapGATv3
                                        # OverlapViT
                                        # OverlapGATNet
dataset_default = os.getenv('DATASET', 'KittiDepthImageTupleDataset')
                                      # KittiPointSparseTupleDataset(LoGG3D)
                                      # KittirtpCoordinateTupleDataset(LOGG3D_kpfcnn)
                                      # GMPointSparseTupleDataset
                                      # NCLTPointSparseTupleDataset
                                      # KittiDepthImageTupleDataset(ot)
                                      # GMDepthImageTupleDataset
                                      # NCLTDepthImageTupleDataset
experiment_name_default = os.getenv('EXPERIMENT_NAME', 'LOGG3D_gm03')

### Training ###
trainer_arg = add_argument_group('Train')
trainer_arg.add_argument('--pipeline', type=str, default=pipeline_default)
trainer_arg.add_argument('--resume_checkpoint', type=str, default='')

# Batch setting
trainer_arg.add_argument('--batch_size', type=int, default=1) # Batch size is limited to 1.
trainer_arg.add_argument('--train_num_workers', type=int, default=8)  # per gpu in dist. try 8
trainer_arg.add_argument('--server', type=bool, default=False)  # per gpu in dist. try 8

### Loss Function ###
loss_arg = add_argument_group('Loss')
# Contrastive
loss_arg.add_argument('--train_loss_function', type=str, default='quadruplet') # quadruplet, triplet
loss_arg.add_argument('--lazy_loss', type=str2bool, default=True)
loss_arg.add_argument('--ignore_zero_loss', type=str2bool, default=False)
loss_arg.add_argument('--positives_per_query', type=int, default=2) # 2
loss_arg.add_argument('--negatives_per_query', type=int, default=2) # 2-18
loss_arg.add_argument('--loss_margin_1', type=float, default=0.5) # 0.5
loss_arg.add_argument('--loss_margin_2', type=float, default=0.3) # 0.3

# Point Contrastive
loss_arg.add_argument('--point_loss_function', type=str, default='contrastive') # infonce, contrastive
loss_arg.add_argument('--point_neg_margin', type=float, default=2.0) # 1.4
loss_arg.add_argument('--point_pos_margin', type=float, default=0.1) # 0.1
loss_arg.add_argument('--point_neg_weight', type=float, default=1.0)
loss_arg.add_argument('--point_loss_weight', type=float, default=1.0) # 0.1
loss_arg.add_argument('--scene_loss_weight', type=float, default=1.0) # 0.1

### Optimizer arguments ###
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='adam')  # 'sgd','adam'
opt_arg.add_argument('--max_epoch', type=int, default=60)  # 20
opt_arg.add_argument('--base_learning_rate', type=float, default=5e-5) # LoGG3D: 1e-3, OT: 5e-6, GAT: 5e-5
opt_arg.add_argument('--momentum', type=float, default=0.8)  # 0.9
opt_arg.add_argument('--scheduler', type=str, default='step2') 
                                                    #cosine #multistep(LoGG3D) #step(ot),. step2, ReduceLROnPlateau

### Dataset specific configurations ###
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default=dataset_default)
data_arg.add_argument('--collation_type', type=str, default='default')  # default # sparcify_list
data_arg.add_argument('--num_points', type=int, default=35000) # kitti 35000, mulran 50000, nclt 10000
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool, default=False, help="Remove ground plane.")
data_arg.add_argument("--pnv_preprocessing", type=str2bool, default=False, help="Preprocessing in dataloader for PNV.")

# Kitti Dataset
data_arg.add_argument('--kitti_dir', type=str, default='/media/vision/SSD1/Datasets/kitti/dataset', help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_3m_json', type=str, default='positive_sequence_D-3_T-0.json')
data_arg.add_argument('--kitti_20m_json', type=str, default='positive_sequence_D-20_T-0.json')
data_arg.add_argument('--kitti_seq_lens', type=dict, default=
                        {   "0": 4541, "1": 1101, "2": 4661, "3": 801, "4": 271, "5": 2761,
                            "6": 1101, "7": 1101, "8": 4071, "9": 1591, "10": 1201})
data_arg.add_argument('--kitti_data_split', type=dict, default=
                        {'train': [0, 3, 4, 5, 6, 7, 9, 10], 'val': [2], 'test': [8]})

# MulRan Dataset
data_arg.add_argument('--mulran_dir', type=str, default='/media/vision/SSD1/Datasets/MulRan', help="Path to the MulRan dataset")
data_arg.add_argument("--mulran_normalize_intensity", type=str2bool, default=False, help="Normalize intensity return.")
data_arg.add_argument('--mulran_3m_json', type=str, default='positive_sequence_D-3_T-0.json')
data_arg.add_argument('--mulran_20m_json', type=str, default='positive_sequence_D-20_T-0.json')
data_arg.add_argument('--mulran_seq_lens', type=dict, default={
    "DCC/DCC_01": 5542, "DCC/DCC_02": 7561, "DCC/DCC_03": 7479,
    "KAIST/KAIST_01": 8226, "KAIST/KAIST_02": 8941, "KAIST/KAIST_03": 8629,
    "Sejong/Sejong_01": 28779, "Sejong/Sejong_02": 27494, "Sejong/Sejong_03": 27215,
    "Riverside/Riverside_01": 5537, "Riverside/Riverside_02": 8157, "Riverside/Riverside_03": 10476
    })
data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
    })

# GM Dataset
data_arg.add_argument('--gm_dir', type=str, default='/media/vision/SSD1/Datasets/gm_datasets', help="Path to the gm dataset")
data_arg.add_argument("--gm_normalize_intensity", type=str2bool, default=False, help="Normalize intensity return.")
data_arg.add_argument('--gm_3m_json', type=str, default='positive_sequence_D-2_T-0.json')
data_arg.add_argument('--gm_20m_json', type=str, default='positive_sequence_D-10_T-0.json')
data_arg.add_argument('--gm_seq_lens', type=dict, default={"0": 4150, "1": 6283, "2": 5340,"3": 5410})
data_arg.add_argument('--gm_data_split', type=dict, default={'train': [0, 1, 2], 'val': [3], 'test': [3]})

# NCLT Dataset
data_arg.add_argument('--nclt_dir', type=str, default='/media/vision/SSD1/Datasets/NCLT', help="Path to the gm dataset")
data_arg.add_argument("--nclt_normalize_intensity", type=str2bool, default=False, help="Normalize intensity return.")
data_arg.add_argument('--nclt_3m_json', type=str, default='positive_sequence_D-3_T-0.json')
data_arg.add_argument('--nclt_20m_json', type=str, default='positive_sequence_D-20_T-0.json')
data_arg.add_argument('--nclt_seq_lens', type=dict, default={
    "2012-01-08": 27903, "2012-01-15": 33463, "2012-01-22": 25998, "2012-02-02": 29213, "2012-02-05":28054
    })
data_arg.add_argument('--nclt_data_split', type=dict, default={
    'train': ["2012-01-08"],
    'val': ["2012-01-15"],
    'test': ["2012-02-05"]
    })

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--train_pickles', type=dict, default={'new_dataset': "/path/to/new_dataset/training_both_5_50.pickle",}) # for general
data_arg.add_argument('--gp_vals', type=dict, default={'apollo': 1.6, 'kitti':1.5, 'mulran':0.9}) # for general
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)


### Misc ###
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--experiment_name', type=str, default=experiment_name_default)
misc_arg.add_argument('--job_id', type=str, default='0')
misc_arg.add_argument('--loss_log_step', type=int, default=1000)
misc_arg.add_argument('--checkpoint_epoch_step', type=int, default=1)


### Evaluation ###
eval_arg = add_argument_group('Eval')
eval_arg.add_argument("--skip_time", default=30, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.001, type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0, type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int, help="Number of thresholds. Number of points on PR curve.")
eval_arg.add_argument("--target_ch", default=64, type=int, help="ch")


### Parse ###
def get_config():
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = get_config()
    dconfig = vars(cfg)
    print(dconfig)
