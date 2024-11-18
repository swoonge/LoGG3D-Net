import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


# Evaluation
eval_arg = add_argument_group('Eval')
eval_arg.add_argument('--pipeline', type=str, default='CVTNet')
eval_arg.add_argument('--kitti_eval_seq', type=int, default=0)
eval_arg.add_argument('--mulran_eval_seq', type=str,
                      default='Riverside/Riverside_02')
eval_arg.add_argument('--checkpoint_name', type=str,
                      default='/kitti_10cm_loo/2021-09-14_03-43-02_3n24h_Kitti_v10_q29_10s0_262447.pth')
eval_arg.add_argument('--batch_size', type=int, default=1)
eval_arg.add_argument('--train_num_workers', type=int, default=1)
eval_arg.add_argument("--random_rotation", type=str2bool,
                      default=False, help="If random rotation. ")
eval_arg.add_argument("--random_occlusion", type=str2bool,
                      default=False, help="If random occlusion. ")

eval_arg.add_argument("--revisit_criteria", default=3,
                      type=float, help="in meters")
eval_arg.add_argument("--not_revisit_criteria",
                      default=20, type=float, help="in meters")
eval_arg.add_argument("--skip_time", default=30, type=float, help="in seconds")
eval_arg.add_argument("--cd_thresh_min", default=0.001,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--cd_thresh_max", default=1.0,
                      type=float, help="Thresholds on cosine-distance to top-1.")
eval_arg.add_argument("--num_thresholds", default=1000, type=int,
                      help="Number of thresholds. Number of points on PR curve.")


# Dataset specific configurations
data_arg = add_argument_group('Data')
# KittiDataset #MulRanDataset #NCLTRiBevDataset
data_arg.add_argument('--dataset', type=str, default='NCLTRiBevDataset')
data_arg.add_argument('--collation_type', type=str,
                      default='default')  # default#sparcify_list
data_arg.add_argument("--train_save_descriptors", type=str2bool, default=False)
data_arg.add_argument("--train_save_counts", type=str2bool, default=False)
data_arg.add_argument("--train_plot_pr_curve", type=str2bool, default=False)
data_arg.add_argument('--num_points', type=int, default=80000)
data_arg.add_argument('--voxel_size', type=float, default=0.10)
data_arg.add_argument("--gp_rem", type=str2bool,
                      default=False, help="Remove ground plane.")
data_arg.add_argument('--train_feature_distance', type=str,
                      default='cosine')  # cosine#euclidean
data_arg.add_argument("--pnv_preprocessing", type=str2bool,
                      default=False, help="Preprocessing in dataloader for PNV.")

data_arg.add_argument('--kitti_dir', type=str, default='/media/vision/SSD1/Datasets/kitti/dataset/',
                      help="Path to the KITTI odometry dataset")
data_arg.add_argument('--kitti_data_split', type=dict, default={
    'train': [3, 4, 5, 6, 7, 8, 9, 10],
    'val': [2],
    'test': [0]
})

data_arg.add_argument('--mulran_dir', type=str,
                      default='/media/vision/SSD1/Datasets/MulRan/', help="Path to the MulRan dataset")
data_arg.add_argument("--mulran_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--mulran_data_split', type=dict, default={
    'train': ['DCC/DCC_01', 'DCC/DCC_02',
              'Riverside/Riverside_01', 'Riverside/Riverside_03'],
    'val': [],
    'test': ['KAIST/KAIST_01']
})

data_arg.add_argument('--gm_dir', type=str,
                      default='/media/vision/SSD1/Datasets/gm_datasets', help="Path to the gm dataset")
data_arg.add_argument("--gm_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--gm_seq_lens', type=dict, default={
    "0": 4150, "1": 6283, "2": 5340,"3": 5410})
data_arg.add_argument('--gm_data_split', type=dict, default={
    'train': [0, 1, 2],
    'val': [3],
    'test': [3]
})

data_arg.add_argument('--nclt_dir', type=str,
                      default='/media/vision/SSD1/Datasets/NCLT', help="Path to the gm dataset")
data_arg.add_argument("--nclt_normalize_intensity", type=str2bool,
                      default=False, help="Normalize intensity return.")
data_arg.add_argument('--nclt_seq_lens', type=dict, default={
    "2012-01-08": 27903, "2012-01-15": 33463, "2012-01-22": 25998, "2012-02-02": 29213, "2012-02-05":28054})
data_arg.add_argument('--nclt_data_split', type=dict, default={
    'train': ["2012-01-15", "2012-01-22", "2012-02-02"],
    'val': ["2012-01-08"],
    'test': ["2012-02-05"]
})

# Data loader configs
data_arg.add_argument('--train_phase', type=str, default="train")
data_arg.add_argument('--val_phase', type=str, default="val")
data_arg.add_argument('--test_phase', type=str, default="test")
data_arg.add_argument('--use_random_rotation', type=str2bool, default=False)
data_arg.add_argument('--rotation_range', type=float, default=360)
data_arg.add_argument('--use_random_occlusion', type=str2bool, default=False)
data_arg.add_argument('--occlusion_angle', type=float, default=30)
data_arg.add_argument('--use_random_scale', type=str2bool, default=False)
data_arg.add_argument('--min_scale', type=float, default=0.8)
data_arg.add_argument('--max_scale', type=float, default=1.2)


def get_config_eval():
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cfg = get_config_eval()
    dconfig = vars(cfg)
    print(dconfig)