from utils import *
import matplotlib.pyplot as plt

class plotter:
    def __init__(self, dataset, dataset_path=None, seq=None, matching_data=None, save_file_name=None) -> None:
        if dataset_path and seq is not None:
            self.data_setting(dataset, dataset_path, seq, matching_data, save_file_name)
        else:
            pass
        
    def data_setting(self, dataset, dataset_path, seq, matching_data, save_file_name):
        if dataset == "kitti":
            self.dataset = "KITTI"
            self.dataset_path = os.path.join(dataset_path, 'sequences', '{:02d}'.format(seq))
        elif dataset == "gm":
            self.dataset = "GM_Cave"
            self.dataset_path = os.path.join(dataset_path, '{:02d}'.format(seq))
        else:
            self.dataset = "Unknown"
        print("****** plot for {} dataset ******".format(save_file_name))

        self.matchings = matching_data
        self.save_file_name = save_file_name
        self.get_poses()

    def get_poses(self):
        # load poses
        poses_file = os.path.join(self.dataset_path, 'poses.txt')
        poses = load_poses(poses_file)

        if self.dataset == "KITTI":
            # load calibrations
            calib_file = os.path.join(self.dataset_path, 'calib.txt')
            T_cam_velo = load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            T_velo_cam = np.linalg.inv(T_cam_velo)

            pose0_inv = np.linalg.inv(poses[0])
            poses_new = []
            for pose in poses:
                poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
            self.poses = np.array(poses_new)
        elif self.dataset == "GM_Cave":
            self.poses = poses

    def plot_total_matching(self, vis=True):
        # 전체 지도 생성 -> x, y 좌표 추출
        x_map, y_map = zip(*[pose[:2, 3] for pose in self.poses])

        # 전체 화면 창 설정
        plt.figure(figsize=(12, 12))

        # 지도 플롯
        plt.plot(x_map, y_map, color='gray', linestyle='-', marker='.', label='Map', alpha=0.5)
        plt.title("Total Matching")
        plt.xlabel('X')
        plt.ylabel('Y')

        # 매칭 플롯
        for match in self.matchings:
            if not match:
                continue
            first_match = match[0]
            x_current, y_current = x_map[int(first_match[0])], y_map[int(first_match[0])]
            x_matching, y_matching = x_map[int(first_match[1])], y_map[int(first_match[1])]

            if first_match[4] == "fp":
                plt.plot([x_current, x_matching], [y_current, y_matching], 'r-', label='False Positive', linewidth=0.5, markevery=[0, -1], marker='x', markersize=3, markeredgewidth=0.2)
            elif first_match[4] == "tp":
                plt.plot([x_current, x_matching], [y_current, y_matching], 'b-', label='True Positive', linewidth=0.5)
            elif first_match[4] == "fn":
                plt.plot(x_current, y_current, 'coral', marker='^', label='False Negative', markersize=1)

        # 범례 설정 (중복 제거)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        # 이미지 저장 (저장 경로와 파일명을 지정해야 합니다)
        plt.savefig(self.save_file_name, dpi=900)
        plt.show() if vis else plt.close()