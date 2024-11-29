import os, sys, logging, time
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
import numpy as np
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.distance import pdist, squareform
import argparse

from utils import *

import plotly.graph_objects as go

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='* [%(asctime)s] %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])
logging.basicConfig(level=logging.INFO, format="")

parser = argparse.ArgumentParser()
parser.add_argument('--gen_ri_bev', action='store_true', help="Generate RI BEV images")
parser.add_argument('--no-gen_ri_bev', dest='gen_ri_bev', action='store_false', help="Do not generate RI BEV images")
parser.add_argument('--gen_ri', action='store_true', help="Generate Range Images")
parser.add_argument('--no-gen_ri', dest='gen_ri', action='store_false', help="Do not generate Range Images")
parser.add_argument('--tuple_mining', action='store_true', help="Enable tuple mining")
parser.add_argument('--no-tuple_mining', dest='tuple_mining', action='store_false', help="Disable tuple mining")
parser.add_argument('--gen_all', action='store_true', help="Generate all outputs")
parser.set_defaults(gen_ri_bev=False, gen_ri=False, tuple_mining=False, gen_all=False)

class NCLT_processor:
    def __init__(self, base_dir, drive_id, d_thresh, t_thresh):
        self.logger = logging.getLogger()
        self.base_dir = base_dir
        self.drive_id = drive_id
        self.d_thresh = d_thresh
        self.t_thresh = t_thresh

        self.fnames, self.poses, self.timestamps = load_nclt_files_poses_timestamps(self.base_dir, drive_id)

        translations = np.array([pose[:3, 3] for pose in self.poses]) # Extract (n, 3) positions
        p_dists = squareform(pdist(translations, metric='euclidean'))
        t_dists = np.abs(self.timestamps[:, np.newaxis] - self.timestamps[np.newaxis, :])

        self.odom = []
        for idx in range(len(self.poses)):
            if idx % 4 != 0:
                continue
            valid_indices = np.where((p_dists[idx] <= d_thresh) & (t_dists[idx] >= t_thresh))[0]
            valid_indices = valid_indices[valid_indices < idx]
            rivisit = 0
            for idx2 in valid_indices:
                dot_product = np.dot(self.poses[idx][:3, 0], self.poses[idx2][:3, 0])
                if dot_product > 0:
                    rivisit = 1
                else:
                    rivisit = 2
            self.odom.append([self.poses[idx], rivisit, valid_indices.tolist()])

        # print(self.timestamps[0], self.timestamps[499]) # 0, 100, 즉 500회동안 100초가 걸렸으니, 1초에 5회씩 샘플링된 것.
        # 따라서 t_dist가 30이면 30*5=150의 간격을 가진다. NCLT는 데이터 특성상 도로의 스케일은 크지만 이동속도가 느리니 더 큰 간격이 필요.
        # self.visualize_odom()
        self.visualize_odom_plotly()
        # self.visualize_odom_real_time()

    def visualize_odom(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # Set background to white
        
        geometries = []
        for i, (pose, revisit, _) in enumerate(tqdm(self.odom)):
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
            mesh.transform(pose)
            if revisit == 1:
                mesh.paint_uniform_color([0, 0, 1]) # Blue for revisited poses
            elif revisit == 2:
                mesh.paint_uniform_color([0, 1, 0]) # Green for inv-revisited poses
            else:
                mesh.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-revisited poses
            geometries.append(mesh)
        
        for geometry in geometries:
            vis.add_geometry(geometry)
        
        vis.run()
        vis.destroy_window()

    def visualize_odom_plotly(self):
        positions = []
        colors = []
        
        for i, (pose, revisit, _) in enumerate(self.odom):
            position = pose[:3, 3]  # Extract translation (x, y, z)
            positions.append(position)
            if revisit == 1:
                colors.append('blue')  # Blue for revisited poses
            elif revisit == 2:
                colors.append('green')  # Green for inv-revisited poses
            else:
                colors.append('gray')  # Gray for non-revisited poses

        positions = np.array(positions)

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,  # Set colors dynamically
                opacity=0.8
            )
        ))

        # Determine the range for each axis
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()

        # Find the maximum range
        max_range = max(x_range, y_range, z_range)

        fig.update_layout(
            title="Odometry Visualization",
            scene=dict(
                xaxis=dict(title="X", range=[positions[:, 0].min(), positions[:, 0].min() + max_range]),
                yaxis=dict(title="Y", range=[positions[:, 1].min(), positions[:, 1].min() + max_range]),
                zaxis=dict(title="Z", range=[positions[:, 2].min(), positions[:, 2].min() + max_range])
            )
        )

        fig.show()

    def visualize_odom_real_time(self):
        # Initialize positions and colors
        positions_x = []
        positions_y = []
        positions_z = []
        colors = []

        # Create a 3D scatter plot using Plotly FigureWidget
        fig = go.FigureWidget()

        scatter = go.Scatter3d(
            x=[],  # Initialize with empty data
            y=[],
            z=[],
            mode='markers',
            marker=dict(
                size=5,
                color=[],  # Empty color list to start
                opacity=0.8
            )
        )

        fig.add_trace(scatter)
        fig.update_layout(
            title="Odometry Visualization (Real-time)",
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z")
            )
        )

        fig.show()

        # Simulate real-time data addition
        for i, (pose, revisit, _) in enumerate(self.odom):
            if i % 5 != 0:
                continue

            # Extract position and determine color
            position = pose[:3, 3]
            positions_x.append(position[0])
            positions_y.append(position[1])
            positions_z.append(position[2])

            if revisit == 1:
                colors.append('blue')  # Blue for revisited poses
            elif revisit == 2:
                colors.append('green')  # Green for inv-revisited poses
            else:
                colors.append('gray')  # Gray for non-revisited poses

            # Update scatter plot data directly
            scatter.x = positions_x
            scatter.y = positions_y
            scatter.z = positions_z
            scatter.marker.color = colors

            # Pause to simulate real-time (adjust the delay as needed)
            time.sleep(0.01)

# Example usage
if __name__ == '__main__':
    args = parser.parse_args()
    base_dir = '/media/vision/SSD1/Datasets/NCLT'
    drive_id = "2012-02-05"
    d_thresh = 3
    t_thresh = 50

    preprocessor = NCLT_processor(base_dir, drive_id, d_thresh, t_thresh)
    
