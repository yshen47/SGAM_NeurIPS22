# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology
import os
import shutil
import torch
from PIL import Image
from sgam.generative_sensing_module.model import VQModel
try:
    import open3d as o3d
except:
    pass
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


class InfiniteSceneGeneration:

    def __init__(self,
                 dynamic_model, data, topk=1, step_size_denom=2, use_rgbd_integration=False, use_discriminator_loss=False,
                 discriminator_loss_weight=0, recon_on_visible=False, offscreen_rendering=True, output_dim=None, seed_index=0, num_src=None):
        self.use_discriminator_loss = use_discriminator_loss
        self.offscreen_rendering = offscreen_rendering
        self.discriminator_loss_weight = discriminator_loss_weight
        self.seed_index = seed_index
        self.topk = topk
        self.recon_on_visible = recon_on_visible
        self.use_rgbd_integration = use_rgbd_integration
        self.step_size_denom = step_size_denom
        self.dynamic_model = dynamic_model
        self.data = data
        name = data + "_seed" + str(seed_index)
        grid_transform_path = Path(f'grid_res/{name}')
        if os.path.exists(grid_transform_path):
            shutil.rmtree(grid_transform_path)

        if data == 'clevr-infinite':
            image_resolution = (256, 256)
            self.output_dim = (20, 20) if output_dim is None else output_dim
            shutil.copytree('templates/clevr-infinite', grid_transform_path)

        elif data == 'google_earth':
            image_resolution = (256, 256)
            self.output_dim = (100, 1) if output_dim is None else output_dim
            os.makedirs(grid_transform_path, exist_ok=True)
            img_fn = sorted(Path(f'templates/google_earth/seed{seed_index}').glob("im*"))[0]
            shutil.copy(img_fn,
                        grid_transform_path / img_fn.name.replace('.png', '_00_00.png'))
            shutil.copy(str(img_fn).replace('im', 'dm').replace('.png', '.npy'),
                        grid_transform_path / img_fn.name.replace('im', 'dm').replace('.png', '_00_00.npy'))
        else:
            raise NotImplementedError
        output_dim = self.output_dim
        self.image_resolution = image_resolution

        if data == 'clevr-infinite':
            self.K = np.array([
                [355.5555, 0, 128],
                [0, 355.5555, 128],
                [0, 0, 1]
            ])
            self.K_inv = np.linalg.inv(self.K)
            trajectory_shape = 'grid'
            self.num_src = (5 if num_src is None else num_src) if isinstance(dynamic_model, VQModel) else 1
            self.curr = 1

            for dm_path in sorted(grid_transform_path.glob('dm*')):
                depth = np.load(dm_path)
                h, w = depth.shape[:2]
                x = np.linspace(0, w - 1, w)
                y = np.linspace(0, h - 1, h)
                xs, ys = np.meshgrid(x, y)
                depth = (depth * self.K[0][0] / np.sqrt(
                    self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2))
                np.save(dm_path, depth)

        elif data == 'google_earth':
            trajectory_shape = 'grid'
            self.K = np.array([
                [497.77774, 0, 256],
                [0, 497.77774, 256],
                [0, 0, 1]
            ])
            self.K[0] = self.K[0] * self.image_resolution[1] / 512
            self.K[1] = self.K[1] * self.image_resolution[0] / 512
            self.num_src = (3 if num_src is None else num_src) if isinstance(dynamic_model, VQModel) else 1
            self.curr = 1
        else:
            raise NotImplementedError

        self.grid_transform_path = grid_transform_path
        self.grid_transform_mask_path = grid_transform_path / "masks"
        os.makedirs(self.grid_transform_mask_path, exist_ok=True)
        self.global_pcd = []
        self.transform_grid = []
        self.anchor_poses = {}
        self.total_inconsistency = 0
        self.trajectory_shape = trajectory_shape
        known_map = self.get_known_map()
        if trajectory_shape == 'grid':
            self.prepare_grid(output_dim, known_map, grid_transform_path)
            self._ordered_grid_coords = self.zig_zag_order()
        elif trajectory_shape == 'spiral':
            self.prepare_spiral(output_dim, known_map, grid_transform_path)
            self._ordered_grid_coords = self.zig_zag_order()
        elif trajectory_shape == 'cylinder':
            self.prepare_ring(output_dim, known_map, grid_transform_path, horizontal_offset=0.002)
            self._ordered_grid_coords = self.zig_zag_order()
        elif trajectory_shape == 'trajectory':
            self._ordered_grid_coords = self.prepare_trajectory(output_dim[0], known_map, grid_transform_path, pose_path=f"{grid_transform_path}/cam0_to_world.txt")
        else:
            raise NotImplementedError
        self.dynamic_model.use_rgbd_integration = self.use_rgbd_integration

        if self.use_rgbd_integration:
            if data == 'clevr-infinite':
                vox_length = 0.05
                self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=vox_length,
                    sdf_trunc=10 * vox_length,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            elif data in ['google_earth']:
                vox_length = 0.01
                self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=vox_length,
                    sdf_trunc=0.03,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            else:
                raise NotImplementedError

            if not self.offscreen_rendering:
                self.vis = o3d.visualization.Visualizer()
                if data == 'clevr-infinite':
                    self.vis.create_window(width=256, height=256, visible=True)
                elif data in ['google_earth']:
                    self.vis.create_window(width=256, height=256, visible=True)
                else:
                    raise NotImplementedError

    def get_known_map(self):
        map = {}
        for file in Path(self.grid_transform_path).glob('dm*'):
            grid_i = int(file.name[3:-4].split('_')[1])
            grid_j = int(file.name[3:-4].split('_')[2])
            orig_frame_idx = int(file.name[3:-4].split('_')[0])
            map[(grid_i, grid_j)] = {
                'rgb_path': str(file).replace('dm', 'im').replace('npy', 'png'),
                'depth_path':str(file),
                'orig_frame_idx': orig_frame_idx
            }
        return map

    def prepare_grid(self, grid_size, known_map, output_folder):
        self.transform_grid = [[]]
        self.anchor_poses = {}
        if self.data == 'google_earth':
            start_transform = np.array([[ 1.,          0.,          0.,         -3.        ],
                               [ 0.,          0.86602527, -0.50000024, -6.        ],
                               [ 0.,          0.50000024,  0.86602527,  2.        ],
                               [ 0.,          0.,          0.,          1.        ]])
            step_unit_i = np.array([0.,         0.11878788, 0.,        ]) / self.step_size_denom
            step_unit_j = np.array([0.12,         0, 0.        ]) / self.step_size_denom
        else:
            start_transform = np.array([[1., 0., 0., -20.],
                                        [0., 0.95533651, -0.29552022, -20.],
                                        [0., 0.29552022, 0.95533651, 0.],
                                        [0., 0., 0., 1.]])
            step_unit_j = np.array([0.81632614, 0, 0., ]) / self.step_size_denom
            step_unit_i = np.array([0, 0.81632614, 0.]) / self.step_size_denom
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                curr_t = start_transform[:3, 3] + step_unit_j * j + step_unit_i * i
                c2w = np.eye(4)
                c2w[:3, :3] = start_transform[:3, :3]
                c2w[:3, 3] = curr_t
                c2w = c2w @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3]
                t = w2c[:3, 3]
                position = -R.T @ t
                curr_transform = {
                    "R": R,
                    "t": t,
                    "K": self.K,
                    "position": position,
                    "rgb_path": known_map[(i, j)]['rgb_path'] if (i, j) in known_map
                        else f"{output_folder}/im_{i*grid_size[1]+j:05d}.png",
                    "depth_path": known_map[(i, j)]['depth_path'] if (i, j) in known_map
                        else f"{output_folder}/dm_{i*grid_size[1]+j:05d}.npy",
                    "R_path": f"{output_folder}/R_{i:05d}.npy",
                    "K_path": f"{output_folder}/K_{i:05d}.npy",
                    "t_path": f"{output_folder}/t_{i:05d}.npy",
                    "visited": (i, j) in known_map,
                    "grid_coord": (i, j),
                }
                if (i, j) in known_map:
                    self.anchor_poses[(i, j)] = curr_transform
                self.transform_grid[-1].append(curr_transform)
            if i != grid_size[0] - 1:
                self.transform_grid.append([])

    def prepare_spiral(self, grid_size, known_map, output_folder):
        self.transform_grid = []
        self.anchor_poses = {}
        frames = []
        if self.data == 'google_earth':
            start_transform = np.array([
                               [ 1.,          0.,          0.,         -3.        ],
                               [ 0.,          0.86602527, -0.50000024, -6.        ],
                               [ 0.,          0.50000024,  0.86602527,  2.        ],
                               [ 0.,          0.,          0.,          1.        ]])
            step_unit_i = np.array([0.,         0.11878788, 0.,        ]) / self.step_size_denom
            step_unit_j = np.array([0.12,         0, 0.        ]) / self.step_size_denom
        else:
            start_transform = np.array([[1., 0., 0., -20.],
                                        [0., 0.95533651, -0.29552022, -20.],
                                        [0., 0.29552022, 0.95533651, 0.],
                                        [0., 0., 0., 1.]])
            step_unit_j = np.array([0.81632614, 0, 0., ]) / self.step_size_denom
            step_unit_i = -np.array([0, 0.81632614, 0.]) / self.step_size_denom

        curr_t = start_transform @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # theta = np.linspace(0, np.pi * 2 * 20, grid_size[1])

        w2c = np.linalg.inv(curr_t)
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        origin = -R.T @ t

        arc = 1
        separation = 1
        r = arc
        b = separation / (2 * np.pi)
        # find the first phi to satisfy distance of `arc` to the second point
        theta = float(r) / b

        for i in range(grid_size[0]):
            # T = np.eye(4)
            # T[:3, 3] = -step_unit_i #+ step_unit_j
            #
            self_rotation = np.eye(4)
            self_rotation[:3, :3] = np.array([
                [np.cos(90-theta), np.sin(90-theta), 0],
                [-np.sin(90-theta), np.cos(90-theta), 0],
                [0, 0, 1]
            ])
            # w2c = np.linalg.inv(curr_t)
            # w2c = T @ self_rotation @ w2c
            c2w = np.eye(4)
            c2w[:3, 3] = origin
            c2w[0, 3] += theta * np.cos(theta) /10
            c2w[1, 3] += theta * np.sin(theta) /10
            c2w[:3, :3] = self_rotation[:3, :3]
            curr_t = c2w
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            theta += float(arc) / r
            r = b * theta
            position = -R.T @ t
            # print(i, R, t, position)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            frame = frame.transform(curr_t)
            frames.append(frame)
            curr_transform = {
                "R": R,
                "t": t,
                "K": self.K,
                "position": position,
                "rgb_path": known_map[(i, 0)]['rgb_path'] if (i, 0) in known_map
                    else f"{output_folder}/im_{i*grid_size[1]:05d}.png",
                "depth_path": known_map[(i, 0)]['depth_path'] if (i, 0) in known_map
                    else f"{output_folder}/dm_{i*grid_size[1]:05d}.npy",
                "R_path": f"{output_folder}/R_{i:05d}.npy",
                "K_path": f"{output_folder}/K_{i:05d}.npy",
                "t_path": f"{output_folder}/t_{i:05d}.npy",
                "visited": (i, 0) in known_map,
                "grid_coord": (i, 0),
            }
            if (i, 0) in known_map:
                self.anchor_poses[(i, 0)] = curr_transform
            self.transform_grid.append([curr_transform])
        o3d.visualization.draw_geometries(frames)

    def prepare_ring(self, grid_size, known_map, output_folder, horizontal_offset=0):
        self.transform_grid = [[]]
        self.anchor_poses = {}
        frames = []
        if self.data == 'google_earth':
            start_transform = np.array([
                [1., 0., 0., -3.],
                [0., 0.86602527, -0.50000024, -6.],
                [0., 0.50000024, 0.86602527, 2.],
                [0., 0., 0., 1.]])
            step_unit_i = np.array([0., 0.11878788, 0., ]) / self.step_size_denom
            step_unit_j = np.array([0.12, 0, 0.]) / self.step_size_denom
        else:
            start_transform = np.array([[1., 0., 0., -20.],
                                        [0., 0.95533651, -0.29552022, -20.],
                                        [0., 0.29552022, 0.95533651, 0.],
                                        [0., 0., 0., 1.]])
            step_unit_j = np.array([0.81632614, 0, 0., ]) / self.step_size_denom
            step_unit_i = -np.array([0, 0.81632614, 0.]) / self.step_size_denom

        curr_t = start_transform @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        theta = np.pi / 80 # np.linspace(np.pi / 30, np.pi / 60, grid_size[1])
        for i in range(grid_size[0]):
            T = np.eye(4)
            T[:3, 3] = -step_unit_i  # + step_unit_j
            T[0, 3] = horizontal_offset
            self_rotation = np.eye(4)

            self_rotation[:3, :3] = np.array([
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)]
            ])

            # self_rotation[:3, :3] = np.array([
            #     [np.cos(theta), 0, np.sin(theta)],
            #     [0, 1, 0],
            #     [-np.sin(theta), 0, np.cos(theta)]
            # ])


            w2c = np.linalg.inv(curr_t)
            w2c = T @ self_rotation @ w2c
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            curr_t = np.linalg.inv(w2c)
            position = -R.T @ t
            # print(i, R, t, position)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            frame = frame.transform(curr_t)
            frames.append(frame)
            curr_transform = {
                "R": R,
                "t": t,
                "K": self.K,
                "position": position,
                "rgb_path": known_map[(i, 0)]['rgb_path'] if (i, 0) in known_map
                else f"{output_folder}/im_{i * grid_size[1]:05d}.png",
                "depth_path": known_map[(i, 0)]['depth_path'] if (i, 0) in known_map
                else f"{output_folder}/dm_{i * grid_size[1]:05d}.npy",
                "R_path": f"{output_folder}/R_{i:05d}.npy",
                "K_path": f"{output_folder}/K_{i:05d}.npy",
                "t_path": f"{output_folder}/t_{i:05d}.npy",
                "visited": (i, 0) in known_map,
                "grid_coord": (i, 0),
            }
            if (i, 0) in known_map:
                self.anchor_poses[(i, 0)] = curr_transform
            self.transform_grid[-1].append(curr_transform)
        o3d.visualization.draw_geometries(frames)

    def load_poses(self, pose_file):
        # load poses of the current camera
        poses = np.loadtxt(pose_file)
        frames = poses[:, 0].astype(np.int)
        poses = np.reshape(poses[:, 1:], (-1, 4, 4))
        res = dict([(k, {"frame_idx": k, "pose": v}) for (k, v) in zip(frames, poses)])
        return res

    def prepare_trajectory(self, trajectory_length, known_map, output_folder, pose_path):
        self.transform_grid = []
        ordered_targets = []
        self.anchor_poses = {}
        # frames = []
        poses = self.load_poses(f'{str(self.grid_transform_path)}/cam0_to_world.txt')

        start_key = sorted(list(known_map.keys()))[0]
        end_key = sorted(list(known_map.keys()))[-1]
        start_known_frame_idx = known_map[start_key]["orig_frame_idx"]
        end_known_frame_idx = known_map[end_key]["orig_frame_idx"]
        orig_known_frame_indices = sorted([v["orig_frame_idx"] for v in known_map.values()])
        rgb_paths = [v["rgb_path"] for v in known_map.values()]
        depth_paths = [v["depth_path"] for v in known_map.values()]
        assert start_known_frame_idx in poses
        pose_frame_idx = sorted(list(poses.keys()))
        pose_pointer = pose_frame_idx.index(start_known_frame_idx)
        assert pose_pointer + trajectory_length < len(pose_frame_idx)
        for i in range(trajectory_length):
            curr_pose = poses[pose_frame_idx[pose_pointer+i]]
            R = np.linalg.inv(curr_pose['pose'])[:3, :3]
            t = np.linalg.inv(curr_pose['pose'])[:3, 3]

            position = -R.T @ t
            # print(i, R, t, position)
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
            # curr_t = np.eye(4)
            # curr_t[:3, :3] = R
            # curr_t[:3, 3] = t
            # frame = frame.transform(np.linalg.inv(curr_t))
            # frames.append(frame)
            curr_transform = {
                "R": R,
                "t": t,
                "K": self.K,
                "position": position,
                "rgb_path": known_map[(i, 0)]['rgb_path'] if (i, 0) in known_map
                    else f"{output_folder}/im_{i:05d}.png",
                "depth_path": known_map[(i, 0)]['depth_path'] if (i, 0) in known_map
                    else f"{output_folder}/dm_{i:05d}.npy",
                "R_path": f"{output_folder}/R_{i:05d}.npy",
                "K_path": f"{output_folder}/K_{i:05d}.npy",
                "t_path": f"{output_folder}/t_{i:05d}.npy",
                "visited": (i, 0) in known_map,
                "grid_coord": (i, 0),
            }

            if (i, 0) in known_map:
                self.anchor_poses[(i, 0)] = curr_transform
            self.transform_grid.append([curr_transform])
            ordered_targets.append((i, 0))
        # o3d.visualization.draw_geometries(frames)
        return ordered_targets

    def get_closest_anchor(self, curr_node):
        min_dis = 99999999
        res = None
        for k in self.anchor_poses:
            curr_dis = np.linalg.norm(self.anchor_poses[k]['position'] - curr_node['position'])
            if curr_dis < min_dis:
                min_dis = curr_dis
                res = self.anchor_poses[k]
        return res

    def scene_expansion(self, return_hs=False):
        for i in tqdm(range(self.output_dim[0] * self.output_dim[1]-1)):
            print('curr: ', i)
            with torch.no_grad():
                tgt_pose_grid_coord = self.next_pose(self.curr)
                self.one_step_prediction(tgt_pose_grid_coord)
                self.curr += 1
        print(f"Successfully unrolling, results saved at {self.grid_transform_path}")
        merged_pcd = self.unproject_to_color_point_cloud()
        merged_pcd_path = str(self.grid_transform_path / "merged_pcds.ply")
        o3d.io.write_point_cloud(merged_pcd_path, merged_pcd)
        print(f"Merged per-view point cloud is saved at {merged_pcd_path}")

        if self.use_rgbd_integration:
            pcd = self.volume.extract_point_cloud()
            rgbd_integrated_path = str(self.grid_transform_path / "rgbd_integrated_mesh.ply")
            o3d.io.write_point_cloud(rgbd_integrated_path, pcd)
            print(f"RGB-D integrated point cloud is saved at {rgbd_integrated_path}")

    def zig_zag_order(self):
        rows = self.output_dim[0]
        columns = self.output_dim[1]


        solution = [[] for i in range(rows + columns - 1)]

        for i in range(rows):
            for j in range(columns):
                sum = i + j
                if (sum % 2 == 0):

                    # add at beginning
                    solution[sum].insert(0, (i, j))
                else:

                    # add at end of the list
                    solution[sum].append((i, j))
        res = []
        for i in solution:
            for j in i:
                res.append(j)
        self.transform_grid[res[0][0]][res[0][1]]['visited'] = True
        return res

    def row_major_order(self):
        rows = self.output_dim[0]
        columns = self.output_dim[1]

        res = []

        for i in range(rows):
            for j in range(columns):
                # add at end of the list
                res.append((i, j if i % 2 == 0 else self.output_dim[1]-j-1))
        self.transform_grid[res[0][0]][res[0][1]]['visited'] = True
        return res

    def column_major_order(self):
        rows = self.output_dim[0]
        columns = self.output_dim[1]

        res = []
        column_indices = list(np.arange(0, columns))

        for j in column_indices:
            for i in range(rows):
                # add at end of the list
                res.append((i if j % 2 == 0 else self.output_dim[0]-i-1, j))
        self.transform_grid[res[0][0]][res[0][1]]['visited'] = True
        return res

    def next_pose(self, curr):
        return self._ordered_grid_coords[curr]

    def get_src_grid_coords(self, tgt_grid_coord):
        # return [self._ordered_grid_coords[self.curr-1]]
        src_grid_coords = []
        tgt_pose = self.transform_grid[tgt_grid_coord[0]][tgt_grid_coord[1]]
        if self.trajectory_shape != 'trajectory':
            for i in range(self.curr):
                candidate_coord = self._ordered_grid_coords[i]
                candidate_pose = self.transform_grid[candidate_coord[0]][candidate_coord[1]]
                if candidate_pose['visited'] and np.linalg.norm(candidate_pose['position']-tgt_pose['position']) <= (0.3 if self.data != 'clevr-infinite' else 1):
                    src_grid_coords.append((candidate_coord, np.linalg.norm(candidate_pose['position']-tgt_pose['position'])))
            src_grid_coords = sorted(src_grid_coords, key=lambda x: x[1])
            src_grid_coords = src_grid_coords[:self.num_src]
            src_grid_coords = [src_grid_coord[0] for src_grid_coord in src_grid_coords]
        else:
            src_grid_coords = [(tgt_grid_coord[0] - i-1, 0) for i in range(self.num_src)]

        # dis_src_grid_coords = sorted(src_grid_coords, key=lambda x: x[2])

        # dis_src_grid_coords = dis_src_grid_coords[:4]
        # closest_anchor = self.get_closest_anchor(tgt_pose)
        # dis_src_grid_coords = [dis_src_grid_coord[0] for dis_src_grid_coord in dis_src_grid_coords]
        # src_grid_coords.append(closest_anchor['grid_coord'])
        # src_grid_coords += [p['grid_coord'] for p in self.anchor_poses.values()]
        # src_grid_coords = list(set(src_grid_coords))
        return src_grid_coords, None

    def prepare_batch_data(self, tgt_node, src_nodes, num_src):
        img_srcs = [np.array(Image.open(src_node['rgb_path']).resize((self.image_resolution[1], self.image_resolution[0]), resample=Image.LANCZOS)) / 127.5 - 1.0 for src_node in src_nodes]
        img_dst = np.zeros_like(img_srcs[0])    # placeholder
        dm_srcs = [F.interpolate(torch.from_numpy(np.load(src_node['depth_path'])[None, None,]), size=self.image_resolution)[0][
            0].numpy().squeeze() for src_node in src_nodes]
        dm_dst = np.zeros_like(dm_srcs[0])      # placeholder
        print([src_node['depth_path'] for src_node in src_nodes], [src_node['rgb_path'] for src_node in src_nodes])

        R_dst = tgt_node["R"]
        t_dst = tgt_node["t"]

        R_rels = []
        t_rels = []
        Ks = []
        K_invs = []
        T_tgt = np.eye(4)
        T_tgt[:3, :3] = R_dst
        T_tgt[:3, 3] = t_dst

        K = self.K
        Rs = [src_node["R"] for src_node in src_nodes]
        ts = [src_node["t"] for src_node in src_nodes]
        T_tgt2srcs = []
        for src_node in src_nodes:
            R_src = src_node["R"]
            t_src = src_node["t"]
            T_src = np.eye(4)
            T_src[:3, :3] = R_src
            T_src[:3, 3] = t_src
            T_rel = T_tgt @ np.linalg.inv(T_src)
            T_tgt2srcs.append(np.linalg.inv(T_rel))
            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]
            R_rels.append(R_rel)
            t_rels.append(t_rel)
            Ks.append(K)
            K_invs.append(np.linalg.inv(K))
        if self.use_rgbd_integration:
            integrated_tgt_depth = self.rgbd_integration(Ks, Rs, ts,
                                                       dm_srcs,
                                                       [src_node['rgb_path'] for src_node in src_nodes],
                                                       K, T_tgt)
            warped_features = self.inverse_warping(torch.tensor(img_srcs).permute(0, 3, 1, 2)[None,].float().cuda(),
                                                   torch.tensor(dm_srcs)[None,].float().cuda(),
                                                   torch.from_numpy(integrated_tgt_depth)[None,].float().cuda(),
                                                   torch.tensor(Ks)[None,].float().cuda(),
                                                   torch.from_numpy(K)[None,].float().cuda(),
                                                   torch.tensor(T_tgt2srcs)[None,].float().cuda())

        for i, dm_src in enumerate(dm_srcs):
            curr_index = int(src_nodes[i]['rgb_path'].split('/')[-1][3:-4])
            if curr_index == 0 and self.data == 'clevr-infinite':
                h, w = dm_src.shape[:2]
                x = np.linspace(0, w - 1, w)
                y = np.linspace(0, h - 1, h)
                xs, ys = np.meshgrid(x, y)
                dm_srcs[i] = (dm_srcs[i] * self.K[0][0] / np.sqrt(
                    self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2))

        batch = {
            "Ks": np.stack(Ks)[None, ],
            "K_invs": np.stack(K_invs)[None, ],
            "R_rels": np.stack(R_rels)[None, ],
            "t_rels": np.stack(t_rels)[None, ],
            "dst_img": img_dst[None, ],
            "src_imgs": np.stack(img_srcs)[None, ],
            "dst_depth": dm_dst[None, ],
            "src_depths": np.stack(dm_srcs)[None, ],
        }
        if self.use_rgbd_integration:
            batch["warped_tgt_features"] = warped_features[None,]
            batch["warped_tgt_depth"] = integrated_tgt_depth[None,]


        for k in batch:
            batch[k] = torch.from_numpy(batch[k].astype(np.float32)).cuda()
        return batch

    def set_id_grid(depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        return pixel_coords

    def pixel2cam(self, depth, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        b, h, w = depth.size()
        pixel_coords = self.set_id_grid(depth)
        current_pixel_coords = pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)

    def cam2pixel(self, cam_coords, proj_c2p_rot, proj_c2p_tr):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2]#.clamp(min=1e-3)

        X_norm = 2 * (X / Z) / (
                    w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
        Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

        pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
        return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, h, w)

    def inverse_warping(self, src_imgs, src_depths, tgt_depth, src_intrinsics, tgt_intrinsic, T_tgt2srcs, padding_mode='zeros',
                        depth_threshold=100):
        """
        Inverse warp a source image to the target image plane.
        Args:
            src_img: the source image (where to sample pixels) -- [B, N, 3, H, W]
            src_depth: depth map of the source image -- [B, N, H, W]
            tgt_depth: depth map of the target image -- [B, H, W]
            src_intrinsics: camera intrinsic matrix -- [B, N, 3, 3]
            tgt_intrinsic: camera intrinsic matrix -- [B, 3, 3]
            T_tgt2srcs: transform from target to source -- [B, N, 4, 4]
        Returns:
            projected_img: Source image warped to the target image plane
            valid_points: Boolean array indicating point validity
        """
        # check_sizes(src_img, 'img', 'B3HW')
        # check_sizes(tgt_depth, 'depth', 'BHW')
        # check_sizes(intrinsics, 'intrinsics', 'B33')

        batch_size, src_num,  _, img_height, img_width = src_imgs.size()

        tgt_depths = tgt_depth.repeat(src_num, 1, 1)
        tgt_intrinsics = tgt_intrinsic.repeat(src_num, 1, 1)
        src_intrinsics = src_intrinsics.view(batch_size*src_num, *src_intrinsics.shape[2:])
        T_tgt2srcs = T_tgt2srcs.view(batch_size*src_num, *T_tgt2srcs.shape[2:])
        src_imgs = src_imgs.view(batch_size*src_num, *src_imgs.shape[2:])
        src_depths = src_depths.view(batch_size*src_num, *src_depths.shape[2:])

        cam_coords = self.pixel2cam(tgt_depths, tgt_intrinsics.inverse())
        # Get projection matrix for tgt camera frame to source pixel frame
        proj_cam_to_src_pixel = src_intrinsics @ T_tgt2srcs[:, :3]

        rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
        src_pixel_coords, warped_src_depths = self.cam2pixel(cam_coords, rot, tr)
        valid_depth_masks_by_src_depth_diff = torch.abs(warped_src_depths - src_depths)

        # cam_coords = self.pixel2cam(src_depths, src_intrinsics.inverse())
        # # Get projection matrix for tgt camera frame to source pixel frame
        # proj_cam_to_tgt_pixel = tgt_intrinsics @ torch.inverse(T_tgt2srcs)[:, :3]
        #
        # rot, tr = proj_cam_to_tgt_pixel[..., :3], proj_cam_to_tgt_pixel[..., -1:]
        # tgt_pixel_coords, warped_tgt_depths = self.cam2pixel(cam_coords, rot, tr)
        # valid_depth_masks_by_tgt_depth_diff = torch.abs(warped_tgt_depths - tgt_depths)


        projected_imgs = F.grid_sample(src_imgs + 2, src_pixel_coords, padding_mode=padding_mode, align_corners=False,
                                       mode='nearest')

        # valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

        projected_imgs = projected_imgs.view(batch_size, src_num, *projected_imgs.shape[1:])
        valid_depth_masks_by_src_depth_diff = valid_depth_masks_by_src_depth_diff.unsqueeze(1)#.repeat(1, projected_imgs.shape[1], 1, 1)
        valid_depth_masks_by_src_depth_diff = valid_depth_masks_by_src_depth_diff.view(batch_size, src_num, *valid_depth_masks_by_src_depth_diff.shape[1:])

        # valid_depth_masks_by_tgt_depth_diff = valid_depth_masks_by_tgt_depth_diff.unsqueeze(1)#.repeat(1, projected_imgs.shape[1], 1, 1)
        # valid_depth_masks_by_tgt_depth_diff = valid_depth_masks_by_tgt_depth_diff.view(batch_size, src_num, *valid_depth_masks_by_tgt_depth_diff.shape[1:])

        warped_result = torch.zeros([batch_size, 3, img_height, img_width]).cuda()
        # warped_result[:] = -1
        z_buffer = torch.zeros([batch_size, 1, img_height, img_width]).cuda()  # the first three columns are for RGB, the final column is for z-buffer
        z_buffer[:] = 99999
        warped_src_depths = warped_src_depths.view(batch_size, src_num, 1, *warped_src_depths.shape[1:])

        for src_i in range(src_num):
            # plt.imshow(((projected_imgs[0, src_i].permute(1, 2, 0)) - 1) / 2)
            # plt.title(f'inverse src_{src_i}')
            # plt.show()
            # plt.imshow(((projected_imgs[0, src_i]).sum(0, keepdims=True) > 0.2)[0])
            # plt.show()
            depth_diff = valid_depth_masks_by_src_depth_diff[:, src_i]
            mask = (depth_diff < z_buffer) * (warped_src_depths[:, src_i] >= 0) \
                   * ((projected_imgs[:, src_i]).sum(1, keepdims=True) > 0)
            # mask = mask | ((warped_result[:,:1] == -1) & (depth_diff < z_buffer))
            z_buffer[:] = mask * depth_diff + ~mask * z_buffer
            mask_reshaped = mask.repeat(1,3,1,1)
            warped_result[:] = (projected_imgs[:, src_i] - 2) * mask_reshaped + ~mask_reshaped * warped_result

        # plt.imshow(((warped_result[0].permute(1, 2, 0))+1)/2)
        # plt.title('inverse warping')
        # plt.show()
        # print()
        return warped_result.cpu().numpy()[0]

    def rgbd_integration(self, src_Ks, src_Rs, src_ts, src_dms, src_ims, tgt_K, tgt_T):

        assert len(src_Rs) == len(src_ts) == len(src_dms) == len(src_ims) == len(src_Ks)


        # this is the volume, choose vox_length = 0.05 for clevr-infinite

        # viz_folder = os.path.join(output_folder, method, scene_name)
        # if not os.path.exists(viz_folder):
        #     os.makedirs(viz_folder)

        scene_pcd = None
        for i in tqdm(range(len(src_Ks))):
            # read data
            depth = src_dms[i].astype(np.float32)
            rgb = cv2.resize(cv2.imread(src_ims[i]), self.image_resolution)
            cv2.imwrite(src_ims[i], rgb)
            rgb = o3d.io.read_image(src_ims[i])

            R = src_Rs[i]
            t = src_ts[i]
            K = src_Ks[i]
            fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth.shape[1], height=depth.shape[0], fx=fx, fy=fy,
                                                          cx=cx, cy=cy)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            d = o3d.geometry.Image(depth)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, d, convert_rgb_to_intensity=False, depth_trunc=20, depth_scale=1)
            self.volume.integrate(rgbd_image, intrinsic, T)

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            pcd.transform(np.linalg.inv(T))
            if scene_pcd is None:
                scene_pcd = pcd
            else:
                scene_pcd += pcd

        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        # o3d.visualization.draw_geometries([mesh])
        fx, fy, cx, cy = tgt_K[0][0], tgt_K[1][1], tgt_K[0][2], tgt_K[1][2]

        if self.offscreen_rendering:
            if self.data == 'clevr-infinite':
                vis = o3d.visualization.rendering.OffscreenRenderer(256, 256, headless=True)
            elif self.data == 'google_earth':
                vis = o3d.visualization.rendering.OffscreenRenderer(256, 256, headless=True)
            else:
                raise NotImplementedError
        else:
            vis = self.vis

        if self.offscreen_rendering:
            mtl = o3d.visualization.rendering.MaterialRecord()
            mtl.shader = "defaultLit"
            vis.scene.add_geometry("mesh", mesh, mtl)
        else:
            vis.add_geometry(mesh)

        camera_param = o3d.camera.PinholeCameraParameters()
        camera_param.intrinsic = o3d.camera.PinholeCameraIntrinsic(width=src_dms[0].shape[1], height=src_dms[0].shape[0], fx=fx, fy=fy,
                                                      cx=cx, cy=cy)
        camera_param.extrinsic = tgt_T

        if self.offscreen_rendering:
            vis.setup_camera(camera_param.intrinsic,
                                  camera_param.extrinsic)
        else:
            control = self.vis.get_view_control()
            control.convert_from_pinhole_camera_parameters(camera_param, allow_arbitrary=True)

        # camera_parameters = control.convert_to_pinhole_camera_parameters()
        # print("{}\n"
        #       "{}".format(camera_parameters.extrinsic, camera_parameters.intrinsic.intrinsic_matrix))

        if self.offscreen_rendering:
            integrated_tgt_depth = vis.render_to_depth_image(z_in_view_space=True)
            integrated_tgt_depth = np.array(integrated_tgt_depth).astype(np.float32)
            integrated_tgt_depth[integrated_tgt_depth == np.Inf] = 0
        else:
            vis.poll_events()
            vis.update_renderer()
            integrated_tgt_depth = vis.capture_depth_float_buffer()
            vis.remove_geometry(mesh)
            integrated_tgt_depth = np.array(integrated_tgt_depth).astype(np.float32)

        # o3d.visualization.draw_geometries([mesh])
        # integrated_tgt_depth[integrated_tgt_depth==0] = 999999

        return integrated_tgt_depth

    def convert_depth_from_nonlinear_to_linear(self, dm_dst):
        h, w = dm_dst.shape[2:]
        x = torch.linspace(0, w - 1, w).to('cuda:0')
        y = torch.linspace(0, h - 1, h).to("cuda:0")
        xs, ys = torch.meshgrid(x, y)
        dm_dst = dm_dst * self.K[0][0] / torch.sqrt(
                        self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2)
        return dm_dst

    def convert_depth_from_linear_to_nonlinear(self, dm_dst):
        h, w = dm_dst.shape[2:]
        x = torch.linspace(0, w - 1, w).to('cuda:0')
        y = torch.linspace(0, h - 1, h).to('cuda:0')
        xs, ys = torch.meshgrid(x, y)
        xs = xs[None, None]
        ys = ys[None, None]
        dm_dst = dm_dst * torch.sqrt(
                    self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2) / self.K[0][0]
        return dm_dst

    def one_step_prediction(self, tgt_pose_grid_coord, save_res_to_disk=True): # return type can be either rgbd or feature
        # tgt_pose_grid_coord = self.next_pose()
        src_pose_grid_coords, _ = self.get_src_grid_coords(tgt_pose_grid_coord)
        print(f"im_{self.curr:05d}.png", 'tgt_pose_grid_coord: ', tgt_pose_grid_coord, 'src_pose_grid_coords: ', src_pose_grid_coords)

        tgt_meta = self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]
        src_metas = [self.transform_grid[src_pose_grid_coord[0]][src_pose_grid_coord[1]] for src_pose_grid_coord in src_pose_grid_coords]

        batch = self.prepare_batch_data(tgt_meta, src_metas, self.num_src)

        batch['src_depths'] = batch['src_depths'][..., None]
        # orig_batch = copy.deepcopy(batch)
        x, x_dst, extrapolation_mask, warped_depth = self.dynamic_model.get_x(batch, self.data,
                                                                              return_extrapolation_mask=True,
                                                                              no_depth_range=True,
                                                                              parallel=True)
        if self.data == 'clevr-infinite':
            x_sample_dets, _, pre_quantized_features, quantized_features = self.dynamic_model(x, topk=self.topk,
                                                                                              extrapolation_mask=extrapolation_mask,
                                                                                              get_pre_quantized_feature=True,
                                                                                              get_quantized_feature=True,
                                                                                              sample_number=1)
            x_sample_dets = x_sample_dets[0] # since sample number is 1
        elif self.data == 'google_earth':
            x_sample_dets, _, pre_quantized_features, quantized_features = self.dynamic_model(x, topk=self.topk,
                                                                                              extrapolation_mask=extrapolation_mask,
                                                                                              get_pre_quantized_feature=True,
                                                                                              get_quantized_feature=True,
                                                                                              sample_number=1)
            x_sample_dets = x_sample_dets[0] # since sample number is 1
        else:
            raise NotImplementedError

        for i, x_sample_det in enumerate(reversed(x_sample_dets)):
            # fake_logits = -self.dynamic_model.loss.discriminator(x_sample_det.contiguous())
            # plt.imshow(fake_logits[0][0].detach().cpu().numpy(), cmap='gray')
            # plt.show()

            rgb = np.clip(((x_sample_det[0][:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                          255)

            rgb = rgb.astype(np.uint8)

            plt.imshow(rgb)
            plt.show()

        if self.data == 'clevr-infinite':
            depth = (1 / ((x_sample_det[0][3] + 1) / 2 * (1 / 7 - 1 / 16) + 1 / 16)).detach().cpu().numpy()
        elif self.data == 'kitti360':
            depth = (1 / ((x_sample_det[0][3] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75)).detach().cpu().numpy()
        elif self.data == 'google_earth':
            depth = (1 / ((x_sample_det.squeeze()[3] + 1) / 2 * (1 / 10.099975586 - 1 / 14.765625) + 1 / 14.765625) - 10).detach().cpu().numpy()

        if save_res_to_disk:
            self.save_to_disk(tgt_pose_grid_coord, rgb, depth)
        return {
            "rgbd": x_sample_dets.squeeze().detach(),
            "feature": quantized_features.squeeze().detach(),
            "pre_quantized_features": pre_quantized_features.squeeze().detach(),
            "fixed": False,
            "x": x.detach(),
            "batch_src_imgs": batch['src_imgs'],
            "batch_src_depths": batch['src_depths'],
            "batch_R_rels": batch['R_rels'],
            "batch_t_rels": batch['t_rels'],
            "warped_depth": warped_depth
        }

    def save_to_disk(self, tgt_pose_grid_coord, rgb, depth):
        index = self.curr  # int(tgt_meta['rgb_path'].split('/')[-1][3:-4])
        tgt_meta = self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]
        # shutil.copy(tgt_meta['rgb_path'], str(self.grid_transform_path / f"im_{index:05d}.png"))
        # plt.imshow(Image.open(tgt_meta['rgb_path']))
        # plt.show()
        # shutil.copy(tgt_meta['depth_path'], str(self.grid_transform_path / f"dm_{index:05d}.npy"))
        suffix = f"_{tgt_pose_grid_coord[0]:02d}_{tgt_pose_grid_coord[1]:02d}"

        np.save(str(self.grid_transform_path / f"R_{index:05d}{suffix}.npy"), tgt_meta['R'])
        np.save(str(self.grid_transform_path / f"t_{index:05d}{suffix}.npy"), tgt_meta['t'])
        # np.save(str(self.grid_transform_path / f"K_{index:05d}.npy"), self.K)

        np.save(str(self.grid_transform_path / f"dm_{index:05d}{suffix}.npy"), depth)
        Image.fromarray(rgb).save(str(self.grid_transform_path / f"im_{index:05d}{suffix}.png"), format='png')
        # Image.fromarray((warped_input * 255).astype(np.uint8)).save(str(self.grid_transform_mask_path / f"im_{index:05d}{suffix}.png"), format='png')

        # shutil.copy(f"dataset/blender_3d_large_postprocessed/diffuse_scene_1647996146/dm_{i:05d}.npy", str(grid_transform_path / f"dm_{i:05d}.npy"))
        # shutil.copy(f"dataset/blender_3d_large_postprocessed/diffuse_scene_1647996146/im_{i:05d}.png", str(grid_transform_path / f"im_{i:05d}.png"))

        # # self.pose_candidates[tgt_pose_key]['rgbd'] = x_sample_det
        self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]['visited'] = True
        self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]["rgb_path"] = str(
            self.grid_transform_path / f"im_{index:05d}{suffix}.png")
        self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]['depth_path'] = str(
            self.grid_transform_path / f"dm_{index:05d}{suffix}.npy")
        self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]['R_path'] = str(
            self.grid_transform_path / f"R_{index:05d}{suffix}.npy")
        self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]['K_path'] = str(
            self.grid_transform_path / f"K_{index:05d}{suffix}.npy")
        self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]['t_path'] = str(
            self.grid_transform_path / f"t_{index:05d}{suffix}.npy")

    def find_extrapolation_region(self, extrapolation_mask):
        visited = np.zeros(extrapolation_mask.shape)
        curr_index = 1
        area_counts = {}
        def check_nearby(curr_coord):
            frontiers = [curr_coord]
            while len(frontiers):
                new_frontier = []
                for curr_coord in frontiers:
                    if 0 <= curr_coord[0] < visited.shape[0] and 0 <= curr_coord[1] < visited.shape[1] and \
                            not visited[curr_coord[0]][curr_coord[1]] and extrapolation_mask[curr_coord[0]][curr_coord[1]]:
                        visited[curr_coord[0]][curr_coord[1]] = curr_index
                        area_counts[curr_index] += 1
                        new_frontier.append((curr_coord[0] + 1, curr_coord[1]))
                        new_frontier.append((curr_coord[0] - 1, curr_coord[1]))
                        new_frontier.append((curr_coord[0], curr_coord[1] + 1))
                        new_frontier.append((curr_coord[0], curr_coord[1] - 1))
                frontiers = new_frontier

        for i in range(len(visited)):
            for j in range(len(visited[0])):
                if not visited[i][j] and extrapolation_mask[i][j] == 1:
                    area_counts[curr_index] = 0
                    check_nearby((i, j))
                    curr_index += 1
        res = np.zeros([256, 256])
        for k, v in area_counts.items():
            if v > 4000:
                res += (visited == k)
        return res != 0

    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1, h, w).type_as(depth)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        return pixel_coords

    def pixel2cam(self, depth, intrinsics_inv):
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        b, h, w = depth.size()
        pixel_coords = self.set_id_grid(depth)
        current_pixel_coords = pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
        cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
        return cam_coords * depth.unsqueeze(1)

    def prepare_pcd(self, depth, color, K, Rt):

        h, w = depth.shape
        x = np.linspace(0, w - 1, w)
        y = np.linspace(0, h - 1, h)
        xs, ys = np.meshgrid(x, y)
        coord_matices = np.stack([xs, ys]).transpose(1, 2, 0)
        coord_matices_homo = np.ones([coord_matices.shape[0], coord_matices.shape[1], 3])
        coord_matices_homo[:, :, :2] = coord_matices
        tgt_2d_in_cam_homo = coord_matices_homo.reshape([w * h, 3]).T
        predicted_depth_map = depth.reshape([h * w, 1]).T
        color = color.reshape([h * w, 3]).T / 255.
        tgt_3d_in_cam = np.linalg.inv(K) @ tgt_2d_in_cam_homo
        tgt_3d_in_cam = np.multiply(predicted_depth_map.repeat(3, 0), tgt_3d_in_cam)
        tgt_3d_in_cam_homo = np.ones([4, tgt_3d_in_cam.shape[1]])
        tgt_3d_in_cam_homo[:3, :] = tgt_3d_in_cam
        tgt_3d_in_world_homo = np.linalg.inv(Rt) @ tgt_3d_in_cam_homo
        tgt_3d_in_world = tgt_3d_in_world_homo[:3]
        selected_points = tgt_3d_in_world
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(selected_points.T)
        pcd.colors = o3d.utility.Vector3dVector(color.T)
        return pcd

    def unproject_to_color_point_cloud(self):
        prediction_path = self.grid_transform_path
        Rs = sorted(prediction_path.glob("R_*_*_*.npy"))

        predicted_pcds = None
        for R_path in tqdm(Rs):
            K = self.K
            R = np.load(str(R_path))
            t_path = str(R_path).replace("R", "t")
            dm_path = str(R_path).replace("R", "dm")
            im_path = str(R_path).replace("R", "im").replace('npy', 'png')
            t = np.load(t_path)
            Rt = np.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t
            predicted_depth = np.load(dm_path)
            predicted_color = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2RGB)

            predicted_pcd = self.prepare_pcd(predicted_depth, predicted_color, K, Rt)
            if predicted_pcds is None:
                predicted_pcds = predicted_pcd
            else:
                predicted_pcds += predicted_pcd

        return predicted_pcds

