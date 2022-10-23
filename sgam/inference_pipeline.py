import copy
import os
import random
import shutil
import torch
from PIL import Image
from sgam.generative_sensing_module.model import VQModel
try:
    import open3d as o3d
except:
    pass
from autolab_core import CameraIntrinsics
from tqdm import tqdm
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


class InfiniteSceneGeneration:

    def __init__(self,
                 dynamic_model, model_type, name, data, scene_dirs, index_i, composite=False, topk=1,
                 output_dim=(30, 30), step_size_denom=1, use_rgbd_integration=False,
                 use_test_time_optimization=False, optimization_iteration_num=64, use_discriminator_loss=False,
                 discriminator_loss_weight=0, recon_on_visible=True, offscreen_rendering=True):
        self.use_discriminator_loss = use_discriminator_loss
        self.offscreen_rendering = offscreen_rendering
        self.discriminator_loss_weight = discriminator_loss_weight
        self.use_test_time_optimization = use_test_time_optimization
        self.optimization_iteration_num = optimization_iteration_num
        self.model_type = model_type
        self.topk = topk
        self.recon_on_visible = recon_on_visible
        self.output_dim = output_dim
        self.use_rgbd_integration = use_rgbd_integration
        self.composite = composite
        self.step_size_denom = step_size_denom
        self.dynamic_model = dynamic_model
        self.data = data

        grid_transform_path = Path(f'grid_res/{name}')
        if os.path.exists(grid_transform_path):
            shutil.rmtree(grid_transform_path)

        if data == 'blender':
            shutil.copytree('templates/template_blender', grid_transform_path)
        elif data == 'kitti360':
            shutil.copytree('templates/template_kitti360_debug_sequence3', grid_transform_path)
        elif data == 'google_earth':
            os.makedirs(grid_transform_path, exist_ok=True)
            img_fn = sorted(Path('templates/google_earth_table2_random_sampled_seeds/seeds').glob("im*"))[index_i]
            shutil.copy(img_fn,
                        grid_transform_path / img_fn.name.replace('.png', '_00_00.png'))
            shutil.copy(str(img_fn).replace('im', 'dm').replace('.png', '.npy'),
                        grid_transform_path / img_fn.name.replace('im', 'dm').replace('.png', '_00_00.npy'))
        else:
            raise NotImplementedError
        if scene_dirs is not None:
            shutil.copy(str(scene_dirs[index_i]) + f'/dm_00000.npy', f'{grid_transform_path}/dm_00000_00_00.npy')
            shutil.copy(str(scene_dirs[index_i]) + f'/im_00000.png', f'{grid_transform_path}/im_00000_00_00.png')
        if data == 'blender':
            self.K = np.array([
                [355.5555, 0, 128],
                [0, 355.5555, 128],
                [0, 0, 1]
            ])
            self.K_inv = np.linalg.inv(self.K)
            trajectory_shape = 'grid'
            self.num_src = 5 if isinstance(dynamic_model, VQModel) else 1
            self.curr = 1
        elif data == 'google_earth':
            trajectory_shape = 'grid'
            if os.path.exists("/media/yuan/T7_red/GoogleEarthDataset/K.npy"):
                self.K = np.load("/media/yuan/T7_red/GoogleEarthDataset/K.npy")
            elif os.path.exists("/shared/rsaas/yshen47/GoogleEarthDataset/K.npy"):
                self.K = np.load("/shared/rsaas/yshen47/GoogleEarthDataset/K.npy")
            else:
                self.K = np.array([
                    [497.77774, 0, 256],
                    [0, 497.77774, 256],
                    [0, 0, 1]
                ])
            self.num_src = 5 if isinstance(dynamic_model, VQModel) else 1
            self.curr = 1
        elif data == 'kitti360':
            trajectory_shape = 'trajectory'
            cam_intr = CameraIntrinsics("0",
                                        fx=552.554261,
                                        fy=552.554261,
                                        cx=682.049453,
                                        cy=238.769549,
                                        height=376,
                                        width=1408)
            crop_cj = round(1408 / 2)
            crop_ci = round(376 / 2)
            self.K = cam_intr.crop(height=256, width=1024, crop_ci=crop_ci, crop_cj=crop_cj).resize(0.25)._K
            self.num_src = 1 if isinstance(dynamic_model, VQModel) else 1
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
        # # reverse order of transform grid
        # self.transform_grid = self.transform_grid[::-1]
        self.dynamic_model.use_rgbd_integration = self.use_rgbd_integration

        if self.use_rgbd_integration:
            if data == 'blender':
                vox_length = 0.05
                self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=vox_length,
                    sdf_trunc=100 * vox_length,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            elif data in ['google_earth', 'kitti360']:
                vox_length = 0.01
                self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=vox_length,
                    sdf_trunc=1,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            else:
                raise NotImplementedError

            if not self.offscreen_rendering:
                self.vis = o3d.visualization.Visualizer()
                if data == 'blender':
                    self.vis.create_window(width=256, height=64, visible=True)
                elif data in ['google_earth', 'kitti360']:
                    self.vis.create_window(width=512, height=512, visible=True)
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
                    "consistency_score": 99999,
                    "grid_coord": (i, j),
                }
                if (i, j) in known_map:
                    self.anchor_poses[(i, j)] = curr_transform
                self.transform_grid[-1].append(curr_transform)
            if i != grid_size[0] - 1:
                self.transform_grid.append([])

    def prepare_spiral(self, grid_size, known_map, output_folder):
        self.transform_grid = [[]]
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
                "consistency_score": 99999,
                "grid_coord": (i, 0),
            }
            if (i, 0) in known_map:
                self.anchor_poses[(i, 0)] = curr_transform
            self.transform_grid[-1].append(curr_transform)
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
                "consistency_score": 99999,
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
                "consistency_score": 99999,
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

    def expand_to_inf(self, return_hs=False):
        for i in tqdm(range(self.output_dim[0] * self.output_dim[1]-1)):
            print('curr: ', i)
            with torch.no_grad():
                tgt_pose_grid_coord = self.next_pose(self.curr)
                if isinstance(self.dynamic_model, VQModel):
                    self.one_step_prediction(tgt_pose_grid_coord)
                else:
                    self.one_step_prediction_infinite_nature(tgt_pose_grid_coord)

                # self.unproject_to_color_point_cloud()
                self.curr += 1
        print(f"Successfully unrolling, results saved at {self.grid_transform_path}")

    def init_latent_features(self):
        variables = []  # latent features to optimize in batch
        num_init_poses = len(list(self.grid_transform_path.glob("im_*_*_*.png")))
        for i in range(num_init_poses):
            batch = self.prepare_batch_data(self.transform_grid[i][0], [self.transform_grid[i][0]], 1)

            batch['src_depths'] = batch['src_depths'][..., None]
            x, x_dst, extrapolation_mask, warped_depth = self.dynamic_model.get_x(batch, self.data, return_extrapolation_mask=True, parallel=True)
            if self.data == 'blender':
                x_sample_dets = self.dynamic_model(x_dst, self.topk, extrapolation_mask=extrapolation_mask,
                                                   sample_number=1)
            elif self.data in ['kitti360', 'google_earth']:
                x_sample_dets, _, pre_quantized_features, quantized_features = self.dynamic_model(x,
                                                                                                  extrapolation_mask=extrapolation_mask,
                                                                                                  topk=self.topk,
                                                                                                  sample_number=1,
                                                                                                  get_pre_quantized_feature=True,
                                                                                                  get_quantized_feature=True)

                rgb = np.clip(((x_sample_dets[0][0][0][:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                              255)

                rgb = rgb.astype(np.uint8)

                plt.imshow(rgb)
                plt.show()
            else:
                raise NotImplementedError
            variables.append({
                "grid_coord": (i, 0),
                "rgbd": x_sample_dets[0].squeeze().detach(),
                "fixed": True,
                "feature": quantized_features.squeeze().detach(),
                "pre_quantized_features": pre_quantized_features.squeeze().detach(),
                "x": x.detach(),
                "batch_src_imgs": batch['src_imgs'],
                "batch_src_depths": batch['src_depths'],
                "batch_R_rels": batch['R_rels'],
                "batch_t_rels": batch['t_rels'],
                "warped_depth": warped_depth,
            })

        # initialization pass
        for i in tqdm(range(self.num_src, self.output_dim[0] * self.output_dim[1])):
            with torch.no_grad():
                tgt_pose_grid_coord = self.next_pose(i)
                res = self.one_step_prediction(tgt_pose_grid_coord,
                                               use_test_time_optimization=False)
                res["grid_coord"] = (i, 0)
                variables.append(res)

                rgb = np.clip(((res['rgbd'][:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                              255).astype(np.uint8)

                src_depth = (1 / ((res['rgbd'][3] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75))
                np.save(str(self.grid_transform_path / f"dm_init_{i:05d}.npy"), src_depth.detach().cpu().numpy())
                Image.fromarray(rgb).save(str(self.grid_transform_path / f"im_init_{i:05d}.png"), format='png')
        return variables

    def batch_optimization(self, batch_length=8, sample_size=500):
        variables = self.init_latent_features()
        assert self.data == 'kitti360'
        # variables2 = self.init_latent_features()
        # assert torch.equal(variables[2]['batch_src_imgs'], variables2[2]['batch_src_imgs'])
        # assert torch.equal(variables[2]['batch_src_depths'], variables2[2]['batch_src_depths'])
        # assert torch.equal(variables[2]['batch_R_rels'], variables2[2]['batch_R_rels'])
        # assert torch.equal(variables[2]['batch_t_rels'], variables2[2]['batch_t_rels'])
        # assert torch.equal(variables[2]['warped_depth'], variables2[2]['warped_depth'])
        #
        # assert torch.equal(variables[2]['x'], variables2[2]['x'])
        # variables_copy = copy.deepcopy(variables)
        # sample batch subsequence
        torch.set_grad_enabled(True)
        # self.dynamic_model.train()
        for _ in tqdm(range(sample_size)):
            start_index = random.randint(0, len(variables) - batch_length)
            # start_index = 0 #TODO: remove this, debugging currently
            curr_optimization_features = []
            curr_optimization_feature_indices = []
            for curr_index in range(start_index, start_index + batch_length):
                if not variables[curr_index]['fixed']:
                    variables[curr_index]['feature'].requires_grad = True
                    curr_optimization_features.append(variables[curr_index]['feature'])
                    curr_optimization_feature_indices.append(curr_index)
                else:
                    variables[curr_index]['feature'].requires_grad = False

            features_opt = torch.optim.Adam(curr_optimization_features,
                                            lr=1e-3, betas=(0.5, 0.9))
            # orig_curr_optimization_features = torch.stack(curr_optimization_features).detach().clone()
            # setup MRF
            total_loss = 0
            for curr_target_index in range(start_index, start_index+batch_length):

                curr_target_grid_coord = variables[curr_target_index]['grid_coord']
                tgt_meta = self.transform_grid[curr_target_grid_coord[0]][curr_target_grid_coord[1]]
                tgt_meta['feature'] = variables[curr_target_index]['feature']
                tgt_meta['fixed'] = variables[curr_target_index]['fixed']

                src_metas = []
                for curr_source_index in range(start_index, start_index+batch_length):
                    if curr_source_index != curr_target_index:
                        curr_src_grid_coord = variables[curr_source_index]['grid_coord']
                        curr_src_meta = self.transform_grid[curr_src_grid_coord[0]][curr_src_grid_coord[1]]
                        curr_src_meta['feature'] = variables[curr_source_index]['feature']
                        curr_src_meta['fixed'] = variables[curr_source_index]['fixed']
                        src_metas.append(curr_src_meta)
                src_metas = random.sample(src_metas, self.num_src)

                batch = self.prepare_batch_from_feature_data(tgt_meta, src_metas, self.num_src)

                # decode source rgb-ds from features
                src_depths = []
                src_images = []
                for src_feature in batch['src_features']:
                    quant, emb_loss, info = self.dynamic_model.quantize(src_feature)
                    src_rgbd = self.dynamic_model.decode(quant)
                    src_rgb = src_rgbd[:, :3]
                    src_depth = (1 / ((src_rgbd[:, 3] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75))
                    src_depths.append(src_depth[..., None])
                    src_images.append(src_rgb)

                # for i, x_sample_det in enumerate(reversed(src_images)):
                #     # fake_logits = -self.dynamic_model.loss.discriminator(x_sample_det.contiguous())
                #     # plt.imshow(fake_logits[0][0].detach().cpu().numpy(), cmap='gray')
                #     # plt.show()
                #
                #     rgb = np.clip(((x_sample_det[0][:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                #                   255)
                #
                #     rgb = rgb.astype(np.uint8)
                #
                #     plt.imshow(rgb)
                #     plt.title(f'src_rgb_{i}')
                #     plt.show()
                #
                #     plt.imshow(src_depths[i].squeeze().detach().cpu().numpy())
                #     plt.title(f'src_depth_{i}')
                #     plt.show()

                quant = self.dynamic_model.quantize(batch['dst_feature'])[0]
                tgt_rgbd = self.dynamic_model.decode(quant)
                batch['dst_img'] = torch.clip(tgt_rgbd[:, :3].permute(0, 2, 3, 1), -1, 1)
                batch['dst_depth'] = (1 / ((tgt_rgbd[:, 3] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75))
                batch['src_depths'] = torch.stack(src_depths).permute(1, 0, 2, 3, 4)
                batch['src_imgs'] = torch.clip(torch.stack(src_images).permute(1, 0, 3, 4, 2), -1, 1)

                # one-step prediction
                x, x_dst, extrapolation_mask, warped_depth = self.dynamic_model.get_x(batch, return_extrapolation_mask=True,
                                                                        no_depth_range=True, parallel=True)
                x_pred = self.dynamic_model(x,
                                              extrapolation_mask,
                                              get_pre_quantized_feature=False,
                                              get_quantized_feature=False)[0]


                if curr_target_index == start_index + batch_length // 2:
                    # rgb = np.clip(((x[0][:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                    #               255)
                    #
                    # rgb = rgb.astype(np.uint8)
                    # plt.imshow(rgb)
                    # plt.title(f'warped_rgb')
                    # plt.show()
                    # rgb = np.clip(((x_pred[0][:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                    #               255)
                    #
                    # rgb = rgb.astype(np.uint8)
                    #
                    # plt.imshow(rgb)
                    # plt.title(f'pred_rgb')
                    # plt.show()
                    variables[curr_target_index]['rgbd'] = x_pred.detach().cpu()[0]
                aeloss, log_dict_ae = self.dynamic_model.loss(emb_loss, x, x_pred, 0,
                                                              self.dynamic_model.global_step,
                                                              last_layer=self.dynamic_model.get_last_layer(),
                                                              split="test_optimization",
                                                              extrapolation_mask=extrapolation_mask,
                                                              recon_on_visible=self.recon_on_visible,
                                                              is_inference=True)
                variables[curr_target_index]['test_optimization/g_loss'] = log_dict_ae['test_optimization/g_loss'].detach()

                # x_rec_features = \
                #     self.dynamic_model.encode(x_pred, encoding_indices=None,
                #                               extrapolation_mask=extrapolation_mask)[-1]
                # feature_loss = F.l1_loss(x_rec_features, pre_quantized_features)
                total_loss = total_loss + aeloss    # + feature_loss
                if self.use_discriminator_loss:
                    total_loss = total_loss + log_dict_ae['test_optimization/g_loss'] * self.discriminator_loss_weight

            # optimize
            features_opt.zero_grad()
            total_loss.backward()
            features_opt.step()
            print("total loss: ", total_loss.item())
            # print("aeloss: ", aeloss.item())
            # print("gloss: ", log_dict_ae['test_optimization/g_loss'].item())
            # sanity check on gradient descent

        torch.set_grad_enabled(False)
        # self.dynamic_model.eval()

        # dump decoded optimized image
        sum_of_g_loss = 0
        for i in tqdm(range(self.num_src, self.output_dim[0] * self.output_dim[1])):
            rgbd = variables[i]['rgbd']
            rgb = np.clip(((rgbd[:3] + 1) / 2 * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                          255).astype(np.uint8)

            src_depth = (1 / ((rgbd[3] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75))
            np.save(str(self.grid_transform_path / f"dm_optimized_{i:05d}.npy"), src_depth.detach().cpu().numpy())
            Image.fromarray(rgb).save(str(self.grid_transform_path / f"im_optimized_{i:05d}.png"), format='png')
            sum_of_g_loss += variables[i]['test_optimization/g_loss']
        print("g_loss avg: ", sum_of_g_loss/(self.output_dim[0] * self.output_dim[1]-self.num_src))
    def expand_in_bfs(self):
        visited = np.zeros(self.output_dim)
        target_grid_coord = np.array([0, 0])
        frontier = {}
        while visited.sum() != 900:
            for opt in np.array([[0, 1], [0, -1], [1, 0], [-1, 0]]):
                curr_coord = target_grid_coord + opt
                if 0 <= curr_coord[0] < self.output_dim[0] - 1 and 0 <= curr_coord[1] < self.output_dim[1] - 1 and not visited[curr_coord[0]][curr_coord[1]]:
                    consistency_mse, rgb, depth = self.one_step_prediction(curr_coord, save_res_to_disk=False)
                    frontier[(curr_coord[0], curr_coord[1])] = (consistency_mse, rgb, depth)
            min_consistency = 999999
            visited[target_grid_coord[0]][target_grid_coord[1]] = 1

            for k, v in frontier.items():
                if v[0] < min_consistency:
                    min_consistency = v[0]
                    target_grid_coord = k
            _, rgb, depth = frontier[(target_grid_coord[0], target_grid_coord[1])]
            self.save_to_disk(target_grid_coord, rgb, depth)
            self.curr += 1
            del frontier[target_grid_coord]
            # plt.imshow(visited)
            # plt.show()
            print()

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
        self.transform_grid[res[0][0]][res[0][1]]['consistency_score'] = 0
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
                if candidate_pose['visited'] and np.linalg.norm(candidate_pose['position']-tgt_pose['position']) <= (0.3 if self.data != 'blender' else 1):
                    src_grid_coords.append((candidate_coord, candidate_pose['consistency_score'], np.linalg.norm(candidate_pose['position']-tgt_pose['position'])))
            src_grid_coords = sorted(src_grid_coords, key=lambda x: x[2])
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

        img_srcs = [np.array(Image.open(src_node['rgb_path'])) / 127.5 - 1.0 for src_node in src_nodes]
        img_dst = np.zeros_like(img_srcs[0])    # placeholder

        dm_srcs = [np.load(src_node['depth_path']).squeeze() for src_node in src_nodes]
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
            if curr_index == 0 and self.data == 'blender':
                h, w = dm_src.shape[:2]
                x = np.linspace(0, w - 1, w)
                y = np.linspace(0, h - 1, h)
                xs, ys = np.meshgrid(x, y)
                dm_srcs[i] = (dm_srcs[i] * self.K[0][0] / np.sqrt(
                    self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2))
        if self.model_type != 'vq_gan':
            while len(K_invs) < num_src:
                Ks.append(np.eye(3))
                K_invs.append(np.eye(3))
                R_rels.append(np.eye(3))
                t_rels.append(np.zeros(3))
                img_srcs.append(np.zeros_like(img_srcs[-1]))
                dm_srcs.append(np.zeros_like(dm_srcs[-1]))
            mask = np.zeros(num_src)
            mask[:len(src_nodes)] = 1
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

        if self.model_type != 'vq_gan':
            batch['src_masks'] = mask[None,]

        # batch['dst_depth'] = 2 * (batch['dst_depth'] - 1 / 16) / (1 / 7 - 1 / 16) - 1
        # batch['src_depths'] = 2 * (batch['src_depths'] - 1 / 16) / (1 / 7 - 1 / 16) - 1

        for k in batch:
            batch[k] = torch.from_numpy(batch[k].astype(np.float32)).cuda()
        return batch

    def prepare_batch_from_feature_data(self, tgt_node, src_nodes, num_src):
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

        if self.model_type != 'vq_gan':
            while len(K_invs) < num_src:
                Ks.append(np.eye(3))
                K_invs.append(np.eye(3))
                R_rels.append(np.eye(3))
                t_rels.append(np.zeros(3))
            mask = np.zeros(num_src)
            mask[:len(src_nodes)] = 1
        batch = {
            "Ks": np.stack(Ks)[None, ],
            "K_invs": np.stack(K_invs)[None, ],
            "R_rels": np.stack(R_rels)[None, ],
            "t_rels": np.stack(t_rels)[None, ],
            "dst_feature": tgt_node['feature'][None, ].detach() if tgt_node['fixed'] else tgt_node['feature'][None, ],
            "src_features": [src_node['feature'][None, ].detach() if src_node['fixed'] else src_node['feature'][None, ]
                             for src_node in src_nodes]
        }

        for k in batch:
            if k not in ['dst_feature', 'src_features']:
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


        # this is the volume, choose vox_length = 0.05 for blender

        # viz_folder = os.path.join(output_folder, method, scene_name)
        # if not os.path.exists(viz_folder):
        #     os.makedirs(viz_folder)

        scene_pcd = None
        for i in tqdm(range(len(src_Ks))):
            # read data
            depth = src_dms[i].astype(np.float32)
            rgb = cv2.imread(src_ims[i])
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
            if self.data == 'blender':
                vis = o3d.visualization.rendering.OffscreenRenderer(256, 64, headless=True)
            elif self.data == 'google_earth':
                vis = o3d.visualization.rendering.OffscreenRenderer(512, 512, headless=True)
            elif self.data == 'kitti360':
                vis = o3d.visualization.rendering.OffscreenRenderer(128, 64, headless=True)
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
        plt.imshow(integrated_tgt_depth)
        plt.show()

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

    def convert_batch_data_for_infinite_nature(self, batch_data):
        batch_data["src_img"] = batch_data["src_imgs"]
        batch_data["src_dm"] = batch_data["src_depths"]
        batch_data['dst_dm'] = batch_data['dst_depth']
        batch_data['T_src2tgt'] = torch.eye(4)[None, None].to(batch_data["src_imgs"].device)
        batch_data["T_src2tgt"][:, :, :3, :3] = batch_data["R_rels"]
        batch_data["T_src2tgt"][:, :, :3, 3] = batch_data["t_rels"]
        return batch_data


    def one_step_prediction_infinite_nature(self, tgt_pose_grid_coord, save_res_to_disk=True,
                            use_test_time_optimization=False):  # return type can be either rgbd or feature
        src_pose_grid_coords, _ = self.get_src_grid_coords(tgt_pose_grid_coord)
        print(f"im_{self.curr:05d}.png", 'tgt_pose_grid_coord: ', tgt_pose_grid_coord, 'src_pose_grid_coords: ',
              src_pose_grid_coords)

        tgt_meta = self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]
        src_metas = [self.transform_grid[src_pose_grid_coord[0]][src_pose_grid_coord[1]] for src_pose_grid_coord in
                     src_pose_grid_coords]
        batch = self.prepare_batch_data(tgt_meta, src_metas, self.num_src)
        batch = self.convert_batch_data_for_infinite_nature(batch)

        batch['src_img'] = batch['src_img'][:, 0]
        batch['src_dm'] = batch['src_dm'][:, 0][..., None]
        batch['T_src2tgt'] = batch['T_src2tgt'][:, 0]
        batch['Ks'] = batch['Ks'][:, 0]
        if self.data == 'blender':
            src_disparity_scaled = (1 / batch['src_dm'] - 1 / 16) / (1 / 7 - 1 / 16)
        elif self.data == 'google_earth':
            src_disparity_scaled = (1 / (batch['src_dm'] + 10) - 1 / 14.765625) / (1 / 10.099975586 - 1 / 14.765625)
        else:
            raise NotImplementedError
        x_src = torch.cat([(batch['src_img'] + 1)/2,
                           src_disparity_scaled], dim=-1).permute(0, 3, 1, 2)

        z, mu, logvar = self.dynamic_model.generator.style_encoding(x_src, return_mulogvar=True)
        rendered_rgbd, extrapolation_mask = self.dynamic_model.render_with_projection(x_src[:, :3][:, None],
                                                          batch['src_dm'].permute(0, 3, 1, 2),
                                                          batch["Ks"],
                                                          batch["Ks"],
                                                          batch['T_src2tgt'])
        print(batch['T_src2tgt'][0])
        predicted_rgbd = self.dynamic_model(rendered_rgbd, extrapolation_mask, z)

        rgb = np.clip((predicted_rgbd[0][:3] * 255.).permute(1, 2, 0).detach().cpu().numpy(), 0,
                      255)

        rgb = rgb.astype(np.uint8)

        plt.imshow(rgb)
        plt.show()

        if self.data == 'blender':
            depth = (1 / (predicted_rgbd[0][3] * (1 / 7 - 1 / 16) + 1 / 16)).detach().cpu().numpy()
        elif self.data == 'google_earth':
            depth = (1 / (predicted_rgbd.squeeze()[3]* (
                        1 / 10.099975586 - 1 / 14.765625) + 1 / 14.765625) - 10).detach().cpu().numpy()
        else:
            raise NotImplementedError

        if save_res_to_disk:
            self.save_to_disk(tgt_pose_grid_coord, rgb, depth)
        return {
            "rgbd": predicted_rgbd.squeeze().detach(),
            "fixed": False,
            "batch_src_imgs": batch['src_imgs'],
            "batch_src_depths": batch['src_depths'],
            "batch_R_rels": batch['R_rels'],
            "batch_t_rels": batch['t_rels']
        }

    def one_step_prediction(self, tgt_pose_grid_coord, save_res_to_disk=True, use_test_time_optimization=False): # return type can be either rgbd or feature
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

        # x2, x_dst2, extrapolation_mask2, warped_depth2 = self.dynamic_model.get_x(orig_batch, return_extrapolation_mask=True)
        # assert torch.equal(x, x2)
        # assert torch.equal(x_dst, x_dst2)
        # assert torch.equal(extrapolation_mask, extrapolation_mask2)
        # assert torch.equal(warped_depth, warped_depth2)

        if self.data == 'blender':
            x_sample_dets, _, pre_quantized_features, quantized_features = self.dynamic_model(x, topk=self.topk,
                                                                                              extrapolation_mask=extrapolation_mask,
                                                                                              get_pre_quantized_feature=True,
                                                                                              get_quantized_feature=True,
                                                                                              sample_number=1)
            x_sample_dets = x_sample_dets[0] # since sample number is 1
        elif self.data == 'kitti360':
            x_sample_dets, _, pre_quantized_features, quantized_features = self.dynamic_model(x,
                                                                                              topk=None,
                                                                                              extrapolation_mask=extrapolation_mask,
                                                                                              get_pre_quantized_feature=True,
                                                                                              get_quantized_feature=True)

            if use_test_time_optimization:
                pre_quantized_features = torch.from_numpy(pre_quantized_features.detach().cpu().numpy()).to(
                    pre_quantized_features.device)
                pre_quantized_features.requires_grad = True
                opt_features = torch.optim.Adam([pre_quantized_features],
                                                lr=1e-3, betas=(0.5, 0.9))
                torch.set_grad_enabled(True)
                # self.dynamic_model.train()
                for i in range(self.optimization_iteration_num):
                    quant, emb_loss, info = self.dynamic_model.quantize(pre_quantized_features)
                    xrec_optimized = self.dynamic_model.decode(quant)
                    aeloss, log_dict_ae = self.dynamic_model.loss(emb_loss, x, xrec_optimized, 0, self.dynamic_model.global_step,
                                                    last_layer=self.dynamic_model.get_last_layer(), split="test_optimization",
                                                    extrapolation_mask=extrapolation_mask, recon_on_visible=self.recon_on_visible)

                    x_rec_features = \
                    self.dynamic_model.encode(xrec_optimized, encoding_indices=None, extrapolation_mask=extrapolation_mask)[-1]
                    feature_loss = F.l1_loss(x_rec_features, pre_quantized_features)
                    total_loss = aeloss + feature_loss
                    if self.use_discriminator_loss:
                        total_loss = total_loss + log_dict_ae['test_optimization/g_loss'] * self.discriminator_loss_weight
                    opt_features.zero_grad()
                    total_loss.backward()
                    print("total_loss: ", total_loss.item())
                    opt_features.step()
                torch.set_grad_enabled(False)
                # self.dynamic_model.eval()
                x_sample_dets = xrec_optimized
            else:
                x_sample_dets = x_sample_dets[None,]
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

        if self.data == 'blender':
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
        # self.transform_grid[tgt_pose_grid_coord[0]][tgt_pose_grid_coord[1]]['consistency_score'] = consistency_score
        # self.total_inconsistency += consistency_score
        # print('average_inconsisntency: ', self.total_inconsistency/self.curr)


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
        res = np.zeros([64, 256])
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

    # def render_tgt_depth_map_from_srcs(self, src_imgs, src_depths, tgt_intrinsic, src_intrinsics, tgt2src_transforms):
    #     cur_fused_pc = None
    #     # prepare fused point cloud based on the depth map of the source images
    #     src2tgt_transform = tgt2src_transforms.inverse()
    #     for nbs_i in range(src_depths.shape[1]):
    #         curr_pc = self.pixel2cam(src_depths[:, nbs_i], src_intrinsics[:, nbs_i].inverse())
    #         cam_coords_flat = curr_pc.reshape(src_depths.shape[0], 3, -1)
    #         curr_pc = src2tgt_transform[:, nbs_i, :3, :3] @ cam_coords_flat + src2tgt_transform[:, nbs_i, :3, 3:]
    #
    #         if cur_fused_pc is None:
    #             cur_fused_pc = curr_pc
    #         else:
    #             cur_fused_pc = torch.cat([cur_fused_pc, curr_pc], dim=2)
    #
    #     cur_fused_pc = cur_fused_pc.permute(0, 2, 1)
    #     rendered_tgt_depths = []
    #     for batch_i in range(src_depths.shape[0]):
    #         fused_pc = Pointclouds(points=[cur_fused_pc[batch_i]], features=[torch.zeros_like(cur_fused_pc[batch_i])])
    #         dmap_all_layers = self.render_projection_from_point_cloud(fused_pc, tgt_intrinsic[batch_i], np.eye(4),
    #                                                              src_imgs.shape[-2], src_imgs.shape[-1],
    #                                                              cur_fused_pc.device, splat_radius=0.006).zbuf
    #         dmap_front = dmap_all_layers[..., 0]
    #         rendered_tgt_depths.append(dmap_front)
    #     rendered_tgt_depths = torch.stack(rendered_tgt_depths)
    #     return rendered_tgt_depths

    # def render_projection_from_srcs(self, src_features, src_depths, tgt_intrinsic, src_intrinsics, src2tgt_transform):
    #     cur_fused_pc = None
    #     cur_fused_features = None
    #     # prepare fused point cloud based on the depth map of the source images
    #     for nbs_i in range(src_depths.shape[1]):
    #         curr_pc = self.pixel2cam(src_depths[:, nbs_i], src_intrinsics[:, nbs_i].inverse())
    #         cam_coords_flat = curr_pc.reshape(src_depths.shape[0], 3, -1)
    #         curr_pc = torch.bmm(src2tgt_transform[:, nbs_i, :3, :3], cam_coords_flat) + src2tgt_transform[:, nbs_i, :3,
    #                                                                                     3:]
    #
    #         curr_feature = src_features[:, nbs_i].reshape(src_features.shape[0], src_features.shape[2], -1)
    #         if cur_fused_pc is None:
    #             cur_fused_pc = curr_pc
    #             cur_fused_features = curr_feature
    #         else:
    #             cur_fused_pc = torch.cat([cur_fused_pc, curr_pc], dim=2)
    #             cur_fused_features = torch.cat([cur_fused_features, curr_feature], dim=2)
    #
    #     cur_fused_pc = cur_fused_pc.permute(0, 2, 1)
    #     cur_fused_features = cur_fused_features.permute(0, 2, 1)
    #     rendered_tgt_depths = []
    #     projected_features = []
    #     for batch_i in range(src_depths.shape[0]):
    #         fused_pc = Pointclouds(points=[cur_fused_pc[batch_i]], features=[cur_fused_features[batch_i]])
    #         projected_feature, dmap_front = self.render_projection_from_point_cloud(fused_pc, tgt_intrinsic[batch_i],
    #                                                                            np.eye(4), src_features.shape[-2],
    #                                                                            src_features.shape[-1],
    #                                                                            cur_fused_pc.device, splat_radius=0.006)
    #         projected_features.append((projected_feature[0]))
    #         rendered_tgt_depths.append(dmap_front[0])
    #     rendered_tgt_depths = torch.stack(rendered_tgt_depths).unsqueeze(-1)
    #     projected_features = torch.stack(projected_features)
    #     rendered_tgt_depths = rendered_tgt_depths.permute(0, 3, 1, 2)
    #     projected_features = projected_features.permute(0, 3, 1, 2)
    #     return rendered_tgt_depths, projected_features

    # def render_projection_from_point_cloud(self, mesh, intrinsics, extrinsics, h, w, device, splat_radius):
    #     # Credit Shengze Wang in his DFVS repo
    #     # read mesh and render a depth map
    #     R_cv = torch.tensor(extrinsics[:3, :3]).unsqueeze(0).float().to(device)
    #     T_cv = torch.tensor(extrinsics[:3, 3]).unsqueeze(0).float().to(device)
    #     # NOTE: fx * aspect ratio, needed likely because of Pytorch3d normalized device coordinate
    #     cam_mat = intrinsics.float().unsqueeze(0)
    #
    #     # NOTE: not needed for newer pytorch3d (e.g. 0.6.1)
    #     # aspect_ratio = w/h
    #     # cam_mat[0,0,0] = aspect_ratio * cam_mat[0,1,1]
    #
    #     image_size = torch.tensor([[h, w]]).float()
    #     camera = cameras_from_opencv_projection(R_cv, T_cv, cam_mat, image_size)
    #
    #     raster_settings = PointsRasterizationSettings(
    #         image_size=(h, w),
    #         radius=0.01,
    #         points_per_pixel=10,
    #         bin_size=0
    #
    #     )
    #
    #     # Create a points renderer by compositing points using an alpha compositor (nearer points
    #     # are weighted more heavily). See [1] for an explanation.
    #     rasterizer = PointsRasterizer(cameras=camera, raster_settings=raster_settings).to(device)
    #     dmap = rasterizer(mesh, eps=1e-7).zbuf[..., 0]
    #     renderer = PointsRenderer(
    #         rasterizer=rasterizer,
    #         compositor=AlphaCompositor().to(device)
    #     )
    #     projected_features = renderer(mesh, eps=1e-7)
    #     return projected_features, dmap

    def prepare_pcd(self, depth, color, K, Rt):
        x = np.linspace(0, 256 - 1, 256)
        y = np.linspace(0, 256 - 1, 256)
        xs, ys = np.meshgrid(x, y)
        depth = (depth * K[0][0] / np.sqrt(
            K[0][0] ** 2 + (K[0][2] - ys - 0.5) ** 2 + (K[1][2] - xs - 0.5) ** 2))

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
        Ks = sorted(prediction_path.glob("K*"))
        Rs = sorted(prediction_path.glob("R*"))
        ts = sorted(prediction_path.glob("t*"))
        dms = sorted(prediction_path.glob("dm*"))
        rgbs = sorted(prediction_path.glob("im*"))

        pcds = []

        predicted_pcds = []
        gt_pcds = []
        for i in tqdm(range(len(rgbs))):
            # print(names[i])
            K = np.load(str(Ks[i]))
            R = np.load(str(Rs[i]))
            t = np.load(str(ts[i]))
            Rt = np.eye(4)
            Rt[:3, :3] = R
            Rt[:3, 3] = t
            predicted_depth = np.load(str(dms[i]))
            predicted_color = cv2.cvtColor(cv2.imread(str(rgbs[i])), cv2.COLOR_BGR2RGB)
            # for j in range(predicted_depth.shape[0]):
            #     for k in range(predicted_depth.shape[1]):
            #         predicted_depth[j, k] = predicted_depth[j, k] * K[0][0] / math.sqrt(K[0][0] ** 2 + (K[0][2] - j - 0.5) ** 2 + (K[1][2] - k - 0.5) ** 2)

            predicted_pcd = self.prepare_pcd(predicted_depth, predicted_color, K, Rt)
            # gt_pcd = prepare_pcd(gt_depth, K, Rt)
            predicted_pcds.append(predicted_pcd)
            # gt_pcds.append(gt_pcd)
            # o3d.visualization.draw_geometries([predicted_pcd])
            #
            # o3d.visualization.draw_geometries([gt_pcd])

        # o3d.visualization.draw_geometries(gt_pcds)
        o3d.visualization.draw_geometries(predicted_pcds)
