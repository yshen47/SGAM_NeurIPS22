# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology

import copy
import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
#import open3d as o3d
from data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from PIL import Image


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.depth_data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        if self.depth_data is not None:
            depth_example = self.depth_data[i]
            rgbd_example = example
            rgbd_example['image'] = np.concatenate([rgbd_example['image'], depth_example['image'][:, :, None]], 2)
            #
            # rgb = o3d.geometry.Image(np.array(Image.open(example['file_path_'])).astype(np.uint8))
            # d = o3d.geometry.Image(np.load(depth_example['file_path_']))
            #
            # # d = o3d.geometry.Image(d)
            # K = np.load(f"/media/yuan/T7_red/GoogleEarthDataset/K.npy").astype(np.float64)
            # fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
            #
            # intrinsic = o3d.camera.PinholeCameraIntrinsic(width=256, height=256,
            #                                                            fx=fx, fy=fy,
            #                                                            cx=cx, cy=cy)
            #
            # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, d)
            # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            # o3d.visualization.draw_geometries([pcd])
            rgbd_example['file_path_'] = rgbd_example['file_path_'].split('.')[0]
            return rgbd_example
        return example


import random

class CustomTrain(CustomBase):
    def __init__(self, image_resolution, images_list_file, use_depth, convert_depth_flag, dataset_dir, dataset, depth_range):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        with open(images_list_file, "r") as f:
            paths = [path for path in f.read().splitlines() if 'chicago' not in path]
        self.data = ImagePaths(paths=paths, image_resolution=image_resolution, random_crop=False, convert_depth_flag=convert_depth_flag,
                               dataset_dir=dataset_dir, dataset=dataset, depth_range=depth_range)
        self.use_depth = use_depth
        if use_depth:
            paths = copy.deepcopy(paths)
            if 'kitti360' == dataset:
                for i in range(len(paths)):
                    paths[i] = paths[i].replace('data_rect', 'disparity') + ".npy"
            else:
                for i in range(len(paths)):
                    paths[i] = paths[i].replace('im', 'dm').replace(".png", ".npy")

            self.depth_data = ImagePaths(paths=paths, image_resolution=image_resolution, random_crop=False, convert_depth_flag=convert_depth_flag,
                                         dataset_dir=dataset_dir, dataset=dataset, depth_range=depth_range)


class CustomValidation(CustomBase):
    def __init__(self, image_resolution, images_list_file, use_depth, convert_depth_flag, dataset_dir, dataset, depth_range):
        super().__init__()
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        with open(images_list_file, "r") as f:
            paths = [path for path in f.read().splitlines() if 'chicago' not in path]
        random.seed(3)
        random.shuffle(paths)
        paths = paths[:2500]
        self.data = ImagePaths(paths=paths, image_resolution=image_resolution, random_crop=False, convert_depth_flag=convert_depth_flag,
                               dataset_dir=dataset_dir, dataset=dataset, depth_range=depth_range)
        if use_depth:
            paths = copy.deepcopy(paths)
            if 'kitti360' == dataset:
                for i in range(len(paths)):
                    paths[i] = paths[i].replace('data_rect', 'disparity') + ".npy"
            else:
                for i in range(len(paths)):
                    paths[i] = paths[i].replace('im', 'dm').replace(".png", ".npy")
            self.depth_data = ImagePaths(paths=paths, image_resolution=image_resolution, random_crop=False, convert_depth_flag=convert_depth_flag,
                                         dataset_dir=dataset_dir, dataset=dataset, depth_range=depth_range)


