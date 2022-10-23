import bisect
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F
import torch
from skimage import io


class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx


class ImagePaths(Dataset):
    def __init__(self, paths, image_resolution=None, random_crop=False, labels=None, convert_depth_flag=True, dataset_dir=None, dataset=None, depth_range=None):
        self.image_resolution = image_resolution
        self.depth_range = depth_range
        self.dataset_dir = dataset_dir
        self.random_crop = random_crop
        self.labels = dict() if labels is None else labels
        self.dataset = dataset
        self.labels["file_path_"] = paths
        self._length = len(paths)
        # self.K = np.load('/shared/rsaas/yshen47/GoogleEarthDataset/K.npy')
        # h, w = np.load("/shared/rsaas/yshen47/GoogleEarthDataset/val/ucla3.glb/dm_00000.npy").shape
        self.convert_depth_flag = convert_depth_flag

        if convert_depth_flag:
            self.K = np.load(self.dataset_dir + '/K.npy')
            self.K[0][0] = self.K[0][0] * self.image_resolution[1] / 256
            self.K[0][2] = self.K[0][2] * self.image_resolution[1] / 256
            self.K[1][1] = self.K[1][1] * self.image_resolution[0] / 256
            self.K[1][2] = self.K[1][2] * self.image_resolution[0] / 256

        if self.image_resolution is not None:
            self.rescaler = albumentations.SmallestMaxSize(max_size=min(image_resolution))
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.image_resolution[0],
                                                         width=self.image_resolution[1])
            else:
                self.cropper = albumentations.RandomCrop(height=self.image_resolution[0],
                                                         width=self.image_resolution[1])
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image_google_earth(self, image_path):
        if 'png' in image_path:
            image = Image.open(image_path).resize(self.image_resolution)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            image = self.preprocessor(image=image)["image"]
            res = (image/127.5 - 1.0).astype(np.float32)
            #print(image.shape)
        elif 'npy' in image_path:
            depth = np.load(image_path)
            depth = F.interpolate(torch.from_numpy(depth[None, None,]), size=self.image_resolution)[0][0].numpy()
            if self.convert_depth_flag:
                h, w = depth.shape[:2]
                x = np.linspace(0, w - 1, w)
                y = np.linspace(0, h - 1, h)
                xs, ys = np.meshgrid(x, y)
                depth = depth * self.K[0][0] / np.sqrt(
                    self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2)
            depth = depth + 10 # to reduce the near contribution after converting to inverse depth
            inverse_depth = 1/depth # between (1/10.5429688, 1/5.099975586)
            scaled_idepth = (inverse_depth - 1/14.765625) / (1/10.099975586 - 1/14.765625)
            res = 2 * scaled_idepth - 1
        else:
            raise NotImplementedError
        return res

    def preprocess_image_clevr_infinite(self, image_path):
        image_path = image_path.replace('/home/yuan/Documents', '/media/yuan/T7_red')
        if 'png' in image_path:
            image = Image.open(image_path).resize(self.image_resolution)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            image = self.preprocessor(image=image)["image"]
            res = (image/127.5 - 1.0).astype(np.float32)
            #print(image.shape)
        elif 'npy' in image_path:
            depth = np.load(image_path)
            depth = F.interpolate(torch.from_numpy(depth[None, None,]), size=self.image_resolution)[0][0].numpy()
            if self.convert_depth_flag:
                h, w = depth.shape[:2]
                x = np.linspace(0, w - 1, w)
                y = np.linspace(0, h - 1, h)
                xs, ys = np.meshgrid(x, y)
                depth = depth * self.K[0][0] / np.sqrt(
                    self.K[0][0] ** 2 + (self.K[0][2] - ys - 0.5) ** 2 + (self.K[1][2] - xs - 0.5) ** 2)
            inverse_depth = 1/depth # between (1/10.5429688, 1/5.099975586)
            scaled_idepth = (inverse_depth - 1/16) / (1/7 - 1/16)
            res = 2 * scaled_idepth - 1
        else:
            raise NotImplementedError
        return res.astype(np.float32)

    def center_crop(self, data, height, width):
        h, w = data.shape[:2]
        top_left_y = (h - height)//2
        top_left_x = (w - width)//2
        cropped_image = data[top_left_y:top_left_y+height, top_left_x:top_left_x+width]
        return cropped_image

    def preprocess_image_kitti360(self, image_path):
        image_path = image_path.replace('/shared', '/projects').replace("data_rect", "data_rect_low_res").replace("disparity", "disparity_low_res")
        if 'data_rect' in image_path:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image = np.array(image).astype(np.uint8)
            if image.shape == (64, 256, 3):
                res = (image/127.5 - 1.0).astype(np.float32)
            else:
                image = self.center_crop(image, 256, 1024)
                # print("image shape: ", image.shape)
                image = np.array(Image.fromarray(image).resize((256, 64)))
                # print("resized image shape: ", image.shape)
                # image = self.preprocessor(image=image)["image"]
                res = (image/127.5 - 1.0).astype(np.float32)
        elif 'disparity' in image_path:
            if 'npy' in image_path:
                # downsampled/postprocessed depth
                depth = np.load(image_path)
            else:

                disparity = io.imread(image_path).astype(np.float32) / 256.
                # disparity_range = io.imread(image_path.replace('disparity', 'max_disparity')).astype(
                #     np.float32) / 256. - io.imread(image_path.replace('disparity', 'min_disparity')).astype(
                #     np.float32) / 256.

                depth = 0.6 * 552.554261 / disparity
            depth = np.clip(depth, self.depth_range[0], self.depth_range[1])
            idepth = 1/depth
            scaled_disparity = (idepth - 1/75) / (1/3 - 1/75)
            res = 2 * scaled_disparity - 1
            # mask = (disparity_range < 1) * (depth > self.depth_range[0]) * (depth < self.depth_range[1])

            if 'npy' not in image_path:
                # downsampled/postprocessed depth
                res = self.center_crop(res, 256, 1024)
                # mask = self.center_crop(mask, 256, 1024)

                res = F.interpolate(torch.from_numpy(res[None, None,]), size=(64, 256))[0][0].numpy()
                # mask = F.interpolate(torch.from_numpy(mask[None, None,]).float(), size=(64, 256))[0][0].bool().numpy()

            return res
        else:
            raise NotImplementedError
        # print(res.min(), res.max())
        return res

    def __getitem__(self, i):
        example = dict()
        if self.dataset == 'kitti360':
            example["image"] = self.preprocess_image_kitti360(self.labels["file_path_"][i])
        elif self.dataset == 'google_earth':
            example["image"] = self.preprocess_image_google_earth(self.labels["file_path_"][i])
        elif self.dataset == 'clevr-infinite':
            example["image"] = self.preprocess_image_clevr_infinite(self.labels["file_path_"][i])
        else:
            raise NotImplementedError
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image_google_earth(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
