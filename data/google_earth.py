# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology

import json
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch
import networkx as nx
from PIL import Image
import pickle
import torch.nn.functional as F


class PRNGMixin(object):
    """Adds a prng property which is a numpy RandomState which gets
    reinitialized whenever the pid changes to avoid synchronized sampling
    behavior when used in conjunction with multiprocessing."""

    @property
    def prng(self, seed=None):
        currentpid = os.getpid()
        if getattr(self, "_initpid", None) != currentpid:
            self._initpid = currentpid
            self._prng = np.random.RandomState(seed=seed) if seed is not None else np.random.RandomState()
        return self._prng


class GoogleEarthBase(Dataset, PRNGMixin):
    def __init__(self, split, n_src=2, dataset_dir=None, dataset=None, image_resolution=None, depth_range=None, use_extrapolation_mask=None):
        self.split = split
        self.src_num = n_src
        self.use_extrapolation_mask = use_extrapolation_mask
        self.depth_range = depth_range
        self.dataset = dataset
        self.image_resolution = image_resolution
        self.dataset_dir = dataset_dir
        if not os.path.exists(self.dataset_dir):
            self.dataset_dir = "/shared/rsaas/yshen47/GoogleEarthDataset"

        if not os.path.exists(self.dataset_dir):
            self.dataset_dir = "/projects/perception/datasets/GoogleEarthDataset"
        os.makedirs(f"{self.dataset_dir}/cache", exist_ok=True)
        self.grids = []
        self.cumulative_sum = [0]
        self.K = np.load(f"{self.dataset_dir}/K.npy")
        self.K[0] = self.K[0] * self.image_resolution[1] / 512
        self.K[1] = self.K[1] * self.image_resolution[0] / 512

        for grid_transform_path in sorted(Path(self.dataset_dir, self.split).glob("*")):
            if 'chicago' in str(grid_transform_path):
                continue
            scene_name = grid_transform_path.name
            print(grid_transform_path)
            with open(str(grid_transform_path / "transforms.json"), 'r') as f:
                curr_transform = json.load(f)
                g = self.build_graph_from_transform(curr_transform['frames'], grid_transform_path)
                self.grids.append(g)
                self.cumulative_sum.append(len(g.nodes) + self.cumulative_sum[-1])

    def build_graph_from_transform(self, transforms, grid_transform_path):
        g_path = f"{self.dataset_dir}/cache/{grid_transform_path.name[:-4]}_graph_{self.split}.txt"
        if os.path.exists(g_path):
           return pickle.load(open(g_path, 'rb'))
        g = nx.Graph()
        for i, transform in enumerate(transforms):
            c2w = np.array(transform['transform_matrix']) @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            w2c = np.linalg.inv(c2w)
            if not transform['is_valid']:
                continue
            frame_id = int(transform['file_path'][-9:-4])
            g.add_nodes_from([(frame_id,
                               {
                                    "frame_id": frame_id,
                                    "R": w2c[:3, :3],
                                    "t": w2c[:3, 3],
                                    "position": c2w[:3, 3],
                                    "rgb_path": str(grid_transform_path / f"im_{frame_id:05d}.png"),
                                    "depth_path": str(grid_transform_path / f"dm_{frame_id:05d}.npy"),
                                    })])
            if len(g) == 900 and self.split != 'train':  #TODO: overfit
                break
        node_keys = sorted(g.nodes)
        for i in tqdm(range(len(g))):
            for j in range(i+1, len(g)):
                node_i = g.nodes[node_keys[i]]
                node_j = g.nodes[node_keys[j]]
                # print(node_i.keys())
                if ((node_i['frame_id'] % 4) == (node_j['frame_id'] % 4)) and node_keys[i] != node_keys[j]: # 4 is due to there is 4 rotation variant at the same grid point
                    v = node_i['position'] - node_j['position']
                    euclidean_dis = np.linalg.norm(v)
                    if euclidean_dis <= 0.3:
                        g.add_edge(node_keys[i], node_keys[j], weight=euclidean_dis)
                        # g.add_edge(j, i, weight=euclidean_dis)
            if len(g[node_keys[i]]) == 0:
                # print(node_keys[i])
                g.remove_node(node_keys[i])
        pickle.dump(g, open(g_path, 'wb'))
        return g

    def __len__(self):
        #return 100 if self.split == 'train' else 10
        return self.cumulative_sum[-1] #if self.split == 'train' else 3000

    def parse_idx(self, idx):
        for img_index, cur_cumsum in enumerate(self.cumulative_sum):
            if idx < self.cumulative_sum[img_index+1]:
                relative_idx = idx - cur_cumsum
                return img_index, list(sorted(self.grids[img_index].nodes))[relative_idx]

    def __getitem__(self, global_index):
        grid_id, idx = self.parse_idx(global_index)
        tgt_node = self.grids[grid_id].nodes[idx]
        tgt_neighbors = sorted(self.grids[grid_id][idx])
        tgt_frame_id = tgt_node['frame_id']
        if self.split == 'train':
            src_num = self.src_num
            src_nodes = [self.grids[grid_id].nodes[tgt_neighbors[k]] for k in self.prng.choice(len(tgt_neighbors), src_num)]
        else:
            state = np.random.RandomState(seed=global_index)
            tgt_neighbors = np.array(tgt_neighbors)
            state.shuffle(tgt_neighbors)
            src_num = self.src_num
            src_nodes = [self.grids[grid_id].nodes[k] for k in tgt_neighbors[:src_num]]
        src_frame_ids = [src_node['frame_id'] for src_node in src_nodes]

        img_dst = Image.open(tgt_node['rgb_path'])
        img_srcs = [Image.open(src_node['rgb_path']) for src_node in src_nodes]

        dm_dst = np.load(tgt_node['depth_path'])
        dm_srcs = [np.load(src_node['depth_path']) for src_node in src_nodes]
        R_dst = tgt_node["R"]
        t_dst = tgt_node["t"]
        R_rels = []
        t_rels = []
        Ks = []
        K_invs = []
        T_tgt = np.eye(4)
        T_tgt[:3, :3] = R_dst
        T_tgt[:3, 3] = t_dst

        ## K
        h, w = img_dst.size[:2]

        for src_node in src_nodes:
            R_src = src_node["R"]
            t_src = src_node["t"]
            T_src = np.eye(4)
            T_src[:3, :3] = R_src
            T_src[:3, 3] = t_src
            T_rel = T_tgt @ np.linalg.inv(T_src)
            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]
            R_rels.append(R_rel)
            t_rels.append(t_rel)
            Ks.append(self.K)
            K_invs.append(np.linalg.inv(self.K))

        if self.image_resolution is not None and (self.image_resolution[0] != h or self.image_resolution[1] != w):
            ## img
            for i in range(len(img_srcs)):
                img_srcs[i] = img_srcs[i].resize((self.image_resolution[1], self.image_resolution[0]), resample=Image.LANCZOS)
            img_dst = img_dst.resize((self.image_resolution[1], self.image_resolution[0]), resample=Image.LANCZOS)

            img_dst = np.array(img_dst) / 127.5 - 1.0
            img_srcs = [np.array(img_src) / 127.5 - 1.0 for img_src in img_srcs]

            ## depth
            for i in range(len(dm_srcs)):
                dm_srcs[i] = F.interpolate(torch.from_numpy(dm_srcs[i][None, None,]), size=self.image_resolution)[0][0].numpy()
                dm_srcs[i][dm_srcs[i] == 65504] = -99999
            dm_dst = F.interpolate(torch.from_numpy(dm_dst[None, None,]), size=self.image_resolution)[0][
                0].numpy()
        else:
            img_dst = np.array(img_dst) / 127.5 - 1.0
            img_srcs = [np.array(img_src) / 127.5 - 1.0 for img_src in img_srcs]

            ## depth
            for i in range(len(dm_srcs)):
                dm_srcs[i][dm_srcs[i] == 65504] = -99999

        mask = np.zeros(self.src_num)
        mask[:src_num] = 1
        while len(K_invs) < self.src_num:
            Ks.append(np.eye(3))
            K_invs.append(np.eye(3))
            R_rels.append(np.eye(3))
            t_rels.append(np.zeros(3))
            img_srcs.append(np.zeros_like(img_srcs[-1]))
            dm_srcs.append(np.zeros_like(dm_srcs[-1]))
            src_frame_ids.append(-1)

        example = {
            "Ks": np.stack(Ks),
            "K_invs": np.stack(K_invs),
            "R_rels": np.stack(R_rels),
            "tgt_frame_id": np.array([tgt_frame_id]),
            "src_frame_ids": np.array(src_frame_ids),
            "t_rels": np.stack(t_rels),
            "dst_img": img_dst, # rgb or rgbid
            "src_imgs": np.stack(img_srcs),
            "dst_depth": dm_dst[..., None],
            "src_depths": np.stack(dm_srcs)[..., None],
            'src_masks': mask,
            'tgt_pixel_mask': (dm_dst != 65504)[ None, ]
        }
        for k in example:
            example[k] = example[k].astype(np.float32)

        # example["dst_rgb_fname"] = tgt_node['rgb_path']
        # example["src_rgb_fnames"] = [src_node['rgb_path'] for src_node in src_nodes]
        # example["dst_depth_fname"] = tgt_node['depth_path']
        # example["src_depth_fnames"] = [src_node['depth_path'] for src_node in src_nodes]

        return example


class GoogleEarthTrain(GoogleEarthBase):
    def __init__(self, size=None, n_src=2, dataset_dir=None, dataset=None, image_resolution=None, depth_range=None, use_extrapolation_mask=None):
        super().__init__(split='train', n_src=n_src, dataset=dataset, dataset_dir=dataset_dir,
                         image_resolution=image_resolution, depth_range=depth_range, use_extrapolation_mask=use_extrapolation_mask)
        self.size = size


class GoogleEarthValidation(GoogleEarthBase):
    def __init__(self, size=None, n_src=2, dataset_dir=None, dataset=None, image_resolution=None, depth_range=None, use_extrapolation_mask=None):
        super().__init__(split='val', n_src=n_src, dataset=dataset,  dataset_dir=dataset_dir,
                         image_resolution=image_resolution, depth_range=depth_range, use_extrapolation_mask=use_extrapolation_mask)
        self.size = size


class GoogleEarthTest(GoogleEarthBase):
    def __init__(self, size=None, n_src=2, dataset_dir=None, dataset=None, image_resolution=None, depth_range=None, use_extrapolation_mask=None):
        super().__init__(split='test', n_src=n_src, dataset_dir=dataset_dir, dataset=dataset,
                         image_resolution=image_resolution, depth_range=depth_range, use_extrapolation_mask=use_extrapolation_mask)
        self.size = size

