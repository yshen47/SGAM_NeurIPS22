import os.path
import random
from data.utils.utils import *
# sys.path.extend(['/home/yuan/PycharmProjects/DynamicPath', '/home/yuan/PycharmProjects/DynamicPath/modules/vq_gan_3d', '/home/yuan/PycharmProjects/DynamicPath/modules/vq_gan'])
from sgam.inference_pipeline import InfiniteSceneGeneration
import argparse
from sgam.generative_sensing_module.model import VQModel
import sys
sys.path.extend(['/home/yuan/PycharmProjects/InfiniteNature_pytorch'])
from modules.model import InfiniteNature
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
# torch.cuda.set_device(0)
from pathlib import Path
import numpy as np

def prepare_vqgan(data):
    if data == 'blender':
        config_path = 'trained_models/blender/custom_vqgan.yaml'
    elif data == 'google_earth':
        config_path = 'trained_models/google_earth/custom_vqgan.yaml'
    elif data == 'kitti360':
        # config_path = '/projects/perception/personals/yuan/projects/SGAM_latest/logs/2022-08-30T18-09-59_kitti360_vqgan/configs/kitti360_vqgan.yaml'
        config_path = 'trained_models/kitti360/conditional_generation_kitti360_n_src_2_with_d_layer_1/configs/custom_vqgan.yaml'
    else:
        raise NotImplementedError
    # init and save configs
    config = OmegaConf.load(config_path)
    config.model.params.data_config = config.data.params
    model = VQModel(**config.model['params'])
    # model = VQModel.load_from_checkpoint(ckpt_path, **config.model['params'])
    return model

def prepare_infinite_nature(data):
    if data == 'blender':
        config_path = '/home/yuan/PycharmProjects/InfiniteNature_pytorch/trained_models/clevr-infinite/clevr-infinite.yaml'
    elif data == 'google_earth':
        config_path = '/home/yuan/PycharmProjects/InfiniteNature_pytorch/trained_models/google_earth/google_earth.yaml'
    else:
        raise NotImplementedError
    config = OmegaConf.load(config_path)
    model = InfiniteNature(**config.model['params'])
    # model = VQModel.load_from_checkpoint(ckpt_path, **config.model['params'])
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',  type=str,
                        default="blender",     # choose from 'kitti360, google_earth, blender'
                        help='an integer for the accumulator')
    parser.add_argument('--use_rgbd_integration',  type=bool,
                        default=False,
                        help='an integer for the accumulator')
    parser.add_argument('--offscreen_rendering',  type=bool,
                        default=False,
                        help='offscreen_rendering is necessary when used in colab or docker environment. '
                             'However, the rendered depth is not exact the same as online open3d visualizer, '
                             'which we used to get model results.')
    parser.add_argument('--model_type', type=str,
                        default="vq_gan",
                        help='vqgan or infinite_nature')
    args = parser.parse_args()
    data = args.dataset
    if args.model_type == 'vq_gan':
        model = prepare_vqgan(data).to('cuda:0').eval()
    elif args.model_type == 'infinite_nature':
        model = prepare_infinite_nature(data).to('cuda:0').eval()
    else:
        raise NotImplementedError
    random.seed(10)
    np.random.seed(29)
    torch.random.manual_seed(3)
    framework = InfiniteSceneGeneration(model, 'vq_gan', f'{data}', data, None, 0,
                                        use_rgbd_integration=args.use_rgbd_integration,
                                        offscreen_rendering=args.offscreen_rendering)
    framework.expand_to_inf()
