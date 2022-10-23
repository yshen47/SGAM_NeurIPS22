# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology

import random
from data.utils.utils import *
from sgam.inference_pipeline import InfiniteSceneGeneration
import argparse
from sgam.generative_sensing_module.model import VQModel
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
import numpy as np


def prepare_vqgan(data):
    if data == 'clevr-infinite':
        config_path = 'trained_models/clevr-infinite/config.yaml'
    elif data == 'google_earth':
        config_path = 'trained_models/google_earth/config.yaml'
    else:
        raise NotImplementedError
    # init and save configs
    config = OmegaConf.load(config_path)
    config.model.params.data_config = config.data.params
    model = VQModel(**config.model['params'])
    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset',  type=str,
                        default="clevr-infinite",     # choose from 'google_earth, clevr-infinite'
                        help='an integer for the accumulator')
    parser.add_argument('--use_rgbd_integration',  type=bool,
                        default=True,
                        help='an integer for the accumulator')
    parser.add_argument('--seed_index',  type=str,
                        default="0",
                        help='seed index for initial rgbd pose')
    parser.add_argument('--offscreen_rendering',  type=bool,
                        default=True,
                        help='offscreen_rendering is necessary when used in colab or docker environment. ')
    args = parser.parse_args()
    data = args.dataset
    model = prepare_vqgan(data).to('cuda:0').eval()
    random.seed(10)
    np.random.seed(29)
    torch.random.manual_seed(3)
    framework = InfiniteSceneGeneration(model, data,
                                        seed_index=args.seed_index,
                                        use_rgbd_integration=args.use_rgbd_integration,
                                        offscreen_rendering=args.offscreen_rendering)
    framework.expand_to_inf()
