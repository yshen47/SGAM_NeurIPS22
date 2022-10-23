# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology

import os
import shutil
import numpy as np
import OpenEXR
from pathlib import Path
from PIL import Image
import Imath
import matplotlib.pyplot as plt

if __name__ == '__main__':
    for folder in os.listdir('dataset/clevr-infinite'):
        if 'scene' in folder:
            blender_3d_scene_path = Path(f'dataset/clevr-infinite/{folder}')
            blender_3d_postprocessed_scene_path = Path(f'dataset/clevr-infinite_large_postprocessed/{folder}')
            os.makedirs(blender_3d_postprocessed_scene_path, exist_ok=True)
            shutil.copy(blender_3d_scene_path / 'transforms.json',
                        blender_3d_postprocessed_scene_path / 'transforms.json')
            files = sorted(blender_3d_scene_path.glob('*.exr'))
            for p in files:
                im = OpenEXR.InputFile(str(p))
                dw = im.header()['dataWindow']
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

                rgb = []
                for c in ['Composite.Combined.R', 'Composite.Combined.G', 'Composite.Combined.B']:
                    color = np.fromstring(im.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                    color.shape = (size[1], size[0])
                    rgb.append(color)

                depth = np.fromstring(im.channel('View Layer.Depth.Z', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                depth.shape = (size[1], size[0])

                instance_map = np.fromstring(im.channel('View Layer.IndexOB.X', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                instance_map.shape = (size[1], size[0])

                rgb = np.stack(rgb[:3], axis=2)
                # rgbd = rgbd.transpose(2, 0, 1).astype(np.float32)
                # linear to standard RGB
                rgb = np.where(rgb <= 0.0031308,
                                         12.92 * rgb,
                                         1.055 * np.power(rgb, 1 / 2.4) - 0.055)
                rgb = np.where(rgb < 0.0, 0.0, np.where(rgb > 1.0, 1, rgb)) * 255
                rgb = Image.fromarray(rgb.astype(np.uint8))
                rgb.save(str(blender_3d_postprocessed_scene_path / f"im_{p.name[:-4]}.png"), format="png")
                np.save(str(blender_3d_postprocessed_scene_path / f"dm_{p.name[:-4]}.npy"), depth)
                np.save(str(blender_3d_postprocessed_scene_path / f"instance_{p.name[:-4]}.npy"), instance_map)


