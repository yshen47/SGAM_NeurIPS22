import cv2
import open3d as o3d
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path


def render_result(mesh, file_name, rgb = None, scene_cam = None,
                  center = [-2.15175792, -4.84732041, -0.09736646],
                  position = [-3.,         -5.76242419,  1.99999998]):
    render = o3d.visualization.rendering.OffscreenRenderer(1920, 1080, headless=True)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.shader = "defaultLit"
    line_mtl = o3d.visualization.rendering.MaterialRecord()
    line_mtl.shader = "unlitLine"
    line_mtl.line_width = 1.5

    render.scene.set_background([255, 255, 255, 255])
    render.scene.add_geometry("mesh", mesh, mtl)
    if scene_cam is not None:
        render.scene.add_geometry("cam", scene_cam, line_mtl)
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (1, 1, 1))
    render.scene.scene.enable_sun_light(False)
    target = center # [-8, -4, -13] # center of the scene
    up = [0, 1, 0] # y-up
    render.setup_camera(45.0, target, position, up)
    img = render.render_to_image()
    img = np.asarray(img)
    if rgb is not None:
        img[50:256+50, 50:256+50, :] = cv2.resize(np.asarray(rgb), [256, 256])
    img = o3d.geometry.Image(img)
    o3d.io.write_image(file_name, img, 9)

def correct_depth(depth, K):
    h, w = depth.shape[:2]
    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, h - 1, h)
    xs, ys = np.meshgrid(x, y)
    depth = (depth * K[0][0] / np.sqrt(K[0][0] ** 2 + (K[0][2] - ys - 0.5) ** 2 + (K[1][2] - xs - 0.5) ** 2))
    return depth

def dump_results(method, scene_folder, output_folder, render_step = True,
                 n_frames = 199, starting_frame = 1, vox_length = 0.05, cam_obj_size = 0.5):

    scene_name = os.path.basename(scene_folder)
    depth_file = 'dm_%05d_%02d_00.npy'
    im_file = 'im_%05d_%02d_00.png'
    K_file = 'K_%05d_%02d_00.npy'
    R_file = 'R_%05d_%02d_00.npy'
    t_file = 't_%05d_%02d_00.npy'
    cam_list = []

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=vox_length,
        sdf_trunc=100*vox_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    # this is the volume, choose vox_length = 0.05 for blender

    viz_folder = os.path.join(output_folder, method, scene_name)
    if not os.path.exists(viz_folder):
        os.makedirs(viz_folder)

    scene_cam = None
    for i in tqdm(range(5, len(list(Path(scene_folder).glob("im_*_*_*.png"))))):
        # read data
        depth = np.load(os.path.join(scene_folder, depth_file % (i, i)))
        rgb = o3d.io.read_image(os.path.join(scene_folder, im_file % (i, i)))
        R = np.load(os.path.join(scene_folder, R_file % (i, i)))
        t = np.load(os.path.join(scene_folder, t_file % (i, i)))
        # K = np.load(os.path.join(scene_folder, K_file % i))
        # K = np.load(os.path.join('/media/yuan/T7_red/GoogleEarthDataset/K.npy'))
        if 'blender' in scene_folder:
            K = np.load(os.path.join('/media/yuan/T7_red/blender_3d_large/K.npy'))
        elif 'google_earth' in scene_folder:
            K = np.load("/media/yuan/T7_red/GoogleEarthDataset/K.npy")
        else:
            raise NotImplementedError
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        if 'blender'  in scene_folder:
            depth = correct_depth(depth, K).astype(np.float32)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width = depth.shape[1], height = depth.shape[0], fx = fx, fy = fy, cx = cx, cy = cy)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        d = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, d, convert_rgb_to_intensity = False, depth_trunc = 20, depth_scale = 1)
        volume.integrate(rgbd_image, intrinsic, T)
        ## TODO(yuan): this is RGBD-integration step
        cam_obj = o3d.geometry.LineSet.create_camera_visualization(depth.shape[1], depth.shape[0], K, T, scale = cam_obj_size)
        cam_obj.paint_uniform_color((1, 0, 0))
        cam_list.append(cam_obj)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        pcd.transform(np.linalg.inv(T))
        if scene_cam is None:
            scene_cam = cam_obj
            scene_pcd = pcd
        else:
            scene_cam += cam_obj
            scene_pcd += pcd

        if render_step:
            ## TODO(yuan): this is the marching cube step which returns a mesh from volume
            mesh = volume.extract_triangle_mesh()
            mesh.compute_vertex_normals()
            render_file = os.path.join(output_folder, method, scene_name, "step_%05d.png" %i)
            render_result(mesh, render_file, rgb, scene_cam)


    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    pcd = volume.extract_voxel_point_cloud()
    mesh_file = os.path.join(output_folder, method, scene_name, "mesh.ply")
    cam_file = os.path.join(output_folder, method, scene_name, "camera.ply")
    vol_file = os.path.join(output_folder, method, scene_name, "vol.ply")
    render_file = os.path.join(output_folder, method, scene_name, "viz_cam.png")
    render_result(mesh, render_file, rgb, scene_cam)
    render_file = os.path.join(output_folder, method, scene_name, "viz_nocam.png")
    render_result(mesh, render_file, None, None)
    o3d.io.write_triangle_mesh(mesh_file, mesh)
    o3d.io.write_point_cloud(vol_file, pcd)
    o3d.io.write_line_set(cam_file, scene_cam)

def batch_blender(data_folder, results_folder, multi_process = False):
    render_step = True
    method = 'gt'
    method_folder = os.path.join(data_folder, method)
    scene_list = glob(os.path.join(method_folder, 'blender_*'))
    scene_list = [data_folder]
    if multi_process:
        # multi process
        processes = 16
        with mp.Pool(processes=processes) as pool:
            for _, scene_folder in enumerate(scene_list):
                pool.apply_async(dump_results, args=(method, scene_folder, results_folder, render_step))
            pool.close()
            pool.join()
    else:
        # single process
        for scene_folder in scene_list:
            dump_results(method, scene_folder, results_folder, render_step = render_step)

if __name__ == "__main__":
    data_folder = 'grid_res/google_earth_3_ppt'
    results_folder = f'{data_folder}/integrated_results'
    batch_blender(data_folder, results_folder)