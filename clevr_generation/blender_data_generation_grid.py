# SGAM2: Building a Virtual 3D World through Simultaneous Generation and Mapping
# authors: Yuan Shen, Wei-Chiu Ma, Shenlong Wang
# CLEVR-Infinite generation scripts
import numpy as np
import json
import os
import random
import time
import bpy

def enable_gpus(device_type):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cuda_devices, opencl_devices = cycles_preferences.get_devices()

    if device_type == "CUDA":
        devices = cuda_devices
    elif device_type == "OPENCL":
        devices = opencl_devices
    else:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []

    for device in devices:
        if device.type == "CPU":
            device.use = False
        else:
            device.use = True
            activated_gpus.append(device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


print(enable_gpus("CUDA"))
# prepare random scene
bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 0), rotation=(0.5, 0, 0))
bpy.ops.transform.translate(value=(0, 0, 13.0377), orient_type='GLOBAL',
                            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                            constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False,
                            proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False,
                            use_proportional_projected=False)

bpy.context.object.data.energy = 1.5

bpy.ops.mesh.primitive_plane_add(enter_editmode=False, location=(0, 0, 0))
bpy.ops.transform.resize(value=(100, 100, 100), orient_type="GLOBAL", orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                         orient_matrix_type="GLOBAL", mirror=True, use_proportional_edit=False,
                         proportional_edit_falloff="SMOOTH", proportional_size=1, use_proportional_connected=False,
                         use_proportional_projected=False)
bpy.ops.transform.translate(value=(-0, -0, -11.695), orient_type="GLOBAL",
                            orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type="GLOBAL",
                            constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False,
                            proportional_edit_falloff="SMOOTH", proportional_size=1, use_proportional_connected=False,
                            use_proportional_projected=False)
bpy.ops.rigidbody.object_add()
bpy.context.object.rigid_body.collision_shape = "MESH"
bpy.context.object.rigid_body.collision_margin = 0
bpy.context.object.rigid_body.type = "PASSIVE"


# Get material

def newShader(id, type, r, g, b):
    mat = newMaterial(id)

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')

    if type == "diffuse":
        shader = nodes.new(type='ShaderNodeBsdfDiffuse')
        nodes["Diffuse BSDF"].inputs[0].default_value = (r, g, b, 1)

    elif type == "glossy":
        shader = nodes.new(type='ShaderNodeBsdfGlossy')
        nodes["Glossy BSDF"].inputs[0].default_value = (r, g, b, 1)
        nodes["Glossy BSDF"].inputs[1].default_value = 0

    links.new(shader.outputs[0], output.inputs[0])

    return mat


def newMaterial(id):
    mat = bpy.data.materials.get(id)

    if mat is None:
        mat = bpy.data.materials.new(name=id)

    mat.use_nodes = True

    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()

    return mat


rgb_materials = ['diffuse', 'diffuse']
sizes = [1.0, 0.7, 0.4]
bpy.context.object.data.materials.append(newShader("white", "diffuse", 1, 1, 1))
meta_data = {}
curr_index = 1
number = 21
counter = 0
for a in range(2):
    counter += 2
    counter2 = 0
    for b in range(number):
        counter2 += 2
        counter3 = 0
        for c in range(number):
            shape_type = np.random.randint(0, 4)
            location = ((counter3 + 2 - (number)) * 2 + 2 * random.random(),
                        (counter2 - 2 - (number)) * 2 + 2 * random.random(),
                        (counter - 2) * 2 + 2 * random.random())
            if shape_type == 0:
                bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=2, enter_editmode=False, location=location)
            elif shape_type == 1:
                bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, location=location)
            elif shape_type == 2:
                bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, enter_editmode=False, location=location)
            elif shape_type == 3:
                bpy.ops.mesh.primitive_ico_sphere_add(radius=1, enter_editmode=False, location=location)

            counter3 += 2
            bpy.ops.rigidbody.object_add()
            bpy.context.object.rigid_body.mass = 20
            bpy.context.object.rigid_body.collision_shape = "MESH"
            bpy.context.object.rigid_body.friction = 1
            bpy.context.object.rigid_body.collision_margin = 0.04
            bpy.context.object.rigid_body.use_margin = True
            bpy.context.object.rigid_body.collision_margin = 0
            bpy.context.object.rigid_body.linear_damping = 0.35
            bpy.context.object.rigid_body.angular_damping = 0.6
            bpy.context.object.pass_index = curr_index
            bpy.context.object.name = f"{curr_index:04d}"
            color_type = np.random.randint(0, 2)
            color = [random.random(), random.random(), random.random()]
            bpy.context.object.data.materials.append(
                newShader(f"material.{curr_index}", rgb_materials[color_type], color[0], color[1], color[2]))

            size = sizes[np.random.randint(0, 2)]
            bpy.ops.transform.resize(value=(size, size, size), orient_type='GLOBAL',
                                     orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL',
                                     mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH',
                                     proportional_size=1, use_proportional_connected=False,
                                     use_proportional_projected=False)
            curr_meta_data = {}
            curr_meta_data['id'] = curr_index
            curr_meta_data['shape_type'] = ['cone', 'cube', 'cylinder', 'sphere'][shape_type]
            curr_meta_data['material_type'] = rgb_materials[color_type]
            curr_meta_data['size'] = size
            curr_meta_data['color'] = color
            meta_data[curr_index] = curr_meta_data
            curr_index += 1

override = {"scene": bpy.context.scene, "point_cache": bpy.context.scene.rigidbody_world.point_cache}
bpy.ops.ptcache.bake(override, bake=False)
bpy.context.scene.frame_current = 500

camera = bpy.data.objects.get('Camera')
if camera is None:
    bpy.ops.object.camera_add(enter_editmode=False, align="VIEW", location=(0, 0, 0), rotation=(0.3, 0, 0))
    camera = bpy.data.objects.get('Camera')

bpy.context.scene.camera = bpy.context.object

bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.image_settings.color_mode = 'RGB'
bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'
bpy.context.scene.cycles.samples = 500
bpy.context.scene.frame_start = 500
bpy.context.scene.frame_end = 510


def get_calibration_matrix_K_from_blender(mode='simple'):
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()

    if mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float32)

    return K

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


scene_num = int(time.time())

out_data = {
    'camera_angle_x': camera.data.angle_x,
}

out_data['frames'] = []
seq_path = os.path.join(f"/root/easy_world_dataset/diffuse_scene_{scene_num}")
os.makedirs(seq_path, exist_ok=True)
bpy.context.scene.frame_current = bpy.context.scene.frame_start
start_location = np.array(camera.location)

i = 0
for grid_i in np.linspace(-20, 20, 50):
    for grid_j in np.linspace(-20, 20, 50):
        camera.location.x = grid_j
        camera.location.y = grid_i

        bpy.context.scene.render.filepath = f"{seq_path}/{i:05d}.exr"

        bpy.ops.render.render(write_still=True)
        frame_data = {
            'file_path': bpy.context.scene.render.filepath,
            'grid_i': grid_i,
            'grid_j': grid_j,
            'transform_matrix': listify_matrix(bpy.data.objects["Camera"].matrix_world)
        }
        out_data['frames'].append(frame_data)
        i += 1

for k in meta_data:
    object = bpy.data.objects.get(f"{k:04d}")
    meta_data[k]['transform_matrix'] = listify_matrix(object.matrix_world)
with open(f'{seq_path}/meta_data.json', 'w') as out_file:
    json.dump(meta_data, out_file, indent=4)

K = get_calibration_matrix_K_from_blender()
out_data['camera_intrinsic'] = K.tolist()

with open(f'{seq_path}/transforms.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)
