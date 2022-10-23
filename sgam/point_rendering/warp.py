# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology

from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
pixel_coords = None
from typing import *

def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)
    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    return pixel_coords

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[..., :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
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
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2), Z.reshape(b, h, w)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(src_img, tgt_depth, src_depth, pose, tgt_intrinsics, src_intrinsics, padding_mode='zeros', depth_threshold=1):
    """
    Inverse warp a source image to the target image plane.

    Args:
        src_img: the source image (where to sample pixels) -- [B, 3, H, W]
        tgt_depth: depth map of the target image -- [B, H, W]
        src_depth: depth map of the source image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    # check_sizes(src_img, 'img', 'B3HW')
    # check_sizes(tgt_depth, 'depth', 'BHW')
    # check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = src_img.size()

    cam_coords = pixel2cam(tgt_depth, tgt_intrinsics.inverse())  # [B,3,H,W]

    # pose_mat = pose_vec2mat(pose, rotation_mode)  # [B, 3, 4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = src_intrinsics @ pose  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
    src_pixel_coords, warped_src_depth = cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]
    valid_depth_mask = (warped_src_depth - src_depth) <= depth_threshold
    projected_img = F.grid_sample(src_img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    valid_points *= valid_depth_mask
    valid_points = valid_points.unsqueeze(1).repeat(1,projected_img.shape[1],1,1)
    projected_img *= valid_points
    return projected_img, valid_points

@torch.no_grad()
def render_projection_from_srcs_fast(src_features, src_depths, tgt_intrinsic, src_intrinsics, src2tgt_transform,
                                     src_num, dynamic_masks=None, depth_range=None, parallel=False):
    # src_features: (4,2,3,512,512)
    # src_depths: (8, 512, 512)
    # tgt_intrinsic: (4, 3, 3)
    # src_intrinsics: (8, 3, 3)
    # src2tgt_transform: (8, 4, 4)
    cur_fused_pc = None
    B, N, H, W = src_depths.size()
    device = src_depths.device
    if dynamic_masks is not None:
        dynamic_masks = dynamic_masks.permute(0, 2, 1, 3, 4)
        src_features = torch.cat([src_features, dynamic_masks], 2)
    # a more efficient implementation of render_projection_from_srcs()
    src_depths = src_depths.view(-1, H, W)
    src_intrinsics = src_intrinsics.view(-1, 3, 3)
    src2tgt_transform = src2tgt_transform.view(-1, 4, 4)
    assert (src2tgt_transform.size(0)==B*N)&(src_depths.size(0)==B*N)&(src_intrinsics.size(0)==B*N)
    curr_pc = pixel2cam(src_depths, src_intrinsics.inverse())

    cam_coords_flat = curr_pc.reshape(B*N, 3, -1)
    curr_pc = torch.bmm(src2tgt_transform[:, :3, :3], cam_coords_flat) + src2tgt_transform[:, :3, 3:]

    cur_fused_pc = curr_pc.view(B, N, 3, -1).permute(0, 2, 3, 1).reshape(B, 3, -1)
    cur_fused_features = src_features.view(B, N, 3, -1).permute(0, 2, 3, 1).reshape(B, 3, -1)
    assert (cur_fused_pc.shape[-1]==(N*W*H))&(cur_fused_features.shape[-1]==(N*W*H))

    # ignore splatting. simple forward warping suffices
    proj = tgt_intrinsic.bmm(cur_fused_pc)
    pix2d = proj[:, :2] / proj[:, 2:]
    # assert H==W, "The following clipping assumes images are square"
    pix_idx = (pix2d+0.5).long()

    # tile is more efficient; however torch 1.7.0 does not have it...
    bidx = torch.arange(B, device=device).view(B,1).expand(B, H*W*src_num).flatten() # 2 is due to 2 src view
    pix_idx = pix_idx.permute(0, 2, 1).reshape(-1, 2)
    idx = torch.cat((bidx.unsqueeze(-1), pix_idx.reshape(-1, 2)), dim=1)

    mask = (idx[:, 1] >= 0) & (idx[:, 1] < W) & (idx[:, 2] >= 0) & (idx[:, 2] < H) #& (src_depths!=0).flatten()

    idx = idx[mask, :]

    # import random
    # random.seed(10)
    # np.random.seed(29)
    # torch.random.manual_seed(3)
    # torch.set_deterministic(True)
    # torch.manual_seed(0)

    cur_fused_features = cur_fused_features.permute(0, 2, 1)
    projected_features = torch.zeros(B, 3, H, W, device=device)

    if not parallel:
        reshaped_fused_features = cur_fused_features.reshape(-1, 3)[mask, :]
        for i, ind in enumerate(idx):
            projected_features[ind[0], :, ind[2], ind[1]] = reshaped_fused_features[i]
    else:
        projected_features[idx[:, 0], :, idx[:, 2], idx[:, 1]] = cur_fused_features.reshape(-1, 3)[mask, :]

    # projected_features2 = torch.zeros(B, 3, H, W, device=device)
    # reshaped_fused_features = cur_fused_features.reshape(-1, 3)[mask, :]
    # for i, ind in enumerate(idx):
    #     projected_features2[ind[0], :, ind[2], ind[1]] = reshaped_fused_features[i]
    #
    # assert torch.equal(projected_features, projected_features2)
    rendered_tgt_depths = torch.zeros(B, 1, H, W, device=device)
    if parallel:
        rendered_tgt_depths[idx[:, 0], :, idx[:, 2], idx[:, 1]] = cur_fused_pc[:, 2].flatten()[mask].unsqueeze(-1)
    else:
        reshaped_cur_fused_pc = cur_fused_pc[:, 2].flatten()[mask].unsqueeze(-1)
        for i, ind in enumerate(idx):
            rendered_tgt_depths[ind[0], :, ind[2], ind[1]] = reshaped_cur_fused_pc[i]

    if dynamic_masks is not None:
        projected_features = projected_features * projected_features[:, 3:] + (-1) * (1-projected_features[:, 3:])
        projected_features = projected_features[:, :3]

    mask_feats = (projected_features == 0)
    mask_depths = (rendered_tgt_depths == 0)
#     torch.set_deterministic(False)
    median_feats = median_blur(projected_features, (3, 3))
    median_depths = median_blur(rendered_tgt_depths, (3, 3))
#     torch.set_deterministic(True)

    merge_feats = mask_feats * median_feats + (~mask_feats) * projected_features
    merge_depths = mask_depths * median_depths + (~mask_depths) * (rendered_tgt_depths)
    if depth_range is not None:
        extrapolation_mask = 1 - (merge_depths <= depth_range[1]).float() * (merge_depths >= depth_range[0]).float()
        merge_depths_arr = merge_depths.repeat(1, 3, 1, 1)
        merge_feats[merge_depths_arr >= depth_range[1]] = 0     # clip content when depth greater than 50 meters
    else:
        extrapolation_mask = (merge_depths <= 0).float()
    return merge_depths, merge_feats, extrapolation_mask.bool(), mask, cur_fused_features, idx, projected_features,


def get_binary_kernel2d(window_size):
    r"""Create a binary kernel to extract the patches. If the window size
    is HxW will create a (H*W)xHxW kernel.
    """
    window_range: int = window_size[0] * window_size[1]
    kernel: torch.Tensor = torch.zeros(window_range, window_range)
    for i in range(window_range):
        kernel[i, i] += 1.0
    return kernel.view(window_range, 1, window_size[0], window_size[1])


def _compute_zero_padding(kernel_size):
    """Utility function that computes zero padding tuple."""
    computed: List[int] = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]


def median_blur(input, kernel_size):
    """Blur an image using the median filter.

    .. image:: _static/img/median_blur.png

    Args:
        input: the input image with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.

    Returns:
        the blurred input tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = median_blur(input, (3, 3))
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    padding: Tuple[int, int] = _compute_zero_padding(kernel_size)

    # prepare kernel
    kernel: torch.Tensor = get_binary_kernel2d(kernel_size).to(input)
    b, c, h, w = input.shape

    # map the local window to single vector
    features: torch.Tensor = F.conv2d(input.reshape(b * c, 1, h, w), kernel, padding=padding, stride=1)
    features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

    # compute the median along the feature axis
    median: torch.Tensor = torch.median(features, dim=2)[0]

    return median
