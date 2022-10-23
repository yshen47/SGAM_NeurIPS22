import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance


class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, predicted_points, predicted_sdfs, predicted_colors, ref_points, ref_sdfs, ref_colors):
        predicted_points = predicted_points.view(-1, *predicted_points.shape[-2:])
        predicted_sdfs = predicted_sdfs.view(-1, *predicted_sdfs.shape[-2:])
        predicted_colors = predicted_colors.view(-1, *predicted_colors.shape[-2:])

        ref_points = ref_points.view(-1, *ref_points.shape[-2:]).clone()
        ref_sdfs = ref_sdfs.view(-1, *ref_sdfs.shape[-2:])
        ref_colors = ref_colors.view(-1, *ref_colors.shape[-2:])

        closest_ref_indices = torch.argmin(torch.cdist(predicted_points, ref_points), dim=1)
        sdf_l1_loss = F.l1_loss(torch.gather(ref_sdfs, 1, closest_ref_indices[..., None].clone()), predicted_sdfs)
        color_l1_loss = F.l1_loss(torch.gather(ref_colors, 1, closest_ref_indices[..., None].repeat(1, 1, 3).clone()), predicted_colors)

        # closest_pred_indices = torch.argmin(torch.cdist(ref_points, predicted_points), dim=1)
        # chamfer_distance = F.l1_loss(torch.gather(ref_points, 1, closest_ref_indices[..., None].repeat(1, 1, 3)).detach(),
        #                              predicted_points)# + F.l1_loss(torch.gather(predicted_points, 1, closest_pred_indices[..., None].repeat(1, 1, 3)), ref_points)
        chamfer = chamfer_distance(predicted_points, ref_points)[0]
        losses = {
            'sdf_l1_loss': sdf_l1_loss,
            'color_l1_loss': color_l1_loss,
            'chamfer_distance_loss': chamfer
        }
        return losses
