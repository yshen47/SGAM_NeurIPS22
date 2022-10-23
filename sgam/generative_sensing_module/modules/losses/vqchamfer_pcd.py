import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from sgam.generative_sensing_module.modules.losses.lpips import LPIPS
from sgam.generative_sensing_module.modules.discriminator.model import NLayerDiscriminator, weights_init
from pillar_codebook_models.modules.pointcloudmodules.model import PointNetCls
from pillar_codebook_models.modules.pointcloudmodules.model import PointNetEncoder, PointNetDecoder, PointNetCls


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake, mask=None):
    if mask is not None:
        loss_real = torch.mean(F.relu(1. - logits_real)*mask)
        loss_fake = torch.mean(F.relu(1. + logits_fake)*mask)
    else:
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQChamferWithDiscriminator(nn.Module):
    def __init__(self, disc_start,
                 color_weight, distance_weight, sdf_weight,
                 pillar_quant_emb_loss_weight=1.0, pseudo_image_quant_emb_loss_weight=1.0,
                 disc_num_layers=2, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, kld_loss_weight=0.05, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.color_weight = color_weight
        self.distance_weight = distance_weight
        self.sdf_weight = sdf_weight
        self.pillar_quant_emb_loss_weight = pillar_quant_emb_loss_weight
        self.pseudo_image_quant_emb_loss_weight = pseudo_image_quant_emb_loss_weight

        self.discriminator = PointNetCls().apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, loss_dict, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", no_dis=False):
        total_loss = None
        rec_loss = None
        for k, v in loss_dict.items():
            if k == 'pillar':
                if total_loss is None:
                    total_loss = v * self.pillar_quant_emb_loss_weight
                else:
                    total_loss = total_loss + v * self.pillar_quant_emb_loss_weight
            elif k == 'pseudo_image':
                if total_loss is None:
                    total_loss = v * self.pseudo_image_quant_emb_loss_weight
                else:
                    total_loss = total_loss + v * self.pseudo_image_quant_emb_loss_weight
            elif k == 'color_l1_loss':
                if total_loss is None:
                    total_loss = v * self.color_weight
                else:
                    total_loss = total_loss + v * self.color_weight
                if rec_loss is None:
                    rec_loss = v * self.color_weight
                else:
                    rec_loss = rec_loss + v * self.color_weight
            elif k == 'sdf_l1_loss':
                if total_loss is None:
                    total_loss = v * self.sdf_weight
                else:
                    total_loss = total_loss + v * self.sdf_weight
                if rec_loss is None:
                    rec_loss = v * self.sdf_weight
                else:
                    rec_loss = rec_loss + v * self.sdf_weight
            elif k == 'chamfer_distance_loss':
                if total_loss is None:
                    total_loss = v * self.distance_weight
                total_loss = total_loss + v * self.distance_weight
                if rec_loss is None:
                    rec_loss = v * self.distance_weight
                else:
                    rec_loss = rec_loss + v * self.distance_weight

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if not no_dis:
                if cond is None:
                    assert not self.disc_conditional
                    logits_fake = self.discriminator(reconstructions)
                else:
                    assert self.disc_conditional
                    logits_fake = self.discriminator(torch.cat((reconstructions, cond), dim=1))
                g_loss = -torch.mean(logits_fake)

                try:
                    d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            #print(extrapolation_mask_weight.sum(), 62 * 62)    
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = total_loss
            if not no_dis:
                loss += d_weight * disc_factor * g_loss

            log = loss_dict
            if not no_dis:
                log["{}/g_loss".format(split)] = g_loss.detach().mean()
                log["{}/d_weight".format(split)] = d_weight
                log["{}/disc_factor".format(split)] = torch.tensor(disc_factor)
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            # extrapolation_mask_weight = F.interpolate(extrapolation_mask.float(), size=(62, 62))
            if cond is None:
                logits_real = self.discriminator(inputs.detach())# * extrapolation_mask_weight
                logits_fake = self.discriminator(reconstructions.detach())# * extrapolation_mask_weight
            else:
                logits_real = self.discriminator(torch.cat((inputs.detach(), cond), dim=1))# * extrapolation_mask_weight
                logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond), dim=1))# * extrapolation_mask_weight

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if global_step % 10 != 0:
               disc_factor = disc_factor * 0.
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean(),
                   # "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   }
            return d_loss, log


class ChamferLoss(nn.Module):

    def __init__(self, color_weight, distance_weight, sdf_weight, **ignore_kwargs):
        super().__init__()
        self.color_weight = color_weight
        self.distance_weight = distance_weight
        self.sdf_weight = sdf_weight

    def forward(self, predicted_points, predicted_sdfs, predicted_colors, ref_points, ref_sdfs, ref_colors):
        predicted_points = predicted_points.view(-1, *predicted_points.shape[-2:])
        predicted_sdfs = predicted_sdfs.view(-1, *predicted_sdfs.shape[-2:])
        predicted_colors = predicted_colors.view(-1, *predicted_colors.shape[-2:])

        ref_points = ref_points.view(-1, *ref_points.shape[-2:])
        ref_sdfs = ref_sdfs.view(-1, *ref_sdfs.shape[-2:]).squeeze(-1)
        ref_colors = ref_colors.view(-1, *ref_colors.shape[-2:])

        chamfer_point_distance = chamfer_distance(predicted_points,
                                                  ref_points)[0]
        closest_ref_indices = torch.argmin(torch.cdist(predicted_points, ref_points), dim=1)
        sdf_l1_loss = F.l1_loss(torch.gather(ref_sdfs, 1, closest_ref_indices), predicted_sdfs)
        color_l1_loss = F.l1_loss(torch.gather(ref_colors, 1, closest_ref_indices[..., None].repeat(1, 1, 3)), predicted_colors)
        losses = {
            'sdf_l1_loss': sdf_l1_loss,
            'color_l1_loss': color_l1_loss,
            'chamfer_distance_loss': chamfer_point_distance
        }
        return sdf_l1_loss * self.sdf_weight + color_l1_loss * self.color_weight + chamfer_point_distance * self.distance_weight, losses