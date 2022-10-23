import torch
import torch.nn as nn
import torch.nn.functional as F

from sgam.generative_sensing_module.modules.losses.lpips import LPIPS
from sgam.generative_sensing_module.modules.discriminator.model import NLayerDiscriminator, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", use_discriminative_loss=False, disp_loss_weight=None, disc_update_every_n_step=None, kernel_width=4):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.use_discriminative_loss = use_discriminative_loss
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf,
                                                 kernel_width=kernel_width
                                                 ).apply(weights_init)
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

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", extrapolation_mask=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs[:, :3].contiguous(), reconstructions[:, :3].contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

# class VQLPIPSWithDiscriminator(nn.Module):
#     def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0, disp_loss_weight=1.0,
#                  disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
#                  perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
#                  disc_ndf=64, kernel_width=4, disc_loss="hinge", use_discriminative_loss=True, dataset=None,
#                  disc_update_every_n_step=None):
#         super().__init__()
#         assert disc_loss in ["hinge", "vanilla"]
#         self.dataset = dataset
#         self.use_discriminative_loss = use_discriminative_loss
#         self.codebook_weight = codebook_weight
#         self.disp_loss_weight = disp_loss_weight
#         self.pixel_weight = pixelloss_weight
#         self.perceptual_loss = LPIPS().eval()
#         self.perceptual_weight = perceptual_weight
#
#         self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
#                                                  n_layers=disc_num_layers,
#                                                  use_actnorm=use_actnorm,
#                                                  ndf=disc_ndf,
#                                                  kernel_width=kernel_width,
#                                                  ).apply(weights_init)
#         self.discriminator_iter_start = disc_start
#         if disc_loss == "hinge":
#             self.disc_loss = hinge_d_loss
#         elif disc_loss == "vanilla":
#             self.disc_loss = vanilla_d_loss
#         else:
#             raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
#         print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
#         self.disc_factor = disc_factor
#         self.discriminator_weight = disc_weight
#         self.disc_conditional = disc_conditional
#         self.disc_update_every_n_step = disc_update_every_n_step
#
#     def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
#         if last_layer is not None:
#             nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
#         else:
#             nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
#             g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]
#
#         d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
#         d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
#         d_weight = d_weight * self.discriminator_weight
#         return d_weight
#
#     def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
#                 global_step, last_layer=None, cond=None, split="train", no_dis=False, extrapolation_mask=None,
#                 recon_on_visible=False, is_inference=False):
#
#         # rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
#         # if recon_on_visible:
#         #     rec_loss = rec_loss * ~extrapolation_mask
#         # disparity_loss = rec_loss[:, 3:]
#         # rgb_loss = rec_loss[:, :3]
#         # if self.perceptual_weight > 0:
#         #     if recon_on_visible:
#         #         p_loss = self.perceptual_loss(inputs[:, :3].contiguous() * ~extrapolation_mask,
#         #                                       reconstructions[:, :3].contiguous() * ~extrapolation_mask).mean()
#         #     else:
#         #         p_loss = self.perceptual_loss(inputs[:, :3].contiguous(), reconstructions[:, :3].contiguous()).mean()
#         #     print("perceptual_weight: ", self.perceptual_weight)
#         #     print("p_loss: ", p_loss.item())
#         #     if recon_on_visible:
#         #         rec_loss = disparity_loss.sum() / (~extrapolation_mask).float().sum() * self.disp_loss_weight \
#         #                    + rgb_loss.sum() / (~extrapolation_mask).float().sum() + self.perceptual_weight * p_loss
#         #         disparity_loss = disparity_loss.sum() / (~extrapolation_mask).float().sum()
#         #         rgb_loss = rgb_loss.sum() / (~extrapolation_mask).float().sum()
#         #     else:
#         #         rec_loss = rgb_loss.mean() + disparity_loss.mean() * self.disp_loss_weight + self.perceptual_weight * p_loss
#         #         disparity_loss = disparity_loss.mean()
#         #         rgb_loss = rgb_loss.mean()
#         # else:
#         #     p_loss = torch.tensor([0.0])
#         #
#         # nll_loss = rec_loss
#
#         rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
#         if self.perceptual_weight > 0:
#             p_loss = self.perceptual_loss(inputs[:, :3].contiguous(), reconstructions[:, :3].contiguous())
#             rec_loss = rec_loss + self.perceptual_weight * p_loss
#         else:
#             p_loss = torch.tensor([0.0])
#
#         nll_loss = rec_loss
#         nll_loss = torch.mean(nll_loss)
#
#         # now the GAN part
#         if optimizer_idx == 0:
#             # generator update
#             self.discriminator.eval()
#             if not no_dis:
#                 if cond is None:
#                     assert not self.disc_conditional
#                     logits_fake = self.discriminator(reconstructions.contiguous())
#                 else:
#                     assert self.disc_conditional
#                     logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
#
#                 if extrapolation_mask is not None and is_inference:
#                     extrapolation_mask_weight = F.interpolate(extrapolation_mask.float(), size=logits_fake.shape[2:])
#                     if not recon_on_visible:
#                         g_loss = -torch.sum(logits_fake * extrapolation_mask_weight) / extrapolation_mask_weight.float().sum()
#                     else:
#                         g_loss = -torch.sum(logits_fake * (~extrapolation_mask_weight.bool()).float()) / (~extrapolation_mask_weight.bool()).float().sum()
#
#                 else:
#                     g_loss = -torch.mean(logits_fake)
#
#                 try:
#                     d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
#                 except RuntimeError:
#                     assert not self.training
#                     d_weight = torch.tensor(0.0)
#
#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#             loss = nll_loss + self.codebook_weight * (codebook_loss.mean() if len(codebook_loss.shape) > 0 else codebook_loss)
#
#             if not no_dis:
#                 loss += d_weight * disc_factor * g_loss
#
#             log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
#                    "{}/quant_loss".format(split): (codebook_loss.detach().mean() if len(codebook_loss.shape) > 0 else codebook_loss),
#                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
#                    "{}/p_loss".format(split): p_loss.detach().mean(),
#                    # "{}/disparity_loss".format(split): disparity_loss.detach(),
#                    # "{}/rgb_loss".format(split): rgb_loss.detach(),
#                    }
#             if not no_dis:
#                 log["{}/g_loss".format(split)] = g_loss.detach().mean()
#                 log["{}/d_weight".format(split)] = d_weight.detach()
#                 log["{}/disc_factor".format(split)] = torch.tensor(disc_factor)
#             return loss, log
#
#         if optimizer_idx == 1:
#             # second pass for discriminator update
#             # if extrapolation_mask is not None:
#             #     if self.dataset == 'kitti360':
#             #         reshape_size = (6, 30)
#             #     elif self.dataset == 'google_earth':
#             #         reshape_size = (6, 6)
#             #     else:
#             #         raise NotImplementedError
#                 # extrapolation_mask_weight = F.interpolate(extrapolation_mask.float(), size=reshape_size)
#
#             # inputs[:, :, :, :220] = reconstructions[:, :, :, :220]
#             self.discriminator.train()
#             if cond is None:
#                 logits_real = self.discriminator(inputs.contiguous().detach()) #* (extrapolation_mask_weight if extrapolation_mask is not None else 1)
#                 logits_fake = self.discriminator(reconstructions.contiguous().detach()) #* (extrapolation_mask_weight if extrapolation_mask is not None else 1)
#             else:
#                 logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1)) #* (extrapolation_mask_weight if extrapolation_mask is not None else 1)
#                 logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1)) #* (extrapolation_mask_weight if extrapolation_mask is not None else 1)
#
#             disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
#             if global_step % self.disc_update_every_n_step != 0:
#                 disc_factor = disc_factor * 0.
#
#             d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
#
#             # inputs[:, :3] = torch.clip((inputs[:, :3] + 1) / 2, 0, 1)
#             # reconstructions[:, :3] = torch.clip((reconstructions[:, :3] + 1) / 2, 0, 1)
#             # import numpy as np
#             # import matplotlib.pyplot as plt
#             # pred = torch.clip(reconstructions[0, :3] * 1, 0, 1).permute(1, 2, 0).detach().cpu().numpy().astype(
#             #     np.float32)
#             # plt.imshow(pred)
#             # plt.title('pred')
#             # # plt.show()
#             # plt.savefig('meeting_assets/pred.png')
#
#             # plt.imshow(logits_fake[0][0].detach().cpu(), cmap='gray')
#             # plt.title('logits_fake')
#             # # plt.show()
#             # plt.savefig('meeting_assets/logits_fake.png')
#
#             # gt = torch.clip(inputs[0, :3] * 1, 0, 1).permute(1, 2, 0).detach().cpu().numpy().astype(
#             #     np.float32)
#             # plt.imshow(gt)
#             # plt.title('gt')
#             # # plt.show()
#             # plt.savefig('meeting_assets/gt.png')
#
#
#             # plt.imshow(logits_real[0][0].detach().cpu(), cmap='gray')
#             # plt.title('logits_real')
#             # # plt.show()
#             # plt.savefig('meeting_assets/logits_real.png')
#             #
#             # plt.imshow(extrapolation_mask[0][0].detach().cpu())
#             # plt.title('extrapolation_mask')
#             # # plt.show()
#             # plt.savefig('meeting_assets/extrapolation_mask.png')
#
#             log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
#                    "{}/logits_real".format(split): logits_real.detach().mean(),
#                    "{}/logits_fake".format(split): logits_fake.detach().mean()
#                    }
#             return d_loss, log