# SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping
# Authored by Yuan Shen, Wei-Chiu Ma and Shenlong Wang
# University of Illinois at Urbana-Champaign and Massachusetts Institute of Technology
import copy

import torch.nn.functional as F
import pytorch_lightning as pl
from data.utils.utils import instantiate_from_config
from sgam.point_rendering.warp import render_projection_from_srcs_fast
from sgam.generative_sensing_module.modules.diffusionmodules.model import Encoder, Decoder
from sgam.generative_sensing_module.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from sgam.generative_sensing_module.modules.misc.metrics import *
from sgam.generative_sensing_module.modules.losses.lpips import LPIPS
import torch
from scipy.cluster.vq import kmeans2


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 data_config,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 phase=None,
                 ckpt_path=None,
                 ignore_keys=['loss.discriminator'],
                 image_key="image",
                 colorize_nlabels=None,
                 logdir=None,
                 use_extrapolation_mask=True,
                 vq_step_threshold=0,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 online_kmeans_config=None,
                 batch_size=None,
                 depth_range=None
                 ):
        super().__init__()
        self.phase = phase
        self.online_kmeans_config = online_kmeans_config
        self.data_config = data_config
        self.logdir = logdir
        self.depth_range = depth_range
        self.n_embed = n_embed
        self.do_online_kmeans_clustering = online_kmeans_config['do_online_kmeans_clustering']
        self.use_extrapolation_mask = use_extrapolation_mask
        self.vq_step_threshold = vq_step_threshold
        self.automatic_optimization = False
        self.image_key = image_key
        self.perceptual_loss = LPIPS().eval()
        if self.use_extrapolation_mask:
            self.conv_in = torch.nn.Conv2d(5, 4, kernel_size=1)
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape,
                                        kmean_init_codebook_path=online_kmeans_config['kmean_init_codebook_path'])

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if self.do_online_kmeans_clustering:
            self.val_codebook_map = None
            self.train_codebook_map = {}
            for i in range(self.n_embed):
                self.train_codebook_map[i] = self.online_kmeans_config["online_kmeans_word_timeout"]
            self.train_sampled_feature_maps = []
        self.criterionFeat = torch.nn.L1Loss()

    def use_vq(self):
        # print('use_vq', self.global_step, self.vq_step_threshold, self.global_step >= self.vq_step_threshold)
        return self.global_step >= self.vq_step_threshold

    def init_from_ckpt(self, path, ignore_keys=['loss'], only_keep_keys=[]):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                #print(ik, k)
                if k.startswith(ik):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        keys = list(sd.keys())
        for k in keys:
            for ik in only_keep_keys:
                #print(ik, k)
                if not ik in k:
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, topk=None,  encoding_indices=None, extrapolation_mask=None, use_old=False, sample_number=1):
        if self.use_extrapolation_mask:
            if extrapolation_mask is not None:
                x = torch.cat([x, extrapolation_mask], 1)
            else:
                extrapolation_mask = torch.zeros([x.shape[0], 1, *x.shape[2:]]).to(x.device)
                x = torch.cat([x, extrapolation_mask], 1)
            x = self.conv_in(x)

        h = self.encoder(x)
        pre_quantized_f = self.quant_conv(h)
        if not self.use_vq():
            return pre_quantized_f
        else:
            if topk is None:
                quant, emb_loss, info = self.quantize(pre_quantized_f, encoding_indices=encoding_indices)
            else:
                quant, emb_loss, info = self.quantize.get_multiple_codewords(pre_quantized_f, topk, sample_number, extrapolation_mask)
            return quant, emb_loss, info, pre_quantized_f

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, topk=None, extrapolation_mask=None, sample_number=1, get_codebook_count=False, get_pre_quantized_feature=False, get_quantized_feature=False):
        res = self.encode(input, topk=topk,
                          encoding_indices=None, extrapolation_mask=extrapolation_mask, sample_number=sample_number)
        if not self.use_vq():
            pre_quant = res
            dec = self.decode(pre_quant)
            return dec, torch.tensor(0).to(dec.device), pre_quant
        else:
            if topk is None:
                quants, diff, info, pre_quant = res
                decoder_input = quants
                decs = self.decode(decoder_input)
            else:
                quants, diff, info, pre_quant = res
                decoder_inputs = quants
                decs = []
                for i in range(sample_number):
                    dec = self.decode(decoder_inputs[:, i])
                    decs.append(dec[None,])
            res = [decs, diff]
            if get_codebook_count:
                res.append(info[-1] if len(info) else {})
            if get_pre_quantized_feature:
                res.append(pre_quant)
            if get_quantized_feature:
                res.append(quants)
            return res

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        elif len(x.shape) == 5:
            x = x.permute(0, 1, 4, 2, 3).to(memory_format=torch.contiguous_format)
        return x

    def get_x(self, batch, dataset, return_extrapolation_mask=False, no_depth_range=False, parallel=True):
        if batch["dst_img"].device != self.device:
            for k in batch:
                if hasattr(batch[k], "to"):
                    batch[k] = batch[k].to(device=self.device)

        x_dst = self.get_input("dst_img", batch)
        x_depth = self.get_input("dst_depth", batch)
        x_src = self.get_input("src_imgs", batch)
        dm_src = self.get_input("src_depths", batch)
        bs, src_n = batch["R_rels"].shape[:2]
        R_rels = batch["R_rels"].view(-1, *batch["R_rels"].shape[2:])
        t_rels = batch["t_rels"].view(-1, *batch["t_rels"].shape[2:])
        T_src2tgt = torch.eye(4)[None, ].repeat(R_rels.shape[0], 1, 1)
        T_src2tgt[:, :3, :3] = R_rels
        T_src2tgt[:, :3, 3] = t_rels
        T_src2tgt = T_src2tgt.view(bs, src_n, *T_src2tgt.shape[1:]).to(self.device)
        if 'warped_tgt_features' in batch:
            x = batch['warped_tgt_features']
            warped_depth = batch['warped_tgt_depth'][:, None]
            extrapolation_mask = (warped_depth <= 0)
        else:
            warped_depth, x, extrapolation_mask, mask, cur_fused_features, idx, projected_features = render_projection_from_srcs_fast(x_src,
                                                            dm_src[:, :, 0],
                                                            batch["Ks"][:, 0].to(self.device),
                                                            batch["Ks"].to(self.device),
                                                            T_src2tgt,
                                                            src_num=x_src.shape[1],
                                                            dynamic_masks=None,
                                                            depth_range=self.depth_range if not no_depth_range else None,
                                                            parallel=parallel)
        if dataset == 'google_earth':
            x_inverse_depth = 1 / (x_depth + 10)
            x_inverse_depth = (x_inverse_depth - 1 / 14.765625) / (1 / 10.099975586 - 1 / 14.765625)
            x_scaled_inverse_depth = 2 * x_inverse_depth - 1

            warped_depth = 1 / (warped_depth + 10)
            warped_depth = (warped_depth - 1 / 14.765625) / (1 / 10.099975586 - 1 / 14.765625)
            warped_depth = 2 * warped_depth - 1
            warped_depth = warped_depth * ~extrapolation_mask + torch.ones_like(warped_depth) * (
                -2) * extrapolation_mask
        elif dataset == 'clevr-infinite':
            x_inverse_depth = 1 / x_depth  # between (1/16, 1/7)
            x_inverse_depth = (x_inverse_depth - 1 / 16) / (1 / 7 - 1 / 16)
            x_scaled_inverse_depth = 2 * x_inverse_depth - 1

            warped_depth = 1 / torch.clip(warped_depth, 1e-7)  # between (1/16, 1/7)
            warped_depth = (warped_depth - 1 / 16) / (1 / 7 - 1 / 16)
            warped_depth = 2 * warped_depth - 1
            warped_depth = warped_depth * ~extrapolation_mask + torch.ones_like(warped_depth) * (
                -2) * extrapolation_mask
        else:
            raise NotImplementedError
        # import matplotlib.pyplot as plt
        # plt.imshow((x[0][:3].permute(1, 2, 0).detach().cpu().numpy() + 1) / 2)
        # plt.title('warped_src')
        # plt.show()

        x = torch.cat([x, warped_depth], 1)
        x_dst = torch.cat([x_dst, x_scaled_inverse_depth], 1)


        # import matplotlib.pyplot as plt
        # plt.imshow(x[0][3].detach().cpu().numpy(), cmap='gray')
        # plt.title('warped_depth')
        # plt.show()

        # plt.imshow((cs[0][:3].detach().permute(1, 2, 0).cpu().numpy() + 1)/2)
        # plt.show()
        # plt.imshow(warped_depth[0].detach().permute(1, 2, 0).cpu().numpy())
        # plt.show()
        #
        # import matplotlib.pyplot as plt
        # plt.imshow((batch['dst_img'][0].squeeze().detach().cpu().numpy() + 0.5)/2)
        # plt.title('dst_gt')
        # plt.show()
        #
        # plt.imshow((cs[0][:3].permute(1, 2, 0).detach().cpu().numpy() + 0.5)/2)
        # plt.title('warped_src')
        # plt.show()
        # # warped_depth[warped_depth == -1] = 10
        # plt.imshow(warped_depth[0][0].detach().cpu().numpy())
        # plt.title('warped_depth')
        # plt.show()
        # plt.imshow(batch['dst_depth'][0].detach().cpu().numpy())
        # plt.title('dst_depth_gt')
        # plt.show()
        if return_extrapolation_mask:
            return x, x_dst, extrapolation_mask, warped_depth
        else:
            return x, x_dst

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()

        if self.do_online_kmeans_clustering and self.global_step >= self.online_kmeans_config["start_global_step"]:

            if self.global_rank == 0:
                inactive_codeword_indices = []
                for k, v in self.train_codebook_map.items():
                    if v <= 0:
                        inactive_codeword_indices.append(k)
                if len(inactive_codeword_indices) / len(self.train_codebook_map) > self.online_kmeans_config["inactive_threshold"] \
                        and len(self.train_sampled_feature_maps) >= self.online_kmeans_config["train_feature_buffer_size"]\
                        and self.global_step % self.online_kmeans_config["frequency"] == 0:
                    #        and batch_idx % self.update_frequency == self.update_frequency - 1:
                    train_sampled_feature_maps = np.stack(self.train_sampled_feature_maps)
                    train_sampled_feature_maps = train_sampled_feature_maps.transpose(0, 2, 3, 1)
                    train_sampled_feature_maps = train_sampled_feature_maps.reshape(-1,
                                                                                    train_sampled_feature_maps.shape[
                                                                                        -1])
                    kd = kmeans2(train_sampled_feature_maps, len(inactive_codeword_indices), minit='points')[0]
                    self.quantize.update_codebook(kd, inactive_codeword_indices)
                    for inactive_codeword_index in inactive_codeword_indices:
                        self.train_codebook_map[inactive_codeword_index] = self.online_kmeans_config["online_kmeans_word_timeout"]
                    print('update at batch', batch_idx)
                self.log("train/codebook active percentage", 1- (len(inactive_codeword_indices) / len(self.train_codebook_map)))
        if self.phase == 'conditional_generation':
            x, x_dst, extrapolation_mask, _ = self.get_x(batch, self.data_config.dataset, return_extrapolation_mask=True)
            xrec, qloss, codebook_indices, pre_quantized_features = self(x, extrapolation_mask=extrapolation_mask,
                                                                         get_codebook_count=True,
                                                                         get_pre_quantized_feature=True)
        elif self.phase == 'codebook':
            x = self.get_input(self.image_key, batch)
            extrapolation_mask = None
            if self.use_vq():
                xrec, qloss, codebook_indices, \
                pre_quantized_features = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True,
                                                                             get_pre_quantized_feature=True)
            else:
                xrec, qloss, pre_quantized_features = self(x, extrapolation_mask, get_codebook_count=False, get_pre_quantized_feature=True)
                codebook_indices = None
            x_dst = x

            if codebook_indices is not None and len(codebook_indices) and self.global_rank == 0 and self.do_online_kmeans_clustering and self.global_step >= self.online_kmeans_config["start_global_step"]:
                values, counts = np.unique(codebook_indices[0].flatten().detach().cpu().numpy(), return_counts=True)
                for i in range(len(values)):
                    v = values[i]
                    self.train_codebook_map[v] = self.online_kmeans_config["online_kmeans_word_timeout"]
                if len(self.train_sampled_feature_maps) > self.online_kmeans_config["train_feature_buffer_size"]:
                    self.train_sampled_feature_maps = self.train_sampled_feature_maps[-self.online_kmeans_config["train_feature_buffer_size"]:]
                self.train_sampled_feature_maps.append(pre_quantized_features[0].detach().cpu().numpy())

                for k in self.train_codebook_map.keys():
                    self.train_codebook_map[k] -= 1
        else:
            raise NotImplementedError

        aeloss, log_dict_ae = self.loss(qloss, x_dst, xrec, 0, self.global_step, extrapolation_mask=extrapolation_mask,
                                        last_layer=self.get_last_layer(), split="train")

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        opt_ae.zero_grad()
        aeloss.backward()
        opt_ae.step()

        discloss, log_dict_disc = self.loss(qloss, x_dst, xrec, 1, self.global_step,
                                        last_layer=self.get_last_layer(), split="train", extrapolation_mask=extrapolation_mask)
        opt_disc.zero_grad()
        discloss.backward()
        opt_disc.step()
        self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

    def test_step(self, batch, batch_idx):
        self.evaluation_loop(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.evaluation_epoch(outputs)

    def validation_step(self, batch, batch_idx):
        self.evaluation_loop(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        self.evaluation_epoch(outputs)

    def evaluation_loop(self, batch, batch_idx):
        if batch_idx == 0 and self.global_rank == 0 and self.do_online_kmeans_clustering and self.global_step >= self.online_kmeans_config["start_global_step"]:
            self.val_codebook_map = {}
            for i in range(self.n_embed):
                self.val_codebook_map[i] = 0

        if self.phase == 'conditional_generation':
            x, x_dst, extrapolation_mask, _ = self.get_x(batch, self.data_config.dataset, return_extrapolation_mask=True)
            xrec, qloss, pre_quantized_features = self(x, extrapolation_mask=extrapolation_mask, get_pre_quantized_feature=True)
        elif self.phase == 'codebook':
            x = self.get_input(self.image_key, batch)
            extrapolation_mask = None
            if self.use_vq():
                xrec, qloss, codebook_indices, pre_quantized_features = self(x, get_codebook_count=True,
                                                                             get_pre_quantized_feature=True)
                if len(codebook_indices) and self.global_rank == 0 and self.do_online_kmeans_clustering and self.global_step >= \
                        self.online_kmeans_config["start_global_step"]:
                    values, counts = np.unique(codebook_indices[0].flatten().detach().cpu().numpy(), return_counts=True)
                    for i in range(len(values)):
                        v = values[i]
                        self.val_codebook_map[v] += counts[i]
            else:
                xrec, qloss = self(x, get_codebook_count=False, get_pre_quantized_feature=False)[:2]
            x_dst = x
        else:
            raise NotImplementedError

        aeloss, log_dict_ae = self.loss(qloss, x_dst, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val",
                                        extrapolation_mask=extrapolation_mask)
        discloss, log_dict_disc = self.loss(qloss, x_dst, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val",
                                            extrapolation_mask=extrapolation_mask)
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(log_dict_disc, rank_zero_only=True)

        rgb_l1 = F.l1_loss(xrec[:, :3], x_dst[:, :3])
        disparity_l1 = F.l1_loss(xrec[:, 3:], x_dst[:, 3:])

        self.log("val/rgb_l1", rgb_l1,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/disparity_l1", disparity_l1,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        del log_dict_ae["val/rec_loss"]
        self.log_dict(log_dict_ae)
        return self.log_dict

    def evaluation_epoch(self, outputs):
        if self.phase == 'codebook' and self.do_online_kmeans_clustering and self.online_kmeans_config["start_global_step"] and self.val_codebook_map is not None:
            self.log("val/codebook_active_percentage", sum(np.array(list(self.val_codebook_map.values())) > 0) / len(self.val_codebook_map.values()))

    def configure_optimizers(self):
        lr = self.learning_rate
        if self.phase == 'codebook':
            opt_ae_parameters = list(self.encoder.parameters()) + \
                                      list(self.decoder.parameters()) + \
                                      list(self.quantize.parameters()) + \
                                      list(self.quant_conv.parameters()) + \
                                      list(self.post_quant_conv.parameters())
            if self.use_extrapolation_mask:
                opt_ae_parameters = opt_ae_parameters + list(self.conv_in.parameters())
            opt_ae = torch.optim.Adam(opt_ae_parameters,
                                      lr=lr, betas=(0.5, 0.9))
        elif self.phase == 'conditional_generation':
            opt_ae = torch.optim.Adam(list(self.encoder.parameters()) + list(self.conv_in.parameters()) if self.use_extrapolation_mask else
                                      list(self.encoder.parameters()),
                                       lr=lr, betas=(0.5, 0.9))
        else:
            raise NotImplementedError
        if self.loss.use_discriminative_loss:
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
            return opt_ae, opt_disc #[opt_ae, opt_disc], []
        else:
            return opt_ae #[opt_ae, ], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        if self.phase == 'conditional_generation':
            x, x_dst, extrapolation_mask, _ = self.get_x(batch, self.data_config.dataset, return_extrapolation_mask=True)
            xrec = self(x, extrapolation_mask=extrapolation_mask)[0]
        elif self.phase == 'codebook':
            x = self.get_input(self.image_key, batch)
            xrec = self(x, get_codebook_count=True,  get_pre_quantized_feature=True)[0]
            x_dst = x
        else:
            raise NotImplementedError

        if x.shape[1] > 3:
            input_disparity = x[:, 3:]
            input_rgb = x[:, :3]
            log["warped_input"] = input_rgb
            log["warped_disparity"] = input_disparity
            log["reconstructions"] = xrec[:, :3]
            log["reconstruction_disparities"] = xrec[:, 3:]
            log["gt_rgb"] = x_dst[:, :3]
            log["gt_disparity"] = x_dst[:, 3:]
        else:
            log["inputs"] = x
            log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x
