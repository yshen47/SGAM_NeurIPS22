import copy

import torch.nn.functional as F
import pytorch_lightning as pl
from data.utils.utils import instantiate_from_config
from sgam.point_rendering.warp import render_projection_from_srcs_fast
from sgam.generative_sensing_module.modules.diffusionmodules.model import Encoder, Decoder
from sgam.generative_sensing_module.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from sgam.generative_sensing_module.modules.misc.metrics import *
from sgam.generative_sensing_module.modules.losses.lpips import LPIPS
import PIL
import torch
from scipy.cluster.vq import kmeans2
from transformers import ViTMAEForPreTraining, ViTMAEConfig
from .customized_vit_decoder import ViTMAEDecoder


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 data_config,
                 lossconfig,
                 mae_config,
                 n_embed,
                 embed_dim,
                 phase=None,
                 ckpt_path=None,
                 mae_ckpt_path=None,
                 ignore_keys=['loss.discriminator'],
                 image_key="image",
                 colorize_nlabels=None,
                 use_cycle_consistency_loss=False,
                 logdir=None,
                 use_extrapolation_mask=True,
                 vq_step_threshold=0,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 online_kmeans_config=None,
                 test_time_optimization_config=None,
                 batch_size=None,
                 depth_range=None
                 ):
        super().__init__()
        self.phase = phase
        self.use_cycle_consistency_loss = use_cycle_consistency_loss
        self.online_kmeans_config = online_kmeans_config
        self.data_config = data_config
        self.test_time_optimization_config = test_time_optimization_config
        if test_time_optimization_config['use_test_time_optimization']:
            self.optimization_iteration_num = test_time_optimization_config['optimization_iteration_count']
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
        if self.phase == 'test_time_optimization':
            self.predicted_image_features = torch.zeros(batch_size, 1, 256)
            self.predicted_image_features.requires_grad = True

        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if 'mae' in self.phase:
            configuration = ViTMAEConfig(image_size=list(self.data_config.image_resolution), num_channels=5,
                                         mask_ratio=mae_config.mask_ratio)
            if data_config.dataset == 'kitti360':
                configuration.patch_size = 4
            configuration.token_id_embedding_size = mae_config.token_id_embedding_size
            self.mae_transformer = ViTMAEForPreTraining(configuration)
            self.mae_transformer.decoder = ViTMAEDecoder(configuration, num_patches=self.mae_transformer.vit.embeddings.num_patches)
            self.mae_linear = torch.nn.Linear(configuration.patch_size * configuration.patch_size * configuration.num_channels,
                                              self.n_embed)
            self.mae_config = mae_config
            self.token_id_embedding = torch.nn.Embedding(self.n_embed, self.mae_config.token_id_embedding_size)
            self.quant_conv.eval()
            self.post_quant_conv.eval()
            self.encoder.eval()
            self.decoder.eval()
            self.quantize.eval()
        if mae_ckpt_path is not None:
            self.init_from_ckpt(mae_ckpt_path, ignore_keys=ignore_keys)
        elif ckpt_path is not None:
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
        # Ks = batch["Ks"].view(-1, *batch["Ks"].shape[2:])
        # x_src = x_src.view(-1, *x_src.shape[2:])
        # dm_src = dm_src.view(-1, *dm_src.shape[2:])
        # orig_x_src = copy.deepcopy(x_src)
        # orig_dm_src = copy.deepcopy(dm_src[:, :, 0])
        # orig_batch_K = copy.deepcopy(batch["Ks"][:, 0])
        # orig_T_src2tgt = copy.deepcopy(T_src2tgt)
        if 'warped_tgt_features' in batch:
            x = batch['warped_tgt_features']
            warped_depth = batch['warped_tgt_depth'][:, None]
            extrapolation_mask = (warped_depth <= 0)
        else:
            # if dataset != 'blender':
            warped_depth, x, extrapolation_mask, mask, cur_fused_features, idx, projected_features = render_projection_from_srcs_fast(x_src,
                                                            dm_src[:, :, 0],
                                                            batch["Ks"][:, 0].to(self.device),
                                                            batch["Ks"].to(self.device),
                                                            T_src2tgt,
                                                            src_num=x_src.shape[1],
                                                            dynamic_masks=None,
                                                            depth_range=self.depth_range if not no_depth_range else None,
                                                            parallel=parallel) #batch["dynamic_mask_srcs"].to(self.device))
            # else:
            #     warped_depth, x = render_projection_from_srcs(x_src,
            #                                                   dm_src[:, :, 0],
            #                                                   batch["Ks"][:, 0].to(self.device),
            #                                                   batch["Ks"].to(self.device),
            #                                                   T_src2tgt)
            #     extrapolation_mask = (warped_depth <= 0)
        if dataset == 'kitti360':
            x_depth = torch.clip(x_depth, 3, 75)
            x_inverse_depth = 1 / x_depth
            x_inverse_depth = (x_inverse_depth - 1 / 75) / (1 / 3 - 1 / 75)
            x_scaled_inverse_depth = 2 * x_inverse_depth - 1

            warped_depth = torch.clip(warped_depth, 3, 75)
            warped_depth = 1 / warped_depth
            warped_depth = (warped_depth - 1 / 75) / (1 / 3 - 1 / 75)
            warped_depth = 2 * warped_depth - 1
            warped_depth = warped_depth * ~extrapolation_mask

        elif dataset == 'google_earth':
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
                #print(len(self.train_feature_buffer))
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
        elif self.phase == 'mae_reconstruction_training':
            with torch.no_grad():
                x = self.get_input(self.image_key, batch)
                extrapolation_mask = None
                x_dst = x
                # get vqgan token
                xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True, get_pre_quantized_feature=False)
            metadata = self.mae_forward(pixel_values=x)
            logits = self.mae_linear(metadata['logits'])
            token_loss = torch.nn.functional.cross_entropy(logits.view(-1, self.n_embed), codebook_indices.flatten())
        elif self.phase == 'mae_outpainting_training':
            with torch.no_grad():
                x, x_dst, extrapolation_mask, _ = self.get_x(batch, self.data_config.dataset, return_extrapolation_mask=True, parallel=True)
                # get vqgan token
                xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True, get_pre_quantized_feature=False)
            metadata = self.mae_forward(pixel_values=x, vqgan_token_ids=codebook_indices, extrapolation_mask=extrapolation_mask)
            logits = self.mae_linear(metadata['logits'])
            gt_codebook_indices = self(x_dst, extrapolation_mask=None, get_codebook_count=True, get_pre_quantized_feature=False)[2]
            token_loss = torch.nn.functional.cross_entropy(logits.view(-1, self.n_embed), gt_codebook_indices.flatten())

            if self.use_cycle_consistency_loss:
                T_src2tgts = torch.eye(4)[None, None].repeat(batch['R_rels'].shape[0], 1, 1, 1)
                T_src2tgts[:, :, :3] = torch.cat([batch['R_rels'], batch['t_rels'][..., None]], -1)
                T_src2tgts = torch.inverse(T_src2tgts)

                reverse_batch = {
                    't_rels': T_src2tgts[:, :, :3, 3],
                    'R_rels': T_src2tgts[:, :, :3, :3],
                    "Ks": batch["Ks"],
                    "src_imgs": xrec[:, :3][:, None].permute(0, 1, 3, 4, 2).detach(),
                    "dst_img": batch['src_imgs'][:, 0],
                    "src_depths": xrec[:, 3][:, None].permute(0, 2, 3, 1)[:, None].detach(),
                    "dst_depth": batch['src_depths'][:, 0],
                }

                if self.data_config.dataset == 'clevr-infinite':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (1 / 7 - 1 / 16) + 1 / 16)).detach()
                elif self.data_config.dataset == 'kitti360':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75)).detach()
                elif self.data_config.dataset == 'google_earth':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (
                                1 / 10.099975586 - 1 / 14.765625) + 1 / 14.765625) - 10)
                else:
                    raise NotImplementedError

                with torch.no_grad():
                    x, x_dst, extrapolation_mask, _ = self.get_x(reverse_batch, self.data_config.dataset,
                                                                 return_extrapolation_mask=True, parallel=True)
                    # get vqgan token
                    xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask,
                                                         get_codebook_count=True, get_pre_quantized_feature=False)
                metadata = self.mae_forward(pixel_values=x, vqgan_token_ids=codebook_indices,
                                            extrapolation_mask=extrapolation_mask)
                logits = self.mae_linear(metadata['logits'])
                gt_codebook_indices = \
                self(x_dst, extrapolation_mask=None, get_codebook_count=True, get_pre_quantized_feature=False)[2]
                token_loss = token_loss + torch.nn.functional.cross_entropy(logits.view(-1, self.n_embed),
                                                               gt_codebook_indices.flatten())
        else:
            raise NotImplementedError

        if 'mae' in self.phase:
            opt_ae.zero_grad()
            token_loss.backward()
            opt_ae.step()
            self.log("train/token_cross_entropy_loss", token_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        else:
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
        elif self.phase == 'mae_reconstruction_training':
            x = self.get_input(self.image_key, batch)
            extrapolation_mask = None
            x_dst = x
            # get vqgan token
            xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True, get_pre_quantized_feature=False)
            metadata = self.mae_forward(pixel_values=x)
            logits = self.mae_linear(metadata['logits'])
            token_loss = torch.nn.functional.cross_entropy(logits.view(-1, self.n_embed), codebook_indices.flatten())
        elif self.phase == 'mae_outpainting_training':
            x, x_dst, extrapolation_mask, _ = self.get_x(batch, self.data_config.dataset, return_extrapolation_mask=True, parallel=True)
            # get vqgan token
            xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True, get_pre_quantized_feature=False)
            metadata = self.mae_forward(pixel_values=x, vqgan_token_ids=codebook_indices, extrapolation_mask=extrapolation_mask)
            logits = self.mae_linear(metadata['logits'])
            gt_codebook_indices = self(x_dst, extrapolation_mask=None, get_codebook_count=True, get_pre_quantized_feature=False)[2]
            token_loss = torch.nn.functional.cross_entropy(logits.view(-1, self.n_embed), gt_codebook_indices.flatten())

            if self.use_cycle_consistency_loss:
                T_src2tgts = torch.eye(4)[None, None].repeat(batch['R_rels'].shape[0], 1, 1, 1)
                T_src2tgts[:, :, :3] = torch.cat([batch['R_rels'], batch['t_rels'][..., None]], -1)
                T_src2tgts = torch.inverse(T_src2tgts)

                reverse_batch = {
                    't_rels': T_src2tgts[:, :, :3, 3],
                    'R_rels': T_src2tgts[:, :, :3, :3],
                    "Ks": batch["Ks"],
                    "src_imgs": xrec[:, :3][:, None].permute(0, 1, 3, 4, 2).detach(),
                    "dst_img": batch['src_imgs'][:, 0],
                    "src_depths": xrec[:, 3][:, None].permute(0, 2, 3, 1)[:, None].detach(),
                    "dst_depth": batch['src_depths'][:, 0],
                }

                if self.data_config.dataset == 'clevr-infinite':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (1 / 7 - 1 / 16) + 1 / 16)).detach()
                elif self.data_config.dataset == 'kitti360':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75)).detach()
                elif self.data_config.dataset == 'google_earth':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (
                                1 / 10.099975586 - 1 / 14.765625) + 1 / 14.765625) - 10)
                else:
                    raise NotImplementedError

                with torch.no_grad():
                    x, x_dst, extrapolation_mask, _ = self.get_x(reverse_batch, self.data_config.dataset,
                                                                 return_extrapolation_mask=True, parallel=True)
                    # get vqgan token
                    xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask,
                                                         get_codebook_count=True, get_pre_quantized_feature=False)
                metadata = self.mae_forward(pixel_values=x, vqgan_token_ids=codebook_indices,
                                            extrapolation_mask=extrapolation_mask)
                logits = self.mae_linear(metadata['logits'])
                gt_codebook_indices = \
                self(x_dst, extrapolation_mask=None, get_codebook_count=True, get_pre_quantized_feature=False)[2]
                token_loss = token_loss + torch.nn.functional.cross_entropy(logits.view(-1, self.n_embed),
                                                               gt_codebook_indices.flatten())
        else:
            raise NotImplementedError

        if 'mae' in self.phase:
            self.log("val/token_cross_entropy_loss", token_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)
        else:
            if self.test_time_optimization_config['use_test_time_optimization'] and self.phase == 'conditional_generation':
                pre_quantized_features = torch.from_numpy(pre_quantized_features.detach().cpu().numpy()).to(pre_quantized_features.device)
                pre_quantized_features.requires_grad = True
                opt_features = torch.optim.Adam([pre_quantized_features],
                                          lr=self.test_time_optimization_config['learning_rate'], betas=(0.5, 0.9))
                torch.set_grad_enabled(True)
                self.train()
                for i in range(self.optimization_iteration_num):
                    if not self.use_vq():
                        xrec_optimized = self.decode(pre_quantized_features)
                        aeloss, log_dict_ae = self.loss(None, x, xrec_optimized, 0, self.global_step,
                                                        last_layer=self.get_last_layer(), split="test_optimization",
                                                        extrapolation_mask=extrapolation_mask, recon_on_visible=True)
                    else:
                        quant, emb_loss, info = self.quantize(pre_quantized_features)
                        xrec_optimized = self.decode(quant)
                        aeloss, log_dict_ae = self.loss(emb_loss, x, xrec_optimized, 0, self.global_step,
                                                        last_layer=self.get_last_layer(), split="test_optimization",
                                                        extrapolation_mask=extrapolation_mask, recon_on_visible=True)

                    x_rec_features = self.encode(xrec_optimized, encoding_indices=None, extrapolation_mask=extrapolation_mask)[-1]
                    feature_loss = F.l1_loss(x_rec_features, pre_quantized_features)
                    total_loss = aeloss + feature_loss
                    opt_features.zero_grad()
                    total_loss.backward()
                    opt_features.step()
                torch.set_grad_enabled(False)
                self.eval()

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
            ssim = SSIM()
            psnr = PSNR()
            ssim_score = 0
            psnr_score = 0
            ssim_visible_score = 0
            psnr_visible_score = 0

            optimized_ssim_score = 0
            optimized_psnr_score = 0
            optimized_ssim_visible_score = 0
            optimized_psnr_visible_score = 0
            # if self.global_rank == 0:
            #     os.makedirs(os.path.join(self.logdir, f'qualitative_res/pred'), exist_ok=True)
            #     os.makedirs(os.path.join(self.logdir, f'qualitative_res/warped_inputs'), exist_ok=True)
            #
            #     os.makedirs(os.path.join(self.logdir, f'qualitative_res/gt'), exist_ok=True)
            #     os.makedirs(os.path.join(self.logdir, f'qualitative_res/extrapolation_mask'), exist_ok=True)
            p_loss = self.perceptual_loss(x_dst[:, :3].contiguous().float(),
                                          xrec[:, :3].contiguous().float())
            x[:, :3] = torch.clip((x[:, :3] + 1) / 2, 0, 1)
            xrec[:, :3] = torch.clip((xrec[:, :3] + 1) / 2, 0, 1)
            x_dst[:, :3] = torch.clip((x_dst[:, :3] + 1) / 2, 0, 1)

            if self.test_time_optimization_config['use_test_time_optimization'] and self.phase == 'conditional_generation':
                os.makedirs(os.path.join(self.logdir, f'qualitative_res/pred_optimized'), exist_ok=True)
                xrec_optimized[:, :3] = torch.clip((xrec_optimized[:, :3] + 1) / 2, 0, 1)

            for i in range(xrec.shape[0]):
                warped = torch.clip(x[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).astype(np.float32)

                pred = torch.clip(xrec[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).astype(np.float32)
                gt = torch.clip(x_dst[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).astype(np.float32)

                res = ssim(pred, gt, ~extrapolation_mask[i].permute(1, 2, 0).repeat(1, 1, 3)
                                                    .detach().cpu().numpy() if extrapolation_mask is not None else None)
                if extrapolation_mask is None:
                    curr_ssim = res
                else:
                    curr_ssim, curr_visible_ssim = res
                    ssim_visible_score += curr_visible_ssim

                res = psnr(pred, gt, ~extrapolation_mask[i].permute(1, 2, 0).repeat(1, 1, 3)
                                                     .detach().cpu().numpy() if extrapolation_mask is not None else None)
                if extrapolation_mask is None:
                    curr_psnr = res
                else:
                    curr_psnr, curr_visilble_psnr = res
                    psnr_visible_score += curr_visilble_psnr

                ssim_score += curr_ssim
                psnr_score += curr_psnr

                if self.test_time_optimization_config[
                    'use_test_time_optimization'] and self.phase == 'conditional_generation':
                    pred_optimized = torch.clip(xrec_optimized[i, :3] * 255, 0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8).astype(np.float32)

                    res = ssim(pred_optimized, gt, ~extrapolation_mask[i].permute(1, 2, 0).repeat(1, 1, 3)
                               .detach().cpu().numpy() if extrapolation_mask is not None else None)
                    if extrapolation_mask is None:
                        curr_ssim = res
                    else:
                        curr_ssim, curr_visible_ssim = res
                        optimized_ssim_visible_score += curr_visible_ssim

                    res = psnr(pred_optimized, gt, ~extrapolation_mask[i].permute(1, 2, 0).repeat(1, 1, 3)
                               .detach().cpu().numpy() if extrapolation_mask is not None else None)
                    if extrapolation_mask is None:
                        curr_psnr = res
                    else:
                        curr_psnr, curr_visilble_psnr = res
                        optimized_psnr_visible_score += curr_visilble_psnr

                    optimized_ssim_score += curr_ssim
                    optimized_psnr_score += curr_psnr
                    if self.global_rank == 0:
                        PIL.Image.fromarray(pred_optimized.astype(np.uint8)).save(
                            os.path.join(self.logdir, f'qualitative_res/pred_optimized/batch_{batch_idx}_index_{i}.png'))
                if extrapolation_mask is not None and self.global_rank == 0:
                    np.save(os.path.join(self.logdir, f'qualitative_res/extrapolation_mask/batch_{batch_idx}_index_{i}.png'),
                            ~extrapolation_mask[i].permute(1, 2, 0).repeat(1, 1, 3).detach().cpu().numpy())
                    PIL.Image.fromarray(warped.astype(np.uint8)).save(
                        os.path.join(self.logdir, f'qualitative_res/warped_inputs/batch_{batch_idx}_index_{i}.png'))
                if self.global_rank == 0:
                    PIL.Image.fromarray(pred.astype(np.uint8)).save(
                        os.path.join(self.logdir, f'qualitative_res/pred/batch_{batch_idx}_index_{i}.png'))

                    PIL.Image.fromarray(gt.astype(np.uint8)).save(
                        os.path.join(self.logdir, f'qualitative_res/gt/batch_{batch_idx}_index_{i}.png'))
            ssim_score /= xrec.shape[0]
            psnr_score /= xrec.shape[0]
            ssim_visible_score /= xrec.shape[0]
            psnr_visible_score /= xrec.shape[0]

            optimized_ssim_score /= xrec.shape[0]
            optimized_psnr_score /= xrec.shape[0]
            optimized_ssim_visible_score /= xrec.shape[0]
            optimized_psnr_visible_score /= xrec.shape[0]

            rgb_l1 = F.l1_loss(xrec[:, :3], x_dst[:, :3])
            disparity_l1 = F.l1_loss(xrec[:, 3:], x_dst[:, 3:])

            self.log("val/rgb_l1", rgb_l1,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/disparity_l1", disparity_l1,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/ssim", ssim_score,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/psnr", psnr_score,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/visible_ssim", ssim_visible_score,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("val/visible_psnr", psnr_visible_score,
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            if self.test_time_optimization_config[
                'use_test_time_optimization'] and self.phase == 'conditional_generation':
                self.log("val/optimized_ssim", optimized_ssim_score,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("val/optimized_psnr", optimized_psnr_score,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("val/optimized_visible_ssim", optimized_ssim_visible_score,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
                self.log("val/optimized_visible_psnr", optimized_psnr_visible_score,
                         prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            self.log("val/LPIPS", p_loss.mean().item(),
                     prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            del log_dict_ae["val/rec_loss"]
            self.log_dict(log_dict_ae)
            # self.log_dict({"val/ssim": ssim_score, "val/psnr": psnr_score})
            return self.log_dict

    def evaluation_epoch(self, outputs):
        # fid_score = get_fid_score(str(os.path.join(self.logdir, f'qualitative_res/gt')),
        #                           str(os.path.join(self.logdir, f'qualitative_res/pred')), batch_size=4)
        # self.log("val/FID", fid_score,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
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
        elif self.phase == 'mae_reconstruction_training':
            opt_ae = torch.optim.Adam(list(self.mae_transformer.parameters()) + list(self.mae_linear.parameters()), lr=lr, betas=(0.5, 0.9))
        elif self.phase == 'mae_outpainting_training':
            opt_ae = torch.optim.Adam(list(self.mae_transformer.parameters()) + list(self.token_id_embedding.parameters())
                                      + list(self.mae_linear.parameters()),
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
        elif self.phase == 'mae_reconstruction_training':
            x = self.get_input(self.image_key, batch)
            extrapolation_mask = None
            x_dst = x
            # get vqgan token
            xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True, get_pre_quantized_feature=False)
            metadata = self.mae_forward(pixel_values=x)
            logits = self.mae_linear(metadata['logits'])
            encoding_indices = torch.argmax(logits, dim=-1)
            z_q = self.quantize.embedding(encoding_indices).view(*codebook_indices.shape, -1).permute(0, 3, 1, 2)
            xrec = self.decode(z_q)
        elif self.phase == 'mae_outpainting_training':
            x, x_dst, extrapolation_mask, _ = self.get_x(batch, self.data_config.dataset, return_extrapolation_mask=True, parallel=True)
            # get vqgan token
            xrec, qloss, codebook_indices = self(x, extrapolation_mask=extrapolation_mask, get_codebook_count=True, get_pre_quantized_feature=False)
            metadata = self.mae_forward(pixel_values=x, vqgan_token_ids=codebook_indices, extrapolation_mask=extrapolation_mask)
            logits = self.mae_linear(metadata['logits'])
            encoding_indices = torch.argmax(logits, dim=-1)
            z_q = self.quantize.embedding(encoding_indices).view(*codebook_indices.shape, -1).permute(0, 3, 1, 2)
            xrec = self.decode(z_q)

            if self.use_cycle_consistency_loss:
                T_src2tgts = torch.eye(4)[None, None].repeat(batch['R_rels'].shape[0], 1, 1, 1)
                T_src2tgts[:, :, :3] = torch.cat([batch['R_rels'], batch['t_rels'][..., None]], -1)
                T_src2tgts = torch.inverse(T_src2tgts)

                reverse_batch = {
                    't_rels': T_src2tgts[:, :, :3, 3],
                    'R_rels': T_src2tgts[:, :, :3, :3],
                    "Ks": batch["Ks"],
                    "src_imgs": xrec[:, :3][:, None].permute(0, 1, 3, 4, 2).detach(),
                    "dst_img": batch['src_imgs'][:, 0],
                    "src_depths": xrec[:, 3][:, None].permute(0, 2, 3, 1)[:, None].detach(),
                    "dst_depth": batch['src_depths'][:, 0],
                }

                if self.data_config.dataset == 'clevr-infinite':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (1 / 7 - 1 / 16) + 1 / 16)).detach()
                elif self.data_config.dataset == 'kitti360':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (1 / 3 - 1 / 75) + 1 / 75)).detach()
                elif self.data_config.dataset == 'google_earth':
                    reverse_batch['src_depths'] = (1 / ((reverse_batch['src_depths'] + 1) / 2 * (
                                1 / 10.099975586 - 1 / 14.765625) + 1 / 14.765625) - 10)
                else:
                    raise NotImplementedError

                with torch.no_grad():
                    x_reverse, x_dst_reverse, extrapolation_mask_reverse, _ = self.get_x(reverse_batch, self.data_config.dataset,
                                                                 return_extrapolation_mask=True, parallel=True)
                    # get vqgan token
                    xrec_reverse, qloss_reverse, codebook_indices_reverse = self(x_reverse, extrapolation_mask=extrapolation_mask_reverse,
                                                         get_codebook_count=True, get_pre_quantized_feature=False)
                metadata_reverse = self.mae_forward(pixel_values=x_reverse, vqgan_token_ids=codebook_indices_reverse,
                                            extrapolation_mask=extrapolation_mask_reverse)
                logits_reverse = self.mae_linear(metadata_reverse['logits'])

                encoding_indices_reverse = torch.argmax(logits_reverse, dim=-1)
                z_q_reverse = self.quantize.embedding(encoding_indices_reverse).view(*codebook_indices_reverse.shape, -1).permute(0, 3, 1, 2)
                xrec_reverse = self.decode(z_q_reverse)
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

            input_disparity_reverse = x_reverse[:, 3:]
            input_rgb_reverse = x_reverse[:, :3]
            log["warped_input_reverse"] = input_rgb_reverse
            log["warped_disparity_reverse"] = input_disparity_reverse
            log["reconstructions_reverse"] = xrec_reverse[:, :3]
            log["reconstruction_disparities_reverse"] = xrec_reverse[:, 3:]
            log["gt_rgb_reverse"] = x_dst_reverse[:, :3]
            log["gt_disparity_reverse"] = x_dst_reverse[:, 3:]
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


    def mae_forward(
        self,
        pixel_values,
        vqgan_token_ids=None,
        noise=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        extrapolation_mask=None
    ):
        if extrapolation_mask is not None:
            pixel_values = torch.cat([pixel_values, extrapolation_mask], 1)
        else:
            extrapolation_mask = torch.zeros([pixel_values.shape[0], 1, *pixel_values.shape[2:]]).to(pixel_values.device)
            pixel_values = torch.cat([pixel_values, extrapolation_mask], 1)

        outputs = self.mae_transformer.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        if vqgan_token_ids is None:
            vqgan_token_ids = torch.randint(0, self.n_embed, mask.shape).long().to(mask.device)

        vqgan_token_embeddings = self.token_id_embedding(vqgan_token_ids).view(vqgan_token_ids.shape[0], -1, self.mae_config.token_id_embedding_size)

        decoder_outputs = self.mae_transformer.decoder(latent, ids_restore, token_id_embeddings=vqgan_token_embeddings)
        logits = decoder_outputs["logits"]  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        # loss = self.mae_transformer.forward_loss(pixel_values, logits, mask)

        return {
            # "loss": loss,
            "logits": logits,
            "mask": mask,
            "ids_restore": ids_restore,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }
