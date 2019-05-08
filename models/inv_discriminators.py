from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
import models
from models.discriminator_base import DiscBase
from models.nn_discriminators import NNDisc
from utils.training_utils import fetch_model

Discriminators = namedtuple('Discriminators', ['s_from_zs', 'y_from_zy', 's_from_zy'])


class InvDisc(DiscBase):
    def make_networks(self, args, x_dim, z_dim_flat):
        """Create the discriminators that enfoce the partition on z"""
        if args.dataset == 'adult':
            z_dim_flat += 1
            args.zs_dim = round(args.zs_frac * z_dim_flat)
            args.zy_dim = z_dim_flat - args.zs_dim
            args.zn_dim = 0
            s_dim = 1
            x_dim += s_dim

            disc_y_from_zy = models.tabular_model(args, input_dim=args.zy_dim)
            disc_s_from_zs = models.tabular_model(args, input_dim=args.zs_dim)
            disc_y_from_zy.to(args.device)
        else:
            z_channels = x_dim * 4 * 4
            wh = z_dim_flat // z_channels
            args.zs_dim = round(args.zs_frac * z_channels)
            args.zy_dim = z_channels - args.zs_dim
            args.zn_dim = 0

<<<<<<< HEAD
        disc_y_from_zy = models.tabular_model(args, input_dim=(wh * args.zy_dim))  # logs-softmax
        disc_y_from_zy.to(args.device)

        # logistic output
        disc_s_from_zs = models.tabular_model(args, input_dim=(wh * args.zs_dim))
        disc_s_from_zy = layers.Mlp([wh * args.zy_dim] + [512, 512] + [10],
                                    activation=nn.ReLU,
                                    output_activation=torch.nn.LogSoftmax)

        disc_s_from_zs.to(args.device)
        discs = Discriminators(s_from_zs=disc_s_from_zs, y_from_zy=disc_y_from_zy,
                               s_from_zy=disc_s_from_zy)

        return fetch_model(args, x_dim), discs

    def compute_loss(self, args, x, s, y, model, discs, return_z=False):
        whole_model = self.assemble_whole_model(args, model, discs)
        zero = x.new_zeros(x.size(0), 1)

        if args.dataset == 'cmnist':
            class_loss = self.multi_class_loss
            indie_loss = F.nll_loss
        else:
            class_loss = self.binary_class_loss
            indie_loss = F.binary_cross_entropy_with_logits
            x = torch.cat((x, s.float()), dim=1)

        z, delta_log_p = whole_model(x, zero)  # run model forward

        log_pz = 0
        # zn = z[:, :args.zn_dim]
        wh = z.size(1) // (args.zy_dim + args.zs_dim)
        zy, zs = z.split(split_size=[args.zy_dim * wh, args.zs_dim * wh], dim=1)

        pred_y_loss = z.new_zeros(1)
        pred_s_from_zs_loss = z.new_zeros(1)

        nn_disc = NNDisc()
        if zy.size(1) > 0:
            log_pz += nn_disc.compute_log_pz(zy)
            if not args.meta_learn:
                pred_y_loss = args.pred_y_weight * class_loss(zy, y)

        pred_s_from_zy = discs.s_from_zy(layers.grad_reverse(zy, lambda_=args.pred_s_from_zy_weight))
        pred_s_from_zy_loss = indie_loss(pred_s_from_zy, s, reduction='mean')
=======
            if not args.meta_learn:
                disc_y_from_zy = models.tabular_model(args, input_dim=(wh * args.zy_dim))  # logs-softmax
                disc_y_from_zy.to(args.device)
            else:
                args.zy_dim = z_channels - args.zs_dim
                args.zn_dim = 0

                disc_y_from_zy = None

            # logistic output
            disc_s_from_zs = models.tabular_model(args, input_dim=(wh * args.zs_dim))

        disc_s_from_zs.to(args.device)
        discs = Discriminators(s_from_zs=disc_s_from_zs, y_from_zy=disc_y_from_zy)

        return fetch_model(args, x_dim), discs

    def compute_loss(self, args, x, s, y, model, discs, return_z=False):
        whole_model = self.assemble_whole_model(args, model, discs)
        zero = x.new_zeros(x.size(0), 1)

        if args.dataset == 'cmnist':
            class_loss = self.multi_class_loss
        else:
            class_loss = self.binary_class_loss
            x = torch.cat((x, s.float()), dim=1)

        z, delta_log_p = whole_model(x, zero)  # run model forward

        log_pz = 0
        # zn = z[:, :args.zn_dim]
        wh = z.size(1) // (args.zy_dim + args.zs_dim)
        zy, zs = z.split(split_size=[args.zy_dim * wh, args.zs_dim * wh], dim=1)

        pred_y_loss = z.new_zeros(1)
        pred_s_from_zs_loss = z.new_zeros(1)

        nn_disc = NNDisc()
        if zy.size(1) > 0:
            log_pz += nn_disc.compute_log_pz(zy)
            if not args.meta_learn:
                pred_y_loss = args.pred_y_weight * class_loss(zy, y)
>>>>>>> 8cd28bbef02c28445281076b1b8dbea404a11696

        if zs.size(1) > 0:
            log_pz += nn_disc.compute_log_pz(zs)
            pred_s_from_zs_loss = args.pred_s_from_zs_weight * class_loss(zs, s)

        log_px = args.log_px_weight * (log_pz - delta_log_p).mean()
<<<<<<< HEAD
        loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss
=======
        loss = -log_px + pred_y_loss + pred_s_from_zs_loss
>>>>>>> 8cd28bbef02c28445281076b1b8dbea404a11696

        if return_z:
            return loss, z

        return loss, -log_px, pred_y_loss, z.new_zeros(1), pred_s_from_zs_loss
