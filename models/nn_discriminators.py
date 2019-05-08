from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
import models
from models.discriminator_base import DiscBase
from utils.training_utils import fetch_model

Discriminators = namedtuple('Discriminators', ['s_from_zs', 's_from_zy', 'y_from_zys'])


class NNDisc(DiscBase):
    def make_networks(self, args, x_dim, z_dim_flat):
        """Create the discriminators that enfoce the partition on z"""
        if args.dataset == 'adult':
            z_dim_flat += 1
            args.zs_dim = round(args.zs_frac * z_dim_flat)
            args.zy_dim = round(args.zy_frac * z_dim_flat)
            s_dim = 1
            x_dim += s_dim
            y_dim = 1
            output_activation = nn.Sigmoid
            hidden_sizes = [40, 40]

            args.zn_dim = z_dim_flat - args.zs_dim - args.zy_dim

            disc_s_from_zy = layers.Mlp([args.zy_dim] + hidden_sizes + [s_dim], activation=nn.ReLU,
                                        output_activation=output_activation)
            hidden_sizes = [40, 40]
            disc_s_from_zs = layers.Mlp([args.zs_dim] + hidden_sizes + [s_dim], activation=nn.ReLU,
                                        output_activation=output_activation)
            disc_y_from_zys = layers.Mlp([z_dim_flat - args.zn_dim] + [100, 100, y_dim],
                                         activation=nn.ReLU, output_activation=None)
            disc_y_from_zys.to(args.device)
        else:
            z_channels = x_dim * 16
            args.zs_dim = round(args.zs_frac * z_channels)
            s_dim = 10
            y_dim = 10
            output_activation = nn.LogSoftmax(dim=1)

            if not args.meta_learn:
                args.zy_dim = round(args.zy_frac * z_channels)
                args.zn_dim = z_channels - args.zs_dim - args.zy_dim

                hidden_sizes = [(args.zy_dim + args.zs_dim * 8), (args.zy_dim + args.zs_dim) * 8]
                disc_y_from_zys = models.MnistConvNet(args.zy_dim + args.zs_dim, y_dim,
                                                      output_activation=nn.LogSoftmax(dim=1),
                                                      hidden_sizes=hidden_sizes)
                disc_y_from_zys.to(args.device)
            else:
                args.zy_dim = z_channels - args.zs_dim
                args.zn_dim = 0

                disc_y_from_zys = None

            hidden_sizes = [args.zs_dim * 16, args.zs_dim * 16]
            disc_s_from_zs = models.MnistConvNet(args.zs_dim, s_dim, hidden_sizes=hidden_sizes,
                                                 output_activation=output_activation)
            hidden_sizes = [args.zy_dim * 16, args.zy_dim * 16, args.zy_dim * 16]
            disc_s_from_zy = models.MnistConvNet(args.zy_dim, s_dim, hidden_sizes=hidden_sizes,
                                                 output_activation=output_activation)

        disc_s_from_zs.to(args.device)
        disc_s_from_zy.to(args.device)
        discs = Discriminators(s_from_zs=disc_s_from_zs, s_from_zy=disc_s_from_zy, y_from_zys=disc_y_from_zys)
        return fetch_model(args, x_dim), discs

    def compute_loss(self, args, x, s, y, model, discs, return_z=False):

        zero = x.new_zeros(x.size(0), 1)

        if args.dataset == 'cmnist':
            # loss_fn = F.l1_loss
            loss_fn = F.nll_loss
            class_loss_fn = F.nll_loss
        else:
            loss_fn = F.binary_cross_entropy
            x = torch.cat((x, s.float()), dim=1)
            class_loss_fn = F.binary_cross_entropy_with_logits

        z, delta_logp = model(x, zero)  # run model forward

        log_pz = self.compute_log_pz(z)
        # zn = z[:, :args.zn_dim]
        zs = z[:, args.zn_dim: (z.size(1) - args.zy_dim)]
        zy = z[:, (z.size(1) - args.zy_dim):]
        # Enforce independence between the fair representation, zy,
        #  and the sensitive attribute, s
        pred_y_loss = z.new_zeros(1)
        pred_s_from_zy_loss = z.new_zeros(1)
        pred_s_from_zs_loss = z.new_zeros(1)

        if discs.y_from_zys is not None and zy.size(1) > 0 and zs.size(1) > 0 and not args.meta_learn:
            pred_y_loss = (args.pred_y_weight
                           * class_loss_fn(discs.y_from_zys(torch.cat((zy, zs), dim=1)), y, reduction='mean'))
        if discs.s_from_zy is not None and zy.size(1) > 0:
            pred_s_from_zy_loss = loss_fn(
                discs.s_from_zy(layers.grad_reverse(zy, lambda_=args.pred_s_from_zy_weight)),
                s, reduction='mean')
        # Enforce independence between the fair, zy, and unfair, zs, partitions

        if discs.s_from_zs is not None and zs.size(1) > 0:
            pred_s_from_zs_loss = args.pred_s_from_zs_weight \
                                  * loss_fn(discs.s_from_zs(zs), s, reduction='mean')

        log_px = args.log_px_weight * (log_pz - delta_logp).mean()
        loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss

        if return_z:
            return loss, z
        return (loss, -log_px, pred_y_loss, args.pred_s_from_zy_weight * pred_s_from_zy_loss,
                pred_s_from_zs_loss)
