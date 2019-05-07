from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import layers
import models
from utils.training_utils import fetch_model
from .nn_discriminators import compute_log_pz, Discriminators


def make_networks(args, x_dim, z_dim_flat):
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
        disc_s_from_zs = layers.Mlp([args.zs_dim] + hidden_sizes + [s_dim], activation=nn.ReLU,
                                    output_activation=output_activation)
        disc_y_from_zys = layers.Mlp([z_dim_flat - args.zn_dim] + [100, 100, y_dim],
                                     activation=nn.ReLU, output_activation=None)
        disc_y_from_zys.to(args.device)
    else:
        z_channels = x_dim * 4 * 4
        wh = z_dim_flat // z_channels
        args.zs_dim = round(args.zs_frac * z_channels)

        if not args.meta_learn:
            disc_y_from_zys = models.tabular_model(args, input_dim=(wh * (args.zy + args.zs)))  # logs-softmax
            disc_y_from_zys.to(args.device)
        else:
            args.zy_dim = z_channels - args.zs_dim
            args.zn_dim = 0

            disc_y_from_zys = None

        # logistic output
        disc_s_from_zs = models.tabular_model(args, input_dim=(wh * args.zs_dim))
        disc_s_from_zy = models.tabular_model(args, input_dim=(wh * args.zy_dim))

    disc_s_from_zs.to(args.device)
    disc_s_from_zy.to(args.device)
    discs = Discriminators(s_from_zs=disc_s_from_zs, s_from_zy=disc_s_from_zy, y_from_zys=disc_y_from_zys)
    return fetch_model(args, x_dim), discs


def compute_loss(args, x, s, y, model, discs, return_z=False):
    zero = x.new_zeros(x.size(0), 1)

    if args.dataset == 'cmnist':

        def class_loss(_logits, _target):
            _preds = F.log_softmax(_logits[:, :10], dim=1)
            return F.nll_loss(_preds, _target, reduction='mean')

        # loss_fn = F.l1_loss
        class_loss_fn = F.nll_loss
    else:
        def class_loss(_logits, _target):
            return F.binary_cross_entropy_with_logits(_logits[:, :1], reduction='mean')

        x = torch.cat((x, s.float()), dim=1)

    z, delta_logp = model(x, zero)  # run model forward

    log_pz = 0
    # zn = z[:, :args.zn_dim]
    zs = z[:, args.zn_dim: (z.size(1) - args.zy_dim)]
    zy = z[:, (z.size(1) - args.zy_dim):]
    # Enforce independence between the fair representation, zy,
    #  and the sensitive attribute, s
    pred_y_loss = z.new_zeros(1)
    pred_s_from_zy_loss = z.new_zeros(1)
    pred_s_from_zs_loss = z.new_zeros(1)

    if discs.y_from_zys is not None and zy.size(1) > 0 and zs.size(1) > 0 and not args.meta_learn:
        logits_y, delta_logp = discs.y_from_zys(torch.cat((zy, zs), dim=1), delta_logp)
        log_pz += compute_log_pz(logits_y)
        pred_y_loss = args.pred_y_weight * class_loss(logits_y, y)

    if discs.s_from_zy is not None and zy.size(1) > 0:
        logits_s, delta_logp = discs.s_from_zy(layers.grad_reverse(zy, lambda_=args.pred_s_from_zy_weight),
                                               delta_logp)
        log_pz += compute_log_pz(logits_s)
        pred_s_from_zy_loss = class_loss(logits_s, s)
    # Enforce independence between the fair, zy, and unfair, zs, partitions

    if discs.s_from_zs is not None and zs.size(1) > 0:
        logits_s, delta_logp = discs.s_from_zs(zs, delta_logp)
        log_pz += compute_log_pz(logits_s)
        pred_s_from_zs_loss = args.pred_s_from_zs_weight * class_loss(logits_s, s)

    log_px = args.log_px_weight * (log_pz - delta_logp).mean()
    loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss

    if return_z:
        return loss, z

    return (loss, -log_px, pred_y_loss, args.pred_s_from_zy_weight * pred_s_from_zy_loss,
            pred_s_from_zs_loss)
