import torch
import torch.nn as nn
import torch.nn.functional as F

from finn import layers
from .discriminator_base import DiscBase, compute_log_pz, fetch_model
from .mnist import MnistConvNet


class NNDisc(DiscBase):
    def __init__(self, args, x_dim, z_dim_flat):
        """Create the discriminators that enfoce the partition on z"""
        if args.dataset == 'adult':
            self.s_dim = 1
            self.x_s_dim = x_dim + self.s_dim
            self.y_dim = 1
            self.z_channels = z_dim_flat + 1
        elif args.dataset == 'cmnist':
            self.s_dim = 10
            self.x_s_dim = x_dim  # s is included in x
            self.y_dim = 10
            self.z_channels = x_dim * 16

        args.zs_dim = round(args.zs_frac * self.z_channels)
        if args.meta_learn:
            args.zy_dim = self.z_channels - args.zs_dim
            args.zn_dim = 0
            disc_y_from_zys = None
        else:
            args.zy_dim = round(args.zy_frac * self.z_channels)
            args.zn_dim = self.z_channels - args.zs_dim - args.zy_dim

        # =========== Define discriminator networks ============
        if args.dataset == 'adult':
            # ==== MLP models ====
            output_activation = nn.Sigmoid
            hidden_sizes = [40, 40]
            disc_s_from_zs = layers.Mlp([args.zs_dim] + hidden_sizes + [self.s_dim],
                                        activation=nn.ReLU, output_activation=output_activation)

            hidden_sizes = [40, 40]
            disc_s_from_zy = layers.Mlp([args.zy_dim] + hidden_sizes + [self.s_dim],
                                        activation=nn.ReLU, output_activation=output_activation)

            if not args.meta_learn:
                disc_y_from_zys = layers.Mlp([self.z_channels - args.zn_dim, 100, 100, self.y_dim],
                                             activation=nn.ReLU, output_activation=None)
                disc_y_from_zys.to(args.device)
        else:
            # ==== CNN models ====
            output_activation = nn.LogSoftmax(dim=1)
            hidden_sizes = [args.zs_dim * 16, args.zs_dim * 16]
            disc_s_from_zs = MnistConvNet(args.zs_dim, self.s_dim, hidden_sizes=hidden_sizes,
                                          output_activation=output_activation)

            hidden_sizes = [args.zy_dim * 16, args.zy_dim * 16, args.zy_dim * 16]
            disc_s_from_zy = MnistConvNet(args.zy_dim, self.s_dim, hidden_sizes=hidden_sizes,
                                          output_activation=output_activation)

            if not args.meta_learn:
                hidden_sizes = [(args.zy_dim + args.zs_dim * 8), (args.zy_dim + args.zs_dim) * 8]
                disc_y_from_zys = MnistConvNet(args.zy_dim + args.zs_dim, self.y_dim,
                                               output_activation=nn.LogSoftmax(dim=1),
                                               hidden_sizes=hidden_sizes)
                disc_y_from_zys.to(args.device)

        disc_s_from_zs.to(args.device)
        disc_s_from_zy.to(args.device)
        self.s_from_zs = disc_s_from_zs
        self.s_from_zy = disc_s_from_zy
        self.y_from_zys = disc_y_from_zys
        self.disc_name_list = ['s_from_zs', 's_from_zy', 'y_from_zys']  # for generating discs_dict
        self.args = args

    @property
    def discs_dict(self):
        return {disc_name: getattr(self, disc_name) for disc_name in self.disc_name_list}

    def create_model(self):
        return fetch_model(self.args, self.x_s_dim)

    def assemble_whole_model(self, trunk):
        return trunk

    def compute_loss(self, x, s, y, model, return_z=False):

        zero = x.new_zeros(x.size(0), 1)

        if self.args.dataset == 'cmnist':
            # loss_fn = F.l1_loss
            loss_fn = F.nll_loss
            class_loss_fn = F.nll_loss
        else:
            loss_fn = F.binary_cross_entropy
            x = torch.cat((x, s.float()), dim=1)
            class_loss_fn = F.binary_cross_entropy_with_logits

        z, delta_logp = model(x, zero)  # run model forward

        log_pz = compute_log_pz(z)
        # zn = z[:, :self.args.zn_dim]
        zs = z[:,  self.args.zn_dim: (z.size(1) - self.args.zy_dim)]
        zy = z[:, (z.size(1) - self.args.zy_dim):]
        # Enforce independence between the fair representation, zy,
        #  and the sensitive attribute, s
        pred_y_loss = z.new_zeros(1)
        pred_s_from_zy_loss = z.new_zeros(1)
        pred_s_from_zs_loss = z.new_zeros(1)

        if self.y_from_zys is not None and zy.size(1) > 0 and zs.size(1) > 0 and not self.args.meta_learn:
            pred_y_loss = (self.args.pred_y_weight
                           * class_loss_fn(self.y_from_zys(torch.cat((zy, zs), dim=1)), y, reduction='mean'))
        if self.s_from_zy is not None and zy.size(1) > 0:
            pred_s_from_zy_loss = loss_fn(
                self.s_from_zy(layers.grad_reverse(zy, lambda_=self.args.pred_s_from_zy_weight)),
                s, reduction='mean')
        # Enforce independence between the fair, zy, and unfair, zs, partitions

        if self.s_from_zs is not None and zs.size(1) > 0:
            pred_s_from_zs_loss = (self.args.pred_s_from_zs_weight
                                   * loss_fn(self.s_from_zs(zs), s, reduction='mean'))

        log_px = self.args.log_px_weight * (log_pz - delta_logp).mean()
        loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss

        if return_z:
            return loss, z
        return (loss, -log_px, pred_y_loss, self.args.pred_s_from_zy_weight * pred_s_from_zy_loss,
                pred_s_from_zs_loss)
