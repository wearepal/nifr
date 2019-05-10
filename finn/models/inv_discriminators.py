import torch
import torch.nn as nn
import torch.nn.functional as F

from finn import layers
from .discriminator_base import DiscBase, compute_log_pz, fetch_model
from .tabular import tabular_model


class InvDisc(DiscBase):
    def __init__(self, args, x_dim, z_dim_flat):
        """Create the discriminators that enfoce the partition on z"""

        if args.dataset == 'adult':
            z_dim_flat += 1
            args.zs_dim = round(args.zs_frac * z_dim_flat)
            args.zy_dim = z_dim_flat - args.zs_dim
            args.zn_dim = 0
            s_dim = 1
            x_dim += s_dim

            disc_y_from_zy = tabular_model(args, input_dim=args.zy_dim,
                                           depth=2, batch_norm=False)
            disc_s_from_zy = layers.Mlp([args.zy_dim] + [200, 200] + [1],
                                        activation=nn.ReLU,
                                        output_activation=None)
            disc_s_from_zs = tabular_model(args, input_dim=args.zs_dim,
                                           depth=2, batch_norm=False)
        else:
            z_channels = x_dim * 4 * 4
            wh = z_dim_flat // z_channels
            args.zs_dim = round(args.zs_frac * z_channels)
            args.zy_dim = z_channels - args.zs_dim
            args.zn_dim = 0

            disc_y_from_zy = tabular_model(args, input_dim=(wh * args.zy_dim),
                                           depth=2, batch_norm=False)
            disc_s_from_zs = tabular_model(args, input_dim=(wh * args.zs_dim),
                                           depth=2, batch_norm=False)
            disc_s_from_zy = layers.Mlp([wh * args.zy_dim] + [512, 512] + [10],
                                        activation=nn.ReLU,
                                        output_activation=torch.nn.LogSoftmax)

        disc_y_from_zy.to(args.device)
        disc_s_from_zy.to(args.device)
        disc_s_from_zs.to(args.device)

        self.s_from_zs = disc_s_from_zs
        self.y_from_zy = disc_y_from_zy
        self.s_from_zy = disc_s_from_zy
        self.disc_name_list = ['s_from_zs', 'y_from_zy', 's_from_zy']  # for generating discs_dict
        self.args = args
        self.x_dim = x_dim

    @property
    def discs_dict(self):
        return {disc_name: getattr(self, disc_name) for disc_name in self.disc_name_list}

    def create_model(self):
        return fetch_model(self.args, self.x_dim)

    def assemble_whole_model(self, trunk):
        chain = [trunk]
        chain += [layers.MultiHead([self.s_from_zs, self.y_from_zy],
                                   split_dim=[self.args.zs_dim, self.args.zy_dim])]
        return layers.SequentialFlow(chain)

    @staticmethod
    def multi_class_loss(_logits, _target):
        _preds = F.log_softmax(_logits[:, :10], dim=1)
        return F.nll_loss(_preds, _target, reduction='mean')

    @staticmethod
    def binary_class_loss(_logits, _target):
        return F.binary_cross_entropy_with_logits(_logits[:, :1], _target, reduction='mean')

    def split_zs_zy(self, z):
        assert z.size(1) % (self.args.zs_dim + self.args.zy_dim) == 0
        width_x_height = z.size(1) // (self.args.zs_dim + self.args.zy_dim)
        return z.split(
            split_size=[self.args.zs_dim * width_x_height, self.args.zy_dim * width_x_height],
            dim=1)

    def compute_loss(self, x, s, y, model, return_z=False):
        whole_model = self.assemble_whole_model(model)
        zero = x.new_zeros(x.size(0), 1)

        if self.args.dataset == 'cmnist':
            class_loss = self.multi_class_loss
            indie_loss = F.nll_loss
        else:
            class_loss = self.binary_class_loss
            indie_loss = F.binary_cross_entropy_with_logits
            x = torch.cat((x, s.float()), dim=1)

        z, delta_log_p = whole_model(x, zero)  # run model forward

        log_pz = 0
        # zn = z[:, :self.args.zn_dim]
        z_sn, z_yn = self.split_zs_zy(z)    # zsn, zyn

        pred_y_loss = z.new_zeros(1)
        pred_s_from_zs_loss = z.new_zeros(1)

        if z_yn.size(1) > 0:
            log_pz += compute_log_pz(z_yn)
            if not self.args.meta_learn:
                pred_y_loss = self.args.pred_y_weight * class_loss(z_yn, y)

            pred_s_from_zy = self.s_from_zy(
                layers.grad_reverse(z_yn, lambda_=self.args.pred_s_from_zy_weight))
            pred_s_from_zy_loss = indie_loss(pred_s_from_zy, s, reduction='mean')

        if z_sn.size(1) > 0:
            log_pz += compute_log_pz(z_sn)
            pred_s_from_zs_loss = self.args.pred_s_from_zs_weight * class_loss(z_sn, s)

        log_px = self.args.log_px_weight * (log_pz - delta_log_p).mean()
        loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss

        if return_z:
            return loss, z

        return loss, -log_px, pred_y_loss, pred_s_from_zy_loss, pred_s_from_zs_loss
