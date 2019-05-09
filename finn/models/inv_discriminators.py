import torch
import torch.nn as nn
import torch.nn.functional as F

from finn import layers
from .discriminator_base import DiscBase, compute_log_pz, fetch_model
from .tabular import tabular_model


class InvDisc(DiscBase):
    def __init__(self, args, x_dim, z_dim_flat):
        super(InvDisc, self).__init__()
        """Create the discriminators that enfoce the partition on z"""

        disc_s_from_zy = None

        if args.dataset == 'adult':
            z_dim_flat += 1
            args.zs_dim = round(args.zs_frac * z_dim_flat)
            args.zy_dim = z_dim_flat - args.zs_dim
            args.zn_dim = 0
            s_dim = 1
            x_dim += s_dim

            disc_y_from_zy = tabular_model(args, input_dim=args.zy_dim)
            disc_s_from_zy = tabular_model(args, input_dim=args.zs_dim)
            disc_s_from_zs = tabular_model(args, input_dim=args.zs_dim)
            disc_y_from_zy.to(args.device)
        else:
            z_channels = x_dim * 4 * 4
            wh = z_dim_flat // z_channels
            args.zs_dim = round(args.zs_frac * z_channels)
            args.zy_dim = z_channels - args.zs_dim
            args.zn_dim = 0

            disc_y_from_zy = tabular_model(args, input_dim=(wh * args.zy_dim))  # logs-softmax
            disc_y_from_zy.to(args.device)

            # logistic output
            disc_s_from_zs = tabular_model(args, input_dim=(wh * args.zs_dim))
            disc_s_from_zy = layers.Mlp([wh * args.zy_dim] + [512, 512] + [10],
                                        activation=nn.ReLU,
                                        output_activation=torch.nn.LogSoftmax)
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
        chain += [layers.MultiHead([self.y_from_zy, self.s_from_zs],
                                   split_dim=[self.args.zy_dim, self.args.zs_dim])]
        return layers.SequentialFlow(chain)

    @staticmethod
    def multi_class_loss(_logits, _target):
        _preds = F.log_softmax(_logits[:, :10], dim=1)
        return F.nll_loss(_preds, _target, reduction='mean')


    @staticmethod
    def binary_class_loss(_logits, _target):
        return F.binary_cross_entropy_with_logits(_logits[:, :1], _target, reduction='mean')

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
        wh = z.size(1) // (self.args.zy_dim + self.args.zs_dim)
        zy, zs = z.split(split_size=[self.args.zy_dim * wh, self.args.zs_dim * wh], dim=1)

        pred_y_loss = z.new_zeros(1)
        pred_s_from_zs_loss = z.new_zeros(1)

        if zy.size(1) > 0:
            log_pz += compute_log_pz(zy)
            if not self.args.meta_learn:
                pred_y_loss = self.args.pred_y_weight * class_loss(zy, y)

            pred_s_from_zy = self.s_from_zy(
                layers.grad_reverse(zy, lambda_=self.args.pred_s_from_zy_weight))
            pred_s_from_zy_loss = indie_loss(pred_s_from_zy, s, reduction='mean')

        if zs.size(1) > 0:
            log_pz += compute_log_pz(zs)
            pred_s_from_zs_loss = self.args.pred_s_from_zs_weight * class_loss(zs, s)

        log_px = self.args.log_px_weight * (log_pz - delta_log_p).mean()
        loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss

        if return_z:
            return loss, z

        return loss, -log_px, pred_y_loss, z.new_zeros(1), pred_s_from_zs_loss
