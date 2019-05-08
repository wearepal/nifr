from abc import ABC, abstractmethod

import torch

from layers import MultiHead, SequentialFlow
import torch.nn.functional as F


class DiscBase(ABC):
    @abstractmethod
    def make_networks(self, args, x_dim, z_dim_flat):
        pass

    @abstractmethod
    def compute_loss(self, args, x, s, y, model, discs, return_z=False):
        pass

    def assemble_whole_model(self, args, trunk, discs):
        chain = [trunk]
        chain += [MultiHead([discs.y_from_zy, discs.s_from_zs],
                                   split_dim=[args.zy_dim, args.zs_dim])]
        return SequentialFlow(chain)

    def multi_class_loss(self, _logits, _target):
        _preds = F.log_softmax(_logits[:, :10], dim=1)
        return F.nll_loss(_preds, _target, reduction='mean')

    def binary_class_loss(self, _logits, _target):
        return F.binary_cross_entropy_with_logits(_logits[:, :1], _target, reduction='mean')

    def compute_log_pz(self, z):
        """Log of the base probability: log(p(z))"""
        log_pz = torch.distributions.Normal(0, 1).log_prob(z).flatten(1).sum(1)
        return log_pz.view(z.size(0), 1)
