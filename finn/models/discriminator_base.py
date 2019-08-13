from abc import ABC, abstractmethod
from itertools import chain

import torch

from finn.utils.distributions import logistic_mixture_logprob
from .model_builder import build_conv_inn
from .tabular import tabular_model


class DiscBase(ABC):
    @abstractmethod
    def __init__(self, args, x_dim, z_dim_flat):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compute_loss(self, x, s, y, model, return_z=False):
        """Compute the loss with the discriminators"""

    @abstractmethod
    def assemble_whole_model(self, trunk):
        pass

    @property
    @abstractmethod
    def discs_dict(self):
        """All discriminators as a dictionary"""

    def parameters(self):
        return chain(*[disc.parameters() for disc in self.discs_dict.values() if disc is not None])


def compute_log_pz(args, z):
    """Log of the base probability: log(p(z))"""
    if args.base_density == 'logistic':
        locs = (-7, -4, 0, 2, 5)
        scales = (0.5, 0.5, 0.5, 0.5, 0.5)
        weights = (2, 3, 2, 1, 4)
        log_pz = logistic_mixture_logprob(z, locs, scales, weights)
    else:
        log_pz = torch.distributions.Normal(0, 1).log_prob(z)

    log_pz = log_pz.flatten(1).sum(1).view(z.size(0), 1)
    return log_pz


def fetch_model(args, x_dim):
    if args.dataset == 'cmnist':
        model = build_conv_inn(args, x_dim).to(args.device)
    elif args.dataset == 'adult':
        model = tabular_model(args, x_dim).to(args.device)
    else:
        raise NotImplementedError(
            "Only works for cmnist and adult - How have you even got" "hererere?????"
        )
    return model
