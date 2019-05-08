from abc import ABC, abstractmethod
from itertools import chain

import torch
import torch.nn.functional as F

from finn.layers import MultiHead, SequentialFlow


class DiscBase(ABC):
    @abstractmethod
    def __init__(self, args, x_dim, z_dim_flat):
        pass

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compute_loss(self, x, s, y, model, eturn_z=False):
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


def compute_log_pz(z):
    """Log of the base probability: log(p(z))"""
    log_pz = torch.distributions.Normal(0, 1).log_prob(z).flatten(1).sum(1)
    return log_pz.view(z.size(0), 1)
