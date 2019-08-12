from typing import Tuple

import torch
import torch.distributions as dist

from finn.models.base import BaseModel
from finn.models.masker import Masker
from finn.utils.distributions import logistic_mixture_logprob


class PartitionedInn(BaseModel):
    """ Base wrapper class for INN models.
    """

    def __init__(
        self,
        args,
        model,
        input_shape,
        optimizer_args: dict = None,
    ) -> None:
        """
        Args:
            args: Namespace. Runtime arguments.
            model: nn.Module. INN model to wrap around.
            input_shape: Tuple or List. Shape (excluding batch dimension) of the
            input data.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        self.base_density = args.base_density
        x_dim = input_shape[0]
        z_channels = x_dim

        if args.dataset == 'adult':
            self.x_s_dim = x_dim
            if args.use_s:
                self.x_s_dim += args.s_dim
                z_channels += args.s_dim
            self.output_shape = self.input_shape

        elif args.dataset == 'cmnist':
            self.x_s_dim = x_dim  # s is included in x
            z_channels *= args.squeeze_factor ** 2
            w = self.input_shape[1] // args.squeeze_factor
            h = self.output_shape[2] // args.squeeze_factor
            self.output_shape = (z_channels, w, h)

        self.zs_dim = round(args.zs_frac * z_channels)
        self.zy_dim = z_channels - args.zs_dim

        super().__init__(
            model,
            optimizer_args=optimizer_args,
        )

    def compute_log_pz(self, z: torch.Tensor) -> torch.Tensor:
        """Log of the base probability: log(p(z))"""
        if self.base_density == 'logistic':
            locs = (-7, -4, 0, 2, 5)
            scales = (0.5, 0.5, 0.5, 0.5, 0.5)
            weights = (2, 3, 2, 1, 4)
            log_pz = logistic_mixture_logprob(z, locs, scales, weights)
        else:
            log_pz = torch.distributions.Normal(0, 1).log_prob(z)

        log_pz = log_pz.flatten(1).sum(1).view(z.size(0), 1)

        return log_pz

    def neg_log_prob(self, z, delta_logp):
        log_pz = self.compute_log_pz(z)
        return -(log_pz - delta_logp).mean()

    def forward(self, inputs, logdet=None, reverse=False):
        outputs = self.model(inputs, logpx=logdet, reverse=reverse)

        return outputs


class SplitInn(PartitionedInn):
    """ Wrapper for classifier models.
    """

    def __init__(
        self,
        args,
        model,
        input_shape,
        optimizer_args: dict = None,
    ) -> None:
        """
        Args:
            args: Namespace. Runtime arguments.
            model: nn.Module. INN model to wrap around.
            input_shape: Tuple or List. Shape (excluding batch dimension) of the
            input data.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """

        super().__init__(
            args,
            model,
            input_shape,
            optimizer_args=optimizer_args,
        )

    def split_encoding(self, z):
        zs, zy = z.split(split_size=(self.zs_dim, self.zy_dim))
        return zs, zy

    def unsplit_encoding(self, zs, zy):
        assert zs.size(1) == self.zs_dim and zy.size(1) == self.zy_dim

        return torch.cat([zs, zy], dim=1)

    def routine(self, data: torch.Tensor) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Training routine for the Split INN.

        Args:
            data: Tensor. Input Data to the INN.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        zero = data.new_zeros(data.size(0), 1)
        z, delta_logp = self.forward(data, logdet=zero, reverse=False)
        neg_log_px = self.neg_log_prob(z, delta_logp)
        z = self.split_encoding(z)

        return z, neg_log_px


class MaskedInn(PartitionedInn):

    def __init__(
        self,
        args,
        model,
        input_shape,
        optimizer_args: dict = None,
        masker_optimizer_args: dict = None
    ) -> None:
        """

        Args:
            args: Namespace. Runtime arguments.
            model: nn.Module. INN model to wrap around.
            input_shape: Tuple or List. Shape (excluding batch dimension) of the
            input data.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.
        """
        super().__init__(
            args=args,
            model=model,
            input_shape=input_shape,
            optimizer_args=optimizer_args,
        )

        self.masker: Masker = Masker(
            shape=self.output_shape,
            prob_1=(1. - args.zs_frac),
            optimizer_args=masker_optimizer_args
        )

    def mask_train(self):
        self.model.eval()
        self.masker.train()

    def train(self):
        self.model.train()
        self.masker.eval()

    def step(self):
        if self.model.training:
            self.model.step()
        if self.masker.training:
            self.masker.step()

    def zero_grad(self):
        if self.model.training:
            self.model.zero_grad()
        if self.masker.training:
            self.masker.zero_grad()

    def routine(self, data: torch.Tensor, threshold: bool = True) \
        -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Training routine for the MaskedINN.

        Args:
            data: Tensor. Input data.
            threshold: Bool. Whether to threshold the mask (hard mask) or
            use the raw probabilities (soft mask)

        Returns:
            Tuple containing the pre-images and the negative log probability
            of the data under the model.
        """
        z = self.forward(data, logdet=None, reverse=False)

        mask = self.masker(threshold=threshold)
        zy = mask * z
        zs = (1 - mask) * z

        zero = data.new_zeros(z.size(0), 1)
        xy_pre, delta_logp = self.forward(zy.detach(), logdet=zero, reverse=True)
        xs_pre = self.forward(zs.detach(), logdet=None, reverse=True)

        neg_log_pz = self.neg_log_prob(xy_pre, delta_logp)

        return (xy_pre, xs_pre), neg_log_pz
