from argparse import Namespace
from typing import Tuple, Union, List, Optional, Sequence
import numpy as np

import torch
import torch.distributions as td
from torch import Tensor

from finn.utils import to_discrete
from finn.utils.distributions import logistic_mixture_logprob
from .base import BaseModel
from .masker import Masker


class PartitionedInn(BaseModel):
    """ Base wrapper class for INN models.
    """

    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module,
        input_shape: Sequence[int],
        feature_groups: Optional[List[slice]] = None,
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
        self.input_shape = input_shape
        self.feature_groups = feature_groups
        self.base_density: str = args.base_density
        x_dim: int = input_shape[0]
        z_channels: int = x_dim

        if args.dataset == 'adult':
            self.x_dim: int = x_dim
            z_channels += args.s_dim
            self.output_shape = self.input_shape

        elif args.dataset == 'cmnist':
            self.x_dim: int = x_dim
            z_channels *= args.squeeze_factor ** 2
            h = self.input_shape[1] // args.squeeze_factor
            w = self.input_shape[2] // args.squeeze_factor
            self.output_shape = (z_channels, w, h)

        super().__init__(
            model,
            optimizer_args=optimizer_args,
        )

    def invert(self, z, discretize: bool = True) -> Tensor:
        x = self.forward(z, reverse=True)

        if discretize and self.feature_groups:
            for group_slice in self.feature_groups["discrete"]:
                one_hot = to_discrete(x[:, group_slice])
                x[:, group_slice] = one_hot

        return x

    def encode(self, data, partials: bool = True) -> Tensor:
        return self.forward(data, reverse=False)

    def decode(self, z, partials: bool = True, discretize: bool = True) -> Tensor:
        return self.invert(z, discretize=discretize)

    def compute_log_pz(self, z: Tensor) -> Tensor:
        """Log of the base probability: log(p(z))"""
        log_pz = torch.distributions.Normal(0, 1).log_prob(z)
        log_pz = log_pz.flatten(1).sum(1)

        return log_pz

    def neg_log_prob(self, z: Tensor, delta_logp: Tensor) -> Tensor:
        log_pz = self.compute_log_pz(z)
        return -(log_pz - delta_logp.view(-1)).mean()

    def forward(
        self, inputs: Tensor, logdet: Tensor = None,
        reverse: bool = False
    ) -> Tensor:
        outputs = self.model(inputs, logpx=logdet, reverse=reverse)

        return outputs


class SplitInn(PartitionedInn):
    """ Wrapper for classifier models.
    """

    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module,
        input_shape: Sequence[int],
        optimizer_args: dict = None,
        feature_groups: Optional[List[slice]] = None,
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
            feature_groups=feature_groups
        )

        self.zs_dim = int(args.zs_frac * self.output_shape[0])
        self.zy_dim = self.output_shape[0] - self.zs_dim

    def split_encoding(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        zy, zs = z.split(split_size=(self.zy_dim, self.zs_dim), dim=1)
        return zy, zs

    def unsplit_encoding(self, zy: Tensor, zs: Tensor) -> Tensor:
        assert zy.size(1) == self.zy_dim and zs.size(1) == self.zs_dim

        return torch.cat([zy, zs], dim=1)

    def zero_mask(self, z):
        zy, zs = self.split_encoding(z)
        zy_m = torch.cat([zy, z.new_zeros(zs.shape)], dim=1)
        zs_m = torch.cat([z.new_zeros(zy.shape), zs], dim=1)

        return zy_m, zs_m

    def encode(
        self, data: Tensor, partials: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        z = self.forward(data, reverse=False)

        if partials:
            zy, zs = self.split_encoding(z)
            return z, zy, zs
        else:
            return z

    def decode(
        self, z: Tensor, partials: bool = True, discretize: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x = super().decode(z, discretize=discretize)
        if partials:
            zy_m, zs_m = self.zero_mask(z)
            xy = super().decode(zy_m, discretize=discretize)
            xs = super().decode(zs_m, discretize=discretize)

            return x, xy, xs
        else:
            return x

    def routine(self, data: torch.Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Training routine for the Split INN.

        Args:
            data: Tensor. Input Data to the INN.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        zero = data.new_zeros(data.size(0), 1)
        z, delta_logp = self.forward(data, logdet=zero, reverse=False)
        neg_log_prob = self.neg_log_prob(z, delta_logp)
        # z = self.split_encoding(z)

        return z, neg_log_prob


class MaskedInn(PartitionedInn):

    def __init__(
        self,
        args: Namespace,
        model: torch.nn.Module,
        input_shape: Sequence[int],
        optimizer_args: dict = None,
        feature_groups: Optional[List[slice]] = None,
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
            feature_groups=feature_groups
        )
        self.masker: Masker = Masker(
            shape=self.output_shape,
            prob_1=(1. - args.zs_frac),
            optimizer_args=masker_optimizer_args
        )

    def mask_train(self) -> None:
        self.model.eval()
        self.masker.train()

    def train(self) -> None:
        super().train()
        self.masker.train()

    def step(self) -> None:
        if self.model.training:
            super().step()
        if self.masker.training:
            self.masker.step()

    def zero_grad(self) -> None:
        if self.model.training:
            super().zero_grad()
        if self.masker.training:
            self.masker.zero_grad()

    def encode(
        self, data, partials: bool = True, threshold: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        z = self.forward(data)
        if partials:
            mask = self.masker(threshold=threshold)
            zy = mask * z
            zs = (1 - mask) * z

            return z, zy, zs
        else:
            return z

    def decode(
        self, z, partials=True, threshold=True
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x = super().decode(z)

        if partials:
            mask = self.masker(threshold=threshold)
            xy = super().decode(mask * z)
            xs = super().decode((1 - mask) * z)

            return x, xy, xs
        else:
            return x

    def routine(
        self, data: Tensor, threshold: bool = True
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Training routine for the MaskedINN.

        Args:
            data: Tensor. Input data.
            threshold: Bool. Whether to threshold the mask (hard mask) or
            use the raw probabilities (soft mask)

        Returns:
            Tuple containing the pre-images and the negative log probability
            of the data under the model.
        """
        zero = data.new_zeros(data.size(0), 1)

        z, delta_logp = self.forward(data, logdet=zero, reverse=False)

        neg_log_prob = self.neg_log_prob(z, delta_logp)

        return z, neg_log_prob
