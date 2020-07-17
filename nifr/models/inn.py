from typing import Dict, List, Optional, Sequence, Tuple, Union, overload

import numpy as np
import torch
import torch.distributions as td
from torch import Tensor
from typing_extensions import Literal

from nifr.configs import InnArgs
from nifr.layers import Bijector
from nifr.utils import DLogistic, MixtureDistribution, logistic_distribution, to_discrete

from .autoencoder import AutoEncoder
from .base import ModelBase

__all__ = ["PartitionedInn", "PartitionedAeInn", "BipartiteInn"]


class BipartiteInn(ModelBase):
    """Base wrapper class for INN models."""

    model: Bijector

    def __init__(
        self,
        args: InnArgs,
        model: Bijector,
        input_shape: Sequence[int],
        feature_groups: Optional[List[slice]] = None,
        optimizer_args: Optional[dict] = None,
    ):
        """
        Args:
            args: Runtime arguments.
            model: nn.Module. INN model to wrap around.
            input_shape: Tuple or List. Shape (excluding batch dimension) of the input data.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """
        self.input_shape = input_shape
        self.feature_groups = feature_groups
        self.base_density: td.Distribution

        if args.idf:
            probs = 5 * [1 / 5]
            dist_params = [(0, 0.5), (2, 0.5), (-2, 0.5), (4, 0.5), (-4, 0.5)]
            components = [DLogistic(loc, scale) for loc, scale in dist_params]
            self.base_density = MixtureDistribution(probs=probs, components=components)
        else:
            if args.base_density == "logistic":
                self.base_density = logistic_distribution(
                    torch.zeros(1, device=args.device),
                    torch.ones(1, device=args.device) * args.base_density_std,
                )
            elif args.base_density == "uniform":
                self.base_density = td.Uniform(
                    low=-torch.ones(1, device=args.device) * args.base_density_std,
                    high=torch.ones(1, device=args.device) * args.base_density_std,
                )
            else:
                self.base_density = td.Normal(0, args.base_density_std)
        x_dim: int = input_shape[0]
        z_channels: int = x_dim

        self.x_dim: int = x_dim
        if len(input_shape) < 2:
            z_channels += args.s_dim
            self.output_dim = self.input_shape[0]
        else:
            self.x_dim = x_dim
            self.output_dim = int(np.product(self.input_shape))

        super().__init__(model, optimizer_kwargs=optimizer_args)

    def invert(self, z, discretize: bool = True) -> Tensor:
        x = self.forward(z, reverse=True)

        if discretize and self.feature_groups:
            for group_slice in self.feature_groups["discrete"]:
                one_hot = to_discrete(x[:, group_slice])
                x[:, group_slice] = one_hot

        return x

    @overload
    def encode(self, data: Tensor, partials: Literal[False] = ...) -> Tensor:
        ...

    @overload
    def encode(self, data: Tensor, partials: Literal[True]) -> Tuple[Tensor, Tensor, Tensor]:
        ...

    def encode(
        self, data: Tensor, partials: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        return self.forward(data, reverse=False)

    def decode(self, z, partials: bool = True, discretize: bool = True) -> Tensor:
        return self.invert(z, discretize=discretize)

    def compute_log_pz(self, z: Tensor) -> Tensor:
        """Log of the base probability: log(p(z))"""
        log_pz = self.base_density.log_prob(z)
        return log_pz

    def nll(self, z: Tensor, sum_logdet: Tensor) -> Tensor:
        log_pz = self.compute_log_pz(z)
        log_px = log_pz.sum() - sum_logdet.sum()
        # if z.dim() > 2:
        #     log_px_per_dim = log_px / z.nelement()
        #     bits_per_dim = -(log_px_per_dim - np.log(256)) / np.log(2)
        #     return bits_per_dim
        # else:
        nll = -log_px / z.nelement()
        return nll

    @overload  # type: ignore[override]
    def forward(self, inputs: Tensor, logdet: None = ..., reverse: bool = ...) -> Tensor:
        ...

    @overload
    def forward(self, inputs: Tensor, logdet: Tensor, reverse: bool = ...) -> Tuple[Tensor, Tensor]:
        ...

    def forward(
        self, inputs: Tensor, logdet: Optional[Tensor] = None, reverse: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tensor]:
        outputs, sum_ldj = self.model(inputs, sum_ldj=logdet, reverse=reverse)

        if sum_ldj is None:
            return outputs
        else:
            return outputs, sum_ldj


class PartitionedInn(BipartiteInn):
    """Wrapper for classifier models."""

    def __init__(
        self,
        args: InnArgs,
        model: Bijector,
        input_shape: Sequence[int],
        optimizer_args: Optional[Dict] = None,
        feature_groups: Optional[List[slice]] = None,
    ) -> None:
        """
        Args:
            args: Runtime arguments.
            model: nn.Module. INN model to wrap around.
            input_shape: Tuple or List. Shape (excluding batch dimension) of the
            input data.
            optimizer_args: Dictionary. Arguments to pass to the optimizer.

        Returns:
            None
        """

        super().__init__(
            args, model, input_shape, optimizer_args=optimizer_args, feature_groups=feature_groups
        )

        self.zs_dim: int = round(args.zs_frac * self.output_dim)
        self.zy_dim: int = self.output_dim - self.zs_dim

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

    def encode_with_partials(self, data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return self.encode(data, partials=True)

    @overload
    def encode(self, data: Tensor, partials: Literal[False] = ...) -> Tensor:
        ...

    @overload
    def encode(self, data: Tensor, partials: Literal[True]) -> Tuple[Tensor, Tensor, Tensor]:
        ...

    def encode(
        self, data: Tensor, partials: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        z = self.forward(data, reverse=False)

        if partials:
            zy, zs = self.split_encoding(z)
            return z, zy, zs
        else:
            return z

    def decode(
        self, z: Tensor, partials: bool = True, discretize: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x = super().decode(z, discretize=discretize)
        if partials:
            zy_m, zs_m = self.zero_mask(z)
            xy = super().decode(zy_m, discretize=discretize)
            xs = super().decode(zs_m, discretize=discretize)

            return x, xy, xs
        else:
            return x

    def routine(self, data: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Training routine for the Split INN.

        Args:
            data: Tensor. Input Data to the INN.

        Returns:
            Tuple of classification loss (Tensor) and accuracy (float)
        """
        zero = data.new_zeros(data.size(0), 1)
        z, sum_ldj = self.forward(data, logdet=zero, reverse=False)
        nll = self.nll(z, sum_ldj)

        return z, nll


class PartitionedAeInn(PartitionedInn):
    def __init__(
        self,
        args: InnArgs,
        model: torch.nn.Module,
        autoencoder: AutoEncoder,
        input_shape: Sequence[int],
        optimizer_args: Optional[Dict] = None,
        feature_groups: Optional[List[slice]] = None,
    ) -> None:
        super().__init__(args, model, input_shape, optimizer_args, feature_groups)
        self.autoencoder = autoencoder

    def train(self):
        self.model.train()
        self.autoencoder.eval()

    def eval(self):
        self.model.eval()
        self.autoencoder.eval()

    def fit_ae(self, train_data, epochs, device, loss_fn=torch.nn.MSELoss()):
        print("===> Fitting Auto-encoder to the training data....")
        self.autoencoder.train()
        self.autoencoder.fit(train_data, epochs, device, loss_fn)
        self.autoencoder.eval()

    def forward(
        self,
        inputs: Tensor,
        logdet: Optional[Tensor] = None,
        reverse: bool = False,
        return_ae_enc: bool = False,
    ) -> Tensor:
        if reverse:
            ae_enc, _ = self.model(inputs, sum_ldj=logdet, reverse=reverse)
            outputs = self.autoencoder.decode(ae_enc)
        else:
            ae_enc = self.autoencoder.encode(inputs)
            outputs, sum_ldj = self.model(ae_enc, sum_ldj=logdet, reverse=reverse)
            if sum_ldj is not None:
                outputs = (outputs, sum_ldj)

        if return_ae_enc:
            return outputs, ae_enc
        return outputs
