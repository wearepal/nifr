from typing import Dict, List, Optional, Sequence, Tuple, Union, overload, NamedTuple

import numpy as np
import torch
import torch.distributions as td
from torch import Tensor
from typing_extensions import Literal

from nosinn.configs import NosinnArgs
from nosinn.layers import Bijector
from nosinn.utils import DLogistic, MixtureDistribution, logistic_distribution, to_discrete

from .autoencoder import AutoEncoder, VAE
from .base import ModelBase

__all__ = ["PartitionedAeInn", "BipartiteInn"]


class EncodingSize(NamedTuple):
    zs: int
    zy: int
    zn: int


class SplitEncoding(NamedTuple):
    s: Tensor
    y: Tensor
    n: Tensor


class Reconstructions(NamedTuple):
    all: Tensor
    rand_s: Tensor  # reconstruction with random s
    rand_y: Tensor  # reconstruction with random y
    zero_s: Tensor
    zero_y: Tensor
    just_s: Tensor


class BipartiteInn(ModelBase):
    """Base wrapper class for INN models."""

    model: Bijector

    def __init__(
        self,
        args: NosinnArgs,
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

    def encode(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs, reverse=False)

    def decode(self, z, discretize: bool = True) -> Tensor:
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


class PartitionedAeInn(BipartiteInn):
    """Wrapper for classifier models."""

    def __init__(
        self,
        args: NosinnArgs,
        model: Bijector,
        autoencoder: AutoEncoder,
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

        zs_dim: int = round(args.zs_frac * self.output_dim)
        zy_dim: int = zs_dim
        zn_dim: int = self.output_dim - zs_dim - zy_dim
        self.encoding_size = EncodingSize(zs=zs_dim, zy=zy_dim, zn=zn_dim)
        self.autoencoder = autoencoder

    def split_encoding(self, z: Tensor) -> SplitEncoding:
        zs, zy, zn = z.split(
            (self.encoding_size.zs, self.encoding_size.zy, self.encoding_size.zn), dim=1
        )
        return SplitEncoding(s=zs, y=zy, n=zn)

    @staticmethod
    def unsplit_encoding(zs: Tensor, zy: Tensor, zn: Tensor) -> Tensor:
        return torch.cat([zs, zy, zn], dim=1)

    def mask(self, z: Tensor, random: bool = False) -> Tuple[Tensor, Tensor]:
        """Split the encoding and mask out zs and zy. This is a cheap function."""
        z = self.split_encoding(z)
        if random:
            # the question here is whether to have one random number per sample
            # or whether to also have distinct random numbers for all the dimensions of zs.
            # if we don't expect s to be complicated, then the former should suffice
            rand_zs = torch.randn((z.s.size(0),) + (z.s.dim() - 1) * (1,), device=z.s.device)
            zs_m = torch.cat([rand_zs + torch.zeros_like(z.s), z.y, z.n], dim=1)
            rand_zy = torch.randn((z.y.size(0),) + (z.y.dim() - 1) * (1,), device=z.y.device)
            zy_m = torch.cat([z.s, rand_zy + torch.zeros_like(z.y), z.n], dim=1)
        else:
            zs_m = torch.cat([torch.zeros_like(z.s), z.y, z.n], dim=1)
            zy_m = torch.cat([z.s, torch.zeros_like(z.y), z.n], dim=1)
        return zs_m, zy_m

    def all_recons(self, z: Tensor, discretize: bool = False) -> Reconstructions:
        rand_s, rand_y = self.mask(z, random=True)
        zero_s, zero_y = self.mask(z)
        splits = self.split_encoding(z)
        just_s = torch.cat(
            [splits.s, torch.zeros_like(splits.y), torch.zeros_like(splits.n)], dim=1
        )
        return Reconstructions(
            all=self.decode(z, discretize=discretize),
            rand_s=self.decode(rand_s, discretize=discretize),
            rand_y=self.decode(rand_y, discretize=discretize),
            zero_s=self.decode(zero_s, discretize=discretize),
            zero_y=self.decode(zero_y, discretize=discretize),
            just_s=self.decode(just_s, discretize=discretize),
        )

    def encode_and_split(self, inputs: Tensor) -> SplitEncoding:
        return self.split_encoding(self.encode(inputs))

    def generate_recon_rand_s(self, inputs: Tensor) -> Tensor:
        zs, zy, zn = self.encode_and_split(inputs)
        zs_m = torch.cat([torch.randn_like(zs), zy, zn], dim=1)
        return self.decode(zs_m)

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

    @overload
    def forward(
        self,
        inputs: Tensor,
        logdet: Optional[Tensor] = ...,
        reverse: bool = ...,
        return_ae_enc: Literal[False] = ...,
    ) -> Tensor:
        ...

    @overload
    def forward(
        self,
        inputs: Tensor,
        *,
        return_ae_enc: Literal[True],
        logdet: Optional[Tensor] = ...,
        reverse: bool = ...,
    ) -> Tuple[Tensor, Tensor]:
        ...

    def forward(
        self,
        inputs: Tensor,
        logdet: Optional[Tensor] = None,
        reverse: bool = False,
        return_ae_enc: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if reverse:
            ae_enc, _ = self.model(inputs, sum_ldj=logdet, reverse=reverse)
            outputs = self.autoencoder.decode(ae_enc)
        else:
            if isinstance(self.autoencoder, VAE):
                ae_enc = self.autoencoder.encode(inputs, stochastic=True)
            else:
                ae_enc = self.autoencoder.encode(inputs)
            outputs, sum_ldj = self.model(ae_enc, sum_ldj=logdet, reverse=reverse)
            if sum_ldj is not None:
                outputs = (outputs, sum_ldj)

        if return_ae_enc:
            return outputs, ae_enc
        return outputs

    def fit_ae(self, train_data, epochs, device, loss_fn=torch.nn.MSELoss()):
        print("===> Fitting Auto-encoder to the training data....")
        self.autoencoder.train()
        self.autoencoder.fit(train_data, epochs, device, loss_fn)
        self.autoencoder.eval()

    def train(self):
        self.model.train()
        self.autoencoder.eval()

    def eval(self):
        self.model.eval()
        self.autoencoder.eval()
