from typing import NamedTuple, Optional

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base import ModelBase

__all__ = ["VaeResults", "AutoEncoder", "VAE"]


class VaeResults(NamedTuple):
    elbo: torch.Tensor
    kl_div: torch.Tensor
    enc_y: torch.Tensor
    enc_s: torch.Tensor
    recon: torch.Tensor


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, decode_with_s=False, optimizer_kwargs=None):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)
        self.decode_with_s = decode_with_s

    def encode(self, inputs):
        return self.encoder(inputs)

    def reconstruct(self, encoding, s=None):
        decoding = self.decode(encoding, s)

        if decoding.dim() == 4 and decoding.size(1) > 3:
            num_classes = 256
            decoding = decoding[:64].view(decoding.size(0), num_classes, -1, *decoding.shape[-2:])
            fac = num_classes - 1
            decoding = decoding.max(dim=1)[1].float() / fac

        return decoding

    def decode(self, encoding, s=None):
        decoder_input = encoding
        if s is not None and self.decode_with_s:
            if encoding.dim() == 4:
                s = s.view(s.size(0), -1, 1, 1).float()
                s = s.expand(-1, -1, decoder_input.size(-2), decoder_input.size(-1))
                decoder_input = torch.cat([decoder_input, s], dim=1)
        decoding = self.decoder(decoder_input)

        return decoding

    def forward(self, inputs, reverse: bool = True):
        if reverse:
            return self.decode(inputs)
        else:
            return self.encode(inputs)

    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def step(self):
        self.encoder.step()
        self.decoder.step()

    def routine(self, x, recon_loss_fn, s=None):
        encoding = self.encoder(x)
        decoding = self.decode(encoding, s=s)
        loss = recon_loss_fn(decoding, x)
        loss /= x.size(0)

        return encoding, decoding, loss

    def fit(self, train_data, epochs, device, loss_fn):

        self.train()

        with tqdm(total=epochs * len(train_data)) as pbar:
            for epoch in range(epochs):

                for x, s, _ in train_data:

                    x = x.to(device)
                    if self.decode_with_s:
                        s = s.to(device)

                    self.zero_grad()
                    _, _, loss = self.routine(x, recon_loss_fn=loss_fn, s=s)
                    loss /= x[0].nelement()

                    loss.backward()
                    self.step()

                    pbar.update()
                    pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())


class VAE(AutoEncoder):
    def __init__(self, encoder, decoder, kl_weight=0.1, decode_with_s=False, optimizer_kwargs=None):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            decode_with_s=decode_with_s,
            optimizer_kwargs=optimizer_kwargs,
        )
        self.encoder: ModelBase = ModelBase(encoder, optimizer_kwargs=optimizer_kwargs)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_kwargs=optimizer_kwargs)

        self.prior = td.Normal(0, 1)
        self.posterior_fn = td.Normal
        self.kl_weight = kl_weight

    def compute_divergence(self, sample, posterior: td.Distribution):
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        kl = (log_q - log_p).sum()

        return kl

    def encode(self, x, stochastic=False, return_posterior=False):
        loc, scale = self.encoder(x).chunk(2, dim=1)

        if stochastic or return_posterior:
            scale = F.softplus(scale)
            posterior = self.posterior_fn(loc, scale)

        sample = posterior.rsample() if stochastic else loc

        if return_posterior:
            return sample, posterior
        else:
            return sample

    def routine(self, x, recon_loss_fn, s=None):
        sample, posterior = self.encode(x, stochastic=True, return_posterior=True)
        kl = self.compute_divergence(sample, posterior)

        decoder_input = sample
        recon = self.decode(decoder_input, s)
        recon_loss = recon_loss_fn(recon, s)

        # denom = x.nelement()
        denom = x.size(0)
        recon_loss /= denom
        kl /= denom

        loss = recon_loss + self.kl_weight * kl

        return sample, recon, loss

    def fit(self, train_data, epochs, device, loss_fn):

        self.train()

        with tqdm(total=epochs * len(train_data)) as pbar:
            for epoch in range(epochs):

                for x, s, _ in train_data:

                    x = x.to(device)
                    if self.decode_with_s:
                        s = s.to(device)

                    self.zero_grad()
                    _, _, loss = self.routine(x, recon_loss_fn=loss_fn, s=s)
                    loss.backward()
                    self.step()

                    pbar.update()
                    pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())

    def standalone_routine(
        self,
        x: torch.Tensor,
        s_oh: Optional[torch.Tensor],
        recon_loss_fn,
        stochastic: bool,
        enc_y_dim: int,
        enc_s_dim: int,
    ) -> VaeResults:
        """Compute ELBO"""

        # Encode the data
        if stochastic:
            encoding, posterior = self.encode(x, stochastic=True, return_posterior=True)
            kl_div = self.compute_divergence(encoding, posterior)
        else:
            encoding = self.encode(x, stochastic=False, return_posterior=False)
            kl_div = x.new_zeros(())

        if enc_s_dim > 0:
            enc_y, enc_s = encoding.split(split_size=(enc_y_dim, enc_s_dim), dim=1)
            decoder_input = torch.cat([enc_y, enc_s.detach()], dim=1)
        else:
            enc_y = encoding
            enc_s = None
            decoder_input = encoding

        recon = self.decode(decoder_input, s_oh)

        # Compute losses
        recon_loss = recon_loss_fn(recon, x)

        recon_loss /= x.size(0)
        kl_div /= x.size(0)

        elbo = recon_loss + self.kl_weight * kl_div
        return VaeResults(elbo=elbo, enc_y=enc_y, enc_s=enc_s, recon=recon, kl_div=kl_div)
