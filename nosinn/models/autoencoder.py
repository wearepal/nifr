import torch
import torch.distributions as td
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelBase


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, decode_with_s=False, optimizer_args=None):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_args=optimizer_args)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_args=optimizer_args)
        self.decode_with_s = decode_with_s

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, encoding, s=None):
        decoder_input = encoding
        if s is not None and self.decode_with_s:
            if encoding.dim() == 4:
                s = s.view(s.size(0), -1, 1, 1).float()
                s = s.expand(-1, -1, decoder_input.size(-2), decoder_input.size(-1))
                decoder_input = torch.cat([decoder_input, s], dim=1)
        decoding = self.decoder(decoder_input)

        if decoding.dim() == 4 and decoding.size(1) > 3:
            num_classes = 256
            decoding = decoding[:64].view(decoding.size(0), num_classes, -1, *decoding.shape[-2:])
            fac = num_classes - 1
            decoding = decoding.max(dim=1)[1].float() / fac

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

    def routine(self, inputs, loss_fn, s=None):
        encoding = self.encoder(inputs)
        decoding = self.decoder(encoding, s=s)
        loss = loss_fn(decoding, inputs)
        loss /= inputs.size(0)

        return loss

    def fit(self, train_data, epochs, device, loss_fn):

        self.train()

        with tqdm(total=epochs * len(train_data)) as pbar:
            for epoch in range(epochs):

                for x, s, _ in train_data:

                    x = x.to(device)
                    if self.decode_with_s:
                        s = s.to(device)

                    self.zero_grad()
                    loss = self.routine(x, loss_fn=loss_fn, s=s)

                    loss.backward()
                    self.step()

                    pbar.update()
                    pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())


class VAE(AutoEncoder):
    def __init__(self, encoder, decoder, kl_weight=0.1, decode_with_s=False, optimizer_args=None):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            decode_with_s=decode_with_s,
            optimizer_args=optimizer_args,
        )
        self.encoder: ModelBase = ModelBase(encoder, optimizer_args=optimizer_args)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_args=optimizer_args)

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
        if s is not None and self.decode_with_s:
            if sample.dim() == 4:
                s = s.view(s.size(0), -1, 1, 1).float()
                s = s.expand(-1, -1, sample.size(-2), sample.size(-1))
                decoder_input = torch.cat([sample, s], dim=1)
        recon = self.decoder(decoder_input)
        recon_loss = recon_loss_fn(recon, x)

        recon_loss /= x.size(0)
        kl /= x.size(0)

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
