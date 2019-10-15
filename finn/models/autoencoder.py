from tqdm import tqdm

import torch.nn as nn

from .base import ModelBase


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder, optimizer_args=None):
        super(AutoEncoder, self).__init__()

        self.encoder: ModelBase = ModelBase(encoder, optimizer_args=optimizer_args)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_args=optimizer_args)

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, encoding):
        return self.decoder(encoding)

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

    def routine(self, inputs, loss_fn=nn.MSELoss()):
        return loss_fn(self.decode(self.encode(inputs)), inputs)

    def fit(self, train_data, epochs, device, loss_fn=nn.MSELoss()):

        self.train()

        with tqdm(total=epochs * len(train_data)) as pbar:
            for epoch in range(epochs):

                for x, _, _ in train_data:

                    x = x.to(device)

                    self.zero_grad()
                    loss = self.routine(x, loss_fn=loss_fn)
                    loss.backward()
                    self.step()

                    pbar.update()
                    pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())


class VAE(AutoEncoder):

    def __init__(self, encoder, decoder, kl_weight=0.1, optimizer_args=None):
        super(AutoEncoder, self).__init__()

        super().__init__(encoder=encoder,
                         decoder=decoder,
                         optimizer_args=optimizer_args)
        self.encoder: ModelBase = ModelBase(encoder, optimizer_args=optimizer_args)
        self.decoder: ModelBase = ModelBase(decoder, optimizer_args=optimizer_args)

        self.prior = td.Normal(0, 1)
        self.posterior_fn = td.Normal
        self.kl_weight = kl_weight

    def compute_kl(self, sample, posterior):
        log_p = self.prior.log_prob(sample)
        log_q = posterior.log_prob(sample)

        kld = log_q.sum() - log_p.sum()
        kld /= sample.size(0)

        return kld

    def encode(self, x, stochastic=True, return_posterior=False):
        loc, scale = self.encoder(x).chunk(2, dim=1)

        if stochastic or return_posterior:
            scale = F.softplus(scale)
            posterior = self.posterior_fn(loc, scale)

        sample = posterior.rsample() if stochastic else loc

        if return_posterior:
            return sample, posterior
        else:
            return sample

    def routine(self, x, recon_loss_fn, stochastic=True):
        sample, posterior = self.encode(x, stochastic=stochastic, return_posterior=True)
        kld = self.compute_kl(sample, posterior)
        recon = self.decoder(sample)
        recon_loss = recon_loss_fn(recon, x)

        loss = recon_loss + self.kl_weight * kld

        return loss

    def fit(self, train_data, epochs, device, loss_fn=nn.MSELoss()):

        self.train()

        with trange(epochs) as pbar:
            for epoch in pbar:

                for x, _, _ in train_data:

                    x = x.to(device)

                    self.zero_grad()
                    loss = self.routine(x, recon_loss_fn=loss_fn)
                    loss.backward()
                    self.step()
                pbar.set_postfix(AE_loss=loss.detach().cpu().numpy())
