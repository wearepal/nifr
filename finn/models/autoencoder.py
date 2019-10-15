from tqdm import trange

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

        with trange(self.epochs) as pbar:
            for epoch in pbar:

                for x, _, _ in train_data:

                    x = x.to(self.device)

                    self.zero_grad()
                    loss = self.routine(x, loss_fn=loss_fn)
                    loss.backward()
                    self.step()
                pbar.set_postfix(MSE_loss=loss)
