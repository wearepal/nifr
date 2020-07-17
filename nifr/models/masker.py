import torch
import torch.nn as nn
from torch.optim import SGD

from nifr.utils import RoundSTE
from nifr.utils.distributions import uniform_bernoulli
from nifr.utils.torch_ops import logit

__all__ = ["Masker"]


class Masker(nn.Module):

    default_args = dict(optimizer_args=dict(lr=1e-3, weight_decay=0))

    def __init__(self, shape, optimizer_args=None, prob_1=0.5):
        super().__init__()

        if not (0.0 <= prob_1 <= 1.0):
            raise ValueError(f"{prob_1} is not a valid probability.")

        optimizer_args = optimizer_args or self.default_args["optimizer_args"]
        self.shape = shape
        self.prob_1 = prob_1
        self.mask = nn.Parameter(torch.empty(shape))
        self.reset_parameters()
        self.optimizer = SGD([self.mask], **optimizer_args)

    def reset_parameters(self) -> None:
        probs = uniform_bernoulli(self.shape, self.prob_1)
        self.mask.data = logit(probs)

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def forward(self, threshold=True) -> torch.Tensor:
        out = self.mask.sigmoid()
        if threshold:
            out = RoundSTE.apply(out)

        return out
