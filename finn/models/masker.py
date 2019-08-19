import torch
import torch.nn as nn

from finn.optimisation.optimizers import AdamExtGrad, RAdam
from finn.utils.distributions import logit, uniform_bernoulli
from finn.utils.utils import RoundSTE


class Masker(nn.Module):

    default_args = dict(optimizer_args=dict(lr=1e-2, weight_decay=0))

    def __init__(self, shape, optimizer_args=None, prob_1=0.5):
        super().__init__()

        if not (0. <= prob_1 <= 1.):
            raise ValueError(f"{prob_1} is not a valid probability.")

        optimizer_args = optimizer_args or self.default_args['optimizer_args']
        self.shape = shape
        self.prob_1 = prob_1

        self.mask = nn.Parameter(torch.empty(shape))
        self.reset_parameters()
        self.optimizer = RAdam([self.mask], **optimizer_args)

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


if __name__ == '__main__':
    masker = Masker((3, 28, 28))
    print(masker.mask)
