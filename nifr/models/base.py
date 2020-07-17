import torch.nn as nn

from nifr.utils.optimizers import RAdam

__all__ = ["ModelBase"]


class ModelBase(nn.Module):

    default_kwargs = dict(optimizer_kwargs=dict(lr=1e-3, weight_decay=0))

    def __init__(self, model, optimizer_kwargs=None):
        super().__init__()
        self.model = model
        optimizer_kwargs = optimizer_kwargs or self.default_kwargs["optimizer_kwargs"]
        self.optimizer = RAdam(self.model.parameters(), **optimizer_kwargs)

    def reset_parameters(self):
        def _reset_parameters(m: nn.Module):
            if hasattr(m.__class__, "reset_parameters") and callable(
                getattr(m.__class__, "reset_parameters")
            ):
                m.reset_parameters()

        self.model.apply(_reset_parameters)

    def step(self, grads=None):
        self.optimizer.step(grads)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def forward(self, inputs):
        return self.model(inputs)
