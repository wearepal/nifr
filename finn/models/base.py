import torch.nn as nn

from finn.optimisation.radam import RAdam


class BaseModel(nn.Module):

    default_args = dict(optimizer_args=dict(lr=1e-3, weight_decay=0))

    def __init__(self, model, optimizer_args=None):
        super(BaseModel, self).__init__()
        optimizer_args = optimizer_args or self.default_args["optimizer_args"]
        self.model = model
        self.optimizer = RAdam(self.model.parameters(), **optimizer_args)

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

    def train(self, *args):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, inputs):
        return self.model(inputs)
