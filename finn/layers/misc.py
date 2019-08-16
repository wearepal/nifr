import torch
import torch.nn as nn
import torch.distributions as td


class _OneHotEncoder(nn.Module):
    def __init__(self, n_dims, index_dim=1):
        super().__init__()
        self.n_dims = n_dims
        self.index_dim = index_dim

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        return td.OneHotCategorical(probs).sample()
