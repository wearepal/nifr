import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F


def uniform_bernoulli(shape, prob_1=0.5):
    nelement = int(np.product(shape))
    bern = td.Bernoulli(probs=prob_1)
    indexes = bern.sample((nelement,)).to(torch.bool)
    samples = torch.ones(nelement)

    ones = samples[indexes]
    ones.uniform_(0.5, 1.0)
    zeros = samples[~indexes]
    zeros.uniform_(0, 0.5)

    samples[indexes] = ones
    samples[~indexes] = zeros

    samples = samples.view(shape)

    return samples
