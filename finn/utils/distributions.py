import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F


def logistic_mixture_logprob(x, locs=(0), scales=(1), weights=(0)):
    assert len(locs) == len(scales) == len(weights)

    weights = F.softmax(torch.tensor(list(weights)), dim=0)
    log_prob = 0
    for mu, sigma, pi in zip(locs, scales, weights):
        exp = -(x - mu) / sigma
        log_prob += pi * (exp - np.log(sigma) - 2 * F.softplus(exp))
    return log_prob


def logit(p, eps=1e-8):
    p = p.clamp(min=eps, max=1.-eps)
    return torch.log(p / (1. - p))


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


if __name__ == '__main__':

    probs = uniform_bernoulli((2, 2))
    print(probs)

