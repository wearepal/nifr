import numpy as np
import torch
import torch.distributions as dist
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
    bern = dist.Bernoulli(probs=prob_1)
    indexes = bern.sample(shape).long()

    samples = torch.empty(shape)
    samples[1 - indexes].data.uniform_(0, 0.5)
    samples[indexes].data.uniform_(0.5, 1)

    return samples
