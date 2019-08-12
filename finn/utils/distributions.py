import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
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


def uniform_bernoulli(shape, prob_1=0.5, eps=1.e-8):
    n_element = (np.product(shape),)
    samples = torch.empty(n_element, dtype=torch.float32)
    bern = dist.Bernoulli(probs=prob_1)
    indexes = bern.sample(n_element).long()
    nn.init.uniform_(samples[1. - indexes], a=0, b=0.5-eps)
    nn.init.uniform_(samples[indexes], a=0.5, b=1.0)§
    samples = samples.view(shape)

    return samples
