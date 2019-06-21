import torch
import torch.nn.functional as F
import numpy as np

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.0 - 1e-5


def logistic_mixture_logprob(x, locs=(0), scales=(1), weights=(0)):
    assert len(locs) == len(scales) == len(weights)

    weights = F.softmax(torch.Tensor(list(weights)), dim=0)
    log_prob = 0
    for mu, sigma, pi in zip(locs, scales, weights):
        exp = -(x - mu) / sigma
        log_prob += pi * (exp - np.log(sigma) - 2 * F.softplus(exp))
    return log_prob


def log_normal_log_sigma(x, mu, logsigma, average=False, reduce=True, dim=None):
    log_norm = (
        float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu) ** 2 / (2 * torch.exp(logsigma) ** 2)
    )

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    # log_norm = x
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * torch.exp(-log_var))
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal(x, mean, inv_covar, logdet_covar, average=False, reduce=True, dim=None):
    log_norm = -0.5 * ((x - mean) @ inv_covar * (x - mean))
    if reduce:
        if average:
            log_norm = torch.mean(log_norm, dim)
        else:
            log_norm = torch.sum(log_norm, dim)
    log_norm += -0.5 * logdet_covar
    return log_norm


def log_bernoulli(y, mean, average=False, reduce=True, dim=None):
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern = y * torch.log(probs) + (1.0 - y) * torch.log(1.0 - probs)
    if reduce:
        if average:
            return torch.mean(log_bern, dim)
        else:
            return torch.sum(log_bern, dim)
    else:
        return log_bern


def log_normal_diag_deriv(x, mu, log_var):
    log_norm_deriv = -(x - mu) * torch.exp(-log_var)
    return log_norm_deriv


def log_normal_deriv(x, mu, inv_covar):
    log_norm_deriv = -(x - mu) @ inv_covar
    return log_norm_deriv


def log_bernoulli_deriv(mean):
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern_deriv = torch.log(probs) - torch.log(1.0 - probs)
    return log_bern_deriv


def sample_gumbel(shape, device, eps=1e-20):
    u = torch.rand(shape).to(device)
    return torch.log(-torch.log(u + eps) + eps)


def sample_gumbel_softmax(logits, temperature):
    y = logits + sample_gumbel(logits.size(), logits.device)
    y = F.softmax(y / temperature, dim=-1)
    return y


def one_hot(y):
    ind = y.argmax(-1)
    y_hard = torch.zeros_like(y).view(-1, y.size(-1))
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view_as(y)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(y.size(0), -1)


def gumbel_softmax(logits, temperature):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = sample_gumbel_softmax(logits, temperature)
    ind = y.argmax(-1)
    y_hard = torch.zeros_like(y).view(-1, y.size(-1))
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view_as(y)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(logits.size(0), -1)
