import torch
import numpy as np

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.-1e-5


def log_normal_log_sigma(x, mu, logsigma, average=False, reduce=True, dim=None):
    log_norm = float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)

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
    log_bern = y * torch.log(probs) + (1. - y) * torch.log(1. - probs)
    if reduce:
        if average:
            return torch.mean(log_bern, dim)
        else:
            return torch.sum(log_bern, dim)
    else:
        return log_bern


def log_normal_diag_deriv(x, mu, log_var):
    log_norm_deriv = - (x - mu) * torch.exp(-log_var)
    return log_norm_deriv


def log_normal_deriv(x, mu, inv_covar):
    log_norm_deriv = - (x - mu) @ inv_covar
    return log_norm_deriv


def log_bernoulli_deriv(mean):
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern_deriv = torch.log(probs) - torch.log(1. - probs)
    return log_bern_deriv
