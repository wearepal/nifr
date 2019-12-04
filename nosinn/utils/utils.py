import os
import logging
import random
from typing import Dict, Any, Optional, List, Iterable

import numpy as np
import torch
import torch.nn as nn
import wandb


LOGGER = None

__all__ = [
    "AverageMeter",
    "RunningAverageMeter",
    "count_parameters",
    "get_logger",
    "random_seed",
    "save_checkpoint",
    "wandb_log",
]


def wandb_log(args, row: Dict[str, Any], commit: bool = True, step: Optional[int] = None):
    """Wrapper around wandb's log function"""
    if args.use_wandb:
        wandb.log(row, commit=commit, step=step)


class BraceString(str):
    def __mod__(self, other):
        return self.format(*other)

    def __str__(self):
        return self


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super(StyleAdapter, self).__init__(logger, extra)

    def process(self, msg, kwargs):
        # if kwargs.pop('style', "%") == "{":  # optional
        msg = BraceString(msg)
        return msg, kwargs


def get_logger(
    logpath: str, filepath: str, package_files: Optional[List] = None, displaying: bool = True,
    saving: bool = True, debug: bool = False
):
    global LOGGER
    if LOGGER is not None:
        return LOGGER
    package_files = package_files or []

    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    LOGGER = StyleAdapter(logger)
    return LOGGER


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: float = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, momentum: float =0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val: float = None
        self.avg: float = 0

    def update(self, val: float):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def save_checkpoint(state: dict, save: str, epoch: int):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "checkpt-%04d.pth" % epoch)
    torch.save(state, filename)


def count_parameters(model: nn.Module):
    """Count all parameters (that have a gradient) in the given model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def random_seed(seed_value: int, use_cuda: bool) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False
