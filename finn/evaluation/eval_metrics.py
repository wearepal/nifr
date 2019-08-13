"""Utility functions for computing metrics"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm

from finn.models import MnistConvNet, InvDisc, compute_log_pz, tabular_model
from finn.utils import utils
from finn.optimisation.training_utils import (
    validate_classifier,
    classifier_training_loop,
    evaluate,
    reconstruct_all,
    log_images,
)


def evaluate_with_classifier(
    args, train_data, test_data, in_channels, pred_s=False, use_s=False, applicative=False
):
    """Evaluate by training a classifier and computing the accuracy on the test set"""
    if args.dataset == 'cmnist':
        meta_clf = MnistConvNet(
            in_channels=in_channels,
            out_dims=10,
            kernel_size=3,
            hidden_sizes=[256, 256],
            output_activation=nn.LogSoftmax(dim=1),
        )
    else:
        meta_clf = nn.Sequential(nn.Linear(in_features=in_channels, out_features=1), nn.Sigmoid())
    meta_clf = meta_clf.to(args.device)
    classifier_training_loop(args, meta_clf, train_data, val_data=test_data)

    if applicative:
        return partial(
            evaluate,
            args=args,
            model=meta_clf,
            batch_size=args.test_batch_size,
            device=args.device,
            pred_s=pred_s,
            use_s=use_s,
            using_x=False,
        )
    else:
        _, acc = validate_classifier(args, meta_clf, test_data, use_s=True, pred_s=False)
        return acc
