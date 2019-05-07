"""Utility functions for computing metrics"""
import torch.nn as nn
from models import MnistConvNet
from utils.training_utils import validate_classifier, classifier_training_loop


def evaluate_with_classifier(args, train_data, test_data, in_channels):
    """Evaluate by training a classifier and computing the accuracy on the test set"""
    if args.dataset == 'cmnist':
        meta_clf = MnistConvNet(in_channels=in_channels, out_dims=10, kernel_size=3,
                                hidden_sizes=[256, 256], output_activation=nn.LogSoftmax(dim=1))
    else:
        meta_clf = nn.Sequential(nn.Linear(in_features=in_channels, out_features=1), nn.Sigmoid())
    meta_clf = meta_clf.to(args.device)
    classifier_training_loop(args, meta_clf, train_data, val_data=test_data)

    _, acc = validate_classifier(args, meta_clf, test_data, use_s=True,
                                 pred_s=False, palette=None)
    return acc
