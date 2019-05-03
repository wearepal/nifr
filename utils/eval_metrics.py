"""Utility functions for computing metrics"""
import torch.nn as nn
from models import MnistConvNet
from utils.training_utils import (validate_classifier, classifier_training_loop,
                                  encode_dataset_no_recon)


def evaluate_metalearner(args, model, train_zy, dagger_data):
    """Evaluate the metalearning model"""
    print('Encoding dagger set...')
    dagger_repr = encode_dataset_no_recon(args, dagger_data, model)

    if args.dataset == 'cmnist':
        meta_clf = MnistConvNet(in_channels=args.zy_dim, out_dims=10, kernel_size=3,
                                hidden_sizes=[256, 256], output_activation=nn.LogSoftmax(dim=1))
    else:
        meta_clf = nn.Sequential(nn.Linear(in_features=args.zy_dim, out_features=1), nn.Sigmoid())
    meta_clf = meta_clf.to(args.device)
    classifier_training_loop(args, meta_clf, train_zy, val_data=dagger_repr['zy'])

    _, acc = validate_classifier(args, meta_clf, dagger_repr['zy'], use_s=True,
                                 pred_s=False, palette=None)
    return acc
