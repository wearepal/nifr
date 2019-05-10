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
from finn.utils.training_utils import validate_classifier, classifier_training_loop, evaluate


def evaluate_with_classifier(args, train_data, test_data, in_channels, pred_s=False, use_s=False, applicative=False):
    """Evaluate by training a classifier and computing the accuracy on the test set"""
    if args.dataset == 'cmnist':
        meta_clf = MnistConvNet(in_channels=in_channels, out_dims=10, kernel_size=3,
                                hidden_sizes=[256, 256], output_activation=nn.LogSoftmax(dim=1))
    else:
        meta_clf = nn.Sequential(nn.Linear(in_features=in_channels, out_features=1), nn.Sigmoid())
    meta_clf = meta_clf.to(args.device)
    classifier_training_loop(args, meta_clf, train_data, val_data=test_data)

    if applicative:
        return partial(evaluate, args=args, model=meta_clf, batch_size=args.test_batch_size,
                       device=args.device, pred_s=pred_s, use_s=use_s, using_x=False)
    else:
        _, acc = validate_classifier(args, meta_clf, test_data, use_s=True, pred_s=False)
        return acc


def train_zy_head(args, trunk, discs, train_data, val_data):
    assert isinstance(discs, InvDisc)
    assert isinstance(train_data, Dataset)
    assert isinstance(val_data, Dataset)
    disc_y_from_zy_copy = tabular_model(args,
                                        input_dim=args.zy_dim,
                                        depth=2,
                                        batch_norm=False)
    disc_y_from_zy_copy.load_state_dict(discs.y_from_zy.state_dict())
    disc_y_from_zy_copy.to(args.device)
    disc_y_from_zy_copy.train()

    discs.assemble_whole_model(trunk).eval()

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size)

    head_optimizer = Adam(disc_y_from_zy_copy.parameters(), lr=args.disc_lr,
                          weight_decay=args.weight_decay)
    standard_normal = torch.distributions.Normal(0, 1)

    n_vals_without_improvement = 0

    best_loss = float('inf')

    if args.dataset == 'cmnist':
        class_loss = discs.multi_class_loss
    else:
        class_loss = discs.binary_class_loss

    for epoch in range(args.clf_epochs):

        if n_vals_without_improvement > args.clf_early_stopping > 0:
            break

        print(f'====> Epoch {epoch} of z_y head training')

        with tqdm(total=len(train_loader)) as pbar:
            for i, (x, s, y) in enumerate(train_loader):

                x = x.to(args.device)
                s = s.to(args.device)
                y = y.to(args.device)

                if args.dataset == 'adult':
                    x = torch.cat((x, s.float()), dim=1)

                head_optimizer.zero_grad()

                zero = x.new_zeros(x.size(0), 1)
                z, delta_log_p = trunk(x, zero)
                _, zy_trunk = discs.split_zs_zy(z)
                zy_new, delta_log_p_ = disc_y_from_zy_copy(zy_trunk, delta_log_p)

                pred_y_loss = args.pred_y_weight * class_loss(zy_new, y)
                log_pz = compute_log_pz(zy_new)

                zy_old = discs.y_from_zy(zy_trunk)

                regularization = (standard_normal.log_prob(zy_new).sum(dim=1) -
                                  standard_normal.log_prob(zy_old).sum(dim=1)).mean()

                log_px = args.log_px_weight * (log_pz - delta_log_p).mean()

                train_loss = -log_px + pred_y_loss + regularization
                train_loss.backward()

                head_optimizer.step()

                pbar.set_postfix(log_px=log_px.item(), pred_y_loss=pred_y_loss.item(),
                                 total_loss=train_loss.item())
                pbar.update()

        print(f'====> Validating...')
        disc_y_from_zy_copy.eval()
        with tqdm(total=len(val_loader)) as pbar:
            with torch.no_grad():
                acc_meter = utils.AverageMeter()
                val_loss_meter = utils.AverageMeter()
                for i, (x, s, y) in enumerate(val_loader):

                    x = x.to(args.device)
                    s = s.to(args.device)
                    y = y.to(args.device)

                    if args.dataset == 'adult':
                        x = torch.cat((x, s.float()), dim=1)

                    zero = x.new_zeros(x.size(0), 1)
                    z, delta_log_p = trunk(x, zero)
                    _, zy_trunk = discs.split_zs_zy(z)
                    zy_new, delta_log_p_ = disc_y_from_zy_copy(zy_trunk, delta_log_p)

                    pred_y_loss = args.pred_y_weight * class_loss(zy_new, y)
                    log_pz = compute_log_pz(zy_new)

                    zy_old = discs.y_from_zy(zy_trunk)

                    regularization = - (standard_normal.log_prob(zy_new).sum()
                                        - standard_normal.log_prob(zy_old).sum())

                    log_px = args.log_px_weight * (log_pz - delta_log_p).mean()

                    val_loss = -log_px + pred_y_loss + regularization

                    if args.dataset == 'adult':
                        acc = torch.sum((zy_new[:, :args.y_dim].sigmoid().round()) == y).item()
                    else:
                        acc = torch.sum(F.softmax(zy_new[:, :args.y_dim], dim=1).argmax(dim=1) == y).item()

                    acc_meter.update(acc / x.size(0), n=x.size(0))
                    val_loss_meter.update(val_loss.item(), n=x.size(0))

                    pbar.set_postfix(acc=acc / x.size(0))
                    pbar.update()

                val_loss = val_loss_meter.avg

                if val_loss < best_loss:
                    best_loss = val_loss
                    n_vals_without_improvement = 0
                else:
                    n_vals_without_improvement += 1

            avg_acc = acc_meter.avg
            print(f'===> Average val accuracy {avg_acc:.4f}')

    return avg_acc
