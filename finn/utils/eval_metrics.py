"""Utility functions for computing metrics"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from finn.models import MnistConvNet
from finn.models.inv_discriminators import assemble_whole_model, multi_class_loss, binary_class_loss
from finn.models.nn_discriminators import compute_log_pz
from finn.utils.training_utils import validate_classifier, classifier_training_loop


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


def train_zy_head(args, trunk, discs, train_data, val_data, experiment):
    whole_model = assemble_whole_model(args, trunk, discs=discs)
    whole_model.eval()

    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size)

    head_optimizer = Adam(discs.y_from_zy.parameters(), lr=args.disc_lr,
                          weight_decay=args.weight_decay)

    n_vals_without_improvement = 0

    best_loss = float('inf')

    if args.dataset == 'cmnist':
        class_loss = multi_class_loss
    else:
        class_loss = binary_class_loss

    for epoch in range(args.clf_epochs):

        if n_vals_without_improvement > args.clf_early_stopping > 0:
            break

        print(f'====> Epoch {epoch} of z_y head training')

        discs.y_from_zy.train()
        with tqdm(total=len(train_loader)) as pbar:
            for i, (x, s, y) in enumerate(train_loader):

                x = x.to(args.device)
                s = s.to(args.device)
                y = y.to(args.device)
                if args.dataset == 'adult':
                    x = torch.cat((x, s.float()), dim=1)

                head_optimizer.zero_grad()

                zero = x.new_zeros(x.size(0), 1)
                z, delta_log_p = whole_model(x, zero)
                wh = z.size(1) // (args.zy_dim + args.zs_dim)
                zy, zs = z.split(split_size=[args.zy_dim * wh, args.zs_dim * wh], dim=1)
                pred_y_loss = args.pred_y_weight * class_loss(zy, y)
                log_pz = compute_log_pz(zy)

                log_px = args.log_px_weight * (log_pz - delta_log_p).mean()

                train_loss = -log_px + pred_y_loss
                train_loss.backward()

                head_optimizer.step()

                pbar.set_postfix(log_px=log_px.item(), pred_y_loss=pred_y_loss.item(), total_loss=train_loss.item())
                pbar.update()

        discs.y_from_zy.eval()

        with tqdm(total=len(val_loader)) as pbar:
            with torch.no_grad():
                acc = 0
                for i, (x, s, y) in enumerate(val_loader):

                    x = x.to(args.device)
                    s = s.to(args.device)
                    y = y.to(args.device)
                    if args.dataset == 'adult':
                        x = torch.cat((x, s.float()), dim=1)

                    zero = x.new_zeros(x.size(0), 1)
                    z, delta_log_p = whole_model(x, zero)

                    wh = z.size(1) // (args.zy_dim + args.zs_dim)
                    zy, zs = z.split(split_size=[args.zy_dim * wh, args.zs_dim * wh], dim=1)
                    pred_y_loss = args.pred_y_weight * class_loss(zy, y)
                    log_pz = compute_log_pz(zy)

                    log_px = args.log_px_weight * (log_pz - delta_log_p).mean()

                    val_loss = -log_px + pred_y_loss

                    acc_ = torch.sum(F.softmax(zy[:, :10], dim=1).argmax(dim=1) == y).item()
                    acc += acc_

                    if val_loss < best_loss:
                        best_loss = val_loss
                        n_vals_without_improvement = 0
                    else:
                        n_vals_without_improvement += 1

                    pbar.set_postfix(acc=acc_)
                    pbar.update()

            acc /= len(val_loader.dataset)
            print(f'===> Average val accuracy {acc:.4f}')

    return acc
