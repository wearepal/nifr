import argparse
import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision

from models import MnistConvClassifier
from functools import partial


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['adult', 'cmnist'], default='cmnist')
    parser.add_argument('--data_pcnt', type=float, metavar='P', default=1.0)

    # Colored MNIST settings
    parser.add_argument('--scale', type=float, default=0.02)
    parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
    parser.add_argument('-bg', '--background', type=eval, default=True, choices=[True, False])
    parser.add_argument('--black', type=eval, default=False, choices=[True, False])
    parser.add_argument('--root', type=str, default="data")

    parser.add_argument('--train_x', metavar="PATH")
    parser.add_argument('--train_s', metavar="PATH")
    parser.add_argument('--train_y', metavar="PATH")
    parser.add_argument('--test_x', metavar="PATH")
    parser.add_argument('--test_s', metavar="PATH")
    parser.add_argument('--test_y', metavar="PATH")

    parser.add_argument('--train_new', metavar="PATH")
    parser.add_argument('--test_new', metavar="PATH")

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--nonlinearity', type=str, default="tanh")
    parser.add_argument('--glow', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--bn_lag', type=float, default=0)

    parser.add_argument('--early_stopping', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--disc_lr', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save', type=str, default='experiments/finn')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--val_freq', type=int, default=4)
    parser.add_argument('--log_freq', type=int, default=10)

    parser.add_argument('--ind-method', type=str, choices=['hsic', 'disc'], default='disc')
    parser.add_argument('--ind-method2t', type=str, choices=['hsic', 'none'], default='none')

    parser.add_argument('--zs_frac', type=int, default=0.33)
    parser.add_argument('--zy_frac', type=int, default=0.33)

    parser.add_argument('--log_px_weight', type=float, default=1.)
    parser.add_argument('--pred_y_weight', type=float, default=1.)
    parser.add_argument('-pszyw', '--pred_s_from_zy_weight', type=float, default=1.)
    parser.add_argument('-pszsw', '--pred_s_from_zs_weight', type=float, default=1.)

    # classifier parameters (for computing fairness metrics)
    parser.add_argument('--clf-epochs', type=int, metavar='N', default=20)
    parser.add_argument('--clf-early-stopping', type=int, metavar='N', default=10)
    parser.add_argument('--clf-val-ratio', type=float, metavar='R', default=0.2)

    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use (if available)')
    parser.add_argument('--use_comet', type=eval, default=False, choices=[True, False],
                        help='whether to use the comet.ml logging')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of iterations without improvement in val loss before'
                             'reducing learning rate.')

    return parser.parse_args()


def find(value, value_list):
    result_list = [[i for i, val in enumerate(value.new_tensor(value_list)) if (s_i == val).all()] for s_i in value]
    return torch.tensor(result_list).flatten()


def run_conv_classifier(args, data, palette, pred_s, use_s):

    # LOGGER = utils.get_logger(logpath=save_dir / 'logs', filepath=Path(__file__).resolve())
    #
    # # ==== check GPU ====
    # args.device = torch.device(f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu")
    # LOGGER.info('{} GPUs available.', torch.cuda.device_count())

    # ==== construct dataset ====
    args.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

    train_len = int((1 - args.clf_val_ratio) * len(data))
    val_length = int(len(data) - train_len)

    lengths = [train_len, val_length]
    train_data, val_data = random_split(data, lengths=lengths)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.test_batch_size)

    in_dim = 3 if use_s else 1
    model = MnistConvClassifier(in_dim).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

    n_vals_without_improvement = 0

    best_loss = float('inf')

    for i in range(args.clf_epochs):

        if n_vals_without_improvement > args.clf_early_stopping > 0:
            break

        model.train()
        for x, s, y in train_loader:

            if pred_s:
                # TODO: do this in EthicML instead
                target = find(s, palette)
                # target = s
            else:
                target = y
            x = x.to(args.device)
            target = target.to(args.device)

            if not use_s:
                x = x.mean(dim=1, keepdim=True)

            optimizer.zero_grad()
            preds = model(x)

            loss = F.nll_loss(preds, target, reduction='sum')

            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = 0
            for x, s, y in val_loader:

                if pred_s:
                    # TODO: do this in EthicML instead
                    target = find(s, palette)
                    # target = s
                else:
                    target = y

                x = x.to(args.device)
                target = target.to(args.device)

                if not use_s:
                    x = x.mean(dim=1, keepdim=True)

                preds = model(x)
                val_loss += F.nll_loss(preds, target, reduction='sum').item()

            val_loss /= len(val_loader)

            if val_loss < best_loss:
                best_loss = val_loss
                n_vals_without_improvement = 0
            else:
                n_vals_without_improvement += 1

        for x, s, y in train_loader:

            if pred_s:
                # TODO: do this in EthicML instead
                target = find(s, palette)
                # target = s
            else:
                target = y

            x = x.to(args.device)
            target = target.to(args.device)

            if not use_s:
                x = x.mean(dim=1, keepdim=True)

            optimizer.zero_grad()
            preds = model(x)

            loss = F.nll_loss(preds, target, reduction='sum')

            loss.backward()

    return partial(evaluate, model=model, palette=palette, batch_size=args.test_batch_size,
                   device=args.device, pred_s=pred_s, use_s=use_s)


def evaluate(test_data, model, palette, batch_size, device, pred_s=False, use_s=True):

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    with torch.no_grad():
        model.eval()

        # generate predictions
        all_preds = []
        all_targets = []
        for x, s, y in test_loader:

            if pred_s:
                # TODO: do this in EthicML instead
                target = find(s, palette)
                # target = s
            else:
                target = y

            x = x.to(device)
            target = target.to(device)

            if not use_s:
                x = x.mean(dim=1, keepdim=True)

            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())
    return pd.DataFrame(all_preds, columns=['preds']), pd.DataFrame(all_targets, columns=['y'])


def get_data_dim(data_loader):
    x, _, _ = next(iter(data_loader))
    x_dim = x.size(1)
    x_dim_flat = np.prod(x.shape[1:]).item()

    return x_dim, x_dim_flat


def metameric_sampling(model, xzx, xzs, zs_dim):
    xzx_dim, xzs_dim = xzx.dim(), xzs.dim()

    if xzx_dim == 1 or xzx_dim == 3:
        xzx = xzx.unsqueeze(0)

    if xzs_dim == 1 or xzs_dim == 3:
        xzs = xzs.unsqueeze(0)

    zx = model(xzx)[:, zs_dim]
    zs = model(xzs)[:, zs_dim:]

    zm = torch.cat((zx, zs), dim=1)
    xm = model(zm, reverse=True)

    return xm


def fetch_model(args, x_dim):
    import models

    if args.dataset == 'cmnist':
        model = models.glow(args, x_dim).to(args.device)
    elif args.dataset == 'adult':
        model = models.tabular_model(args, x_dim).to(args.device)
    else:
        raise NotImplementedError("Only works for cmnist and adult - How have you even got"
                                  "hererere?????")
    return model


def log_images(experiment, image_batch, name, nsamples=64, nrows=8, monochrome=False, train=True):
    """Make a grid of the given images, save them in a file and log them with Comet"""
    prefix = "train_" if train else "test_"
    images = image_batch[:nsamples]
    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    experiment.log_image(torchvision.transforms.functional.to_pil_image(shw), name=prefix + name)
