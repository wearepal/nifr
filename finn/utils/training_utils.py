import argparse
from functools import partial
import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision
from tqdm import tqdm

from finn.models import MnistConvClassifier


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['adult', 'cmnist'], default='cmnist')
    parser.add_argument('--data-pcnt', type=float, metavar='P', default=1.0)
    parser.add_argument('--add-sampling-bias', type=eval, default=False, choices=[True, False],
                        help='if True, a sampling bias is added to the data')

    # Colored MNIST settings
    parser.add_argument('--scale', type=float, default=0.02)
    parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
    parser.add_argument('-bg', '--background', type=eval, default=True, choices=[True, False])
    parser.add_argument('--black', type=eval, default=False, choices=[True, False])
    parser.add_argument('--root', type=str, default="data")

    # path to data
    # parser.add_argument('--train_x', metavar="PATH")
    # parser.add_argument('--train_s', metavar="PATH")
    # parser.add_argument('--train_y', metavar="PATH")
    # parser.add_argument('--test_x', metavar="PATH")
    # parser.add_argument('--test_s', metavar="PATH")
    # parser.add_argument('--test_y', metavar="PATH")

    # parser.add_argument('--train_new', metavar="PATH")
    # parser.add_argument('--test_new', metavar="PATH")

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--nonlinearity', type=str, default="tanh")
    parser.add_argument('--glow', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--bn-lag', type=float, default=0)
    parser.add_argument('--inv-disc', type=eval, default=True, choices=[True, False])

    parser.add_argument('--early-stopping', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--disc-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save', type=str, default='experiments/finn')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--super-val', type=eval, default=False, choices=[True, False],
                        help='Train classifier on encodings as part of validation step.')
    parser.add_argument('--val-freq', type=int, default=4)
    parser.add_argument('--log-freq', type=int, default=10)

    parser.add_argument('--zs-frac', type=float, default=0.33)
    parser.add_argument('--zy-frac', type=float, default=0.33)

    parser.add_argument('--log-px-weight', type=float, default=1.)
    parser.add_argument('--pred-y-weight', type=float, default=1.)
    parser.add_argument('-pszyw', '--pred-s-from-zy-weight', type=float, default=1.)
    parser.add_argument('-pszsw', '--pred-s-from-zs-weight', type=float, default=1.)

    # classifier parameters (for computing fairness metrics)
    parser.add_argument('--clf-epochs', type=int, metavar='N', default=20)
    parser.add_argument('--clf-early-stopping', type=int, metavar='N', default=10)
    parser.add_argument('--clf-val-ratio', type=float, metavar='R', default=0.2)

    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use (if available)')
    parser.add_argument('--use-comet', type=eval, default=False, choices=[True, False],
                        help='whether to use the comet.ml logging')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of iterations without improvement in val loss before'
                             'reducing learning rate.')
    parser.add_argument('--meta-learn', type=eval, default=False, choices=[True, False],
                        help='Use meta learning procedure')

    return parser.parse_args(raw_args)


def find(value, value_list):
    result_list = [[i for i, val in enumerate(value.new_tensor(value_list)) if (s_i == val).all()] for s_i in value]
    return torch.tensor(result_list).flatten(start_dim=1)


def train_classifier(args, model, optimizer, train_data, use_s, pred_s, palette):

    if not isinstance(train_data, DataLoader):
        train_data = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    loss_fn = F.nll_loss if args.dataset == 'cmnist' else F.binary_cross_entropy
    model.train()
    for x, s, y in train_data:

        if pred_s:
            target = s
            # target = s
        else:
            target = y
        x = x.to(args.device)
        target = target.to(args.device)

        if args.dataset == 'adult' and use_s:
            x = torch.cat((x, s), dim=1)
        elif args.dataset == 'mnist' and not use_s:
            x = x.mean(dim=1, keepdim=True)

        optimizer.zero_grad()
        preds = model(x)

        loss = loss_fn(preds.float(), target.float(), reduction='sum')

        loss.backward()
        optimizer.step()


def validate_classifier(args, model, val_data, use_s, pred_s, palette):
    if not isinstance(val_data, DataLoader):
        val_data = DataLoader(val_data, shuffle=False, batch_size=args.test_batch_size)

    loss_fn = F.nll_loss if args.dataset == 'cmnist' else F.binary_cross_entropy

    with torch.no_grad():
        model.eval()
        val_loss = 0
        acc = 0
        for x, s, y in val_data:

            if pred_s:
                # TODO: do this in EthicML instead
                target = s
                # target = s
            else:
                target = y

            x = x.to(args.device)
            target = target.to(args.device)

            if args.dataset == 'adult' and use_s:
                x = torch.cat((x, s), dim=1)
            elif args.dataset == 'mnist' and not use_s:
                x = x.mean(dim=1, keepdim=True)

            preds = model(x)
            val_loss += loss_fn(preds.float(), target.float(), reduction='sum').item()

            if args.dataset == 'adult':
                acc += torch.sum(preds.round().long() == target).item()
            else:
                acc += torch.sum(preds.argmax(dim=1) == target).item()

        acc /= len(val_data.dataset)
        val_loss /= len(val_data.dataset)

        return val_loss, acc


def classifier_training_loop(args, model, train_data, val_data, use_s=True,
                             pred_s=False, palette=None):
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.test_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

    n_vals_without_improvement = 0

    best_loss = float('inf')

    print("Training classifier...")
    for _ in tqdm(range(args.clf_epochs)):

        if n_vals_without_improvement > args.clf_early_stopping > 0:
            break

        train_classifier(args, model, optimizer, train_loader, use_s, pred_s, palette)
        val_loss, _ = validate_classifier(args, model, val_loader, use_s, pred_s, palette)

        if val_loss < best_loss:
            best_loss = val_loss
            n_vals_without_improvement = 0
        else:
            n_vals_without_improvement += 1

    return model


def train_and_evaluate_classifier(args, data, palette, pred_s, use_s, model=None):

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

    in_dim = 3 if use_s else 1
    model = MnistConvClassifier(in_dim) if model is None else model
    model.to(args.device)

    model = classifier_training_loop(args, model, train_data, val_data, use_s=use_s,
                                     pred_s=pred_s, palette=palette)

    return partial(evaluate, args=args, model=model, palette=palette, batch_size=args.test_batch_size,
                   device=args.device, pred_s=pred_s, use_s=use_s)


def evaluate(args, test_data, model, palette, batch_size, device, pred_s=False, use_s=True):

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    with torch.no_grad():
        model.eval()

        # generate predictions
        all_preds = []
        all_targets = []
        for x, s, y in test_loader:

            if pred_s:
                # TODO: do this in EthicML instead
                target = s, palette
            else:
                target = y

            x = x.to(device)
            target = target.to(device)

            if args.dataset == 'adult' and use_s:
                x = torch.cat((x, s), dim=1)
            elif args.dataset == 'mnist' and not use_s:
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
    from finn import models

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


def reconstruct(args, z, model, zero_zy=False, zero_zs=False, zero_zn=False):
    """Reconstruct the input from the representation in various different ways"""
    z_ = z.clone()
    if zero_zy:
        z_[:, z_.size(1) - args.zy_dim:].zero_()
    if zero_zs:
        z_[:, args.zn_dim:z_.size(1) - args.zy_dim].zero_()
    if zero_zn:
        z_[:, :args.zn_dim].zero_()
    recon = model(z_, reverse=True)

    return recon


def reconstruct_all(args, z, model):
    recon_all = reconstruct(args, z, model)

    recon_y = reconstruct(args, z, model, zero_zs=True, zero_zn=True)
    recon_s = reconstruct(args, z, model, zero_zy=True, zero_zn=True)
    recon_n = reconstruct(args, z, model, zero_zy=True, zero_zs=True)

    recon_ys = reconstruct(args, z, model, zero_zn=True)
    recon_yn = reconstruct(args, z, model, zero_zs=True)
    return recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn


def encode_dataset(args, data, model):
    dataloader = DataLoader(data, shuffle=False, batch_size=args.test_batch_size)

    all_s = []
    all_y = []

    representations = ['all_z']
    if args.dataset == 'cmnist':
        representations.extend(['recon_all', 'recon_y', 'recon_s',
                                'recon_n', 'recon_yn', 'recon_ys'])
    representations = {key: [] for key in representations}

    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(dataloader):
            x = x.to(args.device)
            s = s.to(args.device)

            if args.dataset == 'adult':
                x = torch.cat((x, s), dim=1)

            z = model(x)

            if args.dataset == 'cmnist':
                recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(args, z,
                                                                                           model)
                representations['recon_all'].append(recon_all)
                representations['recon_y'].append(recon_y)
                representations['recon_s'].append(recon_s)
                representations['recon_n'].append(recon_n)
                representations['recon_ys'].append(recon_ys)
                representations['recon_yn'].append(recon_yn)

            representations['all_z'].append(z)
            all_s.append(s)
            all_y.append(y)
            # LOGGER.info('Progress: {:.2f}%', itr / len(dataloader) * 100)

    for key, entry in representations.items():
        if entry:
            representations[key] = torch.cat(entry, dim=0).detach().cpu()

    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    if args.dataset == 'cmnist':
        representations['zy'] = torch.utils.data.TensorDataset(
            representations['all_z'][:, z.size(1) - args.zy_dim:], all_s, all_y)
        representations['all_z'] = torch.utils.data.TensorDataset(
            representations['all_z'], all_s, all_y)
        representations['recon_y'] = torch.utils.data.TensorDataset(
            representations['recon_y'], all_s, all_y)
        representations['recon_s'] = torch.utils.data.TensorDataset(
            representations['recon_s'], all_s, all_y)

        return representations

    elif args.dataset == 'adult':
        representations['all_z'] = pd.DataFrame(representations['all_z'].numpy())
        columns = representations['all_z'].columns.astype(str)
        representations['all_z'].columns = columns
        representations['zy'] = representations['all_z'][columns[z.size(1) - args.zy_dim:]]
        representations['zs'] = representations['all_z'][columns[args.zn_dim:z.size(1) - args.zy_dim]]
        representations['zn'] = representations['all_z'][columns[:args.zn_dim]]
        representations['s'] = pd.DataFrame(all_s.cpu().numpy())
        representations['y'] = pd.DataFrame(all_y.cpu().numpy())

        return representations


def encode_dataset_no_recon(args, data, model):
    if not isinstance(data, DataLoader):
        data = DataLoader(data, shuffle=False, batch_size=args.test_batch_size)
    encodings = {'all_z': [], 'all_s': [], 'all_y': []}
    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(data):
            x = x.to(args.device)
            s = s.to(args.device)

            if args.dataset == 'adult':
                x = torch.cat((x, s), dim=1)
            z = model(x)

            encodings['all_z'].append(z)
            encodings['all_s'].append(s)
            encodings['all_y'].append(y)
            # LOGGER.info('Progress: {:.2f}%', itr / len(dataloader) * 100)

    for key, entry in encodings.items():
        if entry:
            encodings[key] = torch.cat(entry, dim=0).detach().cpu()

    if args.dataset == 'cmnist':
        encodings['zy'] = torch.utils.data.TensorDataset(
            encodings['all_z'][:, z.size(1) - args.zy_dim:], encodings['all_s'], encodings['all_y'])
        encodings['all_z'] = torch.utils.data.TensorDataset(
            encodings['all_z'], encodings['all_s'], encodings['all_y'])
        return encodings

    elif args.dataset == 'adult':
        encodings['all_z'] = pd.DataFrame(encodings['all_z'].numpy())
        columns = encodings['all_z'].columns.astype(str)
        encodings['all_z'].columns = columns
        encodings['zy'] = encodings['all_z'][columns[z.size(1) - args.zy_dim:]]
        encodings['zs'] = encodings['all_z'][columns[args.zn_dim:z.size(1) - args.zy_dim]]
        encodings['zn'] = encodings['all_z'][:columns[args.zn_dim]]
        encodings['s'] = pd.DataFrame(encodings['all_s'].cpu().numpy())
        encodings['y'] = pd.DataFrame(encodings['all_y'].cpu().numpy())
        return encodings
