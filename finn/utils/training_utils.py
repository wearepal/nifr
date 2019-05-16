import argparse
from functools import partial
import numpy as np
import pandas as pd
import torch

import torch.nn.functional as F
from ethicml.algorithms.utils import DataTuple
from ethicml.data import Adult
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import torchvision
from tqdm import tqdm

from finn.models import MnistConvClassifier


def parse_arguments(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['adult', 'cmnist'], default='cmnist')
    parser.add_argument('--data-pcnt', type=restricted_float, metavar='P', default=1.0,
                        help="data %% should be a real value > 0, and up to 1")
    parser.add_argument('--add-sampling-bias', type=eval, default=False, choices=[True, False],
                        help='if True, a sampling bias is added to the data')

    # Colored MNIST settings
    parser.add_argument('--scale', type=float, default=0.02)
    parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
    parser.add_argument('-bg', '--background', type=eval, default=False, choices=[True, False])
    parser.add_argument('--black', type=eval, default=True, choices=[True, False])
    parser.add_argument('--binarize', type=eval, default=True, choices=[True, False])
    parser.add_argument('--root', type=str, default="data")

    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dims', type=str, default="100-100")
    parser.add_argument('--glow', type=eval, default=True, choices=[True, False])
    parser.add_argument('--batch-norm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--bn-lag', type=float, default=0)
    parser.add_argument('--inv-disc', type=eval, default=True, choices=[True, False])
    parser.add_argument('--inv-disc-depth', type=int, default=2)

    parser.add_argument('--early-stopping', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--disc-lr', type=float, default=1e-4)
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

    parser.add_argument('--log-px-weight', type=float, default=1.e-3)
    parser.add_argument('-pyzyw', '--pred-y-weight', type=float, default=1.)
    parser.add_argument('-pszyw', '--pred-s-from-zy-weight', type=float, default=1.)
    parser.add_argument('-pszsw', '--pred-s-from-zs-weight', type=float, default=1.)

    # classifier parameters (for computing fairness metrics)
    parser.add_argument('--clf-epochs', type=int, metavar='N', default=50)
    parser.add_argument('--clf-early-stopping', type=int, metavar='N', default=20)
    parser.add_argument('--clf-val-ratio', type=float, metavar='R', default=0.2)
    parser.add_argument('--clf-reg-weight', type=float, metavar='R', default=1.e-7)

    parser.add_argument('--gpu', type=int, default=0, help='Which GPU to use (if available)')
    parser.add_argument('--use-comet', type=eval, default=False, choices=[True, False],
                        help='whether to use the comet.ml logging')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Gamma value for Exponential Learning Rate scheduler. '
                             'Value of 0.95 arbitrarily chosen.')
    parser.add_argument('--meta-learn', type=eval, default=True, choices=[True, False],
                        help='Use meta learning procedure')

    return parser.parse_args(raw_args)


def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x


def find(value, value_list):
    result_list = [[i for i, val in enumerate(value.new_tensor(value_list)) if (s_i == val).all()] for s_i in value]
    return torch.tensor(result_list).flatten(start_dim=1)


def train_classifier(args, model, optimizer, train_data, use_s, pred_s):

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
        target = target.to(args.device).long()

        if args.dataset == 'adult' and use_s:
            x = torch.cat((x, s), dim=1)
        elif args.dataset == 'cmnist' and not use_s:
            x = x.mean(dim=1, keepdim=True)

        optimizer.zero_grad()
        preds = model(x)

        loss = loss_fn(preds.float(), target, reduction='mean')

        loss.backward()
        optimizer.step()


def validate_classifier(args, model, val_data, use_s, pred_s):
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
            elif args.dataset == 'cmnist' and not use_s:
                x = x.mean(dim=1, keepdim=True)

            preds = model(x)
            val_loss += loss_fn(preds.float(), target.long(), reduction='sum').item()

            if args.dataset == 'adult':
                acc += torch.sum(preds.round().long() == target).item()
            else:
                acc += torch.sum(preds.argmax(dim=1) == target).item()

        acc /= len(val_data.dataset)
        val_loss /= len(val_data.dataset)

        return val_loss, acc


def classifier_training_loop(args, model, train_data, val_data, use_s=True,
                             pred_s=False):
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.test_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)

    n_vals_without_improvement = 0

    best_acc = 0

    print("Training classifier...")
    t = tqdm(range(args.clf_epochs))  # we do it like this so we can close in manually
    for _ in t:

        if n_vals_without_improvement > args.clf_early_stopping > 0:
            t.close()  # close it manually
            break

        train_classifier(args, model, optimizer, train_loader, use_s, pred_s)
        _, acc = validate_classifier(args, model, val_loader, use_s, pred_s)

        if acc > best_acc:
            best_acc = acc
            n_vals_without_improvement = 0
        else:
            n_vals_without_improvement += 1

    return model


def train_and_evaluate_classifier(args, experiment, data, pred_s, use_s, model=None, name=None):

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
                                     pred_s=pred_s)

    return partial(evaluate, args=args, model=model, batch_size=args.test_batch_size,
                   device=args.device, pred_s=pred_s, use_s=use_s, experiment=experiment, name=name)


def evaluate(args, test_data, model, batch_size, device, pred_s=False, use_s=True, using_x=True, experiment=None, name=None):
    """
    Evaluate a model on a given test set and return the predictions

    :param args: Our global store
    :param test_data: evaluate gets passed around as a partial function. test_data is
                        the value that is supplied that we evaluate
    :param model: the model that we want to run the test data on
    :param batch_size:
    :param device:
    :param pred_s: whether we want to predict s (in which case set s
                    from the test_data as the target), or not
    :param use_s: Should we include S as input to the model (as well as x)
    :param using_x: are we training the model in x space or z space? If training
                    in x mark this as true, else we'll assume we're running in z space
    :return: a dataframe of predictions
    """

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    with torch.no_grad():
        model.eval()

        # generate predictions
        all_preds = []
        all_targets = []
        for i, (x, s, y) in enumerate(test_loader):

            target = s if pred_s else y

            x = x.to(device)
            target = target.to(device)

            if args.dataset == 'adult' and use_s:
                x = torch.cat((x, s), dim=1)
            elif args.dataset == 'cmnist' and not use_s and using_x:
                x = x.mean(dim=1, keepdim=True)
            if experiment is not None and args.dataset == 'cmnist' and i == 0:
                log_images(experiment, x, f"evaluation on {name}", prefix='eval')


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


def log_images(experiment, image_batch, name, nsamples=64, nrows=8, monochrome=False, prefix=None):
    """Make a grid of the given images, save them in a file and log them with Comet"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]
    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    experiment.log_image(torchvision.transforms.functional.to_pil_image(shw), name=prefix + name)


def reconstruct(args, z, model, zero_zy=False, zero_zs=False, zero_sn=False, zero_yn=False):
    """Reconstruct the input from the representation in various different ways"""
    z_ = z.clone()
    wh = z.size(1) // (args.zs_dim + args.zy_dim + args.zn_dim)


    if zero_zy:
        if args.inv_disc:
            z_[:, (z_.size(1) - args.zy_dim * wh):][:, :args.y_dim].zero_()
        else:
            z_[:, (z_.size(1) - args.zy_dim):].zero_()
    if zero_zs:
        if args.inv_disc:
            z_[:, (args.zn_dim * wh): (z_.size(1) - (args.zy_dim * wh))][:, :args.s_dim].zero_()
        else:
            z_[:, args.zn_dim: (z_.size(1) - args.zy_dim)].zero_()
    if args.inv_disc:
        if zero_yn:
            z_[:, (z_.size(1) - args.zy_dim * wh):][:, args.y_dim:].zero_()
        if zero_sn:
            z_[:, args.zn_dim: (z_.size(1) - args.zy_dim * wh)][:, args.s_dim:].zero_()
    elif zero_sn or zero_yn:
        z_[:, :args.zn_dim].zero_()

    recon = model(z_, reverse=True)

    return recon


def reconstruct_all(args, z, model):
    recon_all = reconstruct(args, z, model, zero_zy=False, zero_zs=False, zero_sn=False, zero_yn=False)

    recon_y = reconstruct(args, z, model, zero_zy=False, zero_zs=True, zero_sn=True, zero_yn=True)
    recon_s = reconstruct(args, z, model, zero_zy=True, zero_zs=False, zero_sn=True, zero_yn=True)
    recon_n = reconstruct(args, z, model, zero_zy=True, zero_zs=True, zero_sn=False, zero_yn=False)

    recon_ys = reconstruct(args, z, model, zero_zy=False, zero_zs=False, zero_sn=True, zero_yn=True)
    recon_yn = reconstruct(args, z, model, zero_zy=False, zero_zs=True, zero_sn=True, zero_yn=False)
    recon_sn = reconstruct(args, z, model, zero_zy=True, zero_zs=False, zero_sn=False, zero_yn=True)
    return recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn, recon_sn


def encode_dataset(args, data, model):
    dataloader = DataLoader(data, shuffle=False, batch_size=args.test_batch_size)

    all_s = []
    all_y = []

    representations = ['all_z']
    representations.extend(['recon_all', 'recon_y', 'recon_s',
                            'recon_n', 'recon_yn', 'recon_ys', 'recon_sn'])
    representations = {key: [] for key in representations}

    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(dataloader):
            x = x.to(args.device)
            s = s.to(args.device)

            if args.dataset == 'adult':
                x = torch.cat((x, s), dim=1)

            z = model(x)

            recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn, recon_sn = reconstruct_all(args, z, model)
            representations['recon_all'].append(recon_all)
            representations['recon_y'].append(recon_y)
            representations['recon_s'].append(recon_s)
            representations['recon_n'].append(recon_n)
            representations['recon_ys'].append(recon_ys)
            representations['recon_yn'].append(recon_yn)
            representations['recon_sn'].append(recon_sn)

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
        representations['zy'] = TensorDataset(
            representations['all_z'][:, z.size(1) - args.zy_dim:], all_s, all_y)
        representations['zs'] = TensorDataset(representations['all_z'][:, args.zn_dim:-args.zy_dim], all_s, all_y)
        representations['zn'] = TensorDataset(representations['all_z'][:, :args.zn_dim], all_s, all_y)
        representations['all_z'] = TensorDataset(representations['all_z'], all_s, all_y)
        representations['recon_y'] = TensorDataset(representations['recon_y'], all_s, all_y)
        representations['recon_s'] = TensorDataset(representations['recon_s'], all_s, all_y)

    elif args.dataset == 'adult':
        representations['all_z'] = pd.DataFrame(representations['all_z'].numpy())
        columns = representations['all_z'].columns.astype(str)
        representations['all_z'].columns = columns

        representations['s'] = pd.DataFrame(all_s.float().cpu().numpy())
        representations['y'] = pd.DataFrame(all_y.float().cpu().numpy())

        representations['zy'] = DataTuple(x=representations['all_z'][columns[z.size(1) - args.zy_dim:]], s=representations['s'], y=representations['y'])
        representations['zs'] = DataTuple(x=representations['all_z'][columns[args.zn_dim:z.size(1) - args.zy_dim]], s=representations['s'], y=representations['y'])
        representations['zn'] = DataTuple(x=representations['all_z'][columns[:args.zn_dim]], s=representations['s'], y=representations['y'])

        representations['all_z'] = DataTuple(x=representations['all_z'], s=representations['s'], y=representations['y'])

        recon_all = 'fff'

        sens_attrs = Adult().feature_split['s']
        recon_all_tuple = pytorch_data_to_dataframe(TensorDataset(representations['recon_all'], all_s, all_y), sens_attrs=sens_attrs)
        recon_y_tuple = pytorch_data_to_dataframe(TensorDataset(representations['recon_y'], all_s, all_y), sens_attrs=sens_attrs)
        recon_s_tuple = pytorch_data_to_dataframe(TensorDataset(representations['recon_s'], all_s, all_y), sens_attrs=sens_attrs)
        recon_n_tuple = pytorch_data_to_dataframe(TensorDataset(representations['recon_n'], all_s, all_y), sens_attrs=sens_attrs)
        recon_ys_tuple = pytorch_data_to_dataframe(TensorDataset(representations['recon_ys'], all_s, all_y), sens_attrs=sens_attrs)
        recon_yn_tuple = pytorch_data_to_dataframe(TensorDataset(representations['recon_yn'], all_s, all_y), sens_attrs=sens_attrs)

        representations['recon_all'] = recon_all_tuple
        representations['recon_y'] = recon_y_tuple
        representations['recon_s'] = recon_s_tuple
        representations['recon_n'] = recon_n_tuple
        representations['recon_ys'] = recon_ys_tuple
        representations['recon_yn'] = recon_yn_tuple

    return representations


def encode_dataset_no_recon(args, data, model, recon_zyn=False):
    if not isinstance(data, DataLoader):
        data = DataLoader(data, shuffle=False, batch_size=args.test_batch_size)
    encodings = {'all_z': [], 'all_s': [], 'all_y': []}
    if recon_zyn:
        encodings['recon_yn'] = []
    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(data):
            x = x.to(args.device)
            s = s.to(args.device)

            if args.dataset == 'adult':
                x = torch.cat((x, s), dim=1)
            z = model(x)
            if recon_zyn:
                recon_yn = reconstruct(args, z, model,
                                       zero_zy=False, zero_zs=True, zero_sn=True, zero_yn=False)
                encodings['recon_yn'].append(recon_yn)

            encodings['all_z'].append(z)
            encodings['all_s'].append(s)
            encodings['all_y'].append(y)
            # LOGGER.info('Progress: {:.2f}%', itr / len(dataloader) * 100)

    for key, entry in encodings.items():
        if entry:
            encodings[key] = torch.cat(entry, dim=0).detach().cpu()

    if args.dataset == 'cmnist':
        all_s, all_y = encodings['all_s'], encodings['all_y']
        encodings['zy'] = TensorDataset(encodings['all_z'][:, z.size(1) - args.zy_dim:], all_s, all_y)
        encodings['all_z'] = TensorDataset(encodings['all_z'], all_s, all_y)
        if recon_zyn:
            encodings['recon_yn'] = TensorDataset(encodings['recon_yn'], all_s, all_y)
        return encodings

    elif args.dataset == 'adult':
        encodings['all_z'] = pd.DataFrame(encodings['all_z'].numpy())
        columns = encodings['all_z'].columns.astype(str)
        encodings['all_z'].columns = columns
        encodings['zy'] = encodings['all_z'][columns[z.size(1) - args.zy_dim:]]
        encodings['zs'] = encodings['all_z'][columns[args.zn_dim:z.size(1) - args.zy_dim]]
        encodings['zn'] = encodings['all_z'][columns[:args.zn_dim]]
        encodings['s'] = pd.DataFrame(encodings['all_s'].cpu().numpy())
        encodings['y'] = pd.DataFrame(encodings['all_y'].cpu().numpy())
        return encodings


def pytorch_data_to_dataframe(dataset, sens_attrs=None):
    """Load a pytorch dataset into a DataTuple consisting of Pandas DataFrames

    Args:
        dataset: PyTorch dataset
        sens_attrs: (optional) list of names of the sensitive attributes
    """
    # create data loader with one giant batch
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    # get the data
    data = next(iter(data_loader))
    # convert it to Pandas DataFrames
    data = [pd.DataFrame(tensor.detach().cpu().numpy()) for tensor in data]
    if sens_attrs:
        data[1].columns = sens_attrs
    # create a DataTuple
    return DataTuple(x=data[0], s=data[1], y=data[2])
