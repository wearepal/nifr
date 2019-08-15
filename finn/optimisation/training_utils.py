import argparse
from functools import partial
from itertools import groupby

import numpy as np
import pandas as pd

import torch
from ethicml.utility.data_structures import DataTuple
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from ethicml.data import Adult

from finn.data import pytorch_data_to_dataframe
from finn import layers


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

            if loss_fn == F.nll_loss:
                target = target.long()

            x = x.to(args.device)
            target = target.to(args.device)

            if args.dataset == 'adult' and use_s and args.use_s:
                x = torch.cat((x, s), dim=1)
            if args.dataset == 'cmnist' and not use_s:
                x = x.mean(dim=1, keepdim=True)

            preds = model(x)
            val_loss += loss_fn(preds.float(), target, reduction='sum').item()

            if args.dataset == 'adult':
                acc += torch.sum(preds.round() == target).item()
            else:
                acc += torch.sum(preds.argmax(dim=1) == target).item()

        acc /= len(val_data.dataset)
        val_loss /= len(val_data.dataset)

        return val_loss, acc


def classifier_training_loop(args, model, train_data, val_data, use_s=True, pred_s=False):
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.test_batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

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

    model = classifier_training_loop(args, model, train_data, val_data, use_s=use_s, pred_s=pred_s)

    return partial(
        evaluate,
        args=args,
        model=model,
        batch_size=args.test_batch_size,
        device=args.device,
        pred_s=pred_s,
        use_s=use_s,
        experiment=experiment,
        name=name,
    )


def evaluate(
    args,
    test_data,
    model,
    batch_size,
    device,
    pred_s=False,
    use_s=True,
    using_x=True,
    experiment=None,
    name=None,
):
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

            if args.dataset == 'adult' and use_s and args.use_s:
                x = torch.cat((x, s), dim=1)
            if args.dataset == 'cmnist' and not use_s and using_x:
                x = x.mean(dim=1, keepdim=True)
            if experiment is not None and i == 0:
                log_images(experiment, x, f"evaluation on {name}", prefix='eval')

            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(target.detach().cpu().numpy())

    return pd.DataFrame(all_preds, columns=['preds']), pd.DataFrame(all_targets, columns=['y'])


def get_data_dim(data_loader):
    x, _, _ = next(iter(data_loader))
    x_dim = x.shape[1:]

    return x_dim


def log_images(experiment, image_batch, name, nsamples=64, nrows=8, monochrome=False, prefix=None):
    """Make a grid of the given images, save them in a file and log them with Comet"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]
    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    experiment.log_image(torchvision.transforms.functional.to_pil_image(shw), name=prefix + name)


def reconstruct(args, z, model, zero_zy=False, zero_zs=False):
    """Reconstruct the input from the representation in various different ways"""
    z_ = z.clone()

    if zero_zy:
        if args.learn_mask:
            z = (1 - model.masker()) * z_
        else:
            z_[:, :model.zy_dim:].zero_()
    if zero_zs:
        if args.learn_mask:
            z = model.masker() * z_
        else:
            z_[:, model.zy_dim:].zero_()

    recon = model(z_, reverse=True)

    if args.dataset == 'adult':
        disc_feats = Adult().discrete_features
        cont_feats = Adult().continuous_features
        assert len(disc_feats) + len(cont_feats) == 101

        if args.drop_native:
            countries = [
                col
                for col in disc_feats
                if (col.startswith('nat') and col != "native-country_United-States")
            ]
            disc_feats = [col for col in disc_feats if col not in countries]
            disc_feats += ["native-country_not_United-States"]
            disc_feats = sorted(disc_feats)
            assert len(disc_feats) + len(cont_feats) == 62

        feats = cont_feats if args.drop_discrete else disc_feats + cont_feats

        def _add_output_layer(feature_group, dataset) -> nn.Sequential:
            n_dims = len(feature_group)
            categorical = (
                n_dims > 1 or feature_group[0] not in dataset.continuous_features
            )  # feature is categorical if it has more than 1 possible output

            if categorical:
                layer = _OneHotEncoder(n_dims)
            else:
                layer = layers.Identity()

            return layer

        grouped_features = [list(group) for key, group in groupby(feats, lambda x: x.split('_')[0])]
        output_layers = nn.ModuleList(
            [_add_output_layer(feature, Adult()) for feature in grouped_features]
        ).to(args.device)

        _recon = []
        start_idx = 0
        for layer, group in zip(output_layers, grouped_features):
            end_idx = len(group)
            _recon.append(layer(recon[:, start_idx: start_idx + end_idx]))
            start_idx += end_idx

        recon = torch.cat(_recon, dim=1)
        if args.drop_native:
            assert recon.size(1) == 5 if args.drop_discrete else 62
        else:
            assert recon.size(1) == 5 if args.drop_discrete else 101

        # recon = torch.cat([layer(recon).flatten(start_dim=1) for layer in output_layers], dim=1)

    return recon


class _OneHotEncoder(nn.Module):
    def __init__(self, n_dims, index_dim=1):
        super().__init__()
        self.n_dims = n_dims
        self.index_dim = index_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indexes = x.argmax(dim=self.index_dim)
        indexes = indexes.type(torch.int64).view(-1, 1)
        n_dims = self.n_dims  # if self.n_dims is not None else int(torch.max(indexes)) + 1
        one_hots = x.new_zeros(indexes.size()[0], n_dims).scatter_(1, indexes, 1)
        one_hots = one_hots.view(x.size(0), -1)
        return one_hots


def reconstruct_all(args, z, model):
    recon_all = reconstruct(
        args, z, model, zero_zy=False, zero_zs=False
    )

    recon_y = reconstruct(args, z, model, zero_zy=False, zero_zs=True)
    recon_s = reconstruct(args, z, model, zero_zy=True, zero_zs=False)

    return recon_all, recon_y, recon_s


def encode_dataset(args, data, model):
    dataloader = DataLoader(data, shuffle=False, batch_size=args.test_batch_size)

    all_s = []
    all_y = []

    representations = ['all_z']
    representations.extend(
        ['recon_all', 'recon_y', 'recon_s', 'recon_n', 'recon_yn', 'recon_ys', 'recon_sn']
    )
    representations = {key: [] for key in representations}

    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(dataloader):
            x = x.to(args.device)
            s = s.to(args.device)

            if args.dataset == 'adult' and args.use_s:
                x = torch.cat((x, s), dim=1)

            z = model(x)

            recon_all, recon_y, recon_s = reconstruct_all(args, z, model)
            representations['recon_all'].append(recon_all)
            representations['recon_y'].append(recon_y)
            representations['recon_s'].append(recon_s)

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
            representations['all_z'][:, :model.zy_dim], all_s, all_y
        )
        representations['zs'] = TensorDataset(
            representations['all_z'][:, model.zy_dim:], all_s, all_y
        )
        representations['all_z'] = TensorDataset(representations['all_z'], all_s, all_y)
        representations['recon_y'] = TensorDataset(representations['recon_y'], all_s, all_y)
        representations['recon_s'] = TensorDataset(representations['recon_s'], all_s, all_y)

    elif args.dataset == 'adult':
        representations['all_z'] = pd.DataFrame(representations['all_z'].numpy())
        columns = representations['all_z'].columns.astype(str)
        representations['all_z'].columns = columns

        representations['s'] = pd.DataFrame(all_s.float().cpu().numpy())
        representations['y'] = pd.DataFrame(all_y.float().cpu().numpy())

        representations['zy'] = DataTuple(
            x=representations['all_z'][columns[:model.zy_dim]],
            s=representations['s'],
            y=representations['y'],
        )
        representations['zs'] = DataTuple(
            x=representations['all_z'][columns[model.zy_dim:]],
            s=representations['s'],
            y=representations['y'],
        )

        representations['all_z'] = DataTuple(
            x=representations['all_z'], s=representations['s'], y=representations['y']
        )

        recon_all = 'fff'

        sens_attrs = Adult().feature_split['s']

        def _convert(*tensors):
            for tensor in tensors:
                yield pytorch_data_to_dataframe(
                    TensorDataset(tensor, all_s, all_y), sens_attrs=sens_attrs
                )

        for key in ['recon_all', 'recon_y', 'recon_s', 'recon_n', 'recon_ys', 'recon_yn']:
            representations[key] = _convert(representations[key])

    return representations


def encode_dataset_no_recon(args, data, model, recon_zyn=False) -> TensorDataset:
    model.eval()
    if not isinstance(data, DataLoader):
        data = DataLoader(data, shuffle=False, batch_size=args.test_batch_size)
    raw_encodings = {'all_z': [], 'all_s': [], 'all_y': []}
    if recon_zyn:
        raw_encodings['recon_yn'] = []
    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(data):
            x = x.to(args.device)
            s = s.to(args.device)

            if args.dataset == 'adult' and args.use_s:
                x = torch.cat((x, s), dim=1)
            z = model(x)
            if recon_zyn:
                recon_yn = reconstruct(
                    args, z, model, zero_zy=False, zero_zs=True, zero_sn=True, zero_yn=False
                )
                raw_encodings['recon_yn'].append(recon_yn)

            raw_encodings['all_z'].append(z)
            raw_encodings['all_s'].append(s)
            raw_encodings['all_y'].append(y)
            # LOGGER.info('Progress: {:.2f}%', itr / len(dataloader) * 100)

    for key, entry in raw_encodings.items():
        if entry:
            raw_encodings[key] = torch.cat(entry, dim=0).detach().cpu()

    encodings = {}
    s, y = raw_encodings['all_s'], raw_encodings['all_y']
    encodings['zy'] = TensorDataset(raw_encodings['all_z'][:, z.size(1) - args.zy_dim :], s, y)
    encodings['all_z'] = TensorDataset(raw_encodings['all_z'], s, y)
    if recon_zyn:
        encodings['recon_yn'] = TensorDataset(raw_encodings['recon_yn'], s, y)
    return encodings

