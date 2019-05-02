"""Main training file"""
import argparse
from itertools import chain
import time
from pathlib import Path

from comet_ml import Experiment
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from pyro.distributions import MixtureOfDiagNormals
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils import utils, metrics, unbiased_hsic, dataloading
from optimisation.custom_optimizers import Adam
import layers
from utils.training_utils import fetch_model, get_data_dim, log_images
import models

NDECS = 0
ARGS = None
LOGGER = None
SUMMARY = None


def convert_data(train_tuple, test_tuple):
    """
    Convert tuples of dataframes to pytorch datasets
    Args:
        train_tuple: tuple of dataframes with the training data
        test_tuple: tuple of dataframes with the test data

    Returns:
        a dictionary with the pytorch datasets
    """
    data = {'trn': TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                   for df in train_tuple]),
            'val': TensorDataset(*[torch.tensor(df.values, dtype=torch.float32)
                                   for df in test_tuple])}
    return data


def compute_loss(x, s, y, model, disc_y_from_zys, disc_s_from_zs, disc_s_from_zy, *, return_z=False):

    zero = x.new_zeros(x.size(0), 1)

    if ARGS.dataset == 'cmnist':
        loss_fn = F.l1_loss
    else:
        loss_fn = F.binary_cross_entropy_with_logits
        x = torch.cat((x, s), dim=1)

    z, delta_logp = model(x, zero)  # run model forward

    log_pz = compute_log_pz(z)
    zn = z[:, :ARGS.zn_dim]
    zs = z[:,  ARGS.zn_dim: (z.size(1) - ARGS.zy_dim)]
    zy = z[:, (z.size(1) - ARGS.zy_dim):]
    # Enforce independence between the fair representation, zy,
    #  and the sensitive attribute, s
    pred_y_loss = 0
    pred_s_from_zy_loss = 0
    pred_s_from_zs_loss = 0

    if zy.size(1) > 0 and zs.size(1) > 0:
        pred_y_loss = ARGS.pred_y_weight \
                      * F.nll_loss(disc_y_from_zys(torch.cat((zy, zs), dim=1)), y, reduction='mean')
    if zy.size(1) > 0:
        pred_s_from_zy_loss = loss_fn(
            layers.grad_reverse(disc_s_from_zy(zy), lambda_=ARGS.pred_s_from_zy_weight),
            s, reduction='mean')
    # Enforce independence between the fair, zy, and unfair, zs, partitions

    if zs.size(1) > 0:
        pred_s_from_zs_loss = ARGS.pred_s_from_zs_weight\
                              * loss_fn(disc_s_from_zs(zs), s, reduction='mean')

    log_px = ARGS.log_px_weight * (log_pz - delta_logp).mean()
    loss = -log_px + pred_y_loss + pred_s_from_zs_loss + pred_s_from_zy_loss

    if return_z:
        return loss, z
    return (loss, -log_px, pred_y_loss, ARGS.pred_s_from_zy_weight * pred_s_from_zy_loss,
            pred_s_from_zs_loss)


def compute_log_pz(z):
    """Log of the base probability: log(p(z))"""
    log_pz = torch.distributions.Normal(0, 1).log_prob(z).flatten(1).sum(1)
    return log_pz.view(z.size(0), 1)


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model


def train(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, optimizer,
          disc_optimizer, dataloader, epoch):
    model.train()

    loss_meter = utils.AverageMeter()
    log_p_x_meter = utils.AverageMeter()
    pred_y_loss_meter = utils.AverageMeter()
    pred_s_from_zy_loss_meter = utils.AverageMeter()
    pred_s_from_zs_loss_meter = utils.AverageMeter()
    time_meter = utils.AverageMeter()
    end = time.time()

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):
        optimizer.zero_grad()

        if ARGS.ind_method == 'disc':
            disc_optimizer.zero_grad()

        # if ARGS.dataset == 'adult':
        x, s, y = cvt(x, s, y)

        loss, log_p_x, pred_y_loss, pred_s_from_zy_loss, pred_s_from_zs_loss =\
            compute_loss(x, s, y, model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, return_z=False)
        loss_meter.update(loss.item())
        log_p_x_meter.update(log_p_x.item())
        pred_y_loss_meter.update(pred_y_loss.item())
        pred_s_from_zy_loss_meter.update(pred_s_from_zy_loss.item())
        pred_s_from_zs_loss_meter.update(pred_s_from_zs_loss.item())

        loss.backward()
        optimizer.step()

        if ARGS.ind_method == 'disc':
            disc_optimizer.step()

        time_meter.update(time.time() - end)

        SUMMARY.set_step(itr)
        SUMMARY.log_metric('Loss log_p_x', log_p_x.item())
        SUMMARY.log_metric('Loss pred_y_from_zys_loss', pred_y_loss.item())
        SUMMARY.log_metric('Loss pred_s_from_zy_loss', pred_s_from_zy_loss.item())
        SUMMARY.log_metric('Loss pred_s_from_zs_loss', pred_s_from_zs_loss.item())
        end = time.time()

    log_images(SUMMARY, x, 'original_x')
    zero = x.new_zeros(x.size(0), 1)
    z, _ = model(x, zero)
    recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(z, model)
    log_images(SUMMARY, recon_all, 'reconstruction_all')
    log_images(SUMMARY, recon_y, 'reconstruction_y')
    log_images(SUMMARY, recon_s, 'reconstruction_s')
    log_images(SUMMARY, recon_n, 'reconstruction_n')
    log_images(SUMMARY, recon_ys, 'reconstruction_ys')
    log_images(SUMMARY, recon_yn, 'reconstruction_yn')

    LOGGER.info("[TRN] Epoch {:04d} | Time {:.4f}({:.4f}) | Loss -log_p_x (surprisal): {:.6f} |"
                "indie_loss: {:.6f} | pred_s_loss: {:.6f} | pred_y_loss {:.6f} ({:.6f})", epoch,
                time_meter.val, time_meter.avg, log_p_x_meter.avg, pred_y_loss_meter.avg,
                pred_s_from_zy_loss_meter.avg, pred_s_from_zs_loss_meter.avg, loss_meter.avg)


def validate(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, dataloader):
    model.eval()
    # start_time = time.time()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in dataloader:
            x_val, s_val, y_val = cvt(x_val, s_val, y_val)
            loss, _, _, _, _ = compute_loss(x_val, s_val, y_val, model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs)

            loss_meter.update(loss.item(), n=x_val.size(0))
    SUMMARY.log_metric("Loss", loss_meter.avg)
    log_images(SUMMARY, x_val, 'original_x', train=False)
    zero = x_val.new_zeros(x_val.size(0), 1)
    z, _ = model(x_val, zero)
    recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(z, model)
    log_images(SUMMARY, recon_all, 'reconstruction_all', train=False)
    log_images(SUMMARY, recon_y, 'reconstruction_y', train=False)
    log_images(SUMMARY, recon_s, 'reconstruction_s', train=False)
    log_images(SUMMARY, recon_n, 'reconstruction_n', train=False)
    log_images(SUMMARY, recon_ys, 'reconstruction_ys', train=False)
    log_images(SUMMARY, recon_yn, 'reconstruction_yn', train=False)
    return loss_meter.avg


def cvt(*tensors):
    """Put tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def main(args, train_data, test_data):
    # ==== initialize globals ====
    global ARGS, LOGGER, SUMMARY
    ARGS = args

    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)

    SUMMARY = Experiment(api_key="Mf1iuvHn2IxBGWnBYbnOqG23h", project_name="finn",
                         workspace="olliethomas", disabled=not ARGS.use_comet, parse_args=False)
    SUMMARY.disable_mp()
    SUMMARY.log_parameters(vars(ARGS))

    save_dir = Path(ARGS.save)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOGGER = utils.get_logger(logpath=save_dir / 'logs', filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)

    # ==== check GPU ====
    ARGS.device = torch.device(f"cuda:{ARGS.gpu}" if (
        torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu")
    LOGGER.info('{} GPUs available.', torch.cuda.device_count())

    # ==== construct dataset ====
    args.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=args.test_batch_size)

    x_dim, z_dim_flat = get_data_dim(train_loader)

    if args.dataset == 'adult':
        ARGS.zs_dim = round(ARGS.zs_frac * z_dim_flat)
        ARGS.zy_dim = round(ARGS.zy_frac * z_dim_flat)
        s_dim = 1
        x_dim += s_dim
        y_dim = 1
        output_activation = None
        hidden_sizes = [40, 40]
        disc_s_from_zy = layers.Mlp([ARGS.zs_dim] + hidden_sizes + [s_dim], activation=nn.ReLU,
                             output_activation=output_activation)
        hidden_sizes = [40, 40]
        disc_s_from_zs = layers.Mlp([ARGS.zs_dim] + hidden_sizes + [y_dim], activation=nn.ReLU,
                             output_activation=nn.Sigmoid)
        disc_y_from_zys = layers.Mlp([z_dim_flat - ARGS.zs_dim] + [100, 100, s_dim], activation=nn.ReLU,
                                     output_activation=output_activation)
    else:
        z_channels = x_dim * 16
        ARGS.zs_dim = round(ARGS.zs_frac * z_channels)
        ARGS.zy_dim = round(ARGS.zy_frac * z_channels)
        s_dim = 3
        y_dim = 10
        output_activation = nn.Sigmoid()

        hidden_sizes = [ARGS.zs_dim * 8, ARGS.zs_dim * 8]
        disc_s_from_zy = models.MnistConvNet(ARGS.zy_dim, s_dim, output_activation=output_activation,
                                             hidden_sizes=hidden_sizes)

        hidden_sizes = [ARGS.zy_dim * 8, ARGS.zy_dim * 8]
        disc_s_from_zs = models.MnistConvNet(ARGS.zs_dim, s_dim, output_activation=output_activation,
                                             hidden_sizes=hidden_sizes)

        ARGS.zn_dim = z_channels - ARGS.zs_dim - ARGS.zy_dim
        hidden_sizes = [(ARGS.zy_dim + ARGS.zs_dim * 8), (ARGS.zy_dim + ARGS.zs_dim) * 8]
        disc_y_from_zys = models.MnistConvNet(ARGS.zy_dim + ARGS.zs_dim, y_dim,
                                              output_activation=nn.LogSoftmax(dim=1),
                                              hidden_sizes=hidden_sizes)

    disc_s_from_zy.to(ARGS.device)
    disc_s_from_zs.to(ARGS.device)
    disc_y_from_zys.to(ARGS.device)

    model = fetch_model(args, x_dim)

    if ARGS.resume is not None:
        checkpt = torch.load(ARGS.resume)
        model.load_state_dict(checkpt['state_dict'])

    SUMMARY.set_model_graph(str(model))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(model))

    if not ARGS.evaluate:
        optimizer = Adam(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
        disc_optimizer = Adam(
            chain(disc_y_from_zys.parameters(), disc_s_from_zy.parameters(), disc_s_from_zs.parameters()),
            lr=ARGS.disc_lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=ARGS.patience,
                                      min_lr=1.e-7, cooldown=1)

        best_loss = float('inf')

        n_vals_without_improvement = 0

        for epoch in range(ARGS.epochs):
            if n_vals_without_improvement > ARGS.early_stopping > 0:
                break

            with SUMMARY.train():
                train(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, optimizer, disc_optimizer, train_loader,
                      epoch)

            if epoch % ARGS.val_freq == 0:
                with SUMMARY.test():
                    # SUMMARY.set_step((epoch + 1) * len(train_loader))
                    val_loss = validate(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, val_loader)

                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save({
                            'ARGS': ARGS,
                            'state_dict': model.state_dict(),
                        }, save_dir / 'checkpt.pth')
                        n_vals_without_improvement = 0
                    else:
                        n_vals_without_improvement += 1

                    scheduler.step(val_loss)

                    log_message = (
                        '[VAL] Epoch {:04d} | Val Loss {:.6f} | '
                        'No improvement during validation: {:02d}'.format(
                            epoch, val_loss, n_vals_without_improvement))
                    LOGGER.info(log_message)

        LOGGER.info('Training has finished.')
        model = restore_model(model, save_dir / 'checkpt.pth').to(ARGS.device)

    model.eval()
    LOGGER.info('Encoding training set...')
    train_encodings = encode_dataset(
        DataLoader(train_data, shuffle=False, batch_size=args.test_batch_size), model, cvt)
    LOGGER.info('Encoding test set...')
    test_encodings = encode_dataset(val_loader, model, cvt)

    return train_encodings, test_encodings


def reconstruct(z, model, zero_zy=False, zero_zs=False, zero_zn=False):
    """Reconstruct the input from the representation in various different ways"""
    z_ = z.clone()
    if zero_zy:
        z_[:, -ARGS.zy_dim:].zero_()
    if zero_zs:
        z_[:, ARGS.zs_dim: z_.size(1) - ARGS.zy_dim:].zero_()
    if zero_zn:
        z_[:, :ARGS.zn_dim].zero_()
    recon, _ = model(z_, z.new_zeros(z.size(0), 1), reverse=True)

    return recon


def reconstruct_all(z, model):
    recon_all = reconstruct(z, model)

    recon_y = reconstruct(z, model, zero_zs=True, zero_zn=True)
    recon_s = reconstruct(z, model, zero_zy=True, zero_zn=True)
    recon_n = reconstruct(z, model, zero_zy=True, zero_zs=True)

    recon_ys = reconstruct(z, model, zero_zn=True)
    recon_yn = reconstruct(z, model, zero_zs=True)
    return recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn


def encode_dataset(dataloader, model, cvt):

    all_s = []
    all_y = []

    representations = ['all_z']
    if ARGS.dataset == 'cmnist':
        representations.extend(['recon_all', 'recon_y', 'recon_s',
                                'recon_n', 'recon_yn', 'recon_ys'])
    representations = {key: [] for key in representations}

    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for x, s, y in tqdm(dataloader):
            x, s = cvt(x, s)

            if ARGS.dataset == 'adult':
                x = torch.cat((x, s), dim=1)
            zero = x.new_zeros(x.size(0), 1)
            z, _ = model(x, zero)

            if ARGS.dataset == 'cmnist':
                recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(z, model)
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

    if ARGS.dataset == 'cmnist':
        all_s = torch.cat(all_s, dim=0)
        all_y = torch.cat(all_y, dim=0)

        representations['all_z'] = torch.utils.data.TensorDataset(representations['all_z'], all_s, all_y)
        representations['recon_y'] = torch.utils.data.TensorDataset(representations['recon_y'], all_s, all_y)
        representations['recon_s'] = torch.utils.data.TensorDataset(representations['recon_s'], all_s, all_y)

        representations['zn'] = torch.utils.data.TensorDataset(
            representations['all_z'][:, :ARGS.zn_dim], all_s, all_y)

        return representations

    elif ARGS.dataset == 'adult':
        representations['all_z'] = pd.DataFrame(representations['all_z'].numpy())
        columns = representations['all_z'].columns.astype(str)
        representations['all_z'].columns = columns
        representations['zy'] = representations['all_z'][columns[z.size(1) - ARGS.zy_dim:]]
        representations['zs'] = representations['all_z'][columns[ARGS.zn_dim: z.size(1) - ARGS.zy_dim:]]
        representations['zn'] = representations['all_z'][:columns[ARGS.zn_dim]]

        return representations


def current_experiment():
    global SUMMARY
    return SUMMARY


if __name__ == '__main__':
    from utils.training_utils import parse_arguments
    main(parse_arguments(), None, None)
