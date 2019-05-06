"""Main training file"""
from itertools import chain
import time
import random
from pathlib import Path

from comet_ml import Experiment
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from optimisation.custom_optimizers import Adam
import layers
from utils import utils  # , unbiased_hsic
from utils.training_utils import fetch_model, get_data_dim, log_images, reconstruct_all
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


def compute_loss(x, s, y, model, *, disc_y_from_zys=None, disc_s_from_zs=None, disc_s_from_zy=None,
                 return_z=False):

    zero = x.new_zeros(x.size(0), 1)

    if ARGS.dataset == 'cmnist':
        # loss_fn = F.l1_loss
        loss_fn = F.nll_loss
        class_loss_fn = F.nll_loss
    else:
        loss_fn = F.binary_cross_entropy
        x = torch.cat((x, s.float()), dim=1)
        class_loss_fn = F.binary_cross_entropy_with_logits

    z, delta_logp = model(x, zero)  # run model forward

    log_pz = compute_log_pz(z)
    # zn = z[:, :ARGS.zn_dim]
    zs = z[:,  ARGS.zn_dim: (z.size(1) - ARGS.zy_dim)]
    zy = z[:, (z.size(1) - ARGS.zy_dim):]
    # Enforce independence between the fair representation, zy,
    #  and the sensitive attribute, s
    pred_y_loss = z.new_zeros(1)
    pred_s_from_zy_loss = z.new_zeros(1)
    pred_s_from_zs_loss = z.new_zeros(1)

    if disc_y_from_zys is not None and zy.size(1) > 0 and zs.size(1) > 0 and not ARGS.meta_learn:
        pred_y_loss = (ARGS.pred_y_weight
                       * class_loss_fn(disc_y_from_zys(torch.cat((zy, zs), dim=1)), y, reduction='mean'))
    if disc_s_from_zy is not None and zy.size(1) > 0:
        pred_s_from_zy_loss = loss_fn(
            disc_s_from_zy(layers.grad_reverse(zy, lambda_=ARGS.pred_s_from_zy_weight)),
            s, reduction='mean')
    # Enforce independence between the fair, zy, and unfair, zs, partitions

    if disc_s_from_zs is not None and zs.size(1) > 0:
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


def save_model(filename, model):
    torch.save({'ARGS': ARGS, 'state_dict': model.state_dict()}, filename)


def restore_model(filename, model):
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

        loss, log_p_x, pred_y_loss, pred_s_from_zy_loss, pred_s_from_zs_loss = compute_loss(
            x, s, y, model,
            disc_y_from_zys=disc_y_from_zys,
            disc_s_from_zs=disc_s_from_zs,
            disc_s_from_zy=disc_s_from_zy,
            return_z=False)
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

    x = torch.cat((x, s), dim=1) if ARGS.dataset == 'adult' else x

    log_images(SUMMARY, x, 'original_x')
    zero = x.new_zeros(x.size(0), 1)
    z, _ = model(x, zero)
    recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(ARGS, z, model)
    log_images(SUMMARY, recon_all, 'reconstruction_all')
    log_images(SUMMARY, recon_y, 'reconstruction_y')
    log_images(SUMMARY, recon_s, 'reconstruction_s')
    log_images(SUMMARY, recon_n, 'reconstruction_n')
    log_images(SUMMARY, recon_ys, 'reconstruction_ys')
    log_images(SUMMARY, recon_yn, 'reconstruction_yn')

    LOGGER.info("[TRN] Epoch {:04d} | Time {:.4f}({:.4f}) | Loss -log_p_x (surprisal): {:.6f} |"
                "pred_y_from_zys: {:.6f} | pred_s_from_zy: {:.6f} | pred_s_from_zs {:.6f} ({:.6f})",
                epoch, time_meter.val, time_meter.avg, log_p_x_meter.avg, pred_y_loss_meter.avg,
                pred_s_from_zy_loss_meter.avg, pred_s_from_zs_loss_meter.avg, loss_meter.avg)


def validate(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, val_loader):
    model.eval()
    # start_time = time.time()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:
            x_val, s_val, y_val = cvt(x_val, s_val, y_val)
            loss, _, _, _, _ = compute_loss(
                x_val, s_val, y_val, model,
                disc_y_from_zys=disc_y_from_zys,
                disc_s_from_zs=disc_s_from_zs,
                disc_s_from_zy=disc_s_from_zy)

            loss_meter.update(loss.item(), n=x_val.size(0))
    SUMMARY.log_metric("Loss", loss_meter.avg)
    log_images(SUMMARY, x_val, 'original_x', train=False)

    x_val = torch.cat((x_val, s_val), dim=1) if ARGS.dataset == 'adult' else x_val

    z = model(x_val[:64])
    recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(ARGS, z, model)
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


def make_networks(x_dim, z_dim_flat):
    """Create the discriminators that enfoce the partition on z"""
    if ARGS.dataset == 'adult':
        z_dim_flat += 1
        ARGS.zs_dim = round(ARGS.zs_frac * z_dim_flat)
        ARGS.zy_dim = round(ARGS.zy_frac * z_dim_flat)
        s_dim = 1
        x_dim += s_dim
        y_dim = 1
        output_activation = nn.Sigmoid
        hidden_sizes = [40, 40]

        ARGS.zn_dim = z_dim_flat - ARGS.zs_dim - ARGS.zy_dim

        disc_s_from_zy = layers.Mlp([ARGS.zy_dim] + hidden_sizes + [s_dim], activation=nn.ReLU,
                                    output_activation=output_activation)
        hidden_sizes = [40, 40]
        disc_s_from_zs = layers.Mlp([ARGS.zs_dim] + hidden_sizes + [s_dim], activation=nn.ReLU,
                                    output_activation=output_activation)
        disc_y_from_zys = layers.Mlp([z_dim_flat - ARGS.zn_dim] + [100, 100, y_dim],
                                     activation=nn.ReLU, output_activation=output_activation)
        disc_y_from_zys.to(ARGS.device)
    else:
        z_channels = x_dim * 16
        ARGS.zs_dim = round(ARGS.zs_frac * z_channels)
        s_dim = 10
        y_dim = 10
        output_activation = nn.LogSoftmax(dim=1)

        if not ARGS.meta_learn:
            ARGS.zy_dim = round(ARGS.zy_frac * z_channels)
            ARGS.zn_dim = z_channels - ARGS.zs_dim - ARGS.zy_dim

            hidden_sizes = [(ARGS.zy_dim + ARGS.zs_dim * 8), (ARGS.zy_dim + ARGS.zs_dim) * 8]
            disc_y_from_zys = models.MnistConvNet(ARGS.zy_dim + ARGS.zs_dim, y_dim,
                                                  output_activation=nn.LogSoftmax(dim=1),
                                                  hidden_sizes=hidden_sizes)
            disc_y_from_zys.to(ARGS.device)
        else:
            ARGS.zy_dim = z_channels - ARGS.zs_dim
            ARGS.zn_dim = 0

            disc_y_from_zys = None

        hidden_sizes = [ARGS.zs_dim * 16, ARGS.zs_dim * 16]
        disc_s_from_zs = models.MnistConvNet(ARGS.zs_dim, s_dim, hidden_sizes=hidden_sizes,
                                             output_activation=output_activation)
        hidden_sizes = [ARGS.zy_dim * 16, ARGS.zy_dim * 16, ARGS.zy_dim * 16]
        disc_s_from_zy = models.MnistConvNet(ARGS.zy_dim, s_dim, hidden_sizes=hidden_sizes,
                                             output_activation=output_activation)

    LOGGER.info('zn_dim: {}, zs_dim: {}, zy_dim: {}', ARGS.zn_dim, ARGS.zs_dim, ARGS.zy_dim)
    disc_s_from_zs.to(ARGS.device)
    disc_s_from_zy.to(ARGS.device)
    model = fetch_model(ARGS, x_dim)
    return model, disc_s_from_zs, disc_s_from_zy, disc_y_from_zys


def main(args, train_data, val_data, test_data, metric_callback):
    """Main function

    Args:
        args: commandline arguments
        train_data: training data
        val_data: validation data
        test_data: test data
        metric_callback: a function that computes metrics

    Returns:
        the trained model
    """
    # ==== initialize globals ====
    global ARGS, LOGGER, SUMMARY
    ARGS = args

    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)

    SUMMARY = Experiment(api_key="Mf1iuvHn2IxBGWnBYbnOqG23h", project_name="finn",
                         workspace="olliethomas", disabled=not ARGS.use_comet, parse_args=False)
    SUMMARY.disable_mp()
    SUMMARY.log_parameters(vars(ARGS))
    SUMMARY.log_dataset_info(name=ARGS.dataset)

    save_dir = Path(ARGS.save) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = save_dir / 'checkpt.pth'
    LOGGER = utils.get_logger(logpath=save_dir / 'logs', filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)

    # ==== check GPU ====
    ARGS.device = torch.device(f"cuda:{ARGS.gpu}" if (
        torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu")
    LOGGER.info('{} GPUs available. Using GPU {}', torch.cuda.device_count(), ARGS.gpu)

    # ==== construct dataset ====
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(train_data, shuffle=True, batch_size=ARGS.batch_size)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=ARGS.test_batch_size)

    # ==== construct networks ====
    x_dim, z_dim_flat = get_data_dim(train_loader)
    model, disc_s_from_zs, disc_s_from_zy, disc_y_from_zys = make_networks(x_dim, z_dim_flat)

    if ARGS.resume is not None:
        checkpt = torch.load(ARGS.resume)
        model.load_state_dict(checkpt['state_dict'])

    SUMMARY.set_model_graph(str(model))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(model))

    optimizer = Adam(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    disc_params = chain(*[disc.parameters() for disc in [disc_y_from_zys, disc_s_from_zy, disc_s_from_zs]
                          if disc is not None])
    disc_optimizer = Adam(disc_params, lr=ARGS.disc_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=ARGS.patience,
                                  min_lr=1.e-7, cooldown=1)

    best_loss = float('inf')

    n_vals_without_improvement = 0

    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        with SUMMARY.train():
            train(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs, optimizer,
                  disc_optimizer, train_loader, epoch)

        if epoch % ARGS.val_freq == 0:
            with SUMMARY.test():
                # SUMMARY.set_step((epoch + 1) * len(train_loader))
                val_loss = validate(model, disc_y_from_zys, disc_s_from_zy, disc_s_from_zs,
                                    val_loader)
                if ARGS.meta_learn:
                    metric_callback(ARGS, SUMMARY, model, train_data, val_data, test_data)

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_model(model_save_path, model)
                    n_vals_without_improvement = 0
                else:
                    n_vals_without_improvement += 1

                scheduler.step(val_loss)

                LOGGER.info('[VAL] Epoch {:04d} | Val Loss {:.6f} | '
                            'No improvement during validation: {:02d}', epoch, val_loss,
                            n_vals_without_improvement)

    LOGGER.info('Training has finished.')
    model = restore_model(model_save_path, model).to(ARGS.device)
    metric_callback(ARGS, SUMMARY, model, train_data, val_data, test_data)

    model.eval()
    return model


if __name__ == '__main__':
    print('This file cannot be run directly.')
