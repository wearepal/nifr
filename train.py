"""Main training file"""
from itertools import chain
import time
import random
from pathlib import Path

from comet_ml import Experiment
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from optimisation.custom_optimizers import Adam
from utils import utils  # , unbiased_hsic
from utils.training_utils import get_data_dim, log_images, reconstruct_all

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


def save_model(save_dir, model, discs):
    filename = save_dir / 'checkpt.pth'
    save_dict = {'ARGS': ARGS,
                 'model': model.state_dict()}

    if discs.s_from_zs is not None:
        save_dict['disc_s_from_zs'] = discs.s_from_zs.state_dict()

    if ARGS.inv_disc:
        if discs.y_from_zy is not None:
            save_dict['disc_y_from_zy'] = discs.y_from_zy.state_dict()
    else:
        if discs.y_from_zys is not None:
            save_dict['disc_y_from_zys'] = discs.y_from_zys.state_dict()
        if discs.s_from_zy is not None:
            save_dict['disc_s_from_zy'] = discs.s_from_zy.state_dict()

    torch.save(save_dict, filename)


def restore_model(filename, model):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt['model'])
    return model


def train(model, discs, optimizer, disc_optimizer, dataloader, epoch):
    model.train()

    loss_meter = utils.AverageMeter()
    log_p_x_meter = utils.AverageMeter()
    pred_y_loss_meter = utils.AverageMeter()
    pred_s_from_zy_loss_meter = utils.AverageMeter()
    pred_s_from_zs_loss_meter = utils.AverageMeter()
    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):
        optimizer.zero_grad()

        disc_optimizer.zero_grad()

        # if ARGS.dataset == 'adult':
        x, s, y = cvt(x, s, y)

        loss, log_p_x, pred_y_loss, pred_s_from_zy_loss, pred_s_from_zs_loss = compute_loss(
            ARGS, x, s, y, model, discs, return_z=False)
        loss_meter.update(loss.item())
        log_p_x_meter.update(log_p_x.item())
        pred_y_loss_meter.update(pred_y_loss.item())
        pred_s_from_zy_loss_meter.update(pred_s_from_zy_loss.item())
        pred_s_from_zs_loss_meter.update(pred_s_from_zs_loss.item())

        loss.backward()
        optimizer.step()

        disc_optimizer.step()

        time_meter.update(time.time() - end)

        SUMMARY.set_step(itr)
        SUMMARY.log_metric('Loss log_p_x', log_p_x.item())
        SUMMARY.log_metric('Loss pred_y_from_zys_loss', pred_y_loss.item())
        SUMMARY.log_metric('Loss pred_s_from_zy_loss', pred_s_from_zy_loss.item())
        SUMMARY.log_metric('Loss pred_s_from_zs_loss', pred_s_from_zs_loss.item())
        end = time.time()

    if ARGS.dataset == 'cmnist':
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

    time_for_epoch = time.time() - start_epoch_time
    LOGGER.info("[TRN] Epoch {:04d} | Duration: {:.3g}s | Batches/s: {:.4g} | "
                "Loss -log_p_x (surprisal): {:.5g} | pred_y_from_zys: {:.5g} | "
                "pred_s_from_zy: {:.5g} | pred_s_from_zs {:.5g} ({:.5g})",
                epoch, time_for_epoch, 1 / time_meter.avg, log_p_x_meter.avg, pred_y_loss_meter.avg,
                pred_s_from_zy_loss_meter.avg, pred_s_from_zs_loss_meter.avg, loss_meter.avg)


def validate(model, discs, val_loader):
    model.eval()
    # start_time = time.time()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:
            x_val, s_val, y_val = cvt(x_val, s_val, y_val)
            loss, _, _, _, _ = compute_loss(ARGS, x_val, s_val, y_val, model, discs)

            loss_meter.update(loss.item(), n=x_val.size(0))
    SUMMARY.log_metric("Loss", loss_meter.avg)

    x_val = torch.cat((x_val, s_val), dim=1) if ARGS.dataset == 'adult' else x_val

    if ARGS.dataset == 'cmnist':
        z = model(x_val[:64])
        recon_all, recon_y, recon_s, recon_n, recon_ys, recon_yn = reconstruct_all(ARGS, z, model)
        log_images(SUMMARY, x_val, 'original_x', train=False)
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
    global ARGS, LOGGER, SUMMARY, make_networks, compute_loss
    ARGS = args

    if args.inv_disc:
        from models.inv_discriminators import make_networks, compute_loss
    else:
        from models.nn_discriminators import make_networks, compute_loss

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
    model, discs = make_networks(ARGS, x_dim, z_dim_flat)
    LOGGER.info('zyn_dim: {}, zs_dim: {}', ARGS.zy_dim, ARGS.zs_dim)
    # LOGGER.info('zn_dim: {}, zs_dim: {}, zy_dim: {}', ARGS.zn_dim, ARGS.zs_dim, ARGS.zy_dim)

    if ARGS.resume is not None:
        checkpt = torch.load(ARGS.resume)
        model.load_state_dict(checkpt['state_dict'])

    SUMMARY.set_model_graph(str(model))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(model))

    optimizer = Adam(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
    disc_params = chain(*[disc.parameters() for disc in discs if disc is not None])
    disc_optimizer = Adam(disc_params, lr=ARGS.disc_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=ARGS.patience,
                                  min_lr=1.e-7, cooldown=1)

    best_loss = float('inf')

    n_vals_without_improvement = 0

    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        with SUMMARY.train():
            train(model, discs, optimizer, disc_optimizer, train_loader, epoch)

        if epoch % ARGS.val_freq == 0:
            with SUMMARY.test():
                # SUMMARY.set_step((epoch + 1) * len(train_loader))
                val_loss = validate(model, discs, val_loader)
                if args.super_val:
                    metric_callback(ARGS, SUMMARY, model, discs, train_data, val_data, test_data)

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_model(save_dir=save_dir, model=model, discs=discs)
                    n_vals_without_improvement = 0
                else:
                    n_vals_without_improvement += 1

                scheduler.step(val_loss)

                LOGGER.info('[VAL] Epoch {:04d} | Val Loss {:.6f} | '
                            'No improvement during validation: {:02d}', epoch, val_loss,
                            n_vals_without_improvement)

    LOGGER.info('Training has finished.')
    model = restore_model(save_dir / 'checkpt.pth', model).to(ARGS.device)
    metric_callback(ARGS, SUMMARY, model, discs, train_data, val_data, test_data)
    save_model(save_dir=save_dir, model=model, discs=discs)
    model.eval()
    return model, discs


if __name__ == '__main__':
    print('This file cannot be run directly.')
