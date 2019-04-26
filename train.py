"""Main training file"""
import argparse
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

from utils import utils, metrics, unbiased_hsic, dataloading
from optimisation.custom_optimizers import Adam
import layers
from utils.training_utils import fetch_model, get_data_dim

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


def compute_loss(x, s, model, disc_zx, disc_zs, *, return_z=False):

    zero = x.new_zeros(x.size(0), 1)

    if ARGS.dataset == 'cmnist':
        loss_fn = F.nll_loss
    else:
        loss_fn = F.binary_cross_entropy_with_logits
        x = torch.cat((x, s), dim=1)

    z, delta_logp = model(x, zero)  # run model forward

    if not ARGS.base_density_zs:
        log_pz, z = compute_log_pz(z, ARGS.base_density)
        zx = z[:, :-ARGS.zs_dim]
        zs = z[:, -ARGS.zs_dim:]
    else:
        # split first and then pass separately through the compute_log_pz function
        zx = z[:, :-ARGS.zs_dim]
        zs = z[:, -ARGS.zs_dim:]
        log_pzx, zx = compute_log_pz(zx, ARGS.base_density)
        log_pzs, zs = compute_log_pz(zs, ARGS.base_density_zs)
        log_pz = log_pzx + log_pzs

    # Enforce independence between the fair representation, zx,
    #  and the sensitive attribute, s
    if ARGS.ind_method == 'disc':
        probs = disc_zx(
            layers.grad_reverse(zx, lambda_=ARGS.independence_weight))
        indie_loss = loss_fn(probs, s)
    else:
        indie_loss = ARGS.independence_weight * unbiased_hsic.variance_adjusted_unbiased_HSIC(zx, s)

    # Enforce independence between the fair, zx, and unfair, zs, partitions
    if ARGS.ind_method2t == 'hsic':
        indie_loss += ARGS.independence_weight_2_towers * unbiased_hsic.variance_adjusted_unbiased_HSIC(zx, zs)

    pred_s_loss = ARGS.pred_s_weight * loss_fn(disc_zs(zs), s)

    log_px = (log_pz - delta_logp).mean()
    loss = -log_px + indie_loss + pred_s_loss

    if return_z:
        return loss, z
    return loss, -log_px, indie_loss * ARGS.independence_weight, pred_s_loss


def compute_log_pz(z, base_density):
    """Log of the base probability: log(p(z))"""
    if base_density == 'binormal':
        ones = z.new_ones(1, z.size(1))
        dist = MixtureOfDiagNormals(torch.cat([-ones, ones], 0), torch.cat([ones, ones], 0),
                                    z.new_ones(2))
        log_pz = dist.log_prob(z)
    elif base_density == 'logitbernoulli':
        temperature = z.new_tensor(.5)
        prob_of_1 = 0.5 * z.new_ones(1, z.size(1))
        dist = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(temperature,
                                                                           probs=prob_of_1)
        log_pz = dist.log_prob(z.clamp(-100, 100)).sum(1)  # not sure why the .sum(1) is needed
        z = z.sigmoid()  # z is logits, so apply sigmoid before feeding to discriminator
    elif base_density == 'bernoulli':
        temperature = z.new_tensor(.5)
        prob_of_1 = 0.5 * z.new_ones(1, z.size(1))
        dist = torch.distributions.RelaxedBernoulli(temperature, probs=prob_of_1)
        log_pz = dist.log_prob(z).sum(1)  # not sure why the .sum(1) is needed
    else:
        log_pz = torch.distributions.Normal(0, 1).log_prob(z).view(z.size(0), -1).sum(1)
    return log_pz.view(z.size(0), 1), z


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model


def train(model, disc_zx, disc_zs, optimizer, disc_optimizer, dataloader, epoch):
    model.train()

    loss_meter = utils.AverageMeter()
    log_p_x_meter = utils.AverageMeter()
    indie_loss_meter = utils.AverageMeter()
    pred_s_loss_meter = utils.AverageMeter()
    time_meter = utils.AverageMeter()
    end = time.time()

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):
        optimizer.zero_grad()

        if ARGS.ind_method == 'disc':
            disc_optimizer.zero_grad()

        # if ARGS.dataset == 'adult':
        x, s = cvt(x, s)

        loss, log_p_x, indie_loss, pred_s_loss = compute_loss(x, s, model, disc_zx, disc_zs,
                                                              return_z=False)
        loss_meter.update(loss.item())
        log_p_x_meter.update(log_p_x.item())
        indie_loss_meter.update(indie_loss.item())
        pred_s_loss_meter.update(pred_s_loss.item())

        loss.backward()
        optimizer.step()

        if ARGS.ind_method == 'disc':
            disc_optimizer.step()

        time_meter.update(time.time() - end)

        SUMMARY.log_metric("Loss log_p_x", log_p_x.item(), step=itr)
        SUMMARY.log_metric("Loss indie_loss", indie_loss.item(), step=itr)
        SUMMARY.log_metric("Loss predict_s_loss", pred_s_loss.item(), step=itr)
        end = time.time()

    LOGGER.info("[TRN] Epoch {:04d} | Time {:.4f}({:.4f}) | Loss -log_p_x (surprisal): {:.6f} "
                "indie_loss: {:.6f} pred_s_loss: {:.6f} ({:.6f}) |", epoch,
                time_meter.val, time_meter.avg, log_p_x_meter.avg, indie_loss_meter.avg,
                pred_s_loss_meter.avg, loss_meter.avg)


def validate(model, disc_zx, disc_zs, dataloader):
    model.eval()
    # start_time = time.time()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, _ in dataloader:
            x_val = cvt(x_val)
            s_val = cvt(s_val)
            loss, _, _, _ = compute_loss(x_val, s_val, model, disc_zx, disc_zs)

            loss_meter.update(loss.item(), n=x_val.size(0))
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
    ARGS.device = torch.device(f"cuda:{ARGS.gpu}" if torch.cuda.is_available() else "cpu")
    LOGGER.info('{} GPUs available.', torch.cuda.device_count())

    # ==== construct dataset ====
    args.test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(test_data, shuffle=False, batch_size=args.test_batch_size)

    x_dim, x_dim_flat = get_data_dim(train_loader)

    if args.dataset == 'adult':
        s_dim = 1
        x_dim += s_dim
    else:
        s_dim = 10

    model = fetch_model(args, x_dim)

    output_activation = None if ARGS.dataset == 'adult' else nn.LogSoftmax
    if ARGS.ind_method == 'disc':
        disc_zx = layers.Mlp([x_dim_flat - ARGS.zs_dim] + [100, 100, s_dim], activation=nn.ReLU,
                             output_activation=output_activation)
        disc_zx.to(ARGS.device)
    else:
        disc_zx = None

    disc_zs = layers.Mlp([ARGS.zs_dim, 40, 40, s_dim], activation=nn.ReLU,
                         output_activation=output_activation)
    disc_zs.to(ARGS.device)

    if ARGS.resume is not None:
        checkpt = torch.load(ARGS.resume)
        model.load_state_dict(checkpt['state_dict'])

    SUMMARY.set_model_graph(str(model))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(model))

    if not ARGS.evaluate:
        optimizer = Adam(model.parameters(), lr=ARGS.lr, weight_decay=ARGS.weight_decay)
        disc_optimizer = Adam(list(disc_zx.parameters()) + list(disc_zs.parameters()),
                              lr=ARGS.disc_lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=ARGS.patience,
                                      min_lr=1.e-7, cooldown=1)

        best_loss = float('inf')

        n_vals_without_improvement = 0

        for epoch in range(ARGS.epochs):
            if n_vals_without_improvement > ARGS.early_stopping > 0:
                break

            with SUMMARY.train():
                train(model, disc_zx, disc_zs, optimizer, disc_optimizer, train_loader, epoch)

            if epoch % ARGS.val_freq == 0:
                with SUMMARY.test():
                    val_loss = validate(model, disc_zx, disc_zs, val_loader)
                    SUMMARY.log_metric("Loss", val_loss, step=(epoch + 1) * len(train_loader))

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

    LOGGER.info('Evaluating model on test set.')
    model.eval()
    test_encodings = encode_dataset(val_loader, model, cvt)
    # df_test.to_feather(ARGS.test_new)
    train_encodings = encode_dataset(
        DataLoader(train_data, shuffle=False, batch_size=args.test_batch_size), model, cvt)
    # df_train.to_feather(ARGS.train_new)
    return train_encodings, test_encodings


def encode_dataset(dataset, model, cvt):
    representation = []
    with torch.no_grad():
        # test_loss = utils.AverageMeter()
        for itr, (x, s, _) in enumerate(dataset):
            x = cvt(x)
            s = cvt(s)

            if ARGS.dataset == 'adult':
                x = torch.cat((x, s), dim=1)
            zero = x.new_zeros(x.size(0), 1)
            z, _ = model(x, zero)
            if ARGS.base_density == 'logitbernoulli':
                z = z.sigmoid()

            if dataset == 'cmnist':
                z[:, -ARGS.zs_dim:].zero_()
                z = model(x, zero, reverse=True)
            # test_loss.update(loss.item(), n=x.shape[0])
            representation.append(z)
            LOGGER.info('Progress: {:.2f}%', itr / len(dataset) * 100)

    representation = torch.cat(representation, dim=0).cpu().detach().numpy()

    if ARGS.dataset == 'cmnist':
        representation = np.reshape(representation, (representation.shape[0], -1))

    df = pd.DataFrame(representation)
    columns = df.columns.astype(str)
    df.columns = columns
    zx = df[columns[:-ARGS.zs_dim]]
    zs = df[columns[-ARGS.zs_dim:]]

    return df, zx, zs


def current_experiment():
    global SUMMARY
    return SUMMARY


if __name__ == '__main__':
    from utils.training_utils import parse_arguments
    main(parse_arguments(), None, None)
