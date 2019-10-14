"""Main training file"""
import time
from pathlib import Path

from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader, TensorDataset

from finn.data import DatasetTriplet
from finn.models.configs import mp_28x28_net
from finn.models.configs.classifiers import fc_net, strided_7x7_net
from finn.models.inn import MaskedInn, BipartiteInn, PartitionedInn
from finn.models.factory import build_fc_inn, build_conv_inn, build_discriminator
from .loss import grad_reverse
from .utils import (
    get_data_dim,
    log_images)
from finn.utils.optimizers import apply_gradients
from finn.utils import utils

NDECS = 0
ARGS = None
LOGGER = None
SUMMARY = None


def save_model(save_dir, inn, discriminator) -> str:
    filename = save_dir / 'checkpt.pth'
    save_dict = {'ARGS': ARGS,
                 'model': inn.state_dict(),
                 'discriminator': discriminator.state_dict()}

    torch.save(save_dict, filename)

    return filename


def restore_model(filename, model, discriminator):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt['model'])
    discriminator.load_state_dict(checkpt['discriminator'])

    return model, discriminator


def train(inn, discriminator, dataloader, epoch):

    inn.train()

    total_loss_meter = utils.AverageMeter()
    log_prob_meter = utils.AverageMeter()
    disc_loss_meter = utils.AverageMeter()

    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):

        x, s, y = to_device(x, s, y)

        enc, nll = inn.routine(x)

        enc_y, enc_s = inn.split_encoding(enc)

        enc_y = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)

        if ARGS.train_on_recon:
            enc_y = inn.invert(enc_y)

        enc_y = grad_reverse(enc_y)
        disc_loss, disc_acc = discriminator.routine(enc_y, s)

        nll *= ARGS.nll_weight
        disc_loss *= ARGS.pred_s_weight

        loss = nll + disc_loss

        inn.zero_grad()
        discriminator.zero_grad()

        loss.backward()

        inn.step()
        discriminator.step()

        total_loss_meter.update(loss.item())
        log_prob_meter.update(nll.item())
        disc_loss_meter.update(disc_loss.item())

        time_meter.update(time.time() - end)

        SUMMARY.set_step(itr)
        SUMMARY.log_metric('Loss NLL', nll.item())
        SUMMARY.log_metric('Loss Adversarial', disc_loss.item())
        end = time.time()

    inn.eval()
    with torch.set_grad_enabled(False):

        log_images(SUMMARY, x, 'original_x')

        z = inn(x[:64])

        recon_all, recon_y, recon_s = inn.decode(z, partials=True)

        log_images(SUMMARY, recon_all, 'reconstruction_all')
        log_images(SUMMARY, recon_y, 'reconstruction_y')
        log_images(SUMMARY, recon_s, 'reconstruction_s')

    time_for_epoch = time.time() - start_epoch_time
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {:.3g}s | Batches/s: {:.4g} | "
        "Loss NLL: {:.5g} | Loss Adv: {:.5g} ({:.5g})",
        epoch,
        time_for_epoch,
        1 / time_meter.avg,
        log_prob_meter.avg,
        disc_loss_meter.avg,
        total_loss_meter.avg,
    )


def validate(inn, discriminator, val_loader):
    inn.eval()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:
            x_val, s_val, y_val = to_device(x_val, s_val, y_val)

            enc, nll = inn.routine(x_val)

            enc_y, enc_s = inn.split_encoding(enc)

            enc_y = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)

            if ARGS.train_on_recon:
                enc_y = inn.invert(enc_y)

            enc_y = grad_reverse(enc_y)

            disc_loss, acc = discriminator.routine(enc_y, s_val)

            nll *= ARGS.nll_weight
            disc_loss *= ARGS.pred_s_weight

            loss = nll + disc_loss

            loss_meter.update(loss.item(), n=x_val.size(0))

    SUMMARY.log_metric("Loss", loss_meter.avg)

    if ARGS.dataset == 'cmnist':

        z = inn(x_val[:64])

        recon_all, recon_y, recon_s = inn.decode(z, partials=True)
        log_images(SUMMARY, x_val, 'original_x', prefix='test')
        log_images(SUMMARY, recon_all, 'reconstruction_all', prefix='test')
        log_images(SUMMARY, recon_y, 'reconstruction_y', prefix='test')
        log_images(SUMMARY, recon_s, 'reconstruction_s', prefix='test')
    else:
        z = inn(x_val[:1000])
        recon_all, recon_y, recon_s = inn.decode(z, partials=True)
        log_images(SUMMARY, x_val, 'original_x', prefix='test')
        log_images(SUMMARY, recon_y, 'reconstruction_yn', prefix='test')
        log_images(SUMMARY, recon_s, 'reconstruction_yn', prefix='test')
        x_recon = inn(inn(x_val), reverse=True)
        x_diff = (x_recon - x_val).abs().mean().item()
        print(f"MAE of x and reconstructed x: {x_diff}")
        SUMMARY.log_metric("reconstruction MAE", x_diff)

    return loss_meter.avg


def to_device(*tensors):
    """Put tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def main(args, datasets, metric_callback):
    """Main function

    Args:
        args: commandline arguments
        datasets: a Dataset object
        metric_callback: a function that computes metrics

    Returns:
        the trained model
    """
    assert isinstance(datasets, DatasetTriplet)
    # ==== initialize globals ====
    global ARGS, LOGGER, SUMMARY
    ARGS = args

    SUMMARY = Experiment(
        api_key="Mf1iuvHn2IxBGWnBYbnOqG23h",
        project_name="finn",
        workspace="olliethomas",
        disabled=not ARGS.use_comet,
        parse_args=False,
    )
    SUMMARY.disable_mp()
    SUMMARY.log_parameters(vars(ARGS))
    SUMMARY.log_dataset_info(name=ARGS.dataset)

    save_dir = Path(ARGS.save) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = utils.get_logger(logpath=save_dir / 'logs', filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS.device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu"
    )
    LOGGER.info('{} GPUs available. Using GPU {}', torch.cuda.device_count(), ARGS.gpu)

    # ==== construct dataset ====
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(datasets.pretrain, shuffle=True, batch_size=ARGS.batch_size)
    val_loader = DataLoader(datasets.task_train, shuffle=False, batch_size=ARGS.test_batch_size)

    # ==== construct networks ====
    input_shape = get_data_dim(train_loader)
    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    feature_groups = None
    if hasattr(datasets.pretrain, "feature_groups"):
        feature_groups = datasets.pretrain.feature_groups

    Module = PartitionedInn
    if len(input_shape) > 2:
        inn = build_conv_inn(args, input_shape[0])
        if args.train_on_recon:
            disc_fn = mp_28x28_net
            disc_kwargs = {}
        else:
            disc_fn = strided_7x7_net
            disc_kwargs = {}
    else:
        inn = build_fc_inn(args, input_shape[0])
        disc_fn = fc_net
        disc_kwargs = {"hidden_dims": args.disc_hidden_dims}

    # Model arguments
    inn_args = {
        'args': args,
        'model': inn,
        'input_shape': input_shape,
        'optimizer_args': optimizer_args,
        'feature_groups': feature_groups,
    }

    # Initialise INN
    inn: BipartiteInn = Module(**inn_args)
    inn.to(args.device)
    # Initialise Discriminator
    disc_optimizer_args = {'lr': args.disc_lr}
    discriminator = build_discriminator(args,
                                        input_shape,
                                        frac_enc=1,
                                        model_fn=disc_fn,
                                        model_kwargs=disc_kwargs,
                                        flatten=False,
                                        optimizer_args=disc_optimizer_args)
    discriminator.to(args.device)

    if ARGS.spectral_norm:
        def spectral_norm(m):
            if hasattr(m, "weight"):
                return torch.nn.utils.spectral_norm(m)
        inn.apply(spectral_norm)

    # Save initial parameters
    save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)

    # Resume from checkpoint
    if ARGS.resume is not None:
        inn, discriminator = restore_model(ARGS.resume, inn, discriminator)
        metric_callback(ARGS, SUMMARY, inn, discriminator, datasets, check_originals=False)
        return

    # Logging
    SUMMARY.set_model_graph(str(inn))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(inn))

    best_loss = float('inf')
    n_vals_without_improvement = 0

    # Train INN for N epochs
    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        with SUMMARY.train():
            train(
                inn,
                discriminator,
                train_loader,
                epoch,
            )

        if epoch % ARGS.val_freq == 0 and epoch != 0:
            with SUMMARY.test():
                val_loss = validate(inn, discriminator, val_loader)
                if args.super_val:
                    metric_callback(ARGS, experiment=SUMMARY, model=inn, data=datasets)

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)
                    n_vals_without_improvement = 0
                else:
                    n_vals_without_improvement += 1

                LOGGER.info(
                    '[VAL] Epoch {:04d} | Val Loss {:.6f} | '
                    'No improvement during validation: {:02d}',
                    epoch,
                    val_loss,
                    n_vals_without_improvement,
                )

    LOGGER.info('Training has finished.')
    inn, discriminator = restore_model(save_dir / 'checkpt.pth', inn, discriminator)
    metric_callback(ARGS, experiment=SUMMARY, model=inn, data=datasets)
    save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)
    inn.eval()


if __name__ == '__main__':
    print('This file cannot be run directly.')
