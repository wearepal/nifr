"""Main training file"""
import time
from pathlib import Path

import torch
from comet_ml import Experiment
from torch.utils.data import DataLoader, TensorDataset

from finn.data import DatasetTuple
from finn.models.configs import mp_28x28_classifier
from finn.models.inn import MaskedInn, PartitionedInn
from finn.models.model_builder import build_fc_inn, build_conv_inn, build_discriminator
from finn.optimisation import grad_reverse
from finn.optimisation.training_utils import (
    get_data_dim,
    log_images,
    reconstruct_all,
)
from finn.utils import utils

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
    data = {
        'trn': TensorDataset(*[torch.tensor(df.values, dtype=torch.float32) for df in train_tuple]),
        'val': TensorDataset(*[torch.tensor(df.values, dtype=torch.float32) for df in test_tuple]),
    }
    return data


def save_model(save_dir, model, discriminator):
    filename = save_dir / 'checkpt.pth'
    save_dict = {'ARGS': ARGS,
                 'model': model.state_dict(),
                 'discriminator': discriminator.state_dict()}

    torch.save(save_dict, filename)


def restore_model(filename, model, discriminator):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt['model'])
    discriminator.load_state_dict(checkpt['discriminator'])

    return model, discriminator


def train(model, discriminator, dataloader, epoch):

    model.train()

    total_loss_meter = utils.AverageMeter()
    log_prob_meter = utils.AverageMeter()
    disc_loss_meter = utils.AverageMeter()

    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):

        x, s, y = to_device(x, s, y)

        (enc_y, enc_s), neg_log_prob = model.routine(x)

        disc_loss = discriminator.routine(grad_reverse(enc_y), s)[0]

        if ARGS.learn_mask:
            disc_loss += discriminator.routine(enc_s, s)[0]

        neg_log_prob *= ARGS.log_prob_weight
        disc_loss *= ARGS.pred_s_weight

        loss = neg_log_prob + disc_loss

        model.zero_grad()
        discriminator.zero_grad()

        loss.backward()

        model.step()
        discriminator.step()

        total_loss_meter.update(loss.item())
        log_prob_meter.update(neg_log_prob.item())
        disc_loss_meter.update(disc_loss.item())

        time_meter.update(time.time() - end)

        SUMMARY.set_step(itr)
        SUMMARY.log_metric('Loss log_p_x', neg_log_prob.item())
        SUMMARY.log_metric('Loss pred_y_from_zys_loss', disc_loss.item())
        end = time.time()

    model.eval()
    with torch.set_grad_enabled(False):

        log_images(SUMMARY, x, 'original_x')

        z = model(x[:64])

        recon_all, recon_y, recon_s = reconstruct_all(
            ARGS, z, model
        )

        log_images(SUMMARY, recon_all, 'reconstruction_all')
        log_images(SUMMARY, recon_y, 'reconstruction_y')
        log_images(SUMMARY, recon_s, 'reconstruction_s')

    time_for_epoch = time.time() - start_epoch_time
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {:.3g}s | Batches/s: {:.4g} | "
        "Loss -log_p_x (surprisal): {:.5g} | disc_loss: {:.5g} ({:.5g})",
        epoch,
        time_for_epoch,
        1 / time_meter.avg,
        log_prob_meter.avg,
        disc_loss_meter.avg,
        total_loss_meter.avg,
    )


def validate(model, discriminator, val_loader):
    model.eval()
    # start_time = time.time()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:
            x_val, s_val, y_val = to_device(x_val, s_val, y_val)

            (enc_y, enc_s), neg_log_prob = model.routine(x_val)
            disc_loss = discriminator.routine(enc_y, s_val)[0]

            if ARGS.learn_mask:
                disc_loss += discriminator.routine(enc_s, s_val)[0]

            neg_log_prob *= ARGS.log_prob_weight
            disc_loss *= ARGS.pred_s_weight

            loss = neg_log_prob + disc_loss

            loss_meter.update(loss.item(), n=x_val.size(0))

    SUMMARY.log_metric("Loss", loss_meter.avg)

    x_val = torch.cat((x_val, s_val), dim=1) if ARGS.dataset == 'adult' and ARGS.use_s else x_val

    if ARGS.dataset == 'cmnist':

        z = model(x_val[:64])

        recon_all, recon_y, recon_s = reconstruct_all(ARGS, z, model)
        log_images(SUMMARY, x_val, 'original_x', prefix='test')
        log_images(SUMMARY, recon_all, 'reconstruction_all', prefix='test')
        log_images(SUMMARY, recon_y, 'reconstruction_y', prefix='test')
        log_images(SUMMARY, recon_s, 'reconstruction_s', prefix='test')
    else:
        z = model(x_val[:1000])
        recon_all, recon_y, recon_s = reconstruct_all(
            ARGS, z, model
        )
        log_images(SUMMARY, x_val, 'original_x', prefix='test')
        log_images(SUMMARY, recon_y, 'reconstruction_yn', prefix='test')
        log_images(SUMMARY, recon_s, 'reconstruction_yn', prefix='test')
        x_recon = model(model(x_val), reverse=True)
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


def train_masker(model, discriminator, dataloader, epoch):

    model.mask_train()
    discriminator.eval()

    for itr, (x, s, y) in enumerate(dataloader, start=epoch * len(dataloader)):

        x, s, y = to_device(x, s, y)

        (enc_y, enc_s), neg_log_prob = model.routine(x)

        disc_loss = discriminator.routine(enc_y, s)[0]

        masker_loss = neg_log_prob - disc_loss

        if ARGS.learn_mask:
            disc_loss_2 = discriminator.routine(enc_s, s)[0]
            disc_loss += disc_loss_2
            masker_loss += disc_loss_2

        model.zero_grad()
        masker_loss.backward()
        model.step()


def main(args, datasets, metric_callback):
    """Main function

    Args:
        args: commandline arguments
        datasets: a Dataset object
        metric_callback: a function that computes metrics

    Returns:
        the trained model
    """
    assert isinstance(datasets, DatasetTuple)
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

    if args.learn_mask:
        Module = MaskedInn
    else:
        Module = PartitionedInn
    if len(input_shape) > 2:
        model = build_conv_inn(args, input_shape[0])
        disc_fn = mp_28x28_classifier
    else:
        model = build_fc_inn(args, input_shape[0])
        disc_fn = torch.nn.Linear

    model = Module(args, model=model, input_shape=input_shape)
    discriminator = build_discriminator(args, input_shape, disc_fn)

    if ARGS.resume is not None:
        model, discriminator = restore_model(ARGS.resume, model, discriminator)
        metric_callback(ARGS, SUMMARY, model, discriminator, datasets, check_originals=False)
        return

    SUMMARY.set_model_graph(str(model))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(model))

    best_loss = float('inf')

    n_vals_without_improvement = 0

    check_originals = True
    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        LOGGER.info("===> Training INN")
        with SUMMARY.train():
            train(
                model,
                discriminator,
                train_loader,
                epoch,
            )

            if args.learn_mask:
                LOGGER.info("===> Training Masker")
                train_masker(
                    model,
                    discriminator,
                    val_loader,
                    epoch
                )
                with torch.set_grad_enabled(False):
                    mask = model.masker(threshold=True)
                    zs_dim = (1 - mask).sum() / mask.nelement()
                LOGGER.info("Zs frac:  {}", zs_dim.item())

        if epoch % ARGS.val_freq == 0 and epoch != 0:
            with SUMMARY.test():
                val_loss = validate(model, discriminator, val_loader)
                if args.super_val:
                    metric_callback(
                        ARGS, SUMMARY, model, discriminator,
                        datasets, check_originals=check_originals
                    )
                    check_originals = False

                if val_loss < best_loss:
                    best_loss = val_loss
                    save_model(save_dir=save_dir, model=model, discriminator=discriminator)
                    n_vals_without_improvement = 0
                else:
                    n_vals_without_improvement += 1

                # scheduler.step(val_loss)

                LOGGER.info(
                    '[VAL] Epoch {:04d} | Val Loss {:.6f} | '
                    'No improvement during validation: {:02d}',
                    epoch,
                    val_loss,
                    n_vals_without_improvement,
                )

    LOGGER.info('Training has finished.')
    model, discs = restore_model(save_dir / 'checkpt.pth', model, discriminator)
    metric_callback(ARGS, SUMMARY, model, discs, datasets, save_to_csv=True)
    save_model(save_dir=save_dir, model=model, discriminator=discriminator)
    model.eval()


if __name__ == '__main__':
    print('This file cannot be run directly.')
