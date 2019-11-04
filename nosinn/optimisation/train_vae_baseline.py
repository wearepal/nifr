"""Main training file"""
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import wandb

from nosinn.data import DatasetTriplet
from nosinn.models.autoencoder import VAE
from nosinn.models.configs import conv_autoencoder, fc_autoencoder
from nosinn.models.configs.classifiers import linear_disciminator
from nosinn.models.factory import build_discriminator
from nosinn.utils import utils

from .evaluation import evaluate
from .loss import PixelCrossEntropy, grad_reverse
from .utils import get_data_dim, log_images

__all__ = ["main_vae"]

NDECS = 0
ARGS = None
LOGGER = None
INPUT_SHAPE = ()


def train(vae, disc_enc_y, disc_enc_s, dataloader, epoch: int, recon_loss_fn) -> int:
    vae.train()
    vae.eval()

    total_loss_meter = utils.AverageMeter()
    elbo_meter = utils.AverageMeter()
    disc_loss_meter = utils.AverageMeter()

    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    start_itr = start = epoch * len(dataloader)
    for itr, (x, s, y) in enumerate(dataloader, start=start_itr):

        x, s, y = to_device(x, s, y)

        s_oh = None
        if ARGS.cond_decoder:  # One-hot encode the sensitive attribute
            s_oh = F.one_hot(s, num_classes=ARGS.s_dim)

        # Encode the data
        encoding, posterior = vae.encode(x, stochastic=True, return_posterior=True)

        if ARGS.enc_s_dim > 0:
            enc_y, enc_s = encoding.split(split_size=(ARGS.enc_y_dim, ARGS.enc_s_dim), dim=1)
            decoder_input = torch.cat([enc_y, enc_s.detach()], dim=1)
        else:
            enc_y = encoding
            decoder_input = encoding

        recon = vae.decode(decoder_input, s_oh)

        # Compute losses
        kl = vae.compute_divergence(encoding, posterior)
        recon_loss = recon_loss_fn(recon, x)

        recon_loss /= x.size(0)
        kl /= x.size(0)

        elbo = recon_loss + vae.kl_weight * kl

        enc_y = grad_reverse(enc_y)
        # Discriminator for zy
        disc_loss, acc = disc_enc_y.routine(enc_y, s)

        # Discriminator for zs if partitioning the latent space
        if disc_enc_s is not None:
            disc_loss += disc_enc_s.routine(enc_s, s)[0]
            disc_enc_s.zero_grad()

        elbo *= ARGS.elbo_weight
        disc_loss *= ARGS.pred_s_weight

        loss = elbo + disc_loss

        # Update model parameters
        vae.zero_grad()
        disc_enc_y.zero_grad()

        loss.backward()
        vae.step()
        disc_enc_y.step()

        if disc_enc_s is not None:
            disc_enc_s.step()

        # Log losses
        total_loss_meter.update(loss.item())
        elbo_meter.update(elbo.item())
        disc_loss_meter.update(disc_loss.item())

        time_meter.update(time.time() - end)

        wandb.log({"Loss NLL": elbo.item()}, step=itr)
        wandb.log({"Loss Adversarial": disc_loss.item()}, step=itr)
        end = time.time()

        # Log images
        if itr % 50 == 0:
            with torch.set_grad_enabled(False):

                enc = vae.encode(x[:64], stochastic=False)

                if s_oh is not None:
                    s_oh = s_oh[:64]

                if ARGS.enc_s_dim > 0:
                    enc_y, enc_s = enc.split(split_size=(ARGS.enc_y_dim, ARGS.enc_s_dim), dim=1)
                    enc_y_m = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)
                    enc_s_m = torch.cat([torch.zeros_like(enc_y), enc_s], dim=1)
                else:
                    enc_y_m = enc
                    enc_s_m = torch.zeros_like(enc)
                    if ARGS.cond_decoder:
                        if ARGS.s_dim == 2:
                            s_oh_flipped = 1 - s_oh
                        else:
                            s_oh_flipped = s_oh[torch.randperm(s_oh.size(0))]
                        recon_s_flipped = vae.reconstruct(enc, s_oh_flipped)
                        log_images(ARGS, recon_s_flipped, "reconstruction_y_flipped_s", step=itr)

                recon_all = vae.reconstruct(enc, s=s_oh)
                recon_y = vae.reconstruct(
                    enc_y_m, s=torch.zeros_like(s_oh) if s_oh is not None else None
                )
                recon_null = vae.reconstruct(
                    torch.zeros_like(enc), s=torch.zeros_like(s_oh) if s_oh is not None else None
                )
                recon_s = vae.reconstruct(enc_s_m, s=s_oh)

                log_images(ARGS, x[:64], "original_x", step=itr)
                log_images(ARGS, recon_all, "reconstruction_all", step=itr)
                log_images(ARGS, recon_y, "reconstruction_y", step=itr)
                log_images(ARGS, recon_s, "reconstruction_s", step=itr)
                log_images(ARGS, recon_null, "reconstruction_null", step=itr)

    time_for_epoch = time.time() - start_epoch_time
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {:.3g}s | Batches/s: {:.4g} | "
        "Loss ELBO: {:.5g} | Loss Adv: {:.5g} ({:.5g})",
        epoch,
        time_for_epoch,
        1 / time_meter.avg,
        elbo_meter.avg,
        disc_loss_meter.avg,
        total_loss_meter.avg,
    )
    return itr


def validate(vae, discriminator, val_loader, itr, recon_loss_fn):
    vae.eval()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:

            x_val, s_val, y_val = to_device(x_val, s_val, y_val)

            s_oh = None
        if ARGS.cond_decoder:
            s_oh = F.one_hot(s_val, num_classes=ARGS.s_dim)
        encoding, posterior = vae.encode(x_val, stochastic=True, return_posterior=True)
        kl = vae.compute_divergence(encoding, posterior)

        if ARGS.enc_s_dim > 0:
            enc_y, enc_s = encoding.split(split_size=(ARGS.enc_y_dim, ARGS.enc_s_dim), dim=1)
            decoder_input = torch.cat([enc_y, enc_s.detach()], dim=1)
        else:
            enc_y = encoding
            decoder_input = encoding

        decoding = vae.decode(decoder_input, s_oh)
        recon_loss = recon_loss_fn(decoding, x_val)

        recon_loss /= x_val.size(0)
        kl /= x_val.size(0)

        elbo = recon_loss + vae.kl_weight * kl

        enc_y = grad_reverse(enc_y)
        disc_loss, acc = discriminator.routine(enc_y, s_val)

        elbo *= ARGS.elbo_weight
        disc_loss *= ARGS.pred_s_weight

        loss = elbo + disc_loss

        loss_meter.update(loss.item(), n=x_val.size(0))

    wandb.log({"Loss": loss_meter.avg}, step=itr)

    return loss_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def encode_dataset(args, vae, data_loader):
    LOGGER.info("Encoding dataset...")
    all_xy = []
    all_s = []
    all_y = []

    with torch.set_grad_enabled(False):
        for x, s, y in data_loader:

            x = x.to(args.device)
            all_s.append(s)
            all_y.append(y)

            enc = vae.encode(x, stochastic=False)

            s_oh = None
            if ARGS.cond_decoder:
                s_oh = x.new_zeros(x.size(0), args.s_dim)

            if ARGS.enc_s_dim > 0:
                enc_y, enc_s = enc.split(split_size=(ARGS.enc_y_dim, ARGS.enc_s_dim), dim=1)
                enc_y_m = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)
            else:
                enc_y_m = enc

            xy = vae.reconstruct(enc_y_m, s=s_oh)
            all_xy.append(xy.detach().cpu())

    all_xy = torch.cat(all_xy, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_xy, all_s, all_y)
    LOGGER.info("Done.")

    return encoded_dataset


def evaluate_vae(args, vae, train_loader, test_loader, step, save_to_csv: Optional[Path] = None):
    train_data = encode_dataset(args, vae, train_loader)
    test_data = encode_dataset(args, vae, test_loader)
    evaluate(ARGS, step, train_data, test_data, "xy", pred_s=False, save_to_csv=save_to_csv)


def main_vae(args, datasets):
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
    global ARGS, LOGGER, INPUT_SHAPE
    ARGS = args

    wandb.init(
        project="nosinn",
        # sync_tensorboard=True,
        config=vars(ARGS),
    )
    # wandb.tensorboard.patch(save=False, pytorch=True)

    save_dir = Path(ARGS.save) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = utils.get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS.device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu"
    )
    LOGGER.info("{} GPUs available. Using GPU {}", torch.cuda.device_count(), ARGS.gpu)

    # ==== construct dataset ====
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(datasets.pretrain, shuffle=True, batch_size=ARGS.batch_size)
    val_loader = DataLoader(datasets.task_train, shuffle=True, batch_size=ARGS.test_batch_size)
    test_loader = DataLoader(datasets.task, shuffle=False, batch_size=ARGS.test_batch_size)

    # ==== construct networks ====
    INPUT_SHAPE = get_data_dim(train_loader)
    is_image_data = len(INPUT_SHAPE) > 2

    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}

    ARGS.s_dim = ARGS.s_dim if ARGS.s_dim > 1 else 2

    if is_image_data:
        decoding_dim = INPUT_SHAPE[0] * 256 if args.recon_loss == "ce" else INPUT_SHAPE[0]
        encoder, decoder, enc_shape = conv_autoencoder(
            INPUT_SHAPE,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_y_dim + ARGS.enc_s_dim,
            decoding_dim=decoding_dim,
            levels=ARGS.levels,
            vae=True,
            s_dim=ARGS.s_dim if ARGS.cond_decoder else 0,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            INPUT_SHAPE,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_y_dim + ARGS.enc_s_dim,
            levels=ARGS.levels,
            vae=ARGS.vae,
            s_dim=ARGS.s_dim if ARGS.cond_decoder else 0,
        )

    if ARGS.recon_loss == "l1":
        recon_loss_fn = nn.L1Loss(reduction="sum")
    elif ARGS.recon_loss == "l2":
        recon_loss_fn = nn.MSELoss(reduction="sum")
    elif ARGS.recon_loss == "huber":
        recon_loss_fn = nn.SmoothL1Loss(reduction="sum")
    elif ARGS.recon_loss == "ce":
        recon_loss_fn = PixelCrossEntropy(reduction="sum")
    else:
        raise ValueError(f"{ARGS.recon_loss} is an invalid reconstruction loss")

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        kl_weight=ARGS.kl_weight,
        optimizer_args=optimizer_args,
        decode_with_s=True,
    )
    vae.to(args.device)

    # Initialise Discriminator
    disc_fn = linear_disciminator

    disc_optimizer_kwargs = {"lr": args.disc_lr}

    disc_enc_y_kwargs = {
        "hidden_channels": ARGS.disc_enc_y_channels,
        "num_blocks": ARGS.disc_enc_y_depth,
        "use_bn": True,
    }

    disc_enc_y = build_discriminator(
        args,
        enc_shape,
        frac_enc=ARGS.enc_y_dim / enc_shape[0],
        model_fn=disc_fn,
        model_kwargs=disc_enc_y_kwargs,
        optimizer_kwargs=disc_optimizer_kwargs,
    )
    disc_enc_y.to(args.device)

    disc_enc_s = None
    if ARGS.enc_s_dim > 0:

        disc_enc_s_kwargs = {
            "hidden_channels": ARGS.disc_enc_s_channels,
            "num_blocks": ARGS.disc_enc_s_depth,
            "use_bn": True,
        }
        disc_enc_s = build_discriminator(
            args,
            enc_shape,
            frac_enc=ARGS.enc_s_dim / enc_shape[0],
            model_fn=disc_fn,
            model_kwargs=disc_enc_s_kwargs,
            optimizer_kwargs=disc_optimizer_kwargs,
        )
        disc_enc_s.to(args.device)

    # Logging
    # wandb.set_model_graph(str(inn))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(vae))

    best_loss = float("inf")
    n_vals_without_improvement = 0

    # Train INN for N epochs
    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(vae, disc_enc_y, disc_enc_s, train_loader, epoch, recon_loss_fn)

        if epoch % ARGS.val_freq == 0 and epoch != 0:
            val_loss = validate(vae, disc_enc_y, val_loader, itr, recon_loss_fn)
            if args.super_val:
                evaluate_vae(args, vae, train_loader=val_loader, test_loader=test_loader, step=itr)

            if val_loss < best_loss:
                best_loss = val_loss
                n_vals_without_improvement = 0
            else:
                n_vals_without_improvement += 1

            LOGGER.info(
                "[VAL] Epoch {:04d} | Val Loss {:.6f} | "
                "No improvement during validation: {:02d}",
                epoch,
                val_loss,
                n_vals_without_improvement,
            )

    LOGGER.info("Training has finished.")
    evaluate_vae(
        args,
        vae,
        train_loader=val_loader,
        test_loader=test_loader,
        step=itr,
        save_to_csv=Path(ARGS.save),
    )