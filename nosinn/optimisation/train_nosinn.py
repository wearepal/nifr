"""Main training file"""
import time
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from nosinn.data import DatasetTriplet
from nosinn.models import AutoEncoder, Classifier, VAE
from nosinn.models.configs import conv_autoencoder, fc_autoencoder
from nosinn.models.configs.classifiers import (
    fc_net,
    linear_disciminator,
    mp_32x32_net,
    mp_64x64_net,
)
from nosinn.models.factory import build_conv_inn, build_discriminator, build_fc_inn
from nosinn.models.inn import PartitionedAeInn, PartitionedInn
from nosinn.utils import AverageMeter, count_parameters, get_logger, wandb_log

from .evaluation import log_metrics
from .loss import PixelCrossEntropy, MixedLoss
from .utils import get_data_dim, log_images, apply_gradients

__all__ = ["main"]

NDECS = 0
ARGS = None
LOGGER = None


def save_model(save_dir, inn, discriminator) -> str:
    filename = save_dir / "checkpt.pth"
    save_dict = {
        "ARGS": ARGS,
        "model": inn.state_dict(),
        "discriminator": discriminator.state_dict(),
    }

    torch.save(save_dict, filename)

    return filename


def restore_model(filename, model, discriminator):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["model"])
    discriminator.load_state_dict(checkpt["discriminator"])

    return model, discriminator


def compute_loss(
    x: torch.Tensor, s: torch.Tensor, inn: PartitionedInn,
    discriminator: Classifier, itr: int
) -> Tuple[torch.Tensor, Dict[str, float]]:

    enc, nll = inn.routine(x)

    enc_y, enc_s = inn.split_encoding(enc)

    # Update the discriminator k-times
    if discriminator.training:
        enc_y_sg = enc_y.detach()
        for _ in range(ARGS.disc_updates):
            logits = discriminator(enc_y_sg)
            disc_loss = discriminator.apply_criterion(logits, s).mean()
            discriminator.zero_grad()
            disc_loss.backward()
            discriminator.step()

    recon_loss = x.new_zeros(())
    if ARGS.train_on_recon:
        enc_y_m = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)
        if ARGS.recon_detach:
            enc_y_m = enc_y_m.detach()
        enc_y = inn.invert(enc_y_m)
        if ARGS.recon_stability_weight > 0:
            recon_loss = F.l1_loss(enc_y, x)
        # enc_y = enc_y.clamp(min=0, max=1)

    logits = discriminator(enc_y)
    probs = logits.softmax(dim=1) if ARGS.s_dim > 1 else logits.sigmoid()
    entropy = -(probs * probs.log()).sum(1).mean()

    if itr < ARGS.warmup_steps:
        pred_s_weight = ARGS.pred_s_weight * np.exp(-7 + 7 * itr / ARGS.warmup_steps)
    else:
        pred_s_weight = ARGS.pred_s_weight

    nll *= ARGS.nll_weight
    entropy *= pred_s_weight
    recon_loss *= ARGS.recon_stability_weight

    inn_loss = nll + entropy + recon_loss

    if inn.training:
        inn.zero_grad()
        inn_params = [param for param in inn.parameters() if param.requires_grad]
        inn_grads = torch.autograd.grad(
            inn_loss,
            inn_params,
            retain_graph=True,
            allow_unused=True)
        apply_gradients(inn_grads, inn_params)
        inn.step()

    logging_dict = {
        "Loss NLL": nll.item(),
        "Loss Adversarial": entropy.item(),
        "Recon L1 loss": recon_loss.item(),
        "Validation loss": (nll + entropy + recon_loss).item(),
    }
    return inn_loss, disc_loss, logging_dict


def train(inn, discriminator, dataloader, epoch: int) -> int:
    inn.train()
    discriminator.train()

    total_loss_meter = AverageMeter()
    loss_meters = {
        "Loss NLL": AverageMeter(),
        "Loss Adversarial": AverageMeter(),
        "Recon L1 loss": AverageMeter(),
        "Validation loss": AverageMeter(),
    }

    time_meter = AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    start_itr = start = epoch * len(dataloader)
    for itr, (x, s, y) in enumerate(dataloader, start=start_itr):

        x, s, y = to_device(x, s, y)

        inn_loss, disc_loss, logging_dict = compute_loss(x, s, inn, discriminator, itr)

        # Log losses
        total_loss_meter.update(inn_loss.item())
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        time_meter.update(time.time() - end)

        wandb_log(ARGS, logging_dict, step=itr)
        end = time.time()

        # Log images
        if itr % ARGS.log_freq == 0:
            with torch.set_grad_enabled(False):
                log_recons(inn, x, itr)

    time_for_epoch = time.time() - start_epoch_time
    log_string = " | ".join(f"{name}: {meter.avg:.5g}" for name, meter in loss_meters.items())
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {:.3g}s | Batches/s: {:.4g} | {} ({:.5g})",
        epoch,
        time_for_epoch,
        1 / time_meter.avg,
        log_string,
        total_loss_meter.avg,
    )
    return itr


def validate(inn: PartitionedInn, discriminator: Classifier, val_loader, itr: int):
    inn.eval()
    discriminator.eval()

    with torch.no_grad():
        loss_meter = AverageMeter()
        for x_val, s_val, y_val in val_loader:

            x_val, s_val, y_val = to_device(x_val, s_val, y_val)

            _, _, logging_dict = compute_loss(x_val, s_val, inn, discriminator, itr)

            loss_meter.update(logging_dict["Validation loss"], n=x_val.size(0))

    wandb_log(ARGS, {"Loss": loss_meter.avg}, step=itr)

    if ARGS.dataset in ("cmnist", "celeba"):
        log_recons(inn, x_val, itr, prefix="test")
    else:
        z = inn(x_val[:1000])
        recon_all, recon_y, recon_s = inn.decode(z, partials=True)
        log_images(ARGS, x_val, "original_x", prefix="test", step=itr)
        log_images(ARGS, recon_y, "reconstruction_yn", prefix="test", step=itr)
        log_images(ARGS, recon_s, "reconstruction_yn", prefix="test", step=itr)
        x_recon = inn(inn(x_val), reverse=True)
        x_diff = (x_recon - x_val).abs().mean().item()
        print(f"MAE of x and reconstructed x: {x_diff}")
        wandb_log(ARGS, {"reconstruction MAE": x_diff}, step=itr)

    return loss_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def log_recons(inn: PartitionedInn, x, itr: int, prefix: Optional[str] = None):
    z = inn(x[:64])
    recon_all, recon_y, recon_s = inn.decode(z, partials=True)
    log_images(ARGS, x, "original_x", prefix=prefix, step=itr)
    log_images(ARGS, recon_all, "reconstruction_all", prefix=prefix, step=itr)
    log_images(ARGS, recon_y, "reconstruction_y", prefix=prefix, step=itr)
    log_images(ARGS, recon_s, "reconstruction_s", prefix=prefix, step=itr)


def main(args, datasets: DatasetTriplet):
    """Main function

    Args:
        args: commandline arguments
        datasets: a Dataset object

    Returns:
        the trained model
    """
    assert isinstance(datasets, DatasetTriplet)
    # ==== initialize globals ====
    global ARGS, LOGGER
    ARGS = args

    if ARGS.use_wandb:
        wandb.init(
            project="nosinn",
            # sync_tensorboard=True,
            config=vars(ARGS),
        )
        # wandb.tensorboard.patch(save=False, pytorch=True)

    save_dir = Path(ARGS.save) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info(ARGS)
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS.device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu"
    )
    LOGGER.info("{} GPUs available. Using device '{}'", torch.cuda.device_count(), ARGS.device)

    # ==== construct dataset ====
    LOGGER.info(
        "Size of pretrain: {}, task_train: {}, task: {}",
        len(datasets.pretrain),
        len(datasets.task_train),
        len(datasets.task),
    )
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(datasets.pretrain, shuffle=True, batch_size=ARGS.batch_size)
    val_loader = DataLoader(datasets.task_train, shuffle=False, batch_size=ARGS.test_batch_size)

    # ==== construct networks ====
    input_shape = get_data_dim(train_loader)
    is_image_data = len(input_shape) > 2

    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    feature_groups = None
    if hasattr(datasets.pretrain, "feature_groups"):
        feature_groups = datasets.pretrain.feature_groups

    # Model constructors and arguments
    if is_image_data:
        inn_fn = build_conv_inn
        if args.train_on_recon:
            if args.dataset == "cmnist":
                disc_fn = mp_32x32_net
            else:
                disc_fn = mp_64x64_net
            disc_kwargs = {"use_bn": not ARGS.spectral_norm}
        else:
            disc_fn = linear_disciminator
            disc_kwargs = {
                "hidden_channels": ARGS.disc_channels,
                "num_blocks": ARGS.disc_depth,
                "use_bn": not ARGS.spectral_norm,
            }
    else:
        inn_fn = build_fc_inn
        disc_fn = fc_net
        disc_kwargs = {"hidden_dims": args.disc_hidden_dims}

    #  Model arguments
    inn_kwargs = {"args": args, "optimizer_args": optimizer_args, "feature_groups": feature_groups}

    # Initialise INN
    if ARGS.autoencode:
        if ARGS.input_noise:
            LOGGER.warn("WARNING: autoencoder and input noise are both turned on!")

        if is_image_data:
            decoding_dim = input_shape[0] * 256 if args.ae_loss == "ce" else input_shape[0]
            encoder, decoder, enc_shape = conv_autoencoder(
                input_shape,
                ARGS.ae_channels,
                encoding_dim=ARGS.ae_enc_dim,
                decoding_dim=decoding_dim,
                levels=ARGS.ae_levels,
                vae=ARGS.vae,
            )
        else:
            encoder, decoder, enc_shape = fc_autoencoder(
                input_shape=input_shape,
                hidden_channels=ARGS.ae_channels,
                encoding_dim=ARGS.ae_enc_dim,
                levels=ARGS.ae_levels,
                vae=ARGS.vae,
            )

        autoencoder: AutoEncoder
        if ARGS.vae:
            autoencoder = VAE(encoder=encoder, decoder=decoder, kl_weight=ARGS.kl_weight)
        else:
            autoencoder = AutoEncoder(encoder=encoder, decoder=decoder)

        inn_kwargs["input_shape"] = enc_shape
        inn_kwargs["autoencoder"] = autoencoder
        inn_kwargs["model"] = inn_fn(args, inn_kwargs["input_shape"])

        inn = PartitionedAeInn(**inn_kwargs)
        inn.to(args.device)

        if ARGS.path_to_ae:
            save_dict = torch.load(ARGS.path_to_ae, map_location=lambda storage, loc: storage)
            autoencoder.load_state_dict(save_dict["model"])
            if "args" in save_dict:
                args_ae = save_dict["args"]
                assert ARGS.ae_channels == args_ae["init_channels"]
                assert ARGS.ae_levels == args_ae["levels"]
        else:
            ae_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
            if ARGS.ae_loss == "l1":
                ae_loss_fn = nn.L1Loss(reduction="sum")
            elif ARGS.ae_loss == "l2":
                ae_loss_fn = nn.MSELoss(reduction="sum")
            elif ARGS.ae_loss == "huber":
                ae_loss_fn = nn.SmoothL1Loss(reduction="sum")
            elif ARGS.ae_loss == "ce":
                ae_loss_fn = PixelCrossEntropy(reduction="sum")
            elif ARGS.ae_loss == "mixed":
                assert feature_groups is not None, "can only do multi loss with feature groups"
                ae_loss_fn = MixedLoss(feature_groups, reduction="sum")
            else:
                raise ValueError(f"{ARGS.ae_loss} is an invalid reconstruction loss")

            inn.fit_ae(train_loader, epochs=ARGS.ae_epochs, device=ARGS.device, loss_fn=ae_loss_fn)
            # the args names follow the convention of the standalone VAE commandline args
            args_ae = {"init_channels": ARGS.ae_channels, "levels": ARGS.ae_levels}
            torch.save(
                {"model": autoencoder.state_dict(), "args": args_ae}, save_dir / "autoencoder"
            )
    else:
        inn_kwargs["input_shape"] = input_shape
        inn_kwargs["model"] = inn_fn(args, input_shape)
        inn = PartitionedInn(**inn_kwargs)
        inn.to(args.device)
        enc_shape = inn.output_dim

    disc_input_shape: Tuple[int, ...] = input_shape if args.train_on_recon else (inn.zy_dim,)

    print(f"zs dim: {inn.zs_dim}")
    print(f"zy dim: {inn.zy_dim}")
    # Initialise Discriminator
    dis_optimizer_kwargs = {"lr": args.disc_lr}
    discriminator = build_discriminator(
        args,
        input_shape=disc_input_shape,
        frac_enc=enc_shape,
        model_fn=disc_fn,
        model_kwargs=disc_kwargs,
        optimizer_kwargs=dis_optimizer_kwargs,
    )
    discriminator.to(args.device)

    if ARGS.spectral_norm:

        def spectral_norm(m):
            if hasattr(m, "weight"):
                return torch.nn.utils.spectral_norm(m)

        inn.apply(spectral_norm)
        discriminator.apply(spectral_norm)

    # Save initial parameters
    save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)

    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring model from checkpoint")
        inn, discriminator = restore_model(ARGS.resume, inn, discriminator)
        log_metrics(ARGS, model=inn, data=datasets, save_to_csv=Path(ARGS.save), step=0)
        return

    # Logging
    # wandb.set_model_graph(str(inn))
    LOGGER.info("Number of trainable parameters: {}", count_parameters(inn))

    best_loss = float("inf")
    n_vals_without_improvement = 0

    # Train INN for N epochs
    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(inn, discriminator, train_loader, epoch)

        if epoch % ARGS.val_freq == 0 and epoch != 0:
            val_loss = validate(inn, discriminator, val_loader, itr)
            if args.super_val:
                log_metrics(ARGS, model=inn, data=datasets, step=itr)

            if val_loss < best_loss:
                best_loss = val_loss
                save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)
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
    inn, discriminator = restore_model(save_dir / "checkpt.pth", inn, discriminator)
    log_metrics(ARGS, model=inn, data=datasets, save_to_csv=Path(ARGS.save), step=itr)
    save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)
    inn.eval()


if __name__ == "__main__":
    print("This file cannot be run directly.")
