"""Main training file"""
import time
from logging import Logger
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import wandb
from nifr import utils
from nifr.configs import VaeArgs
from nifr.data import DatasetTriplet, load_dataset
from nifr.models import VAE, VaeResults, build_discriminator
from nifr.models.configs import conv_autoencoder, fc_autoencoder, linear_disciminator
from nifr.utils import random_seed, wandb_log

from .evaluation import evaluate, evaluate_celeba_all_attrs
from .loss import PixelCrossEntropy, VGGLoss, grad_reverse
from .utils import get_data_dim, log_images

__all__ = ["main_vae"]

ARGS: VaeArgs
LOGGER: Logger
INPUT_SHAPE: Tuple[int, ...]


def compute_losses(
    x, s, s_oh: Optional[torch.Tensor], vae: VAE, disc_enc_y, disc_enc_s, recon_loss_fn
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute all losses"""
    vae_results: VaeResults = vae.standalone_routine(
        x=x,
        s_oh=s_oh,
        recon_loss_fn=recon_loss_fn,
        stochastic=ARGS.stochastic,
        enc_y_dim=ARGS.enc_y_dim,
        enc_s_dim=ARGS.enc_s_dim,
    )
    elbo, enc_y, enc_s = vae_results.elbo, vae_results.enc_y, vae_results.enc_s

    enc_y = grad_reverse(enc_y)
    # Discriminator for zy
    disc_loss, _ = disc_enc_y.routine(enc_y, s)

    # Discriminator for zs if partitioning the latent space
    if ARGS.enc_s_dim > 0:
        disc_loss += disc_enc_s.routine(enc_s, s)[0]
        disc_enc_s.zero_grad()

    elbo *= ARGS.elbo_weight
    disc_loss *= ARGS.pred_s_weight

    loss = elbo + disc_loss
    logging_dict = {
        "ELBO": elbo.item(),
        "Adv loss": disc_loss.item(),
        "KL divergence": vae_results.kl_div.item(),
    }
    return loss, logging_dict


def train(vae, disc_enc_y, disc_enc_s, dataloader, epoch: int, recon_loss_fn) -> int:
    vae.train()
    vae.eval()

    total_loss_meter = utils.AverageMeter()
    loss_meters: Dict[str, utils.AverageMeter] = {
        "ELBO": utils.AverageMeter(),
        "Adv loss": utils.AverageMeter(),
        "KL divergence": utils.AverageMeter(),
    }

    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    start_itr = epoch * len(dataloader)
    for itr, (x, s, y) in enumerate(dataloader, start=start_itr):

        x, s, y = to_device(x, s, y)
        s_oh = None
        if ARGS.cond_decoder:  # One-hot encode the sensitive attribute
            s_oh = F.one_hot(s, num_classes=ARGS.s_dim)

        loss, logging_dict = compute_losses(
            x=x,
            s=s,
            s_oh=s_oh,
            vae=vae,
            disc_enc_y=disc_enc_y,
            disc_enc_s=disc_enc_s,
            recon_loss_fn=recon_loss_fn,
        )

        vae.zero_grad()
        disc_enc_y.zero_grad()

        loss.backward()
        vae.step()
        disc_enc_y.step()

        if disc_enc_s is not None:
            disc_enc_s.step()

        # Log losses
        total_loss_meter.update(loss.item())
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        time_meter.update(time.time() - end)

        wandb_log(ARGS, logging_dict, step=itr)
        end = time.time()

        # Log images
        if itr % ARGS.log_freq == 0:
            with torch.set_grad_enabled(False):
                log_recons(vae=vae, x=x, s_oh=s_oh, itr=itr)

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


def validate(vae: VAE, disc_enc_y, disc_enc_s, val_loader, itr: int, recon_loss_fn):
    vae.eval()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:

            x_val, s_val, y_val = to_device(x_val, s_val, y_val)
            s_oh = None
            if ARGS.cond_decoder:  # One-hot encode the sensitive attribute
                s_oh = F.one_hot(s_val, num_classes=ARGS.s_dim)

            loss, logging_dict = compute_losses(
                x=x_val,
                s=s_val,
                s_oh=s_oh,
                vae=vae,
                disc_enc_y=disc_enc_y,
                disc_enc_s=disc_enc_s,
                recon_loss_fn=recon_loss_fn,
            )

            loss_meter.update(loss.item(), n=x_val.size(0))

    wandb_log(ARGS, {"Loss": loss_meter.avg}, step=itr)

    if ARGS.dataset in ("cmnist", "celeba", "ssrp", "genfaces"):
        log_recons(vae=vae, x=x_val, s_oh=s_oh, itr=itr, prefix="test")

    return loss_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def log_recons(vae: VAE, x, s_oh: Optional[torch.Tensor], itr, prefix=None):
    """Log reconstructed images"""
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
    recon_y = vae.reconstruct(enc_y_m, s=torch.zeros_like(s_oh) if s_oh is not None else None)
    recon_null = vae.reconstruct(
        torch.zeros_like(enc), s=torch.zeros_like(s_oh) if s_oh is not None else None
    )
    recon_s = vae.reconstruct(enc_s_m, s=s_oh)
    log_images(ARGS, x[:64], "original_x", step=itr, prefix=prefix)
    log_images(ARGS, recon_all, "reconstruction_all", step=itr, prefix=prefix)
    log_images(ARGS, recon_y, "reconstruction_y", step=itr, prefix=prefix)
    log_images(ARGS, recon_s, "reconstruction_s", step=itr, prefix=prefix)
    log_images(ARGS, recon_null, "reconstruction_null", step=itr, prefix=prefix)


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

            if args.dataset in ("celeba", "ssrp", "genfaces"):
                xy = 0.5 * xy + 0.5
            xy = xy.clamp(0, 1)

            all_xy.append(xy.detach().cpu())

    all_xy = torch.cat(all_xy, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_xy, all_s, all_y)
    LOGGER.info("Done.")

    return encoded_dataset


def evaluate_vae(
    args,
    vae,
    train_loader,
    test_loader,
    step,
    save_to_csv: Optional[Path] = None,
    all_attrs_celeba: Optional[Path] = None,
):
    train_data = encode_dataset(args, vae, train_loader)
    test_data = encode_dataset(args, vae, test_loader)
    evaluate(ARGS, step, train_data, test_data, name="xy", pred_s=False, save_to_csv=save_to_csv)
    if all_attrs_celeba is not None and args.dataset == "celeba":
        evaluate_celeba_all_attrs(
            args=args,
            train_data=train_loader.dataset,
            test_data=test_loader.dataset,
            test_data_xy=test_data,
            save_dir=all_attrs_celeba,
        )


def main_vae(raw_args=None) -> None:
    """Main function

    Args:
        raw_args: commandline arguments
    """
    args = VaeArgs(explicit_bool=True, underscores_to_dashes=True)
    args.parse_args(raw_args)
    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    # ==== initialize globals ====
    global ARGS, LOGGER, INPUT_SHAPE
    ARGS = args
    args_dict = args.as_dict()

    if ARGS.use_wandb:
        wandb.init(project="nosinn", config=args_dict)

    save_dir = Path(ARGS.save_dir) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = utils.get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info("Namespace(" + ", ".join(f"{k}={args_dict[k]}" for k in sorted(args_dict)) + ")")
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS.device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu"
    )
    LOGGER.info("{} GPUs available. Using device '{}'", torch.cuda.device_count(), ARGS.device)

    # ==== construct dataset ====
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    train_loader = DataLoader(
        datasets.pretrain,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.task_train,
        shuffle=True,
        batch_size=ARGS.test_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        datasets.task,
        shuffle=False,
        batch_size=ARGS.test_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )

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
            vae=ARGS.vae,
            s_dim=ARGS.s_dim if ARGS.cond_decoder else 0,
            level_depth=ARGS.level_depth,
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
    LOGGER.info("Encoding shape: {}", enc_shape)

    recon_loss_fn_: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if ARGS.recon_loss == "l1":
        recon_loss_fn_ = nn.L1Loss(reduction="sum")
    elif ARGS.recon_loss == "l2":
        recon_loss_fn_ = nn.MSELoss(reduction="sum")
    elif ARGS.recon_loss == "huber":
        recon_loss_fn_ = nn.SmoothL1Loss(reduction="sum")
    elif ARGS.recon_loss == "ce":
        recon_loss_fn_ = PixelCrossEntropy(reduction="sum")
    else:
        raise ValueError(f"{ARGS.recon_loss} is an invalid reconstruction loss")

    recon_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if ARGS.vgg_weight != 0:
        vgg_loss = VGGLoss()
        vgg_loss.to(ARGS.device)

        def recon_loss_fn(input_, target):
            return recon_loss_fn_(input_, target) + ARGS.vgg_weight * vgg_loss(input_, target)

    else:
        recon_loss_fn = recon_loss_fn_

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        kl_weight=ARGS.kl_weight,
        optimizer_kwargs=optimizer_args,
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
        input_shape=enc_shape,
        target_dim=datasets.s_dim,
        train_on_recon=ARGS.train_on_recon,
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
            enc_shape,
            target_dim=datasets.s_dim,
            train_on_recon=ARGS.train_on_recon,
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

    epoch_0 = 0
    if args.resume:
        LOGGER.info("Restoring from checkpoint")
        checkpoint = torch.load(args.resume)
        vae.load_state_dict(checkpoint["model"])
        epoch_0 = checkpoint["epoch"]

    itr = epoch_0 * len(train_loader)
    # Train INN for N epochs
    for epoch in range(epoch_0, ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(vae, disc_enc_y, disc_enc_s, train_loader, epoch, recon_loss_fn)

        save_model(save_dir=save_dir, vae=vae, epoch=epoch, prefix="latest")
        if epoch % ARGS.val_freq == 0 and epoch != 0:
            val_loss = validate(vae, disc_enc_y, disc_enc_s, val_loader, itr, recon_loss_fn)
            if args.super_val:
                evaluate_vae(args, vae, train_loader=val_loader, test_loader=test_loader, step=itr)

            if val_loss < best_loss:
                best_loss = val_loss
                n_vals_without_improvement = 0
                save_model(save_dir=save_dir, vae=vae, epoch=epoch)
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
        save_to_csv=Path(ARGS.save_dir),
        all_attrs_celeba=save_dir if ARGS.all_attrs else None,
    )


def save_model(save_dir, vae, epoch, prefix="best") -> str:
    filename = save_dir / f"{prefix}_checkpt.pth"
    save_dict = {"args": ARGS.as_dict(), "model": vae.state_dict(), "epoch": epoch}

    torch.save(save_dict, filename)

    return filename


if __name__ == "__main__":
    main_vae()
