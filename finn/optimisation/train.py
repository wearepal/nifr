"""Main training file"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from finn.data import DatasetTriplet
from finn.models import AutoEncoder
from finn.models.configs import conv_autoencoder, fc_autoencoder
from finn.models.configs.classifiers import fc_net, linear_disciminator, mp_32x32_net
from finn.models.factory import build_fc_inn, build_conv_inn, build_discriminator
from finn.models.inn import PartitionedInn, PartitionedAeInn
from finn.utils import utils
from .loss import grad_reverse
from .utils import get_data_dim, log_images

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


def train(inn, discriminator, dataloader, epoch: int) -> int:
    inn.train()

    total_loss_meter = utils.AverageMeter()
    log_prob_meter = utils.AverageMeter()
    disc_loss_meter = utils.AverageMeter()

    time_meter = utils.AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time
    start_itr = start = epoch * len(dataloader)
    for itr, (x, s, y) in enumerate(dataloader, start=start_itr):

        x, s, y = to_device(x, s, y)

        enc, nll = inn.routine(x)

        enc_y, enc_s = inn.split_encoding(enc)

        if ARGS.train_on_recon:
            enc_y_m = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)
            enc_y = inn.invert(enc_y_m).clamp(min=0, max=1)

        enc_y = grad_reverse(enc_y)
        disc_loss, disc_acc = discriminator.routine(enc_y, s)

        nll *= ARGS.nll_weight

        pred_s_weight = 0 if itr < ARGS.warmup_steps else ARGS.pred_s_weight

        disc_loss *= pred_s_weight

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

        wandb.log({"Loss NLL": nll.item()}, step=itr)
        wandb.log({"Loss Adversarial": disc_loss.item()}, step=itr)
        end = time.time()

        if itr % 50 == 0:
            with torch.set_grad_enabled(False):
                log_images(x, "original_x", step=itr)

                z = inn(x[:64])

                recon_all, recon_y, recon_s = inn.decode(z, partials=True)

                log_images(recon_all, "reconstruction_all", step=itr)
                log_images(recon_y, "reconstruction_y", step=itr)
                log_images(recon_s, "reconstruction_s", step=itr)

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
    return itr


def validate(inn, discriminator, val_loader, itr):
    inn.eval()
    with torch.no_grad():
        loss_meter = utils.AverageMeter()
        for x_val, s_val, y_val in val_loader:

            x_val, s_val, y_val = to_device(x_val, s_val, y_val)

            enc, nll = inn.routine(x_val)
            enc_y, enc_s = inn.split_encoding(enc)

            if ARGS.train_on_recon:
                enc_y = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)
                enc_y = inn.invert(enc_y).clamp(min=0, max=1)

            enc_y = grad_reverse(enc_y)

            disc_loss, acc = discriminator.routine(enc_y, s_val)

            nll *= ARGS.nll_weight
            disc_loss *= ARGS.pred_s_weight

            loss = nll + disc_loss

            loss_meter.update(loss.item(), n=x_val.size(0))

    wandb.log({"Loss": loss_meter.avg}, step=itr)

    if ARGS.dataset == "cmnist":
        z = inn(x_val[:64])
        recon_all, recon_y, recon_s = inn.decode(z, partials=True)
        log_images(x_val, "original_x", prefix="test", step=itr)
        log_images(recon_all, "reconstruction_all", prefix="test", step=itr)
        log_images(recon_y, "reconstruction_y", prefix="test", step=itr)
        log_images(recon_s, "reconstruction_s", prefix="test", step=itr)
    else:
        z = inn(x_val[:1000])
        recon_all, recon_y, recon_s = inn.decode(z, partials=True)
        log_images(x_val, "original_x", prefix="test", step=itr)
        log_images(recon_y, "reconstruction_yn", prefix="test", step=itr)
        log_images(recon_s, "reconstruction_yn", prefix="test", step=itr)
        x_recon = inn(inn(x_val), reverse=True)
        x_diff = (x_recon - x_val).abs().mean().item()
        print(f"MAE of x and reconstructed x: {x_diff}")
        wandb.log({"reconstruction MAE": x_diff}, step=itr)

    return loss_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
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
    global ARGS, LOGGER
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
            disc_fn = mp_32x32_net
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
        if is_image_data:
            encoder, decoder, enc_shape = conv_autoencoder(
                input_shape, ARGS.ae_channels,
                encoded_dim=ARGS.ae_enc_dim,
                levels=ARGS.ae_levels,
                vae=ARGS.vae
            )
        else:
            encoder, decoder, enc_shape = fc_autoencoder(
                input_shape, ARGS.ae_channels,
                encoded_dim=ARGS.ae_enc_dim,
                levels=ARGS.ae_levels,
                vae=ARGS.vae
            )
        autoencoder = AutoEncoder(encoder=encoder, decoder=decoder)

        inn_kwargs["input_shape"] = enc_shape
        inn_kwargs["autoencoder"] = autoencoder
        inn_kwargs["model"] = inn_fn(args, inn_kwargs["input_shape"])

        inn = PartitionedAeInn(**inn_kwargs)
        inn.to(args.device)

        if ARGS.ae_loss == "l1":
            ae_loss_fn = nn.L1Loss()
        elif ARGS.ae_loss == "l2":
            ae_loss_fn = nn.MSELoss()
        elif ARGS.ae_loss == "huber":
            ae_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"{ARGS.ae_loss} is an invalid reconstruction loss")

        inn.fit_ae(train_loader, epochs=ARGS.ae_epochs, device=ARGS.device, loss_fn=ae_loss_fn)
    else:
        inn_kwargs["input_shape"] = input_shape
        inn_kwargs["model"] = inn_fn(args, input_shape)
        inn = PartitionedInn(**inn_kwargs)
        inn.to(args.device)

    print(f"zs dim: {inn.zs_dim}")
    print(f"zy dim: {inn.zy_dim}")
    # Initialise Discriminator
    dis_optimizer_kwargs = {"lr": args.disc_lr}
    discriminator = build_discriminator(
        args,
        inn_kwargs["input_shape"],
        frac_enc=1 - args.zs_frac,
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
        inn, discriminator = restore_model(ARGS.resume, inn, discriminator)
        metric_callback(ARGS, wandb, inn, discriminator, datasets, check_originals=False)
        return

    # Logging
    # wandb.set_model_graph(str(inn))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(inn))

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
                metric_callback(ARGS, model=inn, data=datasets, step=itr)

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
    metric_callback(ARGS, model=inn, data=datasets, save_to_csv=Path(ARGS.save), step=itr)
    save_model(save_dir=save_dir, inn=inn, discriminator=discriminator)
    inn.eval()


if __name__ == "__main__":
    print("This file cannot be run directly.")
