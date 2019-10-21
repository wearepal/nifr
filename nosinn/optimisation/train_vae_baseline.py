"""Main training file"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset

from nosinn.data import DatasetTriplet
from nosinn.models.autoencoder import VAE
from nosinn.models.configs import conv_autoencoder, fc_autoencoder
from nosinn.models.configs.classifiers import linear_disciminator
from nosinn.models.factory import build_discriminator
from nosinn.utils import utils
from .loss import grad_reverse, PixelCrossEntropy
from .utils import get_data_dim, log_images
from .evaluation import fit_classifier

NDECS = 0
ARGS = None
LOGGER = None
INPUT_SHAPE = ()


def train(vae, discriminator, dataloader, epoch: int, recon_loss_fn) -> int:
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

        # enc_y = vae.encode(x)
        # recon = vae.decoder(enc_y)

        # elbo = recon_loss_fn(recon, x) / x.size(0)
        s_oh = F.one_hot(s, num_classes=ARGS.s_dim)
        enc_y, recon, elbo = vae.routine(x, recon_loss_fn=recon_loss_fn, s=s_oh)

        enc_y = grad_reverse(enc_y)
        disc_loss, disc_acc = discriminator.routine(enc_y, s)

        elbo *= ARGS.elbo_weight
        disc_loss *= ARGS.pred_s_weight

        loss = elbo + disc_loss

        vae.zero_grad()
        discriminator.zero_grad()
        loss.backward()
        vae.step()
        discriminator.step()

        total_loss_meter.update(loss.item())
        elbo_meter.update(elbo.item())
        disc_loss_meter.update(disc_loss.item())

        time_meter.update(time.time() - end)

        wandb.log({"Loss ELBO": elbo.item()}, step=itr)
        wandb.log({"Loss Adversarial": disc_loss.item()}, step=itr)
        end = time.time()

        if itr % 50 == 0:
            with torch.set_grad_enabled(False):
                z = vae.encode(x[:64], stochastic=False)

                recon_all = vae.decode(z, s=s_oh[:64])
                recon_y = vae.decode(z, s=torch.zeros_like(s_oh[:64]))
                recon_s = vae.decode(torch.zeros_like(z), s=s_oh[:64])

                log_images(recon_all, "original_x", step=itr)
                log_images(recon_all, "reconstruction_all", step=itr)
                log_images(recon_y, "reconstruction_y", step=itr)
                log_images(recon_s, "reconstruction_s", step=itr)

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

            s_oh = F.one_hot(s_val, num_classes=ARGS.s_dim)
            enc_y, recon, elbo = vae.routine(x_val, recon_loss_fn=recon_loss_fn, s=s_oh)

            enc_y = grad_reverse(enc_y)
            disc_loss, disc_acc = discriminator.routine(enc_y, s_val)

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
    all_xy = []
    all_s = []
    all_y = []

    with torch.set_grad_enabled(False):
        for x, s, y in data_loader:

            x = x.to(args.device)
            all_s.append(s)
            all_y.append(y)

            z = vae.encode(x, stochastic=False)

            xy = vae.decode(z, s=x.new_zeros(x.size(0), args.s_dim))

            if x.dim() > 2:
                xy = xy.clamp(min=0, max=1)

            all_xy.append(xy.detach().cpu())

    all_xy = torch.cat(all_xy, dim=0)
    all_s = torch.cat(all_s, dim=0)
    all_y = torch.cat(all_y, dim=0)

    encoded_dataset = TensorDataset(all_xy, all_s, all_y)

    return encoded_dataset


def evaluate(args, vae, train_loader, test_loader):

    train = encode_dataset(args, vae, train_loader)
    train = DataLoader(train, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    test = encode_dataset(args, vae, test_loader)
    test = DataLoader(test, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    fit_classifier(
        args, INPUT_SHAPE[0], train_data=train, train_on_recon=True, pred_s=False, test_data=test
    )


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

    # Model constructors and arguments
    disc_fn = linear_disciminator
    disc_kwargs = {
        "hidden_channels": ARGS.disc_channels,
        "num_blocks": ARGS.disc_depth,
        "use_bn": True,
    }

    if is_image_data:
        decoding_dim = INPUT_SHAPE[0] * 256 if args.recon_loss == "ce" else INPUT_SHAPE[0]
        encoder, decoder, enc_shape = conv_autoencoder(
            INPUT_SHAPE,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_dim,
            decoding_dim=decoding_dim,
            levels=ARGS.levels,
            vae=True,
            s_dim=ARGS.s_dim,
        )
    else:
        encoder, decoder, enc_shape = fc_autoencoder(
            INPUT_SHAPE,
            ARGS.init_channels,
            encoding_dim=ARGS.enc_dim,
            levels=ARGS.levels,
            vae=ARGS.vae,
            s_dim=ARGS.s_dim,
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
    dis_optimizer_kwargs = {"lr": args.disc_lr}

    discriminator = build_discriminator(
        args,
        enc_shape,
        frac_enc=1,
        model_fn=disc_fn,
        model_kwargs=disc_kwargs,
        optimizer_kwargs=dis_optimizer_kwargs,
    )
    discriminator.to(args.device)

    # Logging
    # wandb.set_model_graph(str(inn))
    LOGGER.info("Number of trainable parameters: {}", utils.count_parameters(vae))

    best_loss = float("inf")
    n_vals_without_improvement = 0

    # Train INN for N epochs
    for epoch in range(ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(vae, discriminator, train_loader, epoch, recon_loss_fn)

        if epoch % ARGS.val_freq == 0 and epoch != 0:
            val_loss = validate(vae, discriminator, val_loader, itr, recon_loss_fn)
            if args.super_val:
                evaluate(args, vae=vae, train_loader=val_loader, test_loader=test_loader)

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
    evaluate(args, vae=vae, train_loader=val_loader, test_loader=test_loader)
