# -*- coding: UTF-8 -*-
"""Main training file"""
import time
from logging import Logger
from pathlib import Path
from itertools import islice
from typing import Callable, Dict, List, Optional, Tuple, Union

import git
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import wandb
from nosinn.configs import NosinnArgs
from nosinn.data import DatasetTriplet, load_dataset
from nosinn.models import (
    VAE,
    AutoEncoder,
    PartitionedAeInn,
    PartitionedInn,
    build_conv_inn,
    build_discriminator,
    build_fc_inn,
)
from nosinn.models.configs import (
    ModelFn,
    conv_autoencoder,
    fc_autoencoder,
    fc_net,
    linear_disciminator,
    mp_32x32_net,
    mp_64x64_net,
)
from nosinn.utils import (
    AverageMeter,
    count_parameters,
    get_logger,
    random_seed,
    readable_duration,
    wandb_log,
    inf_generator,
)
from nosinn.configs import NosinnArgs
>>>>>>> First attempt at the new architecture

from .evaluation import log_metrics
from .loss import MixedLoss, PixelCrossEntropy, grad_reverse
from .utils import get_data_dim, log_images, restore_model, save_model

__all__ = ["main_nosinn"]

NDECS = 0
ARGS: NosinnArgs = None
LOGGER: Logger = None


def compute_loss(
    x_p: torch.Tensor,
    x_t: torch.Tensor,
    inn: Union[PartitionedInn, PartitionedAeInn],
    disc_ensemble: nn.ModuleList,
    itr: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logging_dict = {}

    # the following code is also in inn.routine() but we need to access ae_enc directly
    zero = x_p.new_zeros((x_p.size(0), 1))
    if isinstance(inn, PartitionedAeInn) and ARGS.train_on_recon:
        (enc, sum_ldj), ae_enc = inn.forward(x_p, logdet=zero, reverse=False, return_ae_enc=True)
    else:
        enc, sum_ldj = inn.forward(x_p, logdet=zero, reverse=False)
    nll = inn.nll(enc, sum_ldj)

    enc_y, enc_s = inn.split_encoding(enc)

    if ARGS.mask_disc or ARGS.train_on_recon:
        enc_y = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1)

    recon_loss = x_p.new_zeros(())
    if ARGS.train_on_recon:

        if ARGS.recon_detach:
            enc_y = enc_y.detach()

        if isinstance(inn, PartitionedAeInn):
            enc_y, ae_enc_y = inn.forward(enc_y_m, reverse=True, return_ae_enc=True)
            recon, recon_target = ae_enc_y, ae_enc
            # logging_dict.update({
            #     f"train_xi_min": ae_enc_y.detach().cpu().numpy().min(),
            #     f"train_xi_max": ae_enc_y.detach().cpu().numpy().max(),
            # })
            enc_y_task_train = inn.autoencoder.decode(inn.autoencoder.encode(x_t))
        else:
            enc_y = inn.forward(enc_y_m, reverse=True)
            recon, recon_target = enc_y, x_p
            enc_y_task_train = x_t

        if ARGS.recon_stability_weight > 0:
            recon_loss = ARGS.recon_stability_weight * F.mse_loss(recon, recon_target)
    else:
        enc_task_train = inn.forward(x_t, reverse=False)
        enc_y_task_train, _ = inn.split_encoding(enc_task_train)
        if ARGS.recon_detach:
            enc_y_task_train = enc_y_task_train.detach()

    enc_y = grad_reverse(enc_y)
    disc_loss = x.new_zeros(())
    for disc in disc_ensemble:
        disc_loss_fake, _ = disc.routine(enc_y, x_p.new_zeros((x_p.size(0),)))
        disc_loss_task_train, _ = disc.routine(enc_y_task_train, x_t.new_ones((x_t.size(0),)))
        disc_loss += disc_loss_fake + disc_loss_task_train

    if itr < ARGS.warmup_steps:
        pred_s_weight = ARGS.pred_s_weight * np.exp(-7 + 7 * itr / ARGS.warmup_steps)
    else:
        pred_s_weight = ARGS.pred_s_weight

    nll *= ARGS.nll_weight
    disc_loss *= pred_s_weight
    recon_loss *= ARGS.recon_stability_weight

    loss = nll + disc_loss + recon_loss

    logging_dict.update(
        {
            "Loss NLL": nll.item(),
            "Loss Adversarial": disc_loss.item(),
            "Accuracy Discriminators": disc_acc,
            "Loss Recon": recon_loss.item(),
            "Loss Validation": (nll - disc_loss + recon_loss).item(),
        }
    )
    return loss, logging_dict


def train(
    inn: Union[PartitionedInn, PartitionedAeInn],
    disc_ensemble: nn.ModuleList,
    pretrain_data: DataLoader,
    task_train_data: DataLoader,
    epoch: int,
) -> int:
    inn.train()
    disc_ensemble.train()

    total_loss_meter = AverageMeter()
    loss_meters: Optional[Dict[str, AverageMeter]] = None

    time_meter = AverageMeter()
    start_epoch_time = time.time()
    end = start_epoch_time

    epoch_len = max(len(pretrain_data), len(task_train_data))
    itr = start_itr = (epoch - 1) * epoch_len
    data_iterator = islice(
        zip(inf_generator(pretrain_data), inf_generator(task_train_data)), epoch_len
    )
    for itr, ((x_p, _, _), (x_t, _, _)) in enumerate(data_iterator, start=start_itr):

        x_b, x_u = to_device(x_p, x_t)

        loss, logging_dict = compute_loss(x, s, inn, disc_ensemble, itr)
        loss, logging_dict = compute_loss(x_b, x_u, inn, disc_ensemble, itr)

        inn.zero_grad()

        for disc in disc_ensemble:
            disc.zero_grad()

        loss.backward()
        inn.step()

        if itr % ARGS.skip_disc_steps == 0:
            for disc in disc_ensemble:
                disc.step()

        # Log losses
        total_loss_meter.update(loss.item())
        if loss_meters is None:
            loss_meters = {name: AverageMeter() for name in logging_dict}
        for name, value in logging_dict.items():
            loss_meters[name].update(value)

        time_for_batch = time.time() - end
        time_meter.update(time_for_batch)

        wandb_log(ARGS, logging_dict, step=itr)
        end = time.time()

        # Log images
        if itr % ARGS.log_freq == 0:
            with torch.set_grad_enabled(False):
                log_recons(inn, x_b, itr)
        if itr == 0 and ARGS.jit:
            LOGGER.info(
                "JIT compilation (for training) completed in {}", readable_duration(time_for_batch)
            )

    time_for_epoch = time.time() - start_epoch_time
    assert loss_meters is not None
    log_string = " | ".join(f"{name}: {meter.avg:.5g}" for name, meter in loss_meters.items())
    LOGGER.info(
        "[TRN] Epoch {:04d} | Duration: {} | Batches/s: {:.4g} | {} ({:.5g})",
        epoch,
        readable_duration(time_for_epoch),
        1 / time_meter.avg,
        log_string,
        total_loss_meter.avg,
    )
    return itr


def validate(inn: PartitionedInn, disc_ensemble: nn.ModuleList, val_loader, itr: int):
    inn.eval()
    disc_ensemble.eval()

    with torch.set_grad_enabled(False):
        loss_meter = AverageMeter()
        for val_itr, (x_val, _, _) in enumerate(val_loader):

            x_val = to_device(x_val)

            _, logging_dict = compute_loss(x_val, x_val, inn, disc_ensemble, itr)

            loss_meter.update(logging_dict["Loss Validation"], n=x_val.size(0))

            if val_itr == 0:
                if ARGS.dataset in ("cmnist", "celeba", "ssrp", "genfaces"):
                    log_recons(inn, x_val, itr, prefix="test")
                else:
                    z = inn(x_val[:1000])
                    _, recon_y, recon_s, recon_random = inn.decode(z, partials=True)
                    log_images(ARGS, x_val, "original_x", prefix="test", step=itr)
                    log_images(ARGS, recon_y, "reconstruction_y", prefix="test", step=itr)
                    log_images(ARGS, recon_s, "reconstruction_y", prefix="test", step=itr)
                    log_images(ARGS, recon_random, "reconstruction_random", prefix="test", step=itr)
                    x_recon = inn(inn(x_val), reverse=True)
                    x_diff = (x_recon - x_val).abs().mean().item()
                    print(f"MAE of x and reconstructed x: {x_diff}")
                    wandb_log(ARGS, {"reconstruction MAE": x_diff}, step=itr)

        wandb_log(ARGS, {"Loss": loss_meter.avg}, step=itr)

    return loss_meter.avg


def to_device(*tensors):
    """Place tensors on the correct device and set type to float32"""
    moved = [tensor.to(ARGS.device, non_blocking=True) for tensor in tensors]
    if len(moved) == 1:
        return moved[0]
    return tuple(moved)


def log_recons(inn: PartitionedInn, x, itr: int, prefix: Optional[str] = None) -> None:
    z = inn(x[:64])
    recon_all, recon_y, recon_s, recon_random = inn.decode(z, partials=True)
    log_images(ARGS, x, "original_x", prefix=prefix, step=itr)
    log_images(ARGS, recon_all, "reconstruction_all", prefix=prefix, step=itr)
    log_images(ARGS, recon_y, "reconstruction_y", prefix=prefix, step=itr)
    log_images(ARGS, recon_s, "reconstruction_s", prefix=prefix, step=itr)
    log_images(ARGS, recon_random, "reconstruction_random", prefix=prefix, step=itr)


def main_nosinn(raw_args: Optional[List[str]] = None) -> Union[PartitionedInn, PartitionedAeInn]:
    """Main function

    Args:
        args: commandline arguments
        datasets: a Dataset object

    Returns:
        the trained model
    """
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    args = NosinnArgs(explicit_bool=True, underscores_to_dashes=True)
    args.parse_args(raw_args)
    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    random_seed(args.seed, use_gpu)
    datasets: DatasetTriplet = load_dataset(args)
    # ==== initialize globals ====
    global ARGS, LOGGER
    ARGS = args
    args_dict = args.as_dict()

    if ARGS.use_wandb:
        wandb.init(project="nosinn", config=args_dict)

    save_dir = Path(ARGS.save_dir) / str(time.time())
    save_dir.mkdir(parents=True, exist_ok=True)

    LOGGER = get_logger(logpath=save_dir / "logs", filepath=Path(__file__).resolve())
    LOGGER.info("Namespace(" + ", ".join(f"{k}={args_dict[k]}" for k in sorted(args_dict)) + ")")
    LOGGER.info("Save directory: {}", save_dir.resolve())
    # ==== check GPU ====
    ARGS.device = torch.device(
        f"cuda:{ARGS.gpu}" if (torch.cuda.is_available() and not ARGS.gpu < 0) else "cpu"
    )
    LOGGER.info("{} GPUs available. Using device '{}'", torch.cuda.device_count(), ARGS.device)
    if ARGS.jit:
        LOGGER.info("JIT enabled")

    # ==== construct dataset ====
    LOGGER.info(
        "Size of pretrain: {}, task_train: {}, task: {}",
        len(datasets.pretrain),
        len(datasets.task_train),
        len(datasets.task),
    )
    ARGS.test_batch_size = ARGS.test_batch_size if ARGS.test_batch_size else ARGS.batch_size
    pretrain_loader = DataLoader(
        datasets.pretrain,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    task_train_loader = DataLoader(
        datasets.task_train,
        shuffle=True,
        batch_size=ARGS.batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.task_train,
        shuffle=False,
        batch_size=ARGS.test_batch_size,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )

    # ==== construct networks ====
    input_shape = get_data_dim(pretrain_loader)
    is_image_data = len(input_shape) > 2

    optimizer_args = {"lr": args.lr, "weight_decay": args.weight_decay}
    feature_groups = None
    if hasattr(datasets.pretrain, "feature_groups"):
        feature_groups = datasets.pretrain.feature_groups

    # =================================== discriminator settings ==================================
    disc_fn: ModelFn
    if is_image_data:
        inn_fn = build_conv_inn
        if args.train_on_recon:  # or (args.oxbow_net and not args.autoencode):
            if args.dataset == "cmnist":
                disc_fn = mp_32x32_net
            else:
                disc_fn = mp_64x64_net
            disc_kwargs = {}
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

    # ======================================== INN settings =======================================
    inn_kwargs = {"args": args, "optimizer_args": optimizer_args, "feature_groups": feature_groups}

    # ======================================= initialise INN ======================================
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
            ae_loss_fn: Callable[[Tensor, Tensor], Tensor]
            if ARGS.ae_loss == "l1":
                ae_loss_fn = nn.L1Loss(reduction="sum")
            elif ARGS.ae_loss == "l2":
                ae_loss_fn = nn.MSELoss(reduction="sum")
            elif ARGS.ae_loss == "huber":
                ae_loss_fn = lambda x, y: F.smooth_l1_loss(x * 10, y * 10, reduction="sum")
            elif ARGS.ae_loss == "ce":
                ae_loss_fn = PixelCrossEntropy(reduction="sum")
            elif ARGS.ae_loss == "mixed":
                assert feature_groups is not None, "can only do multi loss with feature groups"
                ae_loss_fn = MixedLoss(feature_groups, reduction="sum")
            else:
                raise ValueError(f"{ARGS.ae_loss} is an invalid reconstruction loss")

            inn.fit_ae(
                pretrain_loader, epochs=ARGS.ae_epochs, device=ARGS.device, loss_fn=ae_loss_fn
            )
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

    if ARGS.train_on_recon:
        disc_input_shape = input_shape
    else:
        if ARGS.mask_disc:
            disc_input_shape = (inn.zy_dim + inn.zs_dim,)
        else:
            disc_input_shape = (inn.zy_dim,)

    print(f"zs dim: {inn.zs_dim}")
    print(f"zy dim: {inn.zy_dim}")

    # Initialise Discriminators
    disc_optimizer_kwargs = {"lr": ARGS.disc_lr}
    disc_ensemble = []

    for k in range(ARGS.num_discs):
        disc = build_discriminator(
            input_shape=disc_input_shape,
            target_dim=1,  # datasets.s_dim,  # this is now always 2
            train_on_recon=ARGS.train_on_recon,
            frac_enc=enc_shape,
            model_fn=disc_fn,
            model_kwargs=disc_kwargs,
            optimizer_kwargs=disc_optimizer_kwargs,
        )
        disc_ensemble.append(disc)
    disc_ensemble = nn.ModuleList(disc_ensemble)
    disc_ensemble.to(args.device)

    if ARGS.spectral_norm:

        def spectral_norm(m):
            if hasattr(m, "weight"):
                return torch.nn.utils.spectral_norm(m)

        inn.apply(spectral_norm)
        for disc in disc_ensemble:
            disc.apply(spectral_norm)

    # Resume from checkpoint
    if ARGS.resume is not None:
        LOGGER.info("Restoring model from checkpoint")
        filename = Path(ARGS.resume)
        inn, _ = restore_model(args, filename, inn=inn, disc_ensemble=disc_ensemble)
        if ARGS.evaluate:
            log_metrics(
                ARGS,
                model=inn,
                data=datasets,
                save_to_csv=Path(ARGS.save_dir),
                step=0,
                feat_attr=ARGS.feat_attr,
            )
            return inn

    # Logging
    # wandb.set_model_graph(str(inn))
    LOGGER.info("Number of trainable parameters: {}", count_parameters(inn))

    best_loss = float("inf")
    n_vals_without_improvement = 0
    super_val_freq = ARGS.super_val_freq or ARGS.val_freq

    itr = 0
    start_epoch = 1  # start at 1 so that the val_freq works correctly
    # Train INN for N epochs
    for epoch in range(start_epoch, start_epoch + ARGS.epochs):
        if n_vals_without_improvement > ARGS.early_stopping > 0:
            break

        itr = train(inn, disc_ensemble, pretrain_loader, task_train_loader, epoch=epoch)

        if epoch % ARGS.val_freq == 0:
            val_loss = validate(inn, disc_ensemble, val_loader, itr)

            if val_loss < best_loss:
                best_loss = val_loss
                save_model(args, save_dir, inn, disc_ensemble, epoch=epoch, sha=sha, best=True)
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
        if ARGS.super_val and epoch % super_val_freq == 0:
            log_metrics(ARGS, model=inn, data=datasets, step=itr)
            save_model(args, save_dir, model=inn, disc_ensemble=disc_ensemble, epoch=epoch, sha=sha)

        for k, disc in enumerate(disc_ensemble):
            if np.random.uniform() < args.disc_reset_prob:
                LOGGER.info("Reinitializing discriminator {}", k)
                disc.reset_parameters()

    LOGGER.info("Training has finished.")
    path = save_model(args, save_dir, model=inn, disc_ensemble=disc_ensemble, epoch=epoch, sha=sha)
    inn, disc_ensemble = restore_model(args, path, inn=inn, disc_ensemble=disc_ensemble)
    log_metrics(
        ARGS, model=inn, data=datasets, save_to_csv=Path(ARGS.save_dir), step=itr, feat_attr=True
    )
    return inn


if __name__ == "__main__":
    main_nosinn()
