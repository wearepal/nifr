from typing import Tuple

import torchvision
import wandb

from nosinn.utils import wandb_log

__all__ = ["get_data_dim", "log_images"]

def apply_gradients(gradients, parameters, detach=True) -> None:
    for grad, param in zip(gradients, parameters):
        if grad is not None and detach:
            grad = grad.detach()
        param.grad = grad

def get_data_dim(data_loader) -> Tuple[int, ...]:
    x, _, _ = next(iter(data_loader))
    x_dim = x.shape[1:]

    return tuple(x_dim)


def log_images(args, image_batch, name, step, nsamples=64, nrows=8, monochrome=False, prefix=None):
    """Make a grid of the given images, save them in a file and log them with W&B"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]
    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    wandb_log(
        args,
        {prefix + name: [wandb.Image(torchvision.transforms.functional.to_pil_image(shw))]},
        step=step,
    )
