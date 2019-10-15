import torchvision
from torch import autograd


def get_data_dim(data_loader):
    x, _, _ = next(iter(data_loader))
    x_dim = x.shape[1:]

    return x_dim


def log_images(experiment, image_batch, name, nsamples=64, nrows=8, monochrome=False, prefix=None):
    """Make a grid of the given images, save them in a file and log them with Comet"""
    prefix = "train_" if prefix is None else f"{prefix}_"
    images = image_batch[:nsamples]
    if monochrome:
        images = images.mean(dim=1, keepdim=True)
    # torchvision.utils.save_image(images, f'./experiments/finn/{prefix}{name}.png', nrow=nrows)
    shw = torchvision.utils.make_grid(images, nrow=nrows).clamp(0, 1).cpu()
    experiment.log_image(torchvision.transforms.functional.to_pil_image(shw), name=prefix + name)
