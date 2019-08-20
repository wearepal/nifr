import os
import shutil

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from finn.data.datasets import TripletDataset
from finn.data.misc import save_image_data_tuple


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


def encode_dataset(args, data, model, recon):

    root = os.path.join('data', 'encodings')
    if os.path.exists(root):
        shutil.rmtree(root)
    os.mkdir(root)

    encodings = ['z', 'zy', 'zs']
    if recon:
        encodings.extend(['x_recon', 'xy', 'xs'])

    filepaths = {key: os.path.join(root, key) for key in encodings}

    data = DataLoader(data, batch_size=1, pin_memory=True)

    with torch.set_grad_enabled(False):
        for i, (x, s, y) in enumerate(iter(data)):
            x = x.to(args.device)
            s = s.to(args.device)

            z, zy, zs = model.encode(x, partials=True)

            save_image_data_tuple(z, s, y,
                                  root=filepaths['z'],
                                  filename=f"image_{i}")
            save_image_data_tuple(zy, s, y,
                                  root=filepaths['zy'],
                                  filename=f"image_{i}")
            save_image_data_tuple(zs, s, y,
                                  root=os.path.join(root, 'zs'),
                                  filename=f"image_{i}")

            if recon:
                x_recon, xy, xs = model.decode(z, partials=True)

                save_image_data_tuple(x_recon, s, y,
                                      root=filepaths['x_recon'],
                                      filename=f"image_{i}")
                save_image_data_tuple(xy, s, y,
                                      root=filepaths['xy'],
                                      filename=f"image_{i}")
                save_image_data_tuple(xs, s, y,
                                      root=filepaths['xs'],
                                      filename=f"image_{i}")

    datasets = {
        key: TripletDataset(root)
        for key, root in filepaths.items()
    }

    return datasets
