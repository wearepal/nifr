import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import save_image

from finn.data import LdColorizer
from finn.data.dataset_wrappers import LdAugmentedDataset
from finn.models import build_conv_inn, build_discriminator, Masker, Classifier
from finn.models.configs import fc_net
from finn.models.inn import PartitionedInn
from finn.optimisation import parse_arguments, grad_reverse


def convnet(in_dim, target_dim):
    layers = []
    layers.extend([
        nn.Conv2d(in_dim, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    ])
    layers.extend([
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
    ])
    layers.append(nn.MaxPool2d(2, 2))

    layers.extend([
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    ])
    layers.extend([
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    ])
    layers.append(nn.MaxPool2d(2, 2))

    layers.extend([
        nn.Flatten(),
        nn.Linear(512, target_dim)
    ])

    return nn.Sequential(*layers)


args = parse_arguments()
args.dataset = "cmnist"
args.y_dim = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(device, *tensors):
    for tensor in tensors:
        yield tensor.to(device)


transforms = [
    ToTensor()
]
transforms = Compose(transforms)

mnist = MNIST(root="data", train=True, download=True, transform=transforms)
mnist, _ = random_split(mnist, lengths=(50000, 10000))
colorizer = LdColorizer(scale=0.0, black=True, background=False)
data = LdAugmentedDataset(mnist, ld_augmentations=colorizer, num_classes=10, li_augmentation=True)
data = DataLoader(data, batch_size=128, pin_memory=True, shuffle=True)

input_shape = (3, 28, 28)

args.depth = 12
args.coupling_dims = 512

args.factor_splits = {}
args.zs_frac = 0.02
args.lr = 3e-4
args.disc_lr = 1e-4
args.glow = False
args.batch_norm = False
args.weight_decay = 1e-5
args.idf = False

model = build_conv_inn(args, input_shape[0])
inn: PartitionedInn = PartitionedInn(args, input_shape=input_shape, model=model)
inn.to(device)

disc_kwargs = {}
disc_optimizer_args = {'lr': args.disc_lr}

args.disc_hidden_dims = [1024]

args.train_on_recon = False

use_conv_disc = True
model_fn = convnet if use_conv_disc else fc_net

discriminator: Classifier = build_discriminator(args,
                                                input_shape,
                                                frac_enc=1,
                                                model_fn=model_fn,
                                                model_kwargs=disc_kwargs,
                                                flatten=not use_conv_disc,
                                                optimizer_args=disc_optimizer_args)

discriminator.to(device)

enc_s_dim = 48

for epoch in range(1000):

    print(f"===> Epoch {epoch} of training")

    inn.model.train()
    discriminator.train()

    for i, (x, s, y) in enumerate(data):

        x, s, y = to_device(device, x, s, y)

        enc, nll = inn.routine(x)
        # # ===== Partition the encoding ======
        enc_y, enc_s = inn.split_encoding(enc)
        # enc_flat = enc.flatten(start_dim=1)
        # enc_y_dim = enc_flat.size(1) - enc_s_dim
        # enc_y, enc_s = enc_flat.split(split_size=(enc_y_dim, enc_s_dim), dim=1)

        enc_s_m = torch.cat([torch.zeros_like(enc_y), enc_s], dim=1)
        enc_y_m = torch.cat([grad_reverse(enc_y), torch.zeros_like(enc_s)], dim=1)

        # if use_conv_disc:
        #     enc_y_m = enc_y_m.view_as(enc)
        #     enc_s_m = enc_s_m.view_as(enc)

        # # ======== Loss computation =========
        pred_s_loss, acc = discriminator.routine(enc_y_m, s)

        inn.zero_grad()
        discriminator.zero_grad()

        loss = nll
        loss += pred_s_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=5)
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5)

        inn.optimizer.step()
        discriminator.step()

        # ============= Logging ==============
        if i % 50 == 0:
            print(f"NLL: {nll:.4f}")
            print(f"Adv Loss: {pred_s_loss:.4f}")

            with torch.set_grad_enabled(False):
                enc = inn(x)

                # enc_flat = enc.flatten(start_dim=1)
                # enc_y_dim = enc_flat.size(1) - enc_s_dim
                # enc_y, enc_s = enc_flat.split(split_size=(enc_y_dim, enc_s_dim), dim=1)
                #
                # enc_y_m = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1).view_as(enc)
                # enc_s_m = torch.cat([torch.zeros_like(enc_y), enc_s], dim=1).view_as(enc)
                # x_recon = inn.invert(enc)
                # xy = inn.invert(enc_y_m, discretize=False)
                # xs = inn.invert(enc_s_m, discretize=False)
                x_recon, xy, xs = inn.decode(enc, partials=True)

                save_image(x_recon[:64], filename="cmnist_recon_x.png")
                save_image(xy[:64], filename="cmnist_recon_xy.png")
                save_image(xs[:64], filename="cmnist_recon_xs.png")
