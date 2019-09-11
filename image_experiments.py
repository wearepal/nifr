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
    layers += [nn.MaxPool2d(2, 2)]

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
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend([
        nn.Flatten(),
        nn.Linear(512, target_dim)
    ])

    return nn.Sequential(*layers)


def depthwise_convnet(in_dim, target_dim):
    layers = []
    h_dim_1 = in_dim * 8
    layers.extend([
        nn.Conv2d(in_dim, h_dim_1, kernel_size=3, stride=1, padding=1, groups=in_dim),
        nn.GroupNorm(num_channels=h_dim_1, num_groups=in_dim),
        nn.ReLU(inplace=True)
    ])
    layers.extend([
        nn.Conv2d(h_dim_1, h_dim_1, kernel_size=3, stride=1, padding=1, groups=in_dim),
        nn.GroupNorm(num_channels=h_dim_1, num_groups=in_dim),
        nn.ReLU(inplace=True)
    ])
    layers += [nn.MaxPool2d(2, 2)]

    h_dim_2 = in_dim * 16
    layers.extend([
        nn.Conv2d(h_dim_1, h_dim_2, kernel_size=3, stride=1, padding=1, groups=in_dim),
        nn.GroupNorm(num_channels=h_dim_2, num_groups=in_dim),
        nn.ReLU(inplace=True)
    ])
    layers.extend([
        nn.Conv2d(h_dim_2, h_dim_2, kernel_size=3, stride=1, padding=1, groups=in_dim),
        nn.GroupNorm(num_channels=h_dim_2, num_groups=in_dim),
        nn.ReLU(inplace=True)
    ])
    layers += [nn.MaxPool2d(2, 2)]

    layers.extend(
        [nn.Conv2d(h_dim_2, out_channels=in_dim * target_dim, kernel_size=1,
                   padding=0, groups=in_dim)]
    )

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
data = LdAugmentedDataset(mnist, ld_augmentations=colorizer, n_labels=10, li_augmentation=True)
data = DataLoader(data, batch_size=256, pin_memory=True, shuffle=True)

input_shape = (3, 28, 28)

args.depth = 12
args.coupling_dims = 512

args.factor_splits = {4: 0.75, 7: 0.75, 10: 0.75}
args.zs_frac = 0.04
args.lr = 3e-4
args.disc_lr = 3e-4
args.glow = True
args.batch_norm = True
args.weight_decay = 0
args.idf = True

model = build_conv_inn(args, input_shape[0])
inn: PartitionedInn = PartitionedInn(args, input_shape=input_shape, model=model)
inn.to(device)

disc_kwargs = {}
disc_optimizer_args = {'lr': args.disc_lr}

args.disc_hidden_dims = [1024, 1024]

args.train_on_recon = False
discriminator: Classifier = build_discriminator(args,
                                                input_shape,
                                                frac_enc=1,
                                                model_fn=convnet,
                                                model_kwargs=disc_kwargs,
                                                flatten=False,
                                                optimizer_args=disc_optimizer_args)

discriminator.to(device)


for epoch in range(1000):

    print(f"===> Epoch {epoch} of training")

    inn.model.train()
    discriminator.train()

    for i, (x, s, y) in enumerate(data):

        x, s, y = to_device(device, x, s, y)

        enc, nll = inn.routine(x)

        enc_y, enc_s = inn.split_encoding(enc)
        # enc = torch.cat([torch.zeros_like(enc_y), enc_s], dim=1)
        enc = torch.cat([grad_reverse(enc_y), torch.zeros_like(enc_s)], dim=1)

        pred_s_loss, acc = discriminator.routine(enc, s)

        inn.optimizer.zero_grad()
        discriminator.zero_grad()

        loss = nll
        # loss += pred_s_loss

        loss.backward()
        inn.optimizer.step()
        discriminator.step()

        if i % 10 == 0:
            print(f"NLL: {nll:.4f}")
            print(f"Adv Loss: {pred_s_loss:.4f}")

            with torch.set_grad_enabled(False):
                enc = inn(x)

                x_recon, xy, xs = inn.decode(enc, partials=True)
                save_image(x_recon[:64], filename="cmnist_recon_x.png")
                save_image(xy[:64], filename="cmnist_recon_xy.png")
                save_image(xs[:64], filename="cmnist_recon_xs.png")
