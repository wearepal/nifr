import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomAffine, Compose
from torchvision.utils import save_image

from finn.data import LdColorizer
from finn.data.datasets import LdAugmentedDataset
from finn.models import build_conv_inn, build_discriminator
from finn.models.configs import fc_net, mp_28x28_net
from finn.models.inn import BipartiteInn
from finn.optimisation import parse_arguments, grad_reverse
from finn.optimisation.misc import contrastive_gradient_penalty

args = parse_arguments()
args.dataset = "cmnist"
args.y_dim = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(device, *tensors):
    for tensor in tensors:
        yield tensor.to(device)


transforms = [
    # RandomAffine(translate=(0.1, 0.1), degrees=(15, 15)),
    ToTensor()
]
transforms = Compose(transforms)

mnist = MNIST(root="data", train=True, download=True, transform=transforms)
mnist, _ = random_split(mnist, lengths=(50000, 10000))
colorizer = LdColorizer(scale=0.0, black=True, background=False)
data = LdAugmentedDataset(mnist, ld_augmentations=colorizer, n_labels=10, li_augmentation=True)
data = DataLoader(data, batch_size=256, pin_memory=True, shuffle=True)

input_shape = (3, 28, 28)

args.depth = 10
args.coupling_dims = 512

args.splits = {4: 0.75, 7: 0.75}
args.zs_frac = 0.02
args.lr = 3e-4
args.disc_lr = 3e-4
args.glow = True
args.batch_norm = False

model = build_conv_inn(args, input_shape[0])
inn: BipartiteInn = BipartiteInn(args, input_shape=input_shape, model=model)
inn.to(device)


disc_kwargs = {}
disc_optimizer_args = {'lr': args.disc_lr}

args.disc_hidden_dims = [1024, 1024]


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


args.train_on_recon = False
discriminator = build_discriminator(args,
                                    input_shape,
                                    frac_enc=args.zs_frac,
                                    model_fn=convnet,
                                    model_kwargs=disc_kwargs,
                                    flatten=False,
                                    optimizer_args=disc_optimizer_args)
# args.disc_hidden_dims = [100, 100]
# discriminator = build_discriminator(args,
#                                     input_shape,
#                                     frac_enc=args.zs_frac,
#                                     model_fn=fc_net,
#                                     model_kwargs=disc_kwargs,
#                                     flatten=True,
#                                     optimizer_args=disc_optimizer_args)
discriminator.to(device)

mod_steps = 1

for epoch in range(1000):

    print(f"===> Epoch {epoch} of training")

    inn.model.train()
    discriminator.train()

    for i, (x, s, y) in enumerate(data):
        x, s, y = to_device(device, x, s, y)

        enc, nll = inn.routine(x)

        enc_y, enc_s = inn.split_encoding(enc)

        loss_enc_y, acc = discriminator.routine(enc_s, s)

        inn.optimizer.zero_grad()
        discriminator.zero_grad()

        nll *= 1e-2
        loss = nll

        loss += loss_enc_y

        loss.backward()

        inn.optimizer.step()
        discriminator.step()

        if i % 10 == 0:
            print(f"NLL: {nll:.4f}")
            print(f"Adv Loss: {loss_enc_y:.4f}")

            with torch.set_grad_enabled(False):
                enc = inn.encode(x, partials=False)
                x_recon, xy, xs = inn.decode(enc, partials=True)
                save_image(x_recon[:64], filename="cmnist_recon_x.png")
                save_image(xy[:64], filename="cmnist_recon_xy.png")
                save_image(xs[:64], filename="cmnist_recon_xs.png")
