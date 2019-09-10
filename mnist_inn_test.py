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
args.coupling_depth = 3

args.splits = {4: 0.74, 7: 0.75}
args.zs_frac = 0.04
args.lr = 3e-4
args.disc_lr = 3e-4
args.train_on_recon = False
args.glow = True
args.batch_norm = True

model = build_conv_inn(args, input_shape[0])
inn: BipartiteInn = BipartiteInn(args, input_shape=input_shape, model=model)
inn.to(device)


def convnet(input_dim, target_dim):

    def _block(in_channels, out_channels):
        block = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]


disc_kwargs = {}
disc_optimizer_args = {'lr': args.disc_lr}

args.disc_hidden_dims = [1024, 1024]
discriminator = build_discriminator(args,
                                    input_shape,
                                    frac_enc=1 - args.zs_frac,
                                    model_fn=fc_net,
                                    model_kwargs=disc_kwargs,
                                    flatten=True,
                                    optimizer_args=disc_optimizer_args)
# discriminator = build_discriminator(args,
#                                     input_shape,
#                                     frac_enc=1 - args.zs_frac,
#                                     model_fn=mp_28x28_net,
#                                     model_kwargs=disc_kwargs,
#                                     flatten=False,
#                                     optimizer_args=disc_optimizer_args)
discriminator.to(device)

mod_steps = 1

for epoch in range(1000):

    print(f"===> Epoch {epoch} of training")

    inn.model.train()
    discriminator.train()
    train_inn = True

    for i, (x, s, y) in enumerate(data):
        x, s, y = to_device(device, x, s, y)

        if train_inn:
            enc, nll = inn.routine(x)

            enc_y, enc_s = inn.split_encoding(enc)
            # enc_y = inn.invert(torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1))
            #
            loss_enc_y, acc = discriminator.routine(grad_reverse(enc_y), s)
            # gp = contrastive_gradient_penalty(input=enc_y, network=discriminator)

            inn.optimizer.zero_grad()
            discriminator.zero_grad()

            nll *= 1e-2
            loss = nll
            # if epoch > 0:
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

            inn.model.eval()
            discriminator.train()

        # else:
        #     x, s, y = to_device(device, x, s, y)
        #
        #     enc = inn.encode(x)
        #     enc_y, enc_s = inn.split_encoding(enc)
        #
        #     loss_enc_y, acc = discriminator.routine(grad_reverse(enc_y), s)
        #
        #     discriminator.zero_grad()
        #
        #     loss = loss_enc_y
        #     loss.backward()
        #
        #     discriminator.step()

        # if i % mod_steps == 0:
        #     train_inn = not train_inn
        #     if train_inn:
        #         print("===> Training INN")
        #         inn.train()
        #         discriminator.eval()
        #     else:
        #         print("===> Training Discriminator")
        #         inn.eval()
        #         discriminator.train()
