import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomAffine, Compose
from torchvision.utils import save_image

from finn.data import LdColorizer
from finn.data.datasets import LdAugmentedDataset
from finn.models import build_conv_inn, build_discriminator
from finn.models.inn import SplitInn
from finn.optimisation import parse_arguments, grad_reverse

args = parse_arguments()
args.dataset = "cmnist"
args.y_dim = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def to_device(device, *tensors):
    for tensor in tensors:
        yield tensor.to(device)

transforms = [
    RandomAffine(translate=(0.1, 0.1), degrees=(15, 15)),
    ToTensor()
]
transforms = Compose(transforms)

mnist = MNIST(root="data", train=True, download=True, transform=transforms)
mnist, _ = random_split(mnist, lengths=(1000, 59000))
colorizer = LdColorizer(scale=0.0, black=True, background=False)
data = LdAugmentedDataset(mnist, ld_augmentations=colorizer, n_labels=10, li_augmentation=True)
data = DataLoader(data, batch_size=100, pin_memory=True)

input_shape = (3, 28, 28)

args.depth = 6
args.splits = {2: 0.5, 3: 0.5, 4: 0.5}
args.zs_frac = 0.10
args.lr = 3e-4
args.disc_lr = 1e-3
args.log_prob_weight = 1e-3
args.learn_mask = True
args.batch_norm = False

inn = build_conv_inn(args, input_shape[0])
inn: SplitInn = SplitInn(args, input_shape=input_shape, model=inn)
inn.to(device)


def disc_net(input_dim, target_dim):
    layers = []

    layers += [nn.Conv2d(input_dim, 64, kernel_size=3, stride=1, padding=1)]
    layers += [nn.MaxPool2d(2, 2)]
    layers += [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)]
    layers += [nn.MaxPool2d(2, 2)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)]
    layers += [nn.MaxPool2d(2, 2)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)]

    layers += [nn.AdaptiveAvgPool2d(1)]
    layers += [nn.Flatten()]
    layers += [nn.Linear(512, 512)]
    layers += [nn.Linear(512, target_dim)]

    return nn.Sequential(*layers)


# disc_fn = mp_28x28_net
disc_fn = disc_net
disc_kwargs = {}
disc_optimizer_args = {'lr': args.disc_lr}

args.disc_hidden_dims = [512, 512]
discriminator = build_discriminator(args,
                                    input_shape,
                                    disc_net,
                                    disc_kwargs,
                                    flatten=not args.learn_mask,
                                    optimizer_args=disc_optimizer_args)
discriminator.to(device)

# Training routine
inn.model.eval()
discriminator.train()

print("===> Training Discriminator")
for epoch in range(0):

    print(f"===> Epoch {epoch}")
    avg_acc = 0
    for x, s, y in data:
        x, s, y = to_device(device, x, s, y)
        loss, acc = discriminator.routine(x, s)
        discriminator.zero_grad()
        loss.backward()
        discriminator.step()

        avg_acc += acc

    avg_acc /= len(data)
    print(avg_acc)


inn.model.train()
discriminator.train()

for epoch in range(10):

    print(f"===> Epoch {epoch} of INN training")
    saved = False
    for x, s, y in data:
        x, s, y = to_device(device, x, s, y)

        z, neg_log_prob = inn.routine(x)
        # z = inn.unsplit_encoding(zy, zs)

        # z = inn.encode(x, partials=False)
        # x_re, enc_y, enc_s = inn.decode(z.detach(), partials=True)
        # loss_enc_y, acc = discriminator.routine(grad_reverse(enc_y), s)
        # loss_enc_s, _ = discriminator.routine(enc_s, s)

        inn.optimizer.zero_grad()
        # discriminator.zero_grad()
        #
        loss = 1e-2 * neg_log_prob
        loss.backward()
        inn.optimizer.step()
        # discriminator.step()
        #
        # torch.nn.utils.clip_grad_norm_(
        #     inn.model.parameters(), max_norm=1)
        # torch.nn.utils.clip_grad_norm_(
        #     discriminator.parameters(), max_norm=1)

        if not saved:
            with torch.set_grad_enabled(False):
                x_recon, xy, xs = inn.decode(z, partials=True)
                save_image(x_recon, filename="test_recon_x.png")
                save_image(xy, filename="test_recon_xy.png")
                save_image(xs, filename="test_recon_xs.png")
                saved = True
