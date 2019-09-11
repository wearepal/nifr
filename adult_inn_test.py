import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomAffine, Compose
from torchvision.utils import save_image

from finn.data import LdColorizer, load_dataset
from finn.data.datasets import LdAugmentedDataset
from finn.models import build_conv_inn, build_discriminator, build_fc_inn
from finn.models.configs import fc_net
from finn.models.inn import PartitionedInn
from finn.optimisation import parse_arguments, grad_reverse
from finn.optimisation.loss import contrastive_gradient_penalty
from finn.utils import to_discrete
from finn.utils.plotting import plot_contrastive

args = parse_arguments()
args.dataset = "adult"
args.y_dim = 10
args.cuda = False

device = torch.device("cuda") if torch.cuda.is_available() and args.cuda\
    else torch.device("cpu")


def to_device(device, *tensors):
    for tensor in tensors:
        yield tensor.to(device)


datasets = load_dataset(args)

input_shape = next(iter(datasets.pretrain))[0].shape

args.depth = 10
args.zs_frac = 0.1
args.lr = 3e-4
args.disc_lr = 3e-4
args.train_on_recon = False
args.batch_norm = False
args.batch_size = 256

inn = build_fc_inn(args, input_dim=input_shape[0])
inn: PartitionedInn = PartitionedInn(args, input_shape=input_shape, model=inn,
                                     feature_groups=datasets.pretrain.feature_groups)
inn.to(device)


disc_kwargs = {}
disc_optimizer_args = {'lr': args.disc_lr}

args.disc_hidden_dims = [100, 100]
discriminator = build_discriminator(args,
                                    (inn.zy_dim,),
                                    fc_net,
                                    disc_kwargs,
                                    flatten=True,
                                    optimizer_args=disc_optimizer_args)
discriminator.to(device)

inn.model.train()
discriminator.train()

pretrain_data = datasets.pretrain
features = datasets.pretrain.disc_features + datasets.pretrain.cont_features
pretrain_data = DataLoader(pretrain_data,
                           batch_size=args.batch_size,
                           pin_memory=True,
                           shuffle=True)

for epoch in range(100):

    print(f"===> Epoch {epoch} of INN training")
    saved = False

    for x, s, y in pretrain_data:
        x, s, y = to_device(device, x, s, y)

        z, neg_log_prob = inn.routine(x)
        zy, zs = inn.split_encoding(z)
        gr_zy = grad_reverse(zy)
        loss_enc_y, acc = discriminator.routine(gr_zy, s)

        inn.optimizer.zero_grad()
        discriminator.zero_grad()
        loss = 1e-3 * neg_log_prob + loss_enc_y
        loss += contrastive_gradient_penalty(discriminator, gr_zy)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=5)

        inn.optimizer.step()
        discriminator.step()

        if not saved:
            with torch.set_grad_enabled(False):
                x_recon, xy, xs = inn.decode(z, partials=True, discretize=False)

                plot_contrastive(original=x[:50],
                                 recon=xy[:50],
                                 columns=features,
                                 filename="adult_xy-x")
                plot_contrastive(original=x[:50],
                                 recon=xs[:50],
                                 columns=features,
                                 filename="adult_xs-x")
                plot_contrastive(original=x[:50],
                                 recon=x_recon[:50],
                                 columns=features,
                                 filename="adult_x_recon-x")
                saved = True

                recon_error = F.l1_loss(x_recon, x).item()
                print(f"Recon error: {recon_error}")
