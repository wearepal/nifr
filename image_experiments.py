import numpy as np
import torch
import kornia
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import save_image

from finn.data import LdColorizer
from finn.data.dataset_wrappers import LdAugmentedDataset
from finn.models import build_conv_inn, build_discriminator, Masker, Classifier
from finn.models.configs import fc_net, mp_28x28_net
from finn.models.inn import PartitionedInn
from finn.optimisation import parse_arguments, grad_reverse
from finn.optimisation.evaluation import encode_dataset
from finn.optimisation.loss import contrastive_gradient_penalty
from finn.utils.optimizers import RAdam


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


args = parse_arguments()
args.dataset = "cmnist"
args.y_dim = 10

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
args.device = device


def to_device(device, *tensors):
    for tensor in tensors:
        yield tensor.to(device)


# ======= Data-loading =========
transforms = [
    ToTensor()
]
transforms = Compose(transforms)

train = CIFAR10(root="data", train=True, download=True, transform=transforms)
test = CIFAR10(root="data", train=False, download=True, transform=transforms)

pretrain_pcnt = 0.95
train_len = len(train)
pretrain_len = round(pretrain_pcnt * train_len)
pretrain, task_train = random_split(train, lengths=(pretrain_len, train_len - pretrain_len))
colorizer = LdColorizer(scale=0.0, black=True, background=False)

pretrain = DataLoader(pretrain, batch_size=64, pin_memory=True, shuffle=True)
task_train = DataLoader(task_train, batch_size=256, pin_memory=True, shuffle=True)
test = DataLoader(test, batch_size=256, pin_memory=True, shuffle=True)

# ======= Define models ===========

input_shape = (3, 28, 28)

args.depth = 10
args.coupling_dims = 64

args.factor_splits = {}  # {4: 0.75, 7: 0.75, 10: 0.75}
args.zs_frac = 0.02
args.lr = 3e-4
args.disc_lr = 3e-4
args.glow = True
args.batch_norm = True
args.weight_decay = 1e-5
args.idf = False


from torchvision.models import AlexNet, ResNet


rotnet: Classifier = Classifier(model=AlexNet(10), num_classes=4)
rotnet.to(device)


for epoch in range(50):

    print(f"Epoch {epoch} of self-supervised training")
    for i, (x, y) in enumerate(pretrain):
        x_0 = x
        y_0 = torch.full_like(y, 0)

        x_90 = kornia.rotate(x, angle=x.new_full((x.size(0),), 90.))
        y_90 = torch.full_like(y, 1)

        x_180 = kornia.rotate(x, angle=x.new_full((x.size(0),), 180.))
        y_180 = torch.full_like(y, 2)

        x_270 = kornia.rotate(x, angle=x.new_full((x.size(0),), 270))
        y_270 = torch.full_like(y, 3)

        x = torch.cat([x_0, x_90, x_180, x_270], dim=0)
        y = torch.cat([y_0, y_90, y_180, y_270], dim=0)
        indexes = torch.randperm(x.size(0))
        x = x[indexes]
        y = y[indexes]

        x, y = to_device(device, x, y)

        loss, acc = rotnet.routine(x, y)
        rotnet.zero_grad()
        loss.backward()
        rotnet.step()

        if i % 10 == 0:
            print(f"Accuracy: {acc:.4f}")

for epoch in range(50):

    rotnet.model.features.train()
    rotnet.model.classifier.train()

    print("===> Training")
    for x, y in task_train:
        x, y = to_device(device, x, y)

        loss, acc = rotnet.routine(x, y)
        rotnet.zero_grad()
        loss.backward()
        rotnet.step()

        # print(f"Train accuracy: {acc:.4f}")

    rotnet.model.features.eval()
    rotnet.model.classifier.eval()
    print("===> Testing")
    for x, y in test:
        x, y = to_device(device, x, y)

        loss, acc = rotnet.routine(x, y)

        print(f"Test accuracy: {acc:.4f}")

# clf.fit(train_data=task_train, test_data=test, epochs=50, device=device, verbose=True)

# model = build_conv_inn(args, input_shape[0])
# inn: PartitionedInn = PartitionedInn(args, input_shape=input_shape, model=model)
# inn.to(device)
#
# disc_kwargs = {}
# disc_optimizer_args = {'lr': args.disc_lr}
#
# args.disc_hidden_dims = [1024, 1024]
#
# args.train_on_recon = False
# discriminator: Classifier = build_discriminator(args,
#                                                 input_shape,
#                                                 frac_enc=1,
#                                                 model_fn=fc_net,
#                                                 model_kwargs=disc_kwargs,
#                                                 flatten=True,
#                                                 optimizer_args=disc_optimizer_args)
#
# discriminator.to(device)
#
# discriminator_adv: Classifier = build_discriminator(args,
#                                                     input_shape,
#                                                     frac_enc=1,
#                                                     model_fn=convnet,
#                                                     model_kwargs=disc_kwargs,
#                                                     flatten=False,
#                                                     optimizer_args=disc_optimizer_args)
#
# discriminator_adv.to(device)
#
# # ==================== Training ========================
# enc_s_dim = 32
#
# for epoch in range(1000):
#
#     print(f"===> Epoch {epoch} of training")
#
#     inn.model.train()
#     discriminator.train()
#     # ==================== Update INN ========================
#     for i, (x, s, y) in enumerate(pretrain):
#         x, s, y = to_device(device, x, s, y)
#
#         enc, nll = inn.routine(x)
#
#         enc_y, enc_s = inn.split_encoding(enc)
#
#         # enc_flat = enc.flatten(start_dim=1)
#         # enc_y_dim = enc_flat.size(1) - enc_s_dim
#         # enc_y, enc_s = enc_flat.split(split_size=(enc_y_dim, enc_s_dim), dim=1)
#
#         # enc_s_m = torch.cat([torch.zeros_like(enc_y), enc_s], dim=1)
#         enc_y_m = torch.cat([grad_reverse(enc_y), torch.zeros_like(enc_s)], dim=1)
#         # enc_y_m = enc_y_m.view_as(enc)
#
#         # pred_s_loss, acc = discriminator.routine(enc_s_m, s)
#         pred_s_loss = discriminator_adv.routine(enc_y_m, s)[0]
#         # gp = contrastive_gradient_penalty(network=discriminator_adv, input=enc_y_m)
#
#         inn.optimizer.zero_grad()
#         # discriminator.zero_grad()
#         discriminator_adv.zero_grad()
#
#         loss = nll
#         loss += pred_s_loss * 1e-1
#
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=5)
#
#         inn.optimizer.step()
#         # discriminator.step()
#         discriminator_adv.step()
#
#         # ==================== Log Images ========================
#         if i % 10 == 0:
#             print(f"NLL: {nll:.4f}")
#             print(f"Adv Loss: {pred_s_loss:.4f}")
#
#             with torch.set_grad_enabled(False):
#                 enc = inn(x)
#                 # enc_y_dim = enc.size(1) - enc_s_dim
#                 # enc_y, enc_s = enc.split(split_size=(enc_y_dim, enc_s_dim), dim=1)
#                 #
#                 # enc_y_m = torch.cat([enc_y, torch.zeros_like(enc_s)], dim=1).view_as(enc)
#                 # enc_s_m = torch.cat([torch.zeros_like(enc_y), enc_s], dim=1).view_as(enc)
#                 # x_recon = inn.invert(enc)
#                 # xy = inn.invert(enc_y_m, discretize=False)
#                 # xs = inn.invert(enc_s_m, discretize=False)
#
#                 x_recon, xy, xs = inn.decode(enc, partials=True)
#                 save_image(x_recon[:64], filename="cmnist_recon_x.png")
#                 save_image(xy[:64], filename="cmnist_recon_xy.png")
#                 save_image(xs[:64], filename="cmnist_recon_xs.png")

    # ======================== Validation ========================
    # if epoch % 10 == 0:
    #     inn.eval()
    #     print("===> Evaluating invariant representation")
    #     print("Encoding task-train dataset")
    #     task_train_encoded = encode_dataset(args,
    #                                         data=task_train.dataset,
    #                                         model=inn,
    #                                         recon=False)
    #     print("Encoding test Dataset")
    #     test_encoded = encode_dataset(args,
    #                                   data=test.dataset,
    #                                   model=inn,
    #                                   recon=False)
    #
    #     clf: Classifier = build_discriminator(args,
    #                                           input_shape,
    #                                           frac_enc=1,
    #                                           model_fn=nn.Linear,
    #                                           model_kwargs=disc_kwargs,
    #                                           flatten=True,
    #                                           optimizer_args={'lr': 1e-3, 'weight_decay': 1e-5})
    #     clf.to(args.device)
    #     clf.fit(train_data=task_train_encoded['zy'],
    #             test_data=test_encoded['zy'],
    #             epochs=5,
    #             device=args.device,
    #             batch_size=512,
    #             test_batch_size=1000)
