import argparse
import os
from pathlib import Path

import torch
from torchvision.transforms import transforms

from data.colorized_mnist import ColorizedMNIST

import matplotlib.pyplot as plt


def load_cmnist_data(args):
    train_data = ColorizedMNIST('./data', download=True, train=True, scale=args.scale,
                                transform=transforms.ToTensor(),
                                cspace=args.cspace, background=args.background, black=args.black)
    test_data = ColorizedMNIST('./data', download=True, train=False, scale=args.scale,
                               transform=transforms.ToTensor(),
                               cspace=args.cspace, background=args.background, black=args.black)

    return train_data, test_data


def make_cmnist_dataset(args, root='../data'):
    # Get the mnist dataset
    # Colorize the dataset

    train_data, test_data = load_cmnist_data(args)

    # Save the dataset with class labels and sensnitive labels
    path = Path(root) / "cmnist"

    if not os.path.exists(path/"train"):
        os.makedirs(path/"train")
    if not os.path.exists(path / "test"):
        os.makedirs(path/"test")

    for i, set in enumerate(train_data):
        with open(os.path.join(path/"train", str(i)), 'wb') as f:
            torch.save(set, f)

    for i, set in enumerate(test_data):
        with open(os.path.join(path/"test", str(i)), 'wb') as f:
            torch.save(set, f)


def parse_args_preprocess():
    parser = argparse.ArgumentParser()
    # Colored MNIST settings
    parser.add_argument('--scale', type=float, default=0.02)
    parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
    parser.add_argument('-bg', '--background', type=eval, default=True, choices=[True, False])
    parser.add_argument('--black', type=eval, default=False, choices=[True, False])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args_preprocess()
    make_cmnist_dataset(args)
