import argparse

import torch
from torchvision import datasets, transforms
import numpy as np
from skimage import color


class MnistColorizer:

    def __init__(self, train, scale, binarize=False, color_space='rgb',
                 background=True, black=True, seed=42):
        self.train = train
        self.scale = scale
        self.binarize = binarize
        self.background = background
        self.black = black

        # create a local random state that won't affect the global random state of the training
        self.random_state = np.random.RandomState(seed)

        self.color_space = color_space
        if color_space == 'rgb':
            colors = [(0, 255, 255),
                      (0, 0, 255),  # blue
                      (255, 0, 255),
                      (0, 128, 0),
                      (0, 255, 0),  # green
                      (128, 0, 0),
                      (0, 0, 128),
                      (128, 0, 128),
                      (255, 0, 0),  # red
                      (255, 255, 0)]  # yellow

            self.palette = [np.divide(color, 255) for color in colors]
            self.scale *= np.eye(3)
        else:
            colors = [0, 35, 60, 78, 125, 165, 190, 235, 270, 300]
            self.palette = np.divide(colors, 360)

    @staticmethod
    def _hsv_colorize(value, hue, saturation_value=1., background=True, black=True):
        """ Add color of the given hue to an RGB image.

        By default, set the saturation to 1 so that the colors pop!
        """
        hw = value.shape[1:]

        # colored background
        if background:
            if black:
                # black digits
                value = 1 - value
                saturation = np.ones_like(value) * saturation_value
            else:
                # white digits
                saturation = 1 - value
                value = np.ones_like(value)
        # colored digits
        else:
            if black:
                # black background
                saturation = np.ones_like(value) * saturation_value
            else:
                # white background
                saturation = value
                value = np.ones_like(value)
        hue = np.tile(hue[..., None, None], (1, *hw))
        hsv = np.stack([hue, saturation, value], axis=-1).reshape(-1, hw[0], 3)

        rgb = color.hsv2rgb(hsv).reshape(-1, *hw, 3)
        rgb = torch.Tensor(rgb).permute(0, 3, 1, 2).contiguous()

        return rgb

    def _sample_color(self, mean_color_values):
        if self.color_space == 'hsv':
            return np.clip(self.random_state.normal(mean_color_values, self.scale), 0, 1)
        else:
            return np.clip(
                self.random_state.multivariate_normal(mean_color_values, self.scale), 0, 1)

    def _transform(self, img, target, train, background=True, black=True):
        if train:
            target = target.numpy()
        else:
            target = self.random_state.randint(0, 10, target.shape)

        mean_values = []
        colors_per_sample = []
        for label in target:
            mean_value = self.palette[label]
            mean_values.append(mean_value)
            colors_per_sample.append(self._sample_color(mean_value))

        if self.binarize:
            img = (img > 0).float()

        if self.color_space == 'hsv':
            img = img.numpy().squeeze()
            if len(img.shape) == 2:
                img = img[None, ...]  # re-add the batch dimension in case it was removed
            colorized_data = self._hsv_colorize(img, np.array(colors_per_sample),
                                                background=background, black=black)
        else:
            if background:
                if black:
                    # colorful background, black digits
                    colorized_data = (1 - img) * torch.Tensor(
                        colors_per_sample).unsqueeze(-1).unsqueeze(-1)
                else:
                    # colorful background, white digits
                    colorized_data = torch.clamp(
                        img + torch.Tensor(colors_per_sample).unsqueeze(-1).unsqueeze(-1), 0, 1)
            else:
                if black:
                    # black background, colorful digits
                    colorized_data = img * torch.Tensor(
                        colors_per_sample).unsqueeze(-1).unsqueeze(-1)
                else:
                    # white background, colorful digits
                    colorized_data = 1 - img * (
                        1 - torch.Tensor(colors_per_sample).unsqueeze(-1).unsqueeze(-1))

        # return colorized_data, torch.Tensor(mean_values)
        return colorized_data, torch.LongTensor(target)

    def __call__(self, img, target):
        return self._transform(img, target, self.train, self.background, self.black)


class ColorizedMNIST(datasets.MNIST):

    def __init__(self, root, train, transform, scale, background=True, black=True, cspace='rgb',
                 download=True):
        super(ColorizedMNIST, self).__init__(root, train=train,
                                             download=download,
                                             transform=transform)
        self.colorizer = MnistColorizer(train=train, scale=scale, background=background,
                                        black=black, color_space=cspace)
        self.palette = self.colorizer.palette

    def swap_train_test_colorization(self):
        self.colorizer.train = not self.colorizer.train

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        data, color = self.colorizer(data, target.view(1))
        return data.squeeze(0), color.squeeze(), target


def test():
    from torch.utils.data import DataLoader

    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--batch-size', type=int, default=64)
        parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion-mnist'],
                            default='mnist')
        parser.add_argument('--scale', type=float, default=0.02)
        parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
        parser.add_argument('-bg', '--background', type=eval, default=True, choices=[True, False])
        parser.add_argument('--black', type=eval, default=False, choices=[True, False])

        return parser.parse_args()

    args = parse_arguments()

    if args.dataset == 'mnist':
        train_dataset = ColorizedMNIST('./data/mnist', train=True,
                                       download=True, transform=transforms.ToTensor(),
                                       scale=args.scale,
                                       cspace=args.cspace,
                                       background=args.background,
                                       black=args.black)

        dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    else:
        train_dataset = datasets.FashionMNIST('./data/fashion-mnist', train=True,
                                              download=True, transform=transforms.ToTensor())

        dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    for data, color, labels in dataloader:
        data = data
        # save_image(data[:64], './colorized.png', nrow=8)
        break


if __name__ == '__main__':
    test()
