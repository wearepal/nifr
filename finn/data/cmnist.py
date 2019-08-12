import torch
from torchvision import datasets, transforms
import numpy as np
from skimage import color as color_conversion


class ColouredMNIST(datasets.MNIST):
    def __init__(
        self,
        root,
        use_train_split,
        assign_color_randomly,
        transform,
        scale,
        background=True,
        black=True,
        cspace='rgb',
        binarize=True,
        download=True,
    ):
        """PyTorch dataset for colorized MNIST

        Args:
            root: A string with the directory where to store the MNIST files
            use_train_split (bool): if True, the official "train" split of MNIST is used
            assign_color_randomly (bool): if True, colors are not assigned according to digit shape
            transform: should always be "transform.ToTensor()"
            scale (float): the scale of the Normal distribution that samples the color
            background (bool): if True, the background is colored instead of the digits themselves
            black (bool): if True, the non-colored parts of the images are black instead of white
            cspace: either "rgb" to use the RGB color space or "hsv" to use HSV
            binarize (bool): if True, make pixels either completely black or white before coloring
            download (bool): if True, download the data if it hasn't been downloaded
        """
        super().__init__(
            root, train=use_train_split, download=download, transform=transform
        )
        self.colorizer = _MnistColorizer(
            assign_color_randomly=assign_color_randomly,
            scale=scale,
            background=background,
            black=black,
            color_space=cspace,
            binarize=binarize,
        )
        self.palette = self.colorizer.palette

    def __getitem__(self, idx):
        data, target = super().__getitem__(idx)

        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)

        data, color = self.colorizer(data, target.view(1))
        return data.squeeze(0), color.squeeze(), target


class _MnistColorizer:
    """Class that takes care of coloring"""

    def __init__(
        self,
        assign_color_randomly,
        scale,
        binarize=False,
        color_space='rgb',
        background=True,
        black=True,
        seed=42,
    ):
        self.assign_color_randomly = assign_color_randomly
        self.scale = scale
        self.binarize = binarize
        self.background = background
        self.black = black

        # create a local random state that won't affect the global random state of the training
        self.random_state = np.random.RandomState(seed)

        self.color_space = color_space
        if color_space == 'rgb':
            # this is the color palette from Kim et al., "Learning not to learn"
            colors = [
                (220, 20, 60),  # crimson
                (0, 128, 128),  # teal
                (253, 233, 16),  # lemon
                (0, 149, 182),  # bondi blue
                (237, 145, 33),  # carrot orange
                (145, 30, 188),  # strong violet
                (70, 240, 240),  # cyan
                (250, 197, 187),  # your pink
                (210, 245, 60),  # lime
                (128, 0, 0),  # maroon
            ]

            self.palette = [np.divide(color, 255) for color in colors]
            self.scale *= np.eye(3)
        elif color_space == 'hsv':
            colors = [0, 35, 60, 78, 125, 165, 190, 235, 270, 300]
            self.palette = np.divide(colors, 360)
        else:
            raise RuntimeError("Unkown color space.")

    @staticmethod
    def _hsv_colorize(value, hue, saturation_value=1.0, background=True, black=True):
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

        rgb = color_conversion.hsv2rgb(hsv).reshape(-1, *hw, 3)
        rgb = torch.Tensor(rgb).permute(0, 3, 1, 2).contiguous()

        return rgb

    def _sample_color(self, mean_color_values):
        if self.color_space == 'hsv':
            return np.clip(self.random_state.normal(mean_color_values, self.scale), 0, 1)
        return np.clip(self.random_state.multivariate_normal(mean_color_values, self.scale), 0, 1)

    def _transform(self, img, target, assign_color_randomly, background=True, black=True):
        if assign_color_randomly:
            target = self.random_state.randint(0, 10, target.shape)
        else:
            target = target.numpy()

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
            colorized_data = self._hsv_colorize(
                img, np.array(colors_per_sample), background=background, black=black
            )
        else:
            colors = torch.Tensor(colors_per_sample).unsqueeze(-1).unsqueeze(-1)
            if background:
                if black:
                    # colorful background, black digits
                    colorized_data = (1 - img) * colors
                else:
                    # colorful background, white digits
                    colorized_data = torch.clamp(img + colors, 0, 1)
            else:
                if black:
                    # black background, colorful digits
                    colorized_data = img * colors
                else:
                    # white background, colorful digits
                    colorized_data = 1 - img * (1 - colors)

        # return colorized_data, torch.Tensor(mean_values)
        return colorized_data, torch.LongTensor(target)

    def __call__(self, img, target):
        return self._transform(img, target, self.assign_color_randomly, self.background, self.black)


def test():
    import argparse
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image

    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', '--batch-size', type=int, default=64)
        parser.add_argument('--scale', type=float, default=0.02)
        parser.add_argument('--cspace', type=str, default='rgb', choices=['rgb', 'hsv'])
        parser.add_argument('-bg', '--background', type=eval, default=True, choices=[True, False])
        parser.add_argument('--black', type=eval, default=False, choices=[True, False])

        return parser.parse_args()

    args = parse_arguments()

    train_dataset = ColouredMNIST(
        './data/mnist',
        use_train_split=True,
        assign_color_randomly=False,
        download=True,
        transform=transforms.ToTensor(),
        scale=args.scale,
        cspace=args.cspace,
        background=args.background,
        black=args.black,
    )

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    for data, color, labels in dataloader:
        save_image(data[:64], './colorized.png', nrow=8)
        break


if __name__ == '__main__':
    test()
