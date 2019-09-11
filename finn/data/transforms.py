import random
from typing import List

import numpy as np
import torch
from skimage import color
from torchvision import transforms


class LdAugmentation(torch.jit.ScriptModule):
    """Base class for label-dependent augmentations.
    """

    @torch.jit.script_method
    def _augment(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Augment the input data in a label-dependent fashion

        Args:
            data: Tensor. Input data to be augmented.
            labels: Tensor. Labels on which the augmentations are conditioned.

        Returns: Tensor. Augmented Data.
        """
        return data

    def __call__(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calls the augment method on the the input data.

        Args:
            data: Tensor. Input data to be augmented.
            labels: Tensor. Labels on which the augmentations are conditioned.

        Returns: Tensor. Augmented data.
        """
        return self._augment(data, labels)


class LdCoordinateCoding(LdAugmentation):
    """Embed the class in the image in the form of a marker
    with unique coordinates for each class.
    """

    def __init__(self, marker_color=(255, 0, 0)):
        """

        Args:
            marker_color: RGB values to use for the marker.
        """
        super(LdCoordinateCoding, self).__init__()

        marker_color = torch.tensor(marker_color) / 255.0  # normalize to [0, 1]
        self.marker_color = torch.jit.Attribute(marker_color, torch.Tensor)

    def _augment(self, data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Augment the input data in a label-dependent fashion

        If the input image is grey-scale, it is first converted to RGB.
        Args:
            data: Tensor. Input data to be augmented.
            labels: Tensor. Labels on which the augmentations are conditioned.

        Returns: Tensor. Augmented data.
        """
        if data.size(-3) == 1:
            data = data.repeat(1, 3, 1, 1)

        b, c, w, h = data.shape
        row = torch.fmod(labels, w)
        col = labels // w

        data[torch.arange(b), :, row, col] = self.marker_color

        return data


class LdColorJitter(LdAugmentation):

    __constants__ = ["scale", "additive"]

    def __init__(self, scale=0.05, multiplicative=False):
        super(LdColorJitter, self).__init__()
        self.scale = scale
        self.multiplicative = multiplicative

    def _augment(self, data, labels):
        if data.size(1) == 1:
            data = data.repeat(1, 3, 1, 1)

        loc = labels.float().view(-1, 1, 1, 1) / (2.0 * labels.max()) + 0.25
        noise = self.scale * torch.randn_like(data) + loc

        data = data - 0.5
        if self.multiplicative:
            data *= noise
        else:
            data += noise
        data.clamp_(0, 1)

        return data


class LdContrastAdjustment(LdAugmentation):
    def __init__(self, sensitivity=1.0):
        super(LdContrastAdjustment, self).__init__()
        self.sensitivity = sensitivity

    def _augment(self, data, labels):
        C = labels.view(-1, 1, 1, 1).float() / labels.max()
        C *= self.sensitivity
        F = (259 * (C + 255)) / (255 * (259 + C))

        data = F * (data - 0.5) + 0.5
        data.clamp_(0, 1)

        return data


class LdGainAdjustment(LdAugmentation):

    __constants__ = ["max_gamma"]

    def __init__(self, max_gamma=1.0e-1):
        super(LdGainAdjustment, self).__init__()

        self.max_gamma = max_gamma

    def _augment(self, data, labels):
        gamma = labels.view(-1, 1, 1, 1).float() * self.max_gamma / labels.max()

        data = torch.exp(torch.log(data.clamp(min=1.0e-8)) * gamma)
        data.clamp_(0, 1)

        return data


class LdColorizer(LdAugmentation):

    __constants__ = ["color_space", "binarize", "black", "background", "seed"]

    def __init__(
        self,
        scale=0.02,
        binarize=False,
        color_space="rgb",
        background=False,
        black=True,
        seed=42,
    ):
        super(LdColorizer, self).__init__()
        self.scale = scale
        self.binarize = binarize
        self.background = background
        self.black = black

        # create a local random state that won't affect the global random state of the training
        self.random_state = np.random.RandomState(seed)

        self.color_space = color_space
        if color_space == "rgb":
            colors = [
                (0, 255, 255),
                (0, 0, 255),  # blue
                (255, 0, 255),
                (0, 128, 0),
                (0, 255, 0),  # green
                (128, 0, 0),
                (0, 0, 128),
                (128, 0, 128),
                (255, 0, 0),  # red
                (255, 255, 0),
            ]  # yellow

            self.palette = [np.divide(color, 255) for color in colors]
            self.scale *= np.eye(3)
        else:
            colors = [0, 35, 60, 78, 125, 165, 190, 235, 270, 300]
            self.palette = np.divide(colors, 360)

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

        rgb = color.hsv2rgb(hsv).reshape(-1, *hw, 3)
        rgb = torch.tensor(rgb).permute(0, 3, 1, 2).contiguous()

        return rgb

    def _sample_color(self, mean_color_values):
        if self.color_space == "hsv":
            return np.clip(
                self.random_state.normal(mean_color_values, self.scale), 0, 1
            )
        else:
            return np.clip(
                self.random_state.multivariate_normal(mean_color_values, self.scale),
                0,
                1,
            )

    def _augment(self, data, labels):

        labels = labels.numpy()

        mean_values = []
        colors_per_sample: List[np.ndarray] = []
        for label in labels:
            mean_value = self.palette[label]
            mean_values.append(mean_value)
            colors_per_sample.append(self._sample_color(mean_value))

        if self.binarize:
            data = (data > 0.5).float()

        if self.color_space == "hsv":
            data = data.numpy().squeeze()
            if len(data.shape) == 2:
                data = data[
                    None, ...
                ]  # re-add the batch dimension in case it was removed
            colorized_data = self._hsv_colorize(
                data,
                np.array(colors_per_sample),
                background=self.background,
                black=self.black,
            )
        else:
            if self.background:
                if self.black:
                    # colorful background, black digits
                    colorized_data = (1 - data) * torch.Tensor(
                        colors_per_sample
                    ).unsqueeze(-1).unsqueeze(-1)
                else:
                    # colorful background, white digits
                    colorized_data = torch.clamp(
                        data
                        + torch.Tensor(colors_per_sample).unsqueeze(-1).unsqueeze(-1),
                        0,
                        1,
                    )
            else:
                if self.black:
                    # black background, colorful digits
                    colorized_data = data * torch.Tensor(colors_per_sample).unsqueeze(
                        -1
                    ).unsqueeze(-1)
                else:
                    # white background, colorful digits
                    colorized_data = 1 - data * (
                        1 - torch.Tensor(colors_per_sample).unsqueeze(-1).unsqueeze(-1)
                    )

        return colorized_data


class RandomChoice:
    def __init__(self, choices):
        super(RandomChoice, self).__init__()
        self.choices = choices

    def _augment(self, data, label):
        choice = random.choice(self.choices)
        return choice(data, label)

    def __call__(self, data, label):
        return self._augment(data, label)
