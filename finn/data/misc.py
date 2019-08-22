import csv
import os

import numpy as np

import torch
from torch.utils.data import Sampler, random_split, Dataset
from torchvision.utils import save_image


def shrink_dataset(dataset, pcnt):
    curr_len = len(dataset)
    new_data_len = int(pcnt * curr_len)
    lengths = [new_data_len, curr_len - new_data_len]
    return random_split(dataset, lengths)


def set_transform(dataset, transform):
    if hasattr(dataset, "dataset"):
        set_transform(dataset.dataset, transform)
    elif isinstance(dataset, Dataset):
        if hasattr(dataset, "transform"):
            dataset.transform = transform


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n, size=(self.num_samples,), dtype=torch.int64
                ).tolist()
            )

        return iter(torch.randperm(n)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


def data_tuple_to_dataset(
    data, sens, target, root: str, filename: str
) -> None:
    """

    Args:
        root: String. Root directory in which to save the dataset.
        filename: String. Filename of the sample being saved

    Returns:

    """
    if data.size(0) == 1:
        data = data.squeeze(0)
    # Create the root directory if it doesn't already exist
    if not os.path.exists(root):
        os.mkdir(root)
    # save the image
    if not filename.lower().endswith(".npz"):
        filename += ".npz"
    np.savez_compressed(os.path.join(root, filename), img=data.cpu().detach().numpy())
    # save filenames
    with open(os.path.join(root, "filename.csv"), 'a', newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow([filename])
    # save sensitive/nuisance labels
    with open(os.path.join(root, 'sens.csv'), 'ab') as f:
        sens = sens.view(-1)
        np.savetxt(f, sens.cpu().detach().numpy(), delimiter=",")
    # save targets
    with open(os.path.join(root, 'target.csv'), 'ab') as f:
        target = target.view(-1)
        np.savetxt(f, target.cpu().detach().numpy(), delimiter=",")
