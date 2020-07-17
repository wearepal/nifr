import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nifr.configs import SharedArgs

from .utils import wandb_log

__all__ = ["plot_contrastive", "plot_histogram"]


def plot_contrastive(original, recon, columns, filename):
    fig, ax = plt.subplots(figsize=(10, 14))
    to_plot = original.cpu()
    recon = recon.cpu()
    diff = to_plot - recon

    im = ax.imshow(diff.numpy())

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(columns)))

    # ax.set_yticks(np.arange(diff.shape[1]))
    # ... and label them with the respective list entries
    ax.set_xticklabels(columns)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    fig.colorbar(im, cax=cax)
    ax.set_title("Difference between Gender reconstructions")
    fig.tight_layout()
    fig.savefig(f"{filename}.png")

    plt.close(fig)


def plot_histogram(
    args: SharedArgs,
    vector: torch.Tensor,
    step: int,
    prefix: str = "train",
    cols: int = 3,
    rows: int = 6,
    bins: int = 30,
):
    """Plot a histogram over the batch"""
    vector = torch.flatten(vector, start_dim=1).detach().cpu()
    vector_np = vector.numpy()
    matplotlib.use("Agg")
    fig, plots = plt.subplots(figsize=(8, 12), ncols=cols, nrows=rows)
    # fig.suptitle("Xi histogram")
    for j in range(rows):
        for i in range(cols):
            _ = plots[j][i].hist(vector_np[:, j * cols + i], bins=np.linspace(-15, 15, bins))
    fig.tight_layout()

    log_dict = {
        f"{prefix}_histogram": fig,
        f"{prefix}_xi_min": vector_np.min(),
        f"{prefix}_xi_max": vector_np.max(),
        f"{prefix}_xi_nans": float(bool(np.isnan(vector_np).any())),
        f"{prefix}_xi_tensor": vector,
    }
    wandb_log(args, log_dict, step=step)
