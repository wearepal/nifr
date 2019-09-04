import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    fig.savefig(f'{filename}.png')

    plt.close(fig)
