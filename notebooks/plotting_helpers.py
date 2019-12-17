from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from matplotlib import pyplot as plt


def from_default_place(file_name: str) -> pd.DataFrame:
    return pd.read_csv(Path("..") / "experiments" / "finn" / file_name)


def nice_plot(
    results: Dict[str, pd.DataFrame],
    y_axis: Union[str, Tuple[str, str]],
    x_axis: Union[str, Tuple[str, str]] = ("Mix_fact", "mixing factor"),
    title: Optional[str] = None,
) -> plt.Axes:
    y_key, y_name = (y_axis[0], y_axis[1]) if isinstance(y_axis, tuple) else (y_axis, y_axis)
    x_key, x_name = (x_axis[0], x_axis[1]) if isinstance(x_axis, tuple) else (x_axis, x_axis)
    fig, plot = plt.subplots(figsize=(3, 2), dpi=200)
    assert len(results) <= 4, "we only have 4 line styles right now"
    for linestyle, (key, result) in zip(["-", "--", ":", "-."], results.items()):
        plot.errorbar(
            result[x_key],
            result[y_key],
            marker="",
            label=key,
            linestyle=linestyle,
            linewidth=2.5,
        )
    plot.grid(True)
    # get handles and labels for the legend
    handles, labels = plot.get_legend_handles_labels()
    # remove the errorbars from the legend if they are there
    handles = [(h[0] if isinstance(h, tuple) else h) for h in handles]
    plot.legend(
        handles, labels, loc='upper left', bbox_to_anchor=(1, 1.05)
    )
    plot.set_xlabel(x_name)
    plot.set_ylabel(y_name)
    if title is not None:
        plot.set_title(title)
    return plot