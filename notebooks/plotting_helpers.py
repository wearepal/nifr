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
    i_hate_fun: bool = False,
    no_lines: bool = False,
    with_markers: bool = False,
    **kwargs,
) -> plt.Axes:
    y_key, y_name = (y_axis[0], y_axis[1]) if isinstance(y_axis, tuple) else (y_axis, y_axis)
    x_key, x_name = (x_axis[0], x_axis[1]) if isinstance(x_axis, tuple) else (x_axis, x_axis)
    plot_kwargs = {"figsize": (3, 2), "dpi": 200, **kwargs}
    fig, plot = plt.subplots(**plot_kwargs, facecolor="white")
    assert i_hate_fun or len(results) <= 5, "we only have 5 line styles right now"
    linestyles = ["-"] * 100 if i_hate_fun else ["-", "--", ":", "-.", (0, (5, 5))]
    markerstyles = ["o", "s", "x", "+", "<", ">"]
    marker = 0
    for linestyle, (key, result) in zip(linestyles, results.items()):
        try:
            if not no_lines:
                plot.plot(
                    result[x_key],
                    result[y_key],
                    marker=markerstyles[marker] if with_markers else "",
                    label=key,
                    linestyle=linestyle,
                    linewidth=2.5,
                )
                marker += 1
            else:
                plot.plot(
                    result[x_key],
                    result[y_key],
                    label=key,
                    marker=markerstyles[marker],
                    linestyle="",
                )
                marker += 1
        except:
            pass
    plot.grid(True)
    # get handles and labels for the legend
    handles, labels = plot.get_legend_handles_labels()
    # remove the errorbars from the legend if they are there
    handles = [(h[0] if isinstance(h, tuple) else h) for h in handles]
    plot.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1.05))
    plot.set_xlabel(x_name)
    plot.set_ylabel(y_name)
    if title is not None:
        plot.set_title(title)
    return fig, plot
