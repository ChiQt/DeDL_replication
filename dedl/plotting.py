"""Paper-style figures from saved measurements only; no smoothing or fitted curves."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

COLORS = {"dedl": "#6d80aa", "sdl": "#7fa393", "lr": "#de806f"}
STYLES = {"dedl": "-", "sdl": (0, (6, 6)), "lr": ":"}


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 11, "axes.linewidth": 0.5, "axes.edgecolor": "#bdbdbd",
        "pdf.fonttype": 42
    })


def figure7(rows, path, description):
    style()
    figure, left = plt.subplots(figsize=(8.6, 3.25))
    right = left.twinx()
    epochs = [row["epoch"] for row in rows]
    left.plot(epochs, [row["validation_mse"] for row in rows], color="#946767",
              linewidth=0.85, marker="D", markersize=2.7, markevery=2, label="Training MSE")
    for method in ["dedl", "sdl", "lr"]:
        right.plot(epochs, [row[f"{method}_mape"] for row in rows],
                   color=COLORS[method], linestyle=STYLES[method], linewidth=0.95,
                   label={"dedl": "DeDL MAPE", "sdl": "SDL MAPE", "lr": "LR MAPE"}[method])
    left.set(xlabel="Training Epoch", ylabel="Training MSE", xlim=(-20, 420))
    left.set_xticks(np.arange(0, 401, 50))
    left.set_ylim(bottom=0)
    left.yaxis.set_major_locator(MaxNLocator(7))
    right.set_ylabel("Estimation MAPE", labelpad=10)
    right.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    maximum = max(row[f"{method}_mape"] for row in rows for method in COLORS)
    right.set_ylim(0, max(104, 10 * np.ceil(maximum / 10)))
    left.grid(color="#c9c9c9", linewidth=0.55)
    left.tick_params(length=0)
    right.tick_params(length=0)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], loc="upper right",
                framealpha=1, fancybox=False, edgecolor="#bdbdbd", fontsize=10)
    figure.subplots_adjust(left=0.09, right=0.91, bottom=0.21, top=0.97)
    figure.savefig(path, metadata={"Title": "Figure 7: synthetic data", "Subject": description,
                                   "CreationDate": None, "ModDate": None})
    figure.savefig(Path(path).with_suffix(".png"), dpi=160)
    plt.close(figure)


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [{key: float(value) if value != "" else np.nan for key, value in row.items()}
                for row in csv.DictReader(handle)]
