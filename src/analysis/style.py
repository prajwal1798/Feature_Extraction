# =============================================================================
# FIGURE STYLE: ONE LOOK FOR EVERY FIGURE OF THE STUDY
#
# Usage:
#     with paper_style():
#         fig, ax = plt.subplots(...)
#         ...
#         save_figure(fig, folder / "name")     -> name.pdf and name.png
#
# PDF is the vector version for the thesis; PNG (600 dpi) for slides.
# Files are written under a temporary name first and then renamed, so an
# interrupted job never leaves a half-written figure behind.
# =============================================================================

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {"deeponet": "#d62728", "fno": "#1f77b4"}
MODEL_NAMES = {"deeponet": "DeepONet", "fno": "FNO"}
FAMILY_NAMES = {0: "circle", 1: "rectangle", 2: "rounded rectangle"}


def paper_style():
    return mpl.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for suffix in (".pdf", ".png"):
        target = path.with_suffix(suffix)
        temporary = target.with_name(target.name + ".partial")
        fig.savefig(temporary, format=suffix[1:], dpi=600, bbox_inches="tight")
        os.replace(temporary, target)

    plt.close(fig)
    print("Saved figure: %s.{pdf,png}" % path, flush=True)
