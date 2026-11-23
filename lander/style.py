"""Shared Matplotlib style for every figure in the project (print-friendly, CVD-validated palette)."""
import matplotlib as mpl

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948",
         ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", surface="#fcfcfb")
SERIES = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["green"], C["violet"]]


def apply():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "axes.titlecolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "legend.frameon": False, "legend.fontsize": 8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
