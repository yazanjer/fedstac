"""Journal figure style: serif (Times), muted print-safe palette, no chartjunk, PDF + 300 dpi PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SINGLE, DOUBLE = 3.5, 7.16  # inches, double-column manuscript
PALETTE = ["#1b4f72", "#b9770e", "#1e8449", "#922b21", "#6c3483", "#5d6d7e", "#a04000", "#117a65", "#283747", "#7d6608"]
METHOD_COLOR = {"fedstap": "#922b21"}  # the proposed method is drawn in one fixed, emphasised colour


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "lines.linewidth": 1.1, "lines.markersize": 3.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def color(method: str, order: list[str]) -> str:
    if method not in METHOD_COLOR:
        METHOD_COLOR[method] = PALETTE[order.index(method) % len(PALETTE)] if method in order else "#5d6d7e"
    return METHOD_COLOR[method]


def save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)
