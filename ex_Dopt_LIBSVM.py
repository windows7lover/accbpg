#!/usr/bin/env python3
"""
Example: D-optimal experiment design with LIBSVM datasets.

Adaptive comparison only:
- left subplot: objective gap vs iteration
- right subplot: objective gap vs time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import accbpg

matplotlib.rcParams.update(
    {"font.size": 18, "legend.fontsize": 14, "font.family": "serif"}
)
# matplotlib.rcParams.update({"text.usetex": True})


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def infer_plot_window(y_vals, x_pad_frac: float = 0.02):
    arrays = [np.asarray(y, dtype=float) for y in y_vals]
    finite_mins = [np.min(a[np.isfinite(a)]) for a in arrays if np.any(np.isfinite(a))]
    if not finite_mins:
        return (-1, 1), (1e-12, 1.0)
    f_star = min(finite_mins)
    pos_gaps = []
    for a in arrays:
        g = a - f_star
        g = g[np.isfinite(g) & (g > 0)]
        if g.size:
            pos_gaps.append(g)
    if pos_gaps:
        all_gaps = np.concatenate(pos_gaps)
        ymin = 10.0 ** np.floor(np.log10(np.min(all_gaps)))
        ymax = 10.0 ** np.ceil(np.log10(np.max(all_gaps)))
        if ymin == ymax:
            ymax = 10.0 * ymin
    else:
        ymin, ymax = 1e-12, 1.0
    kmax = max(max(len(a) - 1, 1) for a in arrays)
    pad = max(1, int(np.ceil(x_pad_frac * kmax)))
    return (-pad, kmax + pad), (ymin, ymax)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D-optimal design experiments on a LIBSVM dataset.")
    parser.add_argument("--filename", type=str, default=r"data\housing.txt", help="LIBSVM dataset path.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory where figures are saved.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures.")
    args = parser.parse_args()

    filename = args.filename
    title_name = Path(filename).stem
    f, h, L, x0 = accbpg.D_opt_libsvm(filename)

    x00, F00, _, T00 = accbpg.BPG(f, h, L, x0, maxitrs=5000, linesearch=False, verbskip=1000)
    xLS, FLS, _, TLS = accbpg.BPG(f, h, L, x0, maxitrs=5000, linesearch=True, ls_ratio=1.2, verbskip=1000)
    x20, F20, _, T20 = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=5000, theta_eq=True, verbskip=1000)
    x2e, F2e, _, _, T2e = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=5000, theta_eq=True, Gmargin=100, verbskip=1000)
    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=5000, G0=0.1, theta_eq=True, verbskip=1000)
    xabra, Fabra, _, _, Tabra = accbpg.ABRA_GD(f, h, L, x0, maxitrs=2000, mu=0.0, restart=False, verbskip=1000)
    xabrag, Fabrag, _, _, Tabrag = accbpg.ABRA_GD(f, h, L, x0, maxitrs=2000, mu=0.0, restart=True, restart_rule="g", verbskip=1000)
    xabraf, Fabraf, _, _, Tabraf = accbpg.ABRA_GD(f, h, L, x0, maxitrs=2000, mu=0.0, restart=True, restart_rule="f", verbskip=1000)

    fig, _ = plt.subplots(1, 2, figsize=(11, 4))
    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"ABRA-GD", r"ABRA-GD g-RS", r"ABRA-GD f-RS"]
    styles = ["k:", "g-", "b-.", "k-", "r--", "c-", "c--", "c:"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], [], [2,2], [1,1]]

    y_vals = [F00, FLS, F20, F2e, F2g, Fabra, Fabrag, Fabraf]
    t_vals = [T00, TLS, T20, T2e, T2g, Tabra, Tabrag, Tabraf]
    iter_xlim, gap_ylim = infer_plot_window(y_vals)

    ax1 = plt.subplot(1, 2, 1)
    accbpg.plot_comparisons(
        ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=list(iter_xlim), ylim=list(gap_ylim),
        xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="best",
        linestyles=styles, linedash=dashes
    )
    plt.title(title_name)

    ax2 = plt.subplot(1, 2, 2)
    accbpg.plot_comparisons(
        ax2, y_vals, labels, x_vals=t_vals, plotdiff=True, yscale="log", ylim=list(gap_ylim),
        xlabel="Time (s)", ylabel=r"$F(x_k)-F_\star$", legendloc="best",
        linestyles=styles, linedash=dashes
    )
    plt.title(title_name)

    plt.tight_layout(w_pad=4)

    if args.save_dir is not None:
        save_figure(fig, args.save_dir / f"{title_name}_adapt.png")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
