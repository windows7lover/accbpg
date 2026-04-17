#!/usr/bin/env python3
"""
Example: Poisson linear inverse problems with random datasets.

Only comparison figures are kept:
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
    {"font.size": 16, "legend.fontsize": 14, "font.family": "serif"}
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


def make_comparison_figure(y_vals, t_vals, labels, styles, dashes, *, title: str):
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))
    iter_xlim, gap_ylim = infer_plot_window(y_vals)

    ax1 = plt.subplot(1, 2, 1)
    accbpg.plot_comparisons(
        ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=list(iter_xlim), ylim=list(gap_ylim),
        xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc="best",
        linestyles=styles, linedash=dashes
    )

    ax2 = plt.subplot(1, 2, 2)
    accbpg.plot_comparisons(
        ax2, y_vals, labels, x_vals=t_vals, plotdiff=True, yscale="log", ylim=list(gap_ylim),
        xlabel="Time (s)", ylabel=r"$F(x_k)-F_\star$", legendloc="best",
        linestyles=styles, linedash=dashes
    )

    fig.suptitle(title)
    plt.tight_layout(w_pad=4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Poisson inverse-problem experiments.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory where figures are saved.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures.")
    args = parser.parse_args()

    figs: list[tuple[plt.Figure, str]] = []

    m = 200
    n = 100
    f, h, L, x0 = accbpg.Poisson_regrL1(m, n, noise=0.0001, lamda=0, randseed=1)
    x00, F00, _, T00 = accbpg.BPG(f, h, L, x0, maxitrs=10000, linesearch=False, verbskip=1000)
    xLS, FLS, _, TLS = accbpg.BPG(f, h, L, x0, maxitrs=10000, linesearch=True, verbskip=1000)
    x20, F20, _, T20 = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=10000, theta_eq=True, verbskip=1000)
    x2e, F2e, _, _, T2e = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=10000, theta_eq=False, Gmargin=3, verbskip=1000)
    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=10000, G0=0.1, theta_eq=False, verbskip=1000)
    xabra, Fabra, _, _, Tabra = accbpg.ABRA_GD(f, h, L, x0, maxitrs=10000, mu=0.0, restart=False, verbskip=1000)
    xabrag, Fabrag, _, _, Tabrag = accbpg.ABRA_GD(f, h, L, x0, maxitrs=10000, mu=0.0, restart=True, restart_rule="g", verbskip=1000)
    xabraf, Fabraf, _, _, Tabraf = accbpg.ABRA_GD(f, h, L, x0, maxitrs=10000, mu=0.0, restart=True, restart_rule="f", verbskip=1000)

    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"ABRA-GD", r"ABRA-GD g-RS", r"ABRA-GD f-RS"]
    styles = ["k:", "g-", "b-.", "k-", "r--", "c-", "c--", "c:"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], [], [2,2], [1,1]]
    y_vals = [F00, FLS, F20, F2e, F2g, Fabra, Fabrag, Fabraf]
    t_vals = [T00, TLS, T20, T2e, T2g, Tabra, Tabrag, Tabraf]
    fig = make_comparison_figure(y_vals, t_vals, labels, styles, dashes, title=f"Poisson L1: m={m}, n={n}")
    figs.append((fig, "Poisson_m200n100_adapt.png"))

    m2 = 100
    n2 = 1000
    f2, h2, L2, x02 = accbpg.Poisson_regrL2(m2, n2, noise=0.001, lamda=0.001, randseed=1)
    x00_, F00_, _, T00_ = accbpg.BPG(f2, h2, L2, x02, maxitrs=10000, linesearch=False, verbskip=1000)
    xLS_, FLS_, _, TLS_ = accbpg.BPG(f2, h2, L2, x02, maxitrs=10000, linesearch=True, ls_ratio=1.5, verbskip=1000)
    x20_, F20_, _, T20_ = accbpg.ABPG(f2, h2, L2, x02, gamma=2.0, maxitrs=10000, theta_eq=False, verbskip=1000)
    x2e_, F2e_, _, _, T2e_ = accbpg.ABPG_expo(f2, h2, L2, x02, gamma0=3, maxitrs=10000, theta_eq=False, Gmargin=1, verbskip=1000)
    x2g_, F2g_, _, _, _, T2g_ = accbpg.ABPG_gain(f2, h2, L2, x02, gamma=2, maxitrs=10000, G0=0.1, ls_inc=1.5, ls_dec=1.5, theta_eq=True, verbskip=1000)
    xabra_, Fabra_, _, _, Tabra_ = accbpg.ABRA_GD(f2, h2, L2, x02, maxitrs=10000, mu=0.0, restart=False, verbskip=1000)
    xabrag_, Fabrag_, _, _, Tabrag_ = accbpg.ABRA_GD(f2, h2, L2, x02, maxitrs=10000, mu=0.0, restart=True, restart_rule="g", verbskip=1000)
    xabraf_, Fabraf_, _, _, Tabraf_ = accbpg.ABRA_GD(f2, h2, L2, x02, maxitrs=10000, mu=0.0, restart=True, restart_rule="f", verbskip=1000)
    y_vals = [F00_, FLS_, F20_, F2e_, F2g_, Fabra_, Fabrag_, Fabraf_]
    t_vals = [T00_, TLS_, T20_, T2e_, T2g_, Tabra_, Tabrag_, Tabraf_]
    fig = make_comparison_figure(y_vals, t_vals, labels, styles, dashes, title=f"Poisson L2: m={m2}, n={n2}")
    figs.append((fig, "Poisson_m100n1000_L2_adapt.png"))

    if args.save_dir is not None:
        for fig, name in figs:
            save_figure(fig, args.save_dir / name)

    if args.no_show:
        for fig, _ in figs:
            plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
