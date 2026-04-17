#!/usr/bin/env python3
"""
Example: D-optimal experiment design with random datasets.

Only comparison figures are kept:
- left subplot: objective gap vs iteration
- right subplot: objective gap vs time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import accbpg


matplotlib.rcParams.update(
    {"font.size": 16, "legend.fontsize": 14, "font.family": "serif"}
)
# matplotlib.rcParams.update({"text.usetex": True})


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def make_comparison_figure(y_vals, t_vals, labels, styles, dashes, *, iter_xlim, gap_ylim, iter_legend, time_legend):
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    ax1 = plt.subplot(1, 2, 1)
    accbpg.plot_comparisons(
        ax1, y_vals, labels, x_vals=[], plotdiff=True, yscale="log", xlim=list(iter_xlim), ylim=list(gap_ylim),
        xlabel=r"Iteration number $k$", ylabel=r"$F(x_k)-F_\star$", legendloc=iter_legend,
        linestyles=styles, linedash=dashes
    )

    ax2 = plt.subplot(1, 2, 2)
    accbpg.plot_comparisons(
        ax2, y_vals, labels, x_vals=t_vals, plotdiff=True, yscale="log", xscale="log", ylim=list(gap_ylim),
        xlabel="Time (s)", ylabel=r"$F(x_k)-F_\star$", legendloc=time_legend,
        linestyles=styles, linedash=dashes
    )

    plt.tight_layout(w_pad=4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run D-optimal design experiments (random data).")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory where figures are saved.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures.")
    args = parser.parse_args()

    figs: list[tuple[plt.Figure, str]] = []

    # ------------------------------------------------------------------
    # Section 1: adaptive comparison + ABRA_GD
    # ------------------------------------------------------------------
    m = 80
    n = 200
    f, h, L, x0 = accbpg.D_opt_design(m, n, randseed=10)

    x00, F00, _, T00 = accbpg.BPG(f, h, L, x0, maxitrs=1000, linesearch=False, verbskip=100)
    xLS, FLS, _, TLS = accbpg.BPG(f, h, L, x0, maxitrs=1000, linesearch=True, verbskip=100)
    x20, F20, _, T20 = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=1000, theta_eq=True, verbskip=100)
    x2e, F2e, _, _, T2e = accbpg.ABPG_expo(f, h, L, x0, gamma0=3, maxitrs=1000, theta_eq=True, verbskip=100)
    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=3000, G0=0.1, theta_eq=True, verbskip=100)
    xabra, Fabra, tk_abra, eta_abra, Tabra = accbpg.ABRA_GD(f, h, L, x0, maxitrs=1000, mu=0.0, verbskip=100)

    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", r"ABRA-GD"]
    styles = ["k:", "g-", "b-.", "k-", "r--", "c-"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]
    y_vals = [F00, FLS, F20, F2e, F2g, Fabra]
    t_vals = [T00, TLS, T20, T2e, T2g, Tabra]

    fig = make_comparison_figure(
        y_vals, t_vals, labels, styles, dashes,
        iter_xlim=(-10, 1000), gap_ylim=(1e-5, 2),
        iter_legend="upper right", time_legend="lower left"
    )
    figs.append((fig, "D_opt_m80n200_adapt.png"))

    # ------------------------------------------------------------------
    # Section 2: restart comparison + ABRA_GD
    # ------------------------------------------------------------------
    ms = 80
    ns = 120
    fs, hs, Ls, x0s = accbpg.D_opt_design(ms, ns, randseed=10)

    xs00, Fs00, _, Ts00 = accbpg.BPG(fs, hs, Ls, x0s, maxitrs=1000, linesearch=False, verbskip=100)
    xsLS, FsLS, _, TsLS = accbpg.BPG(fs, hs, Ls, x0s, maxitrs=1000, linesearch=True, verbskip=100)
    xs20, Fs20, _, Ts20 = accbpg.ABPG(fs, hs, Ls, x0s, gamma=2.0, maxitrs=1000, theta_eq=True, restart=False, verbskip=100)
    xs20rs, Fs20rs, _, Ts20rs = accbpg.ABPG(fs, hs, Ls, x0s, gamma=2.0, maxitrs=1000, theta_eq=True, restart=True, verbskip=100)
    xs2g, Fs2g, _, _, _, Ts2g = accbpg.ABPG_gain(fs, hs, Ls, x0s, gamma=2, maxitrs=3000, G0=0.1, theta_eq=True, restart=False, verbskip=100)
    xs2grs, Fs2grs, _, _, _, Ts2grs = accbpg.ABPG_gain(fs, hs, Ls, x0s, gamma=2, maxitrs=3000, G0=0.1, theta_eq=True, restart=True, verbskip=100)
    xsabra, Fsabra, tks_abra, etas_abra, Tsabra = accbpg.ABRA_GD(fs, hs, Ls, x0s, maxitrs=1000, mu=0.0, verbskip=100)

    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG RS", r"ABPG-g", r"ABPG-g RS", r"ABRA-GD"]
    styles = ["k:", "g-", "b-.", "m-", "k-", "r--", "c-"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [4, 2, 1, 2, 1, 2], [], [4, 2], []]
    y_vals = [Fs00, FsLS, Fs20, Fs20rs, Fs2g, Fs2grs, Fsabra]
    t_vals = [Ts00, TsLS, Ts20, Ts20rs, Ts2g, Ts2grs, Tsabra]

    fig = make_comparison_figure(
        y_vals, t_vals, labels, styles, dashes,
        iter_xlim=(0, 50), gap_ylim=(1e-10, 1),
        iter_legend="upper right", time_legend="upper right"
    )
    figs.append((fig, "D_opt_m80n120_restart.png"))

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
