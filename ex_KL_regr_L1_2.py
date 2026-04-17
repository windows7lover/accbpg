#!/usr/bin/env python3
"""
Example: KL-divergence nonnegative regression with random datasets (restart comparisons).

Adaptive/restart comparison only:
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


def run_experiment(m: int, n: int, *, maxitrs: int = 5000, verbskip: int = 1000):
    f, h, L, x0 = accbpg.KL_nonneg_regr(
        m, n, noise=0.01, lamdaL1=0.001, normalizeA=True, randseed=1
    )

    x00, F00, _, T00 = accbpg.BPG(f, h, L, x0, maxitrs=maxitrs, linesearch=False, verbskip=verbskip)
    xLS, FLS, _, TLS = accbpg.BPG(f, h, L, x0, maxitrs=maxitrs, linesearch=True, ls_ratio=1.2, verbskip=verbskip)
    x20, F20, _, T20 = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=maxitrs, theta_eq=True, restart=False, verbskip=verbskip)
    x20rs, F20rs, _, T20rs = accbpg.ABPG(f, h, L, x0, gamma=2.0, maxitrs=maxitrs, theta_eq=True, restart=True, verbskip=verbskip)
    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=maxitrs, G0=0.1, theta_eq=True, restart=False, verbskip=verbskip)
    x2grs, F2grs, _, _, _, T2grs = accbpg.ABPG_gain(f, h, L, x0, gamma=2, maxitrs=maxitrs, G0=0.1, theta_eq=True, restart=True, restart_rule="f", verbskip=verbskip)
    xabra, Fabra, tk_abra, eta_abra, Tabra = accbpg.ABRA_GD(f, h, L, x0, maxitrs=maxitrs, mu=0.0, restart=False, verbskip=verbskip)
    xabrag, Fabrag, tk_abrag, eta_abrag, Tabrag = accbpg.ABRA_GD(f, h, L, x0, maxitrs=maxitrs, mu=0.0, restart=True, restart_rule="g", verbskip=verbskip)
    xabraf, Fabraf, tk_abraf, eta_abraf, Tabraf = accbpg.ABRA_GD(f, h, L, x0, maxitrs=maxitrs, mu=0.0, restart=True, restart_rule="f", verbskip=verbskip)

    return {
        "BPG": {"F": F00, "T": T00},
        "BPG_LS": {"F": FLS, "T": TLS},
        "ABPG": {"F": F20, "T": T20},
        "ABPG_RS": {"F": F20rs, "T": T20rs},
        "ABPG_g": {"F": F2g, "T": T2g},
        "ABPG_g_RS": {"F": F2grs, "T": T2grs},
        "ABRA_GD": {"F": Fabra, "T": Tabra, "t": tk_abra, "eta": eta_abra},
        "ABRA_GD_g_RS": {"F": Fabrag, "T": Tabrag, "t": tk_abrag, "eta": eta_abrag},
        "ABRA_GD_f_RS": {"F": Fabraf, "T": Tabraf, "t": tk_abraf, "eta": eta_abraf},
    }


def plot_experiment(results: dict, *, title: str):
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG RS", r"ABPG-g", r"ABPG-g RS", r"ABRA-GD", r"ABRA-GD g-RS", r"ABRA-GD f-RS"]
    styles = ["k:", "g-", "b-.", "m-", "k-", "r--", "c-", "c--", "c:"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [4, 2, 1, 2, 1, 2], [], [4, 2], [], [2,2], [1,1]]

    y_vals = [
        results["BPG"]["F"], results["BPG_LS"]["F"], results["ABPG"]["F"],
        results["ABPG_RS"]["F"], results["ABPG_g"]["F"], results["ABPG_g_RS"]["F"],
        results["ABRA_GD"]["F"], results["ABRA_GD_g_RS"]["F"], results["ABRA_GD_f_RS"]["F"],
    ]
    t_vals = [
        results["BPG"]["T"], results["BPG_LS"]["T"], results["ABPG"]["T"],
        results["ABPG_RS"]["T"], results["ABPG_g"]["T"], results["ABPG_g_RS"]["T"],
        results["ABRA_GD"]["T"], results["ABRA_GD_g_RS"]["T"], results["ABRA_GD_f_RS"]["T"],
    ]
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
    parser = argparse.ArgumentParser(description="Run KL nonnegative regression restart comparisons.")
    parser.add_argument("--maxitrs", type=int, default=5000, help="Maximum iterations per method.")
    parser.add_argument("--verbskip", type=int, default=1000, help="Verbosity skip passed to methods.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory where figures are saved.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures.")
    args = parser.parse_args()

    results_1 = run_experiment(1000, 100, maxitrs=args.maxitrs, verbskip=args.verbskip)
    fig1 = plot_experiment(results_1, title="KL nonnegative regression: m=1000, n=100")

    results_2 = run_experiment(100, 1000, maxitrs=args.maxitrs, verbskip=args.verbskip)
    fig2 = plot_experiment(results_2, title="KL nonnegative regression: m=100, n=1000")

    if args.save_dir is not None:
        save_figure(fig1, args.save_dir / "KL_regr_restart_m1000n100.png")
        save_figure(fig2, args.save_dir / "KL_regr_restart_m100n1000.png")

    if args.no_show:
        plt.close(fig1)
        plt.close(fig2)
    else:
        plt.show()


if __name__ == "__main__":
    main()
