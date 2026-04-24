#!/usr/bin/env python3
"""
Example: KL-divergence nonnegative regression with random datasets.

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
    {
        "font.size": 16,
        "legend.fontsize": 14,
        "font.family": "serif",
    }
)
# matplotlib.rcParams.update({"text.usetex": True})


def save_figure(fig, path: Path):
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



def infer_metric_ylim(series_list, logscale=False):
    vals = []
    for s in series_list:
        a = np.asarray(s, dtype=float)
        if logscale:
            a = a[np.isfinite(a) & (a > 0)]
        else:
            a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return None
    allv = np.concatenate(vals)
    if logscale:
        ymin = 10.0 ** np.floor(np.log10(np.min(allv)))
        ymax = 10.0 ** np.ceil(np.log10(np.max(allv)))
        if ymin == ymax:
            ymax = 10.0 * ymin
        return ymin, ymax
    ymin = np.min(allv)
    ymax = np.max(allv)
    if ymin == ymax:
        pad = 1.0 if ymin == 0 else 0.1 * abs(ymin)
        return ymin - pad, ymax + pad
    pad = 0.05 * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_abra_diagnostics(results: dict, *, title: str | None = None):
    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))
    keys = ["ABRA_GD", "ABRA_GD_g_RS", "ABRA_GD_f_RS"]
    labels = [r"ABRA-GD", r"ABRA-GD g-RS", r"ABRA-GD f-RS"]
    styles = ["c-", "c--", "c:"]
    dashes = [[6, 2], [2, 2], [1, 1]]

    xvals = [np.arange(len(results[k]["t"])) for k in keys]

    t_series = [results[k]["t"] for k in keys]
    c_series = [results[k]["c"] for k in keys]
    alpha_series = [results[k]["alpha"] for k in keys]
    eta_series = [np.where(np.isfinite(results[k]["eta"]), results[k]["eta"], np.nan) for k in keys]
    L_series = [results[k]["L"] for k in keys]

    panels = [
        (axes[0, 0], t_series, r"$t_k$", "linear"),
        (axes[0, 1], c_series, r"$c_k$", "log"),
        (axes[1, 0], alpha_series, r"$\alpha_k$", "log"),
        (axes[1, 1], eta_series, r"$\eta_k$", "log"),
        (axes[2, 0], L_series, r"$L_k$", "log"),
    ]

    xmax = max(max(len(s) - 1, 1) for s in t_series)
    xpad = max(1, int(np.ceil(0.02 * xmax)))
    xlim = (-xpad, xmax + xpad)

    for ax, series, ylabel, yscale in panels:
        for xi, yi, lab, sty, dash in zip(xvals, series, labels, styles, dashes):
            ax.plot(xi, yi, sty, label=lab, dashes=dash)
        ax.set_xlim(xlim)
        ax.set_xlabel(r"Iteration number $k$")
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ylim = infer_metric_ylim(series, logscale=(yscale == "log"))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend(loc="best")

    axes[2, 1].axis("off")

    if title:
        fig.suptitle(title)
    plt.tight_layout(w_pad=3.0, h_pad=2.0)
    return fig

def run_experiment(
    m: int,
    n: int,
    *,
    noise: float = 0.0,
    lamdaL1: float = 0.0,
    normalizeA: bool = True,
    randseed: int = 1,
    maxitrs: int = 5000,
    verbskip: int = 1000,
):
    f, h, L, x0 = accbpg.KL_nonneg_regr(
        m,
        n,
        noise=noise,
        lamdaL1=lamdaL1,
        normalizeA=normalizeA,
        randseed=randseed,
    )

    x00, F00, _, T00 = accbpg.BPG(
        f, h, L, x0, maxitrs=maxitrs, linesearch=False, verbskip=verbskip
    )
    xLS, FLS, _, TLS = accbpg.BPG(
        f, h, L, x0, maxitrs=maxitrs, linesearch=True, ls_ratio=1.2, verbskip=verbskip
    )
    x20, F20, _, T20 = accbpg.ABPG(
        f, h, L, x0, gamma=2.0, maxitrs=maxitrs, theta_eq=True,
        restart=False, verbskip=verbskip
    )
    x20rs, F20rs, _, T20rs = accbpg.ABPG(
        f, h, L, x0, gamma=2.0, maxitrs=maxitrs, theta_eq=True,
        restart=True, verbskip=verbskip
    )
    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(
        f, h, L, x0, gamma=2, maxitrs=maxitrs, G0=0.1, theta_eq=True,
        restart=False, verbskip=verbskip
    )
    x2grs, F2grs, _, _, _, T2grs = accbpg.ABPG_gain(
        f, h, L, x0, gamma=2, maxitrs=maxitrs, G0=0.1, theta_eq=True,
        restart=True, restart_rule="f", verbskip=verbskip
    )
    xabra, Fabra, tk_abra, eta_abra, ck_abra, alpha_abra, L_abra, Tabra = accbpg.ABRA_GD(
        f, h, L, x0, maxitrs=maxitrs, mu=0.0, epsilon=1e-11,
        restart=False, verbose=True, verbskip=verbskip
    )
    xabrag, Fabrag, tk_abrag, eta_abrag, ck_abrag, alpha_abrag, L_abrag, Tabrag = accbpg.ABRA_GD(
        f, h, L, x0, maxitrs=maxitrs, mu=0.0, epsilon=1e-11,
        restart=True, restart_rule="g", verbose=True, verbskip=verbskip
    )
    xabraf, Fabraf, tk_abraf, eta_abraf, ck_abraf, alpha_abraf, L_abraf, Tabraf = accbpg.ABRA_GD(
        f, h, L, x0, maxitrs=maxitrs, mu=0.0, epsilon=1e-11,
        restart=True, restart_rule="f", verbose=True, verbskip=verbskip
    )

    return {
        "problem": {"m": m, "n": n, "noise": noise, "lamdaL1": lamdaL1, "L": L},
        "BPG": {"x": x00, "F": F00, "T": T00},
        "BPG_LS": {"x": xLS, "F": FLS, "T": TLS},
        "ABPG": {"x": x20, "F": F20, "T": T20},
        "ABPG_RS": {"x": x20rs, "F": F20rs, "T": T20rs},
        "ABPG_g": {"x": x2g, "F": F2g, "T": T2g},
        "ABPG_g_RS": {"x": x2grs, "F": F2grs, "T": T2grs},
        "ABRA_GD": {"x": xabra, "F": Fabra, "T": Tabra, "t": tk_abra, "eta": eta_abra, "c": ck_abra, "alpha": alpha_abra, "L": L_abra},
        "ABRA_GD_g_RS": {"x": xabrag, "F": Fabrag, "T": Tabrag, "t": tk_abrag, "eta": eta_abrag, "c": ck_abrag, "alpha": alpha_abrag, "L": L_abrag},
        "ABRA_GD_f_RS": {"x": xabraf, "F": Fabraf, "T": Tabraf, "t": tk_abraf, "eta": eta_abraf, "c": ck_abraf, "alpha": alpha_abraf, "L": L_abraf},
    }


def plot_experiment(results: dict, *, title: str | None = None):
    fig, _ = plt.subplots(1, 2, figsize=(11, 4))

    labels = [
        r"BPG", r"BPG-LS", r"ABPG", r"ABPG RS", r"ABPG-g", r"ABPG-g RS",
        r"ABRA-GD", r"ABRA-GD g-RS", r"ABRA-GD f-RS"
    ]
    styles = ["k:", "g-", "b-.", "m-", "k-", "r--", "c-", "c--", "c:"]
    dashes = [
        [1, 2], [], [4, 2, 1, 2], [4, 2, 1, 2, 1, 2], [], [4, 2],
        [6, 2], [2, 2], [1, 1]
    ]

    y_vals = [
        results["BPG"]["F"],
        results["BPG_LS"]["F"],
        results["ABPG"]["F"],
        results["ABPG_RS"]["F"],
        results["ABPG_g"]["F"],
        results["ABPG_g_RS"]["F"],
        results["ABRA_GD"]["F"],
        results["ABRA_GD_g_RS"]["F"],
        results["ABRA_GD_f_RS"]["F"],
    ]
    t_vals = [
        results["BPG"]["T"],
        results["BPG_LS"]["T"],
        results["ABPG"]["T"],
        results["ABPG_RS"]["T"],
        results["ABPG_g"]["T"],
        results["ABPG_g_RS"]["T"],
        results["ABRA_GD"]["T"],
        results["ABRA_GD_g_RS"]["T"],
        results["ABRA_GD_f_RS"]["T"],
    ]
    iter_xlim, gap_ylim = infer_plot_window(y_vals)

    ax1 = plt.subplot(1, 2, 1)
    accbpg.plot_comparisons(
        ax1,
        y_vals,
        labels,
        x_vals=[],
        plotdiff=True,
        yscale="log",
        xlim=list(iter_xlim),
        ylim=list(gap_ylim),
        xlabel=r"Iteration number $k$",
        ylabel=r"$F(x_k)-F_\star$",
        legendloc="best",
        linestyles=styles,
        linedash=dashes,
    )

    ax2 = plt.subplot(1, 2, 2)
    accbpg.plot_comparisons(
        ax2,
        y_vals,
        labels,
        x_vals=t_vals,
        plotdiff=True,
        yscale="log",
        ylim=list(gap_ylim),
        xlabel="Time (s)",
        ylabel=r"$F(x_k)-F_\star$",
        legendloc="best",
        linestyles=styles,
        linedash=dashes,
    )

    if title:
        fig.suptitle(title)
    plt.tight_layout(w_pad=4)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Run KL-divergence nonnegative regression experiments.")
    parser.add_argument("--maxitrs", type=int, default=5000, help="Maximum iterations per method.")
    parser.add_argument("--verbskip", type=int, default=1000, help="Verbosity skip passed to methods.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory where figures will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Do not display figures.")
    args = parser.parse_args()
    
    m, n = 1000, 100
    results_1 = run_experiment(m, n, maxitrs=args.maxitrs, verbskip=args.verbskip)
    fig1 = plot_experiment(results_1, title=f"KL nonnegative regression: m={m}, n={n}")
    fig1_diag = plot_abra_diagnostics(
        results_1,
        title=f"ABRA diagnostics: KL nonnegative regression, m={m}, n={n}",
    )

    m, n = 100, 1000
    results_2 = run_experiment(m, n, maxitrs=args.maxitrs, verbskip=args.verbskip)
    fig2 = plot_experiment(results_2, title=f"KL nonnegative regression: m={m}, n={n}")
    fig2_diag = plot_abra_diagnostics(
        results_2,
        title=f"ABRA diagnostics: KL nonnegative regression, m={m}, n={n}",
    )

    if args.save_dir is not None:
        save_figure(fig1, args.save_dir / "KL_regr_m1000n100.png")
        save_figure(fig1_diag, args.save_dir / "KL_regr_m1000n100_abra_diag.png")
        save_figure(fig2, args.save_dir / "KL_regr_m100n1000.png")
        save_figure(fig2_diag, args.save_dir / "KL_regr_m100n1000_abra_diag.png")

    if args.no_show:
        plt.close(fig1)
        plt.close(fig1_diag)
        plt.close(fig2)
        plt.close(fig2_diag)
    else:
        plt.show()


if __name__ == "__main__":
    main()
