#!/usr/bin/env python3
"""
Example: relatively strongly convex D-optimal experiment design with a LIBSVM dataset.

We convexify the smooth part by replacing
    f(x) <- f0(x) + mu * h(x),
where h is the Burg entropy / Legendre kernel used by the D-optimal design problem.

If f0 is L0-smooth relative to h, then f = f0 + mu h is
(L0 + mu)-smooth and mu-strongly convex relative to h.

Kept methods:
- BPG
- BPG-LS
- ABPG
- ABPG-e
- ABPG-g
- ABRA-GD

Removed:
- restarted ABRA-GD variants
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


# ---------------------------------------------------------------------
# Forced comparison-plot axes
# ---------------------------------------------------------------------

ITER_XLIM = (0, 2500)
ORACLE_XLIM = (0, 7500)
GAP_YLIM = (1e-8, 1e1)


class RelStrongConvexified:
    """
    Wrap f0 into f = f0 + mu * h.

    This adds relative strong convexity mu w.r.t. h and increases the
    relative smoothness constant from L0 to L0 + mu.
    """

    def __init__(self, f0, h, mu: float):
        if mu < 0:
            raise ValueError("mu must be nonnegative.")
        self.f0 = f0
        self.h = h
        self.mu = float(mu)
        self.n = getattr(f0, "n", None)
        self.m = getattr(f0, "m", None)

    def __call__(self, x):
        return self.func_grad(x, flag=0)

    def gradient(self, x):
        return self.func_grad(x, flag=1)

    def func_grad(self, x, flag=2):
        if flag == 0:
            return self.f0(x) + self.mu * self.h(x)
        if flag == 1:
            return self.f0.gradient(x) + self.mu * self.h.gradient(x)

        f0x, g0x = self.f0.func_grad(x, flag=2)
        return f0x + self.mu * self.h(x), g0x + self.mu * self.h.gradient(x)


def _lim(lim):
    """
    accbpg.plot_comparisons calls len(xlim) / len(ylim), so None crashes.
    Use [] for automatic limits, otherwise list(lim).
    """
    return [] if lim is None else list(lim)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


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
    """
    Diagnostic plot for non-restarted ABRA-GD only.
    """

    fig, axes = plt.subplots(3, 2, figsize=(11, 10.5))

    all_styles = {
        "ABRA_GD": (r"ABRA-GD", "c-", []),
    }

    keys = [k for k in all_styles if k in results]
    if not keys:
        raise ValueError("No ABRA result found for diagnostics.")

    xvals = [np.arange(len(results[k]["t"])) for k in keys]

    t_series = [results[k]["t"] for k in keys]
    M_series = [results[k]["M"] for k in keys]
    alpha_series = [results[k]["alpha"] for k in keys]
    eta_series = [
        np.where(np.isfinite(results[k]["eta"]), results[k]["eta"], np.nan)
        for k in keys
    ]
    L_series = [results[k]["L"] for k in keys]

    panels = [
        (axes[0, 0], t_series, r"$t_k$", "linear"),
        (axes[0, 1], M_series, r"$M_k$", "log"),
        (axes[1, 0], alpha_series, r"$\alpha_k$", "log"),
        (axes[1, 1], eta_series, r"$\eta_k$", "log"),
        (axes[2, 0], L_series, r"$L_k$", "log"),
    ]

    xmax = max(max(len(s) - 1, 1) for s in t_series)
    xpad = max(1, int(np.ceil(0.02 * xmax)))
    xlim = (-xpad, xmax + xpad)

    for ax, series, ylabel, yscale in panels:
        for xi, yi, key in zip(xvals, series, keys):
            lab, sty, dash = all_styles[key]
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


def make_comparison_figure(
    y_vals,
    t_vals,
    labels,
    styles,
    dashes,
    *,
    title: str,
    iter_xlim=ITER_XLIM,
    oracle_xlim=ORACLE_XLIM,
    gap_ylim=GAP_YLIM,
):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax1 = axes[0]
    accbpg.plot_comparisons(
        ax1,
        y_vals,
        labels,
        x_vals=[],
        plotdiff=True,
        yscale="log",
        xlim=_lim(iter_xlim),
        ylim=_lim(gap_ylim),
        xlabel=r"Iteration number $k$",
        ylabel=r"$F(x_k)-F_\star$",
        legendloc="best",
        linestyles=styles,
        linedash=dashes,
    )
    ax1.set_title(title)

    # Keep the legend only on the right graph.
    left_legend = ax1.get_legend()
    if left_legend is not None:
        left_legend.remove()

    ax2 = axes[1]
    accbpg.plot_comparisons(
        ax2,
        y_vals,
        labels,
        x_vals=t_vals,
        plotdiff=True,
        yscale="log",
        xlim=_lim(oracle_xlim),
        ylim=_lim(gap_ylim),
        xlabel="Oracle calls",
        ylabel=r"$F(x_k)-F_\star$",
        legendloc="best",
        linestyles=styles,
        linedash=dashes,
    )
    ax2.set_title(title)

    # Force limits after plotting too, in case plot_comparisons overrides them.
    if iter_xlim is not None:
        ax1.set_xlim(*iter_xlim)
    if oracle_xlim is not None:
        ax2.set_xlim(*oracle_xlim)
    if gap_ylim is not None:
        ax1.set_ylim(*gap_ylim)
        ax2.set_ylim(*gap_ylim)

    plt.tight_layout(w_pad=4)
    return fig


def make_problem(filename: str, *, mu: float):
    f0, h, L0, x0 = accbpg.D_opt_libsvm(filename)
    f = RelStrongConvexified(f0, h, mu)
    L = L0 + mu
    return f, h, L, x0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run relatively strongly convex D-optimal design experiments on a LIBSVM dataset."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=r"data\housing.txt",
        help="LIBSVM dataset path.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1e-4,
        help="Relative strong convexity added as f <- f + mu * h.",
    )
    parser.add_argument(
        "--maxitrs",
        type=int,
        default=4000,
        help="Maximum number of iterations for each method.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory where figures are saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures.",
    )
    args = parser.parse_args()

    filename = args.filename
    title_name = Path(filename).stem
    mu = args.mu
    maxitrs = args.maxitrs

    f, h, L, x0 = make_problem(filename, mu=mu)

    xabra, Fabra, tk_abra, eta_abra, M_abra, alpha_abra, L_abra, Tabra = accbpg.ABRA_GD(
        f,
        h,
        L,
        x0,
        maxitrs=maxitrs,
        mu=mu,
        restart=False,
        verbskip=1000,
    )

    x00, F00, _, T00 = accbpg.BPG(
        f,
        h,
        L,
        x0,
        maxitrs=maxitrs,
        linesearch=False,
        verbskip=1000,
    )

    xLS, FLS, _, TLS = accbpg.BPG(
        f,
        h,
        L,
        x0,
        maxitrs=maxitrs,
        linesearch=True,
        ls_ratio=1.2,
        verbskip=1000,
    )

    x20, F20, _, T20 = accbpg.ABPG(
        f,
        h,
        L,
        x0,
        gamma=2.0,
        maxitrs=maxitrs,
        theta_eq=True,
        verbskip=1000,
    )

    x2e, F2e, _, _, T2e = accbpg.ABPG_expo(
        f,
        h,
        L,
        x0,
        gamma0=3,
        maxitrs=maxitrs,
        theta_eq=True,
        Gmargin=100,
        verbskip=1000,
    )

    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(
        f,
        h,
        L,
        x0,
        gamma=2,
        maxitrs=maxitrs,
        G0=0.1,
        theta_eq=True,
        verbskip=1000,
    )

    labels = [
        r"BPG",
        r"BPG-LS",
        r"ABPG",
        r"ABPG-e",
        r"ABPG-g",
        r"ABRA-GD",
    ]

    styles = ["k:", "g-", "b-.", "k-", "r--", "c-"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

    y_vals = [F00, FLS, F20, F2e, F2g, Fabra]
    t_vals = [T00, TLS, T20, T2e, T2g, Tabra]

    title = fr"{title_name}, $\mu={mu:g}$"

    fig = make_comparison_figure(
        y_vals,
        t_vals,
        labels,
        styles,
        dashes,
        title=title,
        iter_xlim=ITER_XLIM,
        oracle_xlim=ORACLE_XLIM,
        gap_ylim=GAP_YLIM,
    )

    abra_results = {
        "ABRA_GD": {
            "t": tk_abra,
            "M": M_abra,
            "alpha": alpha_abra,
            "eta": eta_abra,
            "L": L_abra,
        },
    }
    fig_diag = plot_abra_diagnostics(
        abra_results,
        title=f"ABRA diagnostics: {title_name}, mu={mu:g}",
    )

    if args.save_dir is not None:
        save_figure(fig, args.save_dir / f"{title_name}_relSC_mu{mu:g}_accel_no_restart.png")
        save_figure(fig_diag, args.save_dir / f"{title_name}_relSC_mu{mu:g}_abra_diag.png")

    if args.no_show:
        plt.close(fig)
        plt.close(fig_diag)
    else:
        plt.show()


if __name__ == "__main__":
    main()
