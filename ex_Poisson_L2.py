#!/usr/bin/env python3
"""
Example: Poisson linear inverse problems with random datasets.

Figures:
- left subplot: objective gap vs iteration
- right subplot: objective gap vs oracle calls
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


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def _lim(lim):
    """
    accbpg.plot_comparisons expects xlim/ylim to support len(...).
    Passing None crashes because plot_comparisons calls len(xlim).

    Use:
    - [] for automatic limits;
    - list(lim) for forced limits.
    """
    return [] if lim is None else list(lim)


def make_comparison_figure(
    y_vals,
    t_vals,
    labels,
    styles,
    dashes,
    *,
    title: str,
    iter_xlim=None,
    oracle_xlim=None,
    gap_ylim=None,
):
    if not (len(y_vals) == len(t_vals) == len(labels) == len(styles) == len(dashes)):
        raise ValueError(
            "Mismatched plotting inputs: y_vals, t_vals, labels, styles, "
            "and dashes must have the same length."
        )

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

    # Keep the legend only on the right plot.
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

    # Force limits again after plotting in case plot_comparisons modifies them.
    if iter_xlim is not None:
        ax1.set_xlim(*iter_xlim)
    if oracle_xlim is not None:
        ax2.set_xlim(*oracle_xlim)
    if gap_ylim is not None:
        ax1.set_ylim(*gap_ylim)
        ax2.set_ylim(*gap_ylim)

    fig.suptitle(title)
    plt.tight_layout(w_pad=4)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Poisson inverse-problem experiments."
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

    figs: list[tuple[plt.Figure, str]] = []

    labels = [
        r"BPG",
        r"BPG-LS",
        r"ABPG",
        r"ABPG-e",
        r"ABPG-g",
        r"ABRA-GD",
        # r"ABRA-GD g-RS",  # CRASH if Fabrag / Tabrag are not computed.
        # r"ABRA-GD f-RS",  # CRASH if Fabraf / Tabraf are not computed.
    ]

    styles = [
        "k:",
        "g-",
        "b-.",
        "k-",
        "r--",
        "c-",
        # "c--",
        # "c:",
    ]

    dashes = [
        [1, 2],
        [],
        [4, 2, 1, 2],
        [],
        [4, 2],
        [],
        # [2, 2],
        # [1, 1],
    ]

    # ================================================================
    # Experiment 1: Poisson L1
    # Forced axes:
    #   iterations:  [0, 8000]
    #   oracle calls: [0, 32000]
    #   objective gap: [1e-7, 1e0]
    # ================================================================

    m = 200
    n = 100

    f, h, L, x0 = accbpg.Poisson_regrL1(
        m,
        n,
        noise=0.0001,
        lamda=0.1,
        randseed=1,
    )

    x00, F00, _, T00 = accbpg.BPG(
        f,
        h,
        L,
        x0,
        maxitrs=8000,
        linesearch=False,
        verbskip=1000,
    )

    xLS, FLS, _, TLS = accbpg.BPG(
        f,
        h,
        L,
        x0,
        maxitrs=8000,
        linesearch=True,
        verbskip=1000,
    )

    x20, F20, _, T20 = accbpg.ABPG(
        f,
        h,
        L,
        x0,
        gamma=2.0,
        maxitrs=8000,
        theta_eq=True,
        verbskip=1000,
    )

    x2e, F2e, _, _, T2e = accbpg.ABPG_expo(
        f,
        h,
        L,
        x0,
        gamma0=3,
        maxitrs=8000,
        theta_eq=False,
        Gmargin=3,
        verbskip=1000,
    )

    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(
        f,
        h,
        L,
        x0,
        gamma=2,
        maxitrs=8000,
        G0=0.1,
        theta_eq=False,
        verbskip=1000,
    )

    xabra, Fabra, tk_abra, eta_abra, ck_abra, alpha_abra, L_abra, Tabra = accbpg.ABRA_GD(
        f,
        h,
        L,
        x0,
        maxitrs=8000,
        mu=0.0,
        restart=False,
        verbskip=1000,
    )

    # Optional restart variants.
    # CRASH if these stay commented but Fabrag/Fabraf/Tabrag/Tabraf are used below.
    # xabrag, Fabrag, tk_abrag, eta_abrag, ck_abrag, alpha_abrag, L_abrag, Tabrag = accbpg.ABRA_GD(
    #     f,
    #     h,
    #     L,
    #     x0,
    #     maxitrs=8000,
    #     mu=0.0,
    #     restart=True,
    #     restart_rule="g",
    #     verbskip=1000,
    # )
    # xabraf, Fabraf, tk_abraf, eta_abraf, ck_abraf, alpha_abraf, L_abraf, Tabraf = accbpg.ABRA_GD(
    #     f,
    #     h,
    #     L,
    #     x0,
    #     maxitrs=8000,
    #     mu=0.0,
    #     restart=True,
    #     restart_rule="f",
    #     verbskip=1000,
    # )

    y_vals = [
        F00,
        FLS,
        F20,
        F2e,
        F2g,
        Fabra,
        # Fabrag,  # CRASH if ABRA-GD g-RS was not run.
        # Fabraf,  # CRASH if ABRA-GD f-RS was not run.
    ]

    t_vals = [
        T00,
        TLS,
        T20,
        T2e,
        T2g,
        Tabra,
        # Tabrag,  # CRASH if ABRA-GD g-RS was not run.
        # Tabraf,  # CRASH if ABRA-GD f-RS was not run.
    ]

    fig = make_comparison_figure(
        y_vals,
        t_vals,
        labels,
        styles,
        dashes,
        title=f"Poisson L1: m={m}, n={n}",
        iter_xlim=(0, 8000),
        oracle_xlim=(0, 32000),
        gap_ylim=(1e-7, 1e0),
    )
    figs.append((fig, "Poisson_m200n100_adapt.png"))

    # ================================================================
    # Experiment 2: Poisson L2
    # Forced axes:
    #   iterations:  [0, 1000]
    #   oracle calls: [0, 4000]
    #   objective gap: [1e-6, 1e0]
    # ================================================================

    m2 = 100
    n2 = 1000

    f2, h2, L2, x02 = accbpg.Poisson_regrL2(
        m2,
        n2,
        noise=0.001,
        lamda=0.001,
        randseed=1,
    )

    x00_, F00_, _, T00_ = accbpg.BPG(
        f2,
        h2,
        L2,
        x02,
        maxitrs=8000,
        linesearch=False,
        verbskip=1000,
    )

    xLS_, FLS_, _, TLS_ = accbpg.BPG(
        f2,
        h2,
        L2,
        x02,
        maxitrs=8000,
        linesearch=True,
        ls_ratio=1.5,
        verbskip=1000,
    )

    x20_, F20_, _, T20_ = accbpg.ABPG(
        f2,
        h2,
        L2,
        x02,
        gamma=2.0,
        maxitrs=8000,
        theta_eq=False,
        verbskip=1000,
    )

    x2e_, F2e_, _, _, T2e_ = accbpg.ABPG_expo(
        f2,
        h2,
        L2,
        x02,
        gamma0=3,
        maxitrs=8000,
        theta_eq=False,
        Gmargin=1,
        verbskip=1000,
    )

    x2g_, F2g_, _, _, _, T2g_ = accbpg.ABPG_gain(
        f2,
        h2,
        L2,
        x02,
        gamma=2,
        maxitrs=8000,
        G0=0.1,
        ls_inc=1.5,
        ls_dec=1.5,
        theta_eq=True,
        verbskip=1000,
    )

    xabra_, Fabra_, tk_abra_, eta_abra_, ck_abra_, alpha_abra_, L_abra_, Tabra_ = accbpg.ABRA_GD(
        f2,
        h2,
        L2,
        x02,
        maxitrs=8000,
        mu=0.0,
        restart=False,
        verbskip=1000,
    )

    # Optional restart variants.
    # CRASH if these stay commented but Fabrag_/Fabraf_/Tabrag_/Tabraf_ are used below.
    # xabrag_, Fabrag_, tk_abrag_, eta_abrag_, ck_abrag_, alpha_abrag_, L_abrag_, Tabrag_ = accbpg.ABRA_GD(
    #     f2,
    #     h2,
    #     L2,
    #     x02,
    #     maxitrs=8000,
    #     mu=0.0,
    #     restart=True,
    #     restart_rule="g",
    #     verbskip=1000,
    # )
    # xabraf_, Fabraf_, tk_abraf_, eta_abraf_, ck_abraf_, alpha_abraf_, L_abraf_, Tabraf_ = accbpg.ABRA_GD(
    #     f2,
    #     h2,
    #     L2,
    #     x02,
    #     maxitrs=8000,
    #     mu=0.0,
    #     restart=True,
    #     restart_rule="f",
    #     verbskip=1000,
    # )

    y_vals = [
        F00_,
        FLS_,
        F20_,
        F2e_,
        F2g_,
        Fabra_,
        # Fabrag_,  # CRASH if ABRA-GD g-RS was not run.
        # Fabraf_,  # CRASH if ABRA-GD f-RS was not run.
    ]

    t_vals = [
        T00_,
        TLS_,
        T20_,
        T2e_,
        T2g_,
        Tabra_,
        # Tabrag_,  # CRASH if ABRA-GD g-RS was not run.
        # Tabraf_,  # CRASH if ABRA-GD f-RS was not run.
    ]

    fig = make_comparison_figure(
        y_vals,
        t_vals,
        labels,
        styles,
        dashes,
        title=f"Poisson L2: m={m2}, n={n2}",
        iter_xlim=(0, 1000),
        oracle_xlim=(0, 4000),
        gap_ylim=(1e-6, 1e0),
    )
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
