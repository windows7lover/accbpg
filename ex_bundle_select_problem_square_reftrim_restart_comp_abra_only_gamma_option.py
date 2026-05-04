#!/usr/bin/env python3
"""
Bundle launcher: LIBSVM RelSC D-optimal design and Poisson inverse problems.

Use --problem to choose what to run:
- dopt: LIBSVM D-optimal datasets: abalone, bodyfat, mpg, housing
- poisson-l1: Poisson L1 bundle
- poisson-l2: Poisson L2 bundle
- poisson-all: Poisson L1 + Poisson L2
- all: D-opt + Poisson L1 + Poisson L2

For the LIBSVM D-optimal rows, we convexify the smooth part by replacing
    f(x) = f0(x) + mu * h(x),
    L    = L0 + mu,
where h is the Burg entropy / Legendre kernel used by the D-optimal problem.

For the Poisson rows, the original problem construction is used with mu=0.

Default behavior:
- ABRA-GD is run without restart.
- Use --restart to enable one restarted ABRA-GD variant, with --restart-rule in {g, f}.
- Use --restart-comp / --restart_comp to compare all ABRA-GD restart strategies.

Normal comparison mode:
- BPG
- BPG-LS
- ABPG
- ABPG-e
- ABPG-g
- ABRA-GD, optionally restarted if --restart is passed

Acceleration exponent:
- Use --gamma VALUE to override gamma everywhere acceleration uses an exponent:
  ABPG(gamma), ABPG-e(gamma0), and ABPG-g(gamma).
- If --gamma is omitted, the original per-problem defaults are kept.

Restart-comparison mode (--restart-comp):
- ABRA-GD without restart
- ABRA-GD with g-restart
- ABRA-GD with f-restart
- All three columns compare only these ABRA-GD variants.

Figure format:
- One paper-scale figure is produced, with one row per problem and three near-square columns:
    1. objective gap vs iteration
    2. objective gap vs oracle calls
    3. ABRA-GD M_k-like diagnostic vs iteration
- Axes are inferred adaptively for each row.
- Methods run for run_factor * maxitrs, but plots are trimmed to maxitrs;
  F_ref is computed from the longer histories to reduce end-point reference artifacts.
- Only the middle graph in each row has a title: "problem, mu=...".
- By default, figures are saved as PNG and EPS.
"""

from __future__ import annotations

import argparse
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import accbpg

matplotlib.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "font.family": "serif",
    }
)
# matplotlib.rcParams.update({"text.usetex": True})


LIBSVM_REGRESSION_BASE_URL = (
    "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression"
)

LIBSVM_DATASETS = {
    "abalone": "abalone_scale",
    "bodyfat": "bodyfat_scale",
    "mpg": "mpg_scale",
    "housing": "housing_scale",
}

POISSON_PROBLEMS = {"L1", "L2"}


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


@dataclass
class ExperimentResult:
    title: str
    mu: float
    labels: list[str]
    styles: list[str]
    dashes: list[list[int]]
    y_vals: list[np.ndarray]
    t_vals: list[np.ndarray]
    m_labels: list[str]
    m_styles: list[str]
    m_dashes: list[list[int]]
    m_vals: list[np.ndarray]
    f_ref: float
    restart: bool
    restart_rule: str
    restart_comp: bool
    plot_itrs: int
    run_itrs: int


def save_figure_formats(fig, base_path: Path, formats: list[str]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out = base_path.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if fmt not in {"eps", "pdf", "svg"}:
            kwargs["dpi"] = 300
        fig.savefig(out, **kwargs)
        print(f"Saved {out}")


def download_file(url: str, path: Path) -> None:
    """
    Robust downloader for Windows/Python SSL issues.

    Order:
    1. verified HTTPS
    2. unverified HTTPS
    3. plain HTTP fallback
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    attempts: list[tuple[str, ssl.SSLContext | None]] = [
        (url, None),
        (url, ssl._create_unverified_context()),
    ]
    if url.startswith("https://"):
        attempts.append(("http://" + url[len("https://") :], None))

    last_error: Exception | None = None
    for attempt_url, context in attempts:
        try:
            print(f"Downloading {attempt_url} -> {path}")
            with urlopen(attempt_url, context=context, timeout=60) as response:
                data = response.read()
            path.write_bytes(data)
            return
        except (URLError, OSError, TimeoutError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to download {url}.\n"
        f"Last error: {last_error}\n\n"
        f"Manual fallback from repo root:\n"
        f"  mkdir data\n"
        f"  curl.exe -L -k -o {path} {url}"
    )


def ensure_libsvm_dataset(dataset_key: str, data_dir: Path, *, download: bool) -> Path:
    if dataset_key not in LIBSVM_DATASETS:
        valid = ", ".join(LIBSVM_DATASETS)
        raise ValueError(f"Unknown LIBSVM dataset '{dataset_key}'. Valid choices: {valid}.")

    filename = LIBSVM_DATASETS[dataset_key]
    path = data_dir / filename

    if path.exists():
        return path

    if not download:
        raise FileNotFoundError(
            f"Missing dataset file: {path}. Re-run with --download or place it manually."
        )

    url = f"{LIBSVM_REGRESSION_BASE_URL}/{filename}"
    download_file(url, path)
    return path


def make_libsvm_relsc_problem(filename: str | Path, *, mu: float):
    f0, h, L0, x0 = accbpg.D_opt_libsvm(str(filename))
    f = RelStrongConvexified(f0, h, mu)
    L = L0 + mu
    return f, h, L, x0


def make_poisson_problem(kind: str):
    kind = kind.upper()
    if kind == "L1":
        m, n = 200, 100
        f, h, L, x0 = accbpg.Poisson_regrL1(
            m,
            n,
            noise=0.0001,
            lamda=0.1,
            randseed=1,
        )
        return f, h, L, x0, f"Poisson L1, m={m}, n={n}", 0.0, {
            "bpg_ls": {},
            "abpg": {"gamma": 2.0, "theta_eq": True},
            "expo": {"gamma0": 3, "theta_eq": False, "Gmargin": 3},
            "gain": {"gamma": 2, "G0": 0.1, "theta_eq": False},
        }

    if kind == "L2":
        m, n = 100, 1000
        f, h, L, x0 = accbpg.Poisson_regrL2(
            m,
            n,
            noise=0.001,
            lamda=0.001,
            randseed=1,
        )
        return f, h, L, x0, f"Poisson L2, m={m}, n={n}", 0.0, {
            "bpg_ls": {"ls_ratio": 1.5},
            "abpg": {"gamma": 2.0, "theta_eq": False},
            "expo": {"gamma0": 3, "theta_eq": False, "Gmargin": 1},
            "gain": {
                "gamma": 2,
                "G0": 0.1,
                "ls_inc": 1.5,
                "ls_dec": 1.5,
                "theta_eq": True,
            },
        }

    raise ValueError("Poisson kind must be 'L1' or 'L2'.")


def override_accel_gamma(method_kwargs: dict, accel_gamma: float | None) -> dict:
    """
    Optionally override all acceleration exponents with one value.

    This affects:
    - ABPG:      gamma
    - ABPG-e:    gamma0
    - ABPG-g:    gamma

    If accel_gamma is None, the original problem-specific defaults are kept.
    """
    out = {name: dict(kwargs) for name, kwargs in method_kwargs.items()}

    if accel_gamma is None:
        return out

    if "abpg" in out:
        out["abpg"]["gamma"] = float(accel_gamma)
    if "expo" in out:
        out["expo"]["gamma0"] = float(accel_gamma)
    if "gain" in out:
        out["gain"]["gamma"] = float(accel_gamma)

    return out


def finite_values(a) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr[np.isfinite(arr)]


def positive_finite(a) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    return arr[np.isfinite(arr) & (arr > 0)]


def compute_gaps(y_vals, f_ref=None):
    arrays = [np.asarray(y, dtype=float) for y in y_vals]

    if f_ref is None:
        finite_mins = [np.min(finite_values(a)) for a in arrays if finite_values(a).size]
        if not finite_mins:
            raise ValueError("No finite objective values found.")
        f_ref = min(finite_mins)

    gaps = []
    for a in arrays:
        g = a - f_ref
        # Log-scale cannot display zero or negative values.
        g = np.where(np.isfinite(g) & (g > 0), g, np.nan)
        gaps.append(g)

    return gaps, float(f_ref)


def trim_history(history, plot_itrs: int) -> np.ndarray:
    """Keep iterations 0, ..., plot_itrs when available."""
    arr = np.asarray(history, dtype=float)
    keep = min(arr.shape[0], max(plot_itrs, 0) + 1)
    return arr[:keep]


def run_length(plot_itrs: int, run_factor: float) -> int:
    """Number of iterations used to estimate F_ref more reliably."""
    if run_factor < 1.0:
        raise ValueError("run_factor must be >= 1.0.")
    return int(np.ceil(float(plot_itrs) * float(run_factor)))


def infer_x_upper(x_series, *, default=1.0):
    xmax = 0.0
    for x in x_series:
        arr = finite_values(x)
        if arr.size:
            xmax = max(xmax, float(np.max(arr)))
    return max(xmax, default)


def infer_log_ylim(series_list, *, default=(1e-12, 1e0)):
    vals = []
    for s in series_list:
        p = positive_finite(s)
        if p.size:
            vals.append(p)

    if not vals:
        return default

    allv = np.concatenate(vals)
    ymin = 10.0 ** np.floor(np.log10(np.min(allv)))
    ymax = 10.0 ** np.ceil(np.log10(np.max(allv)))

    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin <= 0:
        return default
    if ymin == ymax:
        ymax = 10.0 * ymin

    return float(ymin), float(ymax)


def plot_log_series(ax, x_vals, y_vals, labels, styles, dashes):
    for x, y, label, style, dash in zip(x_vals, y_vals, labels, styles, dashes):
        (line,) = ax.plot(x, y, style, label=label)
        if dash:
            line.set_dashes(dash)


def run_methods(
    *,
    f,
    h,
    L,
    x0,
    title: str,
    mu_for_abra: float,
    plot_itrs: int,
    run_factor: float,
    restart: bool,
    restart_rule: str,
    restart_comp: bool,
    verbskip: int,
    method_kwargs: dict,
) -> ExperimentResult:
    run_itrs = run_length(plot_itrs, run_factor)
    print(f"\n=== Experiment: {title} ===")
    print(f"Running {run_itrs} iterations; plotting first {plot_itrs} iterations.")

    if restart_comp:
        print("Restart-comparison mode: ABRA-GD, ABRA-GD g-RS, ABRA-GD f-RS.")

        xabra, Fabra, tk_abra, eta_abra, M_abra, alpha_abra, L_abra, Tabra = accbpg.ABRA_GD(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            mu=mu_for_abra,
            restart=False,
            verbskip=verbskip,
        )

        xabrag, Fabrag, tk_abrag, eta_abrag, M_abrag, alpha_abrag, L_abrag, Tabrag = accbpg.ABRA_GD(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            mu=mu_for_abra,
            restart=True,
            restart_rule="g",
            verbskip=verbskip,
        )

        xabraf, Fabraf, tk_abraf, eta_abraf, M_abraf, alpha_abraf, L_abraf, Tabraf = accbpg.ABRA_GD(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            mu=mu_for_abra,
            restart=True,
            restart_rule="f",
            verbskip=verbskip,
        )

        labels = [r"ABRA-GD", r"ABRA-GD g-RS", r"ABRA-GD f-RS"]
        styles = ["C2-", "C1--", "C3:"]
        dashes = [[], [4, 2], [1, 2]]

        y_vals_full = [Fabra, Fabrag, Fabraf]
        t_vals_full = [Tabra, Tabrag, Tabraf]

        m_labels = labels.copy()
        m_styles = styles.copy()
        m_dashes = [d.copy() for d in dashes]
        m_vals_full = [M_abra, M_abrag, M_abraf]

    else:
        bpg_ls_kwargs = method_kwargs.get("bpg_ls", {})
        abpg_kwargs = method_kwargs.get("abpg", {})
        expo_kwargs = method_kwargs.get("expo", {})
        gain_kwargs = method_kwargs.get("gain", {})

        x00, F00, _, T00 = accbpg.BPG(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            linesearch=False,
            verbskip=verbskip,
        )

        x20, F20, _, T20 = accbpg.ABPG(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            verbskip=verbskip,
            **abpg_kwargs,
        )

        xLS, FLS, _, TLS = accbpg.BPG(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            linesearch=True,
            verbskip=verbskip,
            **bpg_ls_kwargs,
        )

        x2e, F2e, _, _, T2e = accbpg.ABPG_expo(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            verbskip=verbskip,
            **expo_kwargs,
        )

        x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            verbskip=verbskip,
            **gain_kwargs,
        )

        xabra, Fabra, tk_abra, eta_abra, M_abra, alpha_abra, L_abra, Tabra = accbpg.ABRA_GD(
            f,
            h,
            L,
            x0,
            maxitrs=run_itrs,
            mu=mu_for_abra,
            restart=restart,
            restart_rule=restart_rule,
            verbskip=verbskip,
        )

        abra_label = rf"ABRA-GD {restart_rule}-RS" if restart else r"ABRA-GD"
        labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", abra_label]
        styles = ["k:", "g-", "b-.", "k-", "r--", "c-"]
        dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

        y_vals_full = [F00, FLS, F20, F2e, F2g, Fabra]
        t_vals_full = [T00, TLS, T20, T2e, T2g, Tabra]

        m_labels = [abra_label]
        m_styles = ["c-"]
        m_dashes = [[]]
        m_vals_full = [M_abra]

    # Use the full run_factor histories to estimate F_ref, then trim plots.
    _, f_ref = compute_gaps(y_vals_full)

    y_vals_plot = [trim_history(v, plot_itrs) for v in y_vals_full]
    t_vals_plot = [trim_history(v, plot_itrs) for v in t_vals_full]
    m_vals_plot = [trim_history(v, plot_itrs) for v in m_vals_full]

    return ExperimentResult(
        title=title,
        mu=mu_for_abra,
        labels=labels,
        styles=styles,
        dashes=dashes,
        y_vals=y_vals_plot,
        t_vals=t_vals_plot,
        m_labels=m_labels,
        m_styles=m_styles,
        m_dashes=m_dashes,
        m_vals=m_vals_plot,
        f_ref=f_ref,
        restart=restart,
        restart_rule=restart_rule,
        restart_comp=restart_comp,
        plot_itrs=plot_itrs,
        run_itrs=run_itrs,
    )


def run_one_libsvm_dataset(
    *,
    dataset_key: str,
    filename: Path,
    mu: float,
    plot_itrs: int,
    run_factor: float,
    restart: bool,
    restart_rule: str,
    restart_comp: bool,
    accel_gamma: float | None,
    verbskip: int,
) -> ExperimentResult:
    f, h, L, x0 = make_libsvm_relsc_problem(filename, mu=mu)
    title = dataset_key
    return run_methods(
        f=f,
        h=h,
        L=L,
        x0=x0,
        title=title,
        mu_for_abra=mu,
        plot_itrs=plot_itrs,
        run_factor=run_factor,
        restart=restart,
        restart_rule=restart_rule,
        restart_comp=restart_comp,
        verbskip=verbskip,
        method_kwargs=override_accel_gamma(
            {
                "bpg_ls": {"ls_ratio": 1.2},
                "abpg": {"gamma": 2.0, "theta_eq": True},
                "expo": {"gamma0": 3, "theta_eq": True, "Gmargin": 100},
                "gain": {"gamma": 2, "G0": 0.1, "theta_eq": True},
            },
            accel_gamma,
        ),
    )


def run_one_poisson_problem(
    *,
    kind: str,
    plot_itrs: int,
    run_factor: float,
    restart: bool,
    restart_rule: str,
    restart_comp: bool,
    accel_gamma: float | None,
    verbskip: int,
) -> ExperimentResult:
    f, h, L, x0, title, mu, method_kwargs = make_poisson_problem(kind)
    method_kwargs = override_accel_gamma(method_kwargs, accel_gamma)
    return run_methods(
        f=f,
        h=h,
        L=L,
        x0=x0,
        title=title,
        mu_for_abra=mu,
        plot_itrs=plot_itrs,
        run_factor=run_factor,
        restart=restart,
        restart_rule=restart_rule,
        restart_comp=restart_comp,
        verbskip=verbskip,
        method_kwargs=method_kwargs,
    )


def set_box_aspect_safe(ax, aspect: float | None) -> None:
    """Set subplot box aspect when supported by the installed Matplotlib."""
    if aspect is None or aspect <= 0:
        return
    try:
        ax.set_box_aspect(aspect)
    except AttributeError:
        # Older Matplotlib versions do not support set_box_aspect.
        pass


def make_big_figure(
    results: list[ExperimentResult],
    *,
    fig_width: float = 8.4,
    row_height: float = 2.35,
    panel_aspect: float | None = 1.0,
):
    if not results:
        raise ValueError("No experiment results to plot.")

    nrows = len(results)
    fig_height = max(row_height * nrows, 2.2)
    fig, axes = plt.subplots(nrows, 3, figsize=(fig_width, fig_height), squeeze=False)

    for row, result in enumerate(results):
        ax1, ax2, ax3 = axes[row]

        gaps, _ = compute_gaps(result.y_vals, f_ref=result.f_ref)
        iter_x = [np.arange(len(g), dtype=float) for g in gaps]
        oracle_x = [np.asarray(t, dtype=float) for t in result.t_vals]

        m_clean = []
        m_x = []
        for m in result.m_vals:
            m_arr = np.asarray(m, dtype=float)
            m_arr = np.where(np.isfinite(m_arr) & (m_arr > 0), m_arr, np.nan)
            m_clean.append(m_arr)
            m_x.append(np.arange(len(m_arr), dtype=float))

        gap_ylim = infer_log_ylim(gaps)
        m_ylim = infer_log_ylim(m_clean)
        iter_xlim = (0.0, infer_x_upper(iter_x))
        oracle_xlim = (0.0, infer_x_upper(oracle_x))
        m_xlim = (0.0, infer_x_upper(m_x))

        plot_log_series(ax1, iter_x, gaps, result.labels, result.styles, result.dashes)
        ax1.set_yscale("log")
        ax1.set_xlim(*iter_xlim)
        ax1.set_ylim(*gap_ylim)
        if row == nrows - 1:
            ax1.set_xlabel(r"Iteration number $k$")
        else:
            ax1.set_xlabel("")
            ax1.tick_params(labelbottom=False)
        ax1.set_ylabel(r"$F-F_{\mathrm{ref}}$")

        plot_log_series(ax2, oracle_x, gaps, result.labels, result.styles, result.dashes)
        ax2.set_yscale("log")
        ax2.set_xlim(*oracle_xlim)
        ax2.set_ylim(*gap_ylim)
        if row == nrows - 1:
            ax2.set_xlabel("Oracle calls")
        else:
            ax2.set_xlabel("")
            ax2.tick_params(labelbottom=False)
        ax2.set_ylabel("")
        ax2.set_title(fr"{result.title}, $\mu={result.mu:g}$", pad=2)

        comparison_handles = None
        comparison_labels = None
        if row == 0:
            comparison_handles, comparison_labels = ax2.get_legend_handles_labels()

        plot_log_series(ax3, m_x, m_clean, result.m_labels, result.m_styles, result.m_dashes)
        ax3.set_yscale("log")
        ax3.set_xlim(*m_xlim)
        ax3.set_ylim(*m_ylim)
        if row == nrows - 1:
            ax3.set_xlabel(r"Iteration number $k$")
        else:
            ax3.set_xlabel("")
            ax3.tick_params(labelbottom=False)
        ax3.set_ylabel(r"$M_k$")

        # Single global legend: top-right panel.
        if row == 0:
            if result.restart_comp:
                handles, labels = ax3.get_legend_handles_labels()
                ax3.legend(handles, labels, loc="best", frameon=False)
            elif comparison_handles is not None and comparison_labels is not None:
                ax3.legend(comparison_handles, comparison_labels, loc="best", frameon=False)

        for ax in (ax1, ax2, ax3):
            set_box_aspect_safe(ax, panel_aspect)
            ax.tick_params(axis="both", which="major", length=2.5, pad=1.5)
            ax.tick_params(axis="both", which="minor", length=1.5, pad=1.5)

    fig.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.06,
        top=0.975,
        wspace=0.30,
        hspace=0.38,
    )
    return fig


def parse_libsvm_dataset_list(raw: str) -> list[str]:
    raw = raw.strip().lower()
    if raw in {"", "none", "no", "false"}:
        return []
    if raw == "all":
        return list(LIBSVM_DATASETS.keys())

    out = []
    for item in raw.split(","):
        item = item.strip().lower()
        if item:
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run bundled experiments with one selectable problem family: D-opt, Poisson L1, Poisson L2, or all."
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        choices=["all", "dopt", "poisson-l1", "poisson-l2", "poisson-all"],
        help=(
            "Which problem family to run. "
            "'dopt' runs the LIBSVM D-optimal bundle; "
            "'poisson-l1' runs the Poisson L1 bundle; "
            "'poisson-l2' runs the Poisson L2 bundle; "
            "'poisson-all' runs both Poisson bundles; "
            "'all' runs D-opt + Poisson L1 + Poisson L2."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "LIBSVM D-optimal subset among abalone,bodyfat,mpg,housing, 'all', or 'none'. "
            "Used only when --problem is dopt or all."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing/downloading LIBSVM files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing LIBSVM files automatically.",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=1e-4,
        help="Relative strong convexity added to LIBSVM D-optimal rows as f <- f + mu * h.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help=(
            "Override the acceleration exponent everywhere it appears: "
            "ABPG gamma, ABPG-e gamma0, and ABPG-g gamma. "
            "If omitted, the original per-problem defaults are used. "
            "Use --gamma 1 for the exponent-one setting."
        ),
    )
    parser.add_argument(
        "--maxitrs",
        type=int,
        default=4000,
        help="Number of iterations shown in the plots.",
    )
    parser.add_argument(
        "--run-factor",
        type=float,
        default=1.3,
        help="Run each method this many times longer than maxitrs to compute F_ref, then trim the plots. Default: 1.3.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run only debug-itrs iterations. Default debug-itrs is 500.",
    )
    parser.add_argument(
        "--debug-itrs",
        type=int,
        default=500,
        help="Iterations used when --debug is enabled.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Enable one restarted ABRA-GD variant. Ignored when --restart-comp is used.",
    )
    parser.add_argument(
        "--restart-rule",
        type=str,
        default="g",
        choices=["g", "f"],
        help="Restart rule used only when --restart is enabled and --restart-comp is not used.",
    )
    parser.add_argument(
        "--restart-comp",
        "--restart_comp",
        dest="restart_comp",
        action="store_true",
        help=(
            "Compare only ABRA-GD without restart, with g-restart, and with f-restart. "
            "No BPG, BPG-LS, ABPG, ABPG-e, or ABPG-g baselines are included in this mode."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("figures_bundle"),
        help="Directory where the combined figure is saved.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=8.4,
        help="Figure width in inches. Default is tuned for near-square panels with three columns.",
    )
    parser.add_argument(
        "--row-height",
        type=float,
        default=2.35,
        help="Height in inches per experiment row. Increase this if panels are still too wide.",
    )
    parser.add_argument(
        "--panel-aspect",
        type=float,
        default=1.0,
        help="Axes box aspect height/width. 1.0 gives square subplot panels.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,eps",
        help="Comma-separated output formats, e.g. png,eps,pdf.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display figures.",
    )
    args = parser.parse_args()

    problem = args.problem.lower()

    if problem in {"dopt", "all"}:
        libsvm_keys = parse_libsvm_dataset_list(args.datasets)
    else:
        libsvm_keys = []

    if problem == "all":
        poisson_keys = ["L1", "L2"]
    elif problem == "poisson-all":
        poisson_keys = ["L1", "L2"]
    elif problem == "poisson-l1":
        poisson_keys = ["L1"]
    elif problem == "poisson-l2":
        poisson_keys = ["L2"]
    else:
        poisson_keys = []

    if not libsvm_keys and not poisson_keys:
        raise ValueError(
            "No experiments selected. Use --problem dopt, poisson-l1, poisson-l2, poisson-all, or all."
        )

    plot_itrs = args.debug_itrs if args.debug else args.maxitrs
    run_itrs_for_logging = run_length(plot_itrs, args.run_factor)
    verbskip = max(1, min(1000, run_itrs_for_logging // 5 if run_itrs_for_logging >= 5 else 1))

    if args.restart_comp and args.restart:
        print("Warning: --restart is ignored because --restart-comp is enabled.")

    results: list[ExperimentResult] = []

    for dataset_key in libsvm_keys:
        filename = ensure_libsvm_dataset(dataset_key, args.data_dir, download=args.download)
        result = run_one_libsvm_dataset(
            dataset_key=dataset_key,
            filename=filename,
            mu=args.mu,
            plot_itrs=plot_itrs,
            run_factor=args.run_factor,
            restart=args.restart,
            restart_rule=args.restart_rule,
            restart_comp=args.restart_comp,
            accel_gamma=args.gamma,
            verbskip=verbskip,
        )
        results.append(result)

    for poisson_key in poisson_keys:
        result = run_one_poisson_problem(
            kind=poisson_key,
            plot_itrs=plot_itrs,
            run_factor=args.run_factor,
            restart=args.restart,
            restart_rule=args.restart_rule,
            restart_comp=args.restart_comp,
            accel_gamma=args.gamma,
            verbskip=verbskip,
        )
        results.append(result)

    fig = make_big_figure(
        results,
        fig_width=args.fig_width,
        row_height=args.row_height,
        panel_aspect=args.panel_aspect,
    )

    suffix = "debug" if args.debug else f"{plot_itrs}shown_{args.run_factor:g}xrun"
    if args.restart_comp:
        restart_suffix = "restartComp"
    else:
        restart_suffix = f"{args.restart_rule}RS" if args.restart else "noRS"
    gamma_suffix = f"_gamma{args.gamma:g}" if args.gamma is not None else ""
    libsvm_suffix = "_".join(libsvm_keys) if libsvm_keys else "noDopt"
    poisson_suffix = "_".join(poisson_keys) if poisson_keys else "noPoisson"
    out_base = (
        args.save_dir
        / f"bundle_square_{problem}_{libsvm_suffix}_{poisson_suffix}_mu{args.mu:g}{gamma_suffix}_{restart_suffix}_{suffix}"
    )
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    save_figure_formats(fig, out_base, formats)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
