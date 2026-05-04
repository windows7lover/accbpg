#!/usr/bin/env python3
"""
Bundle experiment: relatively strongly convex D-optimal design on LIBSVM datasets.

Datasets:
- abalone_scale
- bodyfat_scale
- mpg_scale
- housing_scale

Construction:
    f(x) = f0(x) + mu * h(x)
    L    = L0 + mu

Default behavior:
- ABRA-GD is run without restart.
- Use --restart to enable restart, with --restart-rule in {g, f}.
- One paper-scale figure is produced, with one row per dataset and three near-square columns:
    1. objective gap vs iteration
    2. objective gap vs oracle calls
    3. ABRA-GD M_k vs iteration
- Axes are inferred adaptively for each row.
- Methods run for run_factor * maxitrs, but plots are trimmed to maxitrs;
  F_ref is computed from the longer histories to reduce end-point reference artifacts.
- Only the middle graph in each row has a title: "dataset, mu=...".
- By default, figures are saved as PNG and EPS with near-square subplot panels.
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

DATASETS = {
    "abalone": "abalone_scale",
    "bodyfat": "bodyfat_scale",
    "mpg": "mpg_scale",
    "housing": "housing_scale",
}


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
class DatasetResult:
    dataset_key: str
    mu: float
    labels: list[str]
    styles: list[str]
    dashes: list[list[int]]
    y_vals: list[np.ndarray]
    t_vals: list[np.ndarray]
    M_vals: np.ndarray
    f_ref: float
    restart: bool
    restart_rule: str
    plot_itrs: int
    run_itrs: int


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)


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


def ensure_dataset(dataset_key: str, data_dir: Path, *, download: bool) -> Path:
    if dataset_key not in DATASETS:
        valid = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset '{dataset_key}'. Valid choices: {valid}.")

    filename = DATASETS[dataset_key]
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


def make_problem(filename: str | Path, *, mu: float):
    f0, h, L0, x0 = accbpg.D_opt_libsvm(str(filename))
    f = RelStrongConvexified(f0, h, mu)
    L = L0 + mu
    return f, h, L, x0


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


def run_one_dataset(
    *,
    dataset_key: str,
    filename: Path,
    mu: float,
    plot_itrs: int,
    run_factor: float,
    restart: bool,
    restart_rule: str,
    verbskip: int,
) -> DatasetResult:
    run_itrs = run_length(plot_itrs, run_factor)
    print(f"\n=== Dataset: {dataset_key} ({filename}) ===")
    print(f"Running {run_itrs} iterations; plotting first {plot_itrs} iterations.")
    f, h, L, x0 = make_problem(filename, mu=mu)

    x00, F00, _, T00 = accbpg.BPG(
        f,
        h,
        L,
        x0,
        maxitrs=run_itrs,
        linesearch=False,
        verbskip=verbskip,
    )

    xLS, FLS, _, TLS = accbpg.BPG(
        f,
        h,
        L,
        x0,
        maxitrs=run_itrs,
        linesearch=True,
        ls_ratio=1.2,
        verbskip=verbskip,
    )

    x20, F20, _, T20 = accbpg.ABPG(
        f,
        h,
        L,
        x0,
        gamma=2.0,
        maxitrs=run_itrs,
        theta_eq=True,
        verbskip=verbskip,
    )

    x2e, F2e, _, _, T2e = accbpg.ABPG_expo(
        f,
        h,
        L,
        x0,
        gamma0=3,
        maxitrs=run_itrs,
        theta_eq=True,
        Gmargin=100,
        verbskip=verbskip,
    )

    x2g, F2g, _, _, _, T2g = accbpg.ABPG_gain(
        f,
        h,
        L,
        x0,
        gamma=2,
        maxitrs=run_itrs,
        G0=0.1,
        theta_eq=True,
        verbskip=verbskip,
    )

    xabra, Fabra, tk_abra, eta_abra, M_abra, alpha_abra, L_abra, Tabra = accbpg.ABRA_GD(
        f,
        h,
        L,
        x0,
        maxitrs=run_itrs,
        mu=mu,
        restart=restart,
        restart_rule=restart_rule,
        verbskip=verbskip,
    )

    abra_label = rf"ABRA-GD {restart_rule}-RS" if restart else r"ABRA-GD"
    labels = [r"BPG", r"BPG-LS", r"ABPG", r"ABPG-e", r"ABPG-g", abra_label]
    styles = ["k:", "g-", "b-.", "k-", "r--", "c-"]
    dashes = [[1, 2], [], [4, 2, 1, 2], [], [4, 2], []]

    # Use the full 1.3x histories to estimate F_ref, then trim plots.
    y_vals_full = [F00, FLS, F20, F2e, F2g, Fabra]
    t_vals_full = [T00, TLS, T20, T2e, T2g, Tabra]
    _, f_ref = compute_gaps(y_vals_full)

    y_vals_plot = [trim_history(v, plot_itrs) for v in y_vals_full]
    t_vals_plot = [trim_history(v, plot_itrs) for v in t_vals_full]
    M_plot = trim_history(M_abra, plot_itrs)

    return DatasetResult(
        dataset_key=dataset_key,
        mu=mu,
        labels=labels,
        styles=styles,
        dashes=dashes,
        y_vals=y_vals_plot,
        t_vals=t_vals_plot,
        M_vals=M_plot,
        f_ref=f_ref,
        restart=restart,
        restart_rule=restart_rule,
        plot_itrs=plot_itrs,
        run_itrs=run_itrs,
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
    results: list[DatasetResult],
    *,
    fig_width: float = 8.4,
    row_height: float = 2.35,
    panel_aspect: float | None = 1.0,
):
    if not results:
        raise ValueError("No dataset results to plot.")

    nrows = len(results)
    fig_height = max(row_height * nrows, 2.2)
    fig, axes = plt.subplots(nrows, 3, figsize=(fig_width, fig_height), squeeze=False)

    for row, result in enumerate(results):
        ax1, ax2, ax3 = axes[row]

        gaps, _ = compute_gaps(result.y_vals, f_ref=result.f_ref)
        iter_x = [np.arange(len(g), dtype=float) for g in gaps]
        oracle_x = [np.asarray(t, dtype=float) for t in result.t_vals]

        M = np.asarray(result.M_vals, dtype=float)
        M = np.where(np.isfinite(M) & (M > 0), M, np.nan)
        M_x = np.arange(len(M), dtype=float)

        gap_ylim = infer_log_ylim(gaps)
        M_ylim = infer_log_ylim([M])
        iter_xlim = (0.0, infer_x_upper(iter_x))
        oracle_xlim = (0.0, infer_x_upper(oracle_x))
        M_xlim = (0.0, infer_x_upper([M_x]))

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
        ax2.set_title(fr"{result.dataset_key}, $\mu={result.mu:g}$", pad=2)

        comparison_handles = None
        comparison_labels = None
        if row == 0:
            comparison_handles, comparison_labels = ax2.get_legend_handles_labels()

        abra_label = rf"ABRA-GD {result.restart_rule}-RS" if result.restart else r"ABRA-GD"
        ax3.plot(M_x, M, "c-", label=abra_label)
        ax3.set_yscale("log")
        ax3.set_xlim(*M_xlim)
        ax3.set_ylim(*M_ylim)
        if row == nrows - 1:
            ax3.set_xlabel(r"Iteration number $k$")
        else:
            ax3.set_xlabel("")
            ax3.tick_params(labelbottom=False)
        ax3.set_ylabel(r"$M_k$")

        if row == 0 and comparison_handles is not None and comparison_labels is not None:
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


def parse_dataset_list(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DATASETS.keys())

    out = []
    for item in raw.split(","):
        item = item.strip().lower()
        if item:
            out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bundle of RelSC D-optimal LIBSVM experiments."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="abalone,bodyfat,mpg,housing",
        help="Comma-separated subset among abalone,bodyfat,mpg,housing, or 'all'.",
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
        help="Relative strong convexity added as f <- f + mu * h.",
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
        help="Enable restarted ABRA-GD. Default is no restart.",
    )
    parser.add_argument(
        "--restart-rule",
        type=str,
        default="g",
        choices=["g", "f"],
        help="Restart rule used only when --restart is enabled.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("figures_libsvm_bundle"),
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
        help="Height in inches per dataset row. Increase this if panels are still too wide.",
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

    dataset_keys = parse_dataset_list(args.datasets)
    plot_itrs = args.debug_itrs if args.debug else args.maxitrs
    run_itrs_for_logging = run_length(plot_itrs, args.run_factor)
    verbskip = max(1, min(1000, run_itrs_for_logging // 5 if run_itrs_for_logging >= 5 else 1))

    results: list[DatasetResult] = []
    for dataset_key in dataset_keys:
        filename = ensure_dataset(dataset_key, args.data_dir, download=args.download)
        result = run_one_dataset(
            dataset_key=dataset_key,
            filename=filename,
            mu=args.mu,
            plot_itrs=plot_itrs,
            run_factor=args.run_factor,
            restart=args.restart,
            restart_rule=args.restart_rule,
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
    restart_suffix = f"{args.restart_rule}RS" if args.restart else "noRS"
    dataset_suffix = "_".join(dataset_keys)
    out_base = args.save_dir / f"bundle_square_{dataset_suffix}_relSC_mu{args.mu:g}_{restart_suffix}_{suffix}"
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]

    if args.save_dir is not None:
        save_figure_formats(fig, out_base, formats)

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
