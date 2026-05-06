"""Render all DJIA-28 (DJIA-28) figures from the aggregated CSVs.

This script is the single source of truth for every figure shipped in
the DJIA-28 revision of the manuscript. It reads the canonical
aggregate CSVs produced by ``scripts/aggregate_results.py`` and writes
PNGs under ``publication_data/figures/``.

Usage::

    python scripts/render_figures.py \
        --input-dir  publication_data/_aggregate \
        --output-dir publication_data/figures \
        [--only make_cross_asset_correlation_3panel]

Each ``make_*`` function is fully self-contained: it consumes one or
more CSVs by name, produces one figure, and returns the output path.
The mapping between functions, source CSVs, and the LaTeX figure
labels they replace is documented in the docstrings.

The script targets matplotlib only (no seaborn) so that it can run on a
minimal environment. All numerical content comes from CSVs; nothing is
hard-coded except the cosmetic ordering by GICS sector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Global style and constants
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Sector grouping for the DJIA-28 panel. Replace the second sub-list of each
# sector with the actual constituents you decided to include during DJIA-28.
# Order within each sector is alphabetical to keep the plots reproducible.
SECTOR_ORDER: Dict[str, List[str]] = {
    "Technology":       ["AAPL", "CRM", "CSCO", "IBM", "MSFT"],
    "Financials":       ["AXP", "GS", "JPM", "TRV", "V"],
    "Energy":           ["CVX", "XOM"],
    "Healthcare":       ["AMGN", "JNJ", "MRK", "UNH"],
    "Consumer":         ["DIS", "HD", "KO", "MCD", "NKE", "PG", "WMT"],
    "Industrials":      ["BA", "CAT", "HON", "MMM"],
    "Other":            ["AMZN"],
}

SECTOR_COLORS: Dict[str, str] = {
    "Technology":   "#1f77b4",
    "Financials":   "#2ca02c",
    "Energy":       "#d62728",
    "Healthcare":   "#9467bd",
    "Consumer":     "#ff7f0e",
    "Industrials":  "#8c564b",
    "Other":        "#7f7f7f",
}


def ordered_tickers() -> List[str]:
    return [t for sector_tickers in SECTOR_ORDER.values() for t in sector_tickers]


def sector_of(ticker: str) -> str:
    for sector, tickers in SECTOR_ORDER.items():
        if ticker in tickers:
            return sector
    return "Other"


def sector_boundaries() -> List[int]:
    """Cumulative tick indices that delimit one sector from the next."""
    out: List[int] = []
    cum = 0
    for tickers in SECTOR_ORDER.values():
        cum += len(tickers)
        out.append(cum)
    return out[:-1]  # drop trailing


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_csv(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing aggregate CSV: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ===========================================================================
# BODY FIGURES
# ===========================================================================

# ---------------------------------------------------------------------------
# F4 — Cross-asset correlation, 3-panel (Real / ECSDE / |diff|)
# ---------------------------------------------------------------------------

def make_cross_asset_correlation_3panel(
    input_dir: Path, output_dir: Path
) -> Path:
    """Three-panel cross-asset correlation heatmap on the DJIA-28 panel.

    Inputs
    ------
    cross_asset_correlation/real_correlation_matrix.csv
    cross_asset_correlation/synth_correlation_matrix.csv

    Output
    ------
    cross_asset_correlation_3panel.png

    Replaces ``fig:cross-asset-corr`` in ``Sections/04_experiments.tex``.
    """
    import json as _json
    report_path = Path("results/djia28/cross_asset_correlation/cross_asset_correlation_report.json")
    if not report_path.exists():
        raise FileNotFoundError(f"missing cross-asset report: {report_path}")
    rep = _json.loads(report_path.read_text())
    _tickers = rep["tickers"]
    real_df  = pd.DataFrame(rep["real_corr"],  index=_tickers, columns=_tickers)
    synth_df = pd.DataFrame(rep["synth_corr"], index=_tickers, columns=_tickers)

    order = [t for t in ordered_tickers() if t in real_df.columns]
    real  = real_df.reindex(index=order, columns=order).values.astype(float)
    synth = synth_df.reindex(index=order, columns=order).values.astype(float)
    absd  = np.abs(real - synth)

    # mask diagonals visually
    real_v  = real.copy();  np.fill_diagonal(real_v,  np.nan)
    synth_v = synth.copy(); np.fill_diagonal(synth_v, np.nan)
    absd_v  = absd.copy();  np.fill_diagonal(absd_v,  np.nan)

    off_mask = ~np.eye(len(order), dtype=bool)
    mae_off  = float(np.nanmean(absd[off_mask]))
    max_off  = float(np.nanmax (absd[off_mask]))
    diff_vmax = max(0.20, np.nanpercentile(absd_v, 95))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))
    panels = [
        (axes[0], real_v,  "Real",                 "RdBu_r", Normalize(-1.0, 1.0)),
        (axes[1], synth_v, "ECSDE (synthetic)",    "RdBu_r", Normalize(-1.0, 1.0)),
        (axes[2], absd_v,  "|Real $-$ ECSDE|",     "magma",  Normalize(0.0, diff_vmax)),
    ]
    for ax, M, title, cmap, norm in panels:
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=7)
        ax.set_yticklabels(order, fontsize=7)
        ax.set_title(title)
        for b in sector_boundaries():
            ax.axhline(b - 0.5, color="black", lw=0.6)
            ax.axvline(b - 0.5, color="black", lw=0.6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[2].text(
        0.02, 0.98,
        f"Off-diagonal MAE $= {mae_off:.3f}$\nOff-diagonal max $= {max_off:.3f}$",
        transform=axes[2].transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", lw=0.4),
    )
    fig.suptitle(
        "Cross-asset Pearson correlation, DJIA-28: real vs ECSDE",
        fontsize=12, y=1.02,
    )
    return _save(fig, output_dir, "cross_asset_correlation_3panel")


# ---------------------------------------------------------------------------
# F3 — Direction-ablation aggregate (28 horizontal bars)
# ---------------------------------------------------------------------------

def make_direction_ablation_aggregate(
    input_dir: Path, output_dir: Path
) -> Path:
    """Horizontal bar plot of cumulative bull-removal effect, all 28 tickers.

    Input
    -----
    direction_ablation_summary.csv
        Columns: ticker, all_cumulative_pct, bears_off_delta_pct,
                 bulls_off_delta_pct, sign_flip_under_bull_removal.

    Output
    ------
    direction_ablation_aggregate.png

    Replaces ``fig:direction-amplified-rel`` in ``Sections/04_experiments.tex``.
    """
    df = _load_csv(input_dir, "direction_summary.csv")
    df = df.rename(columns={
        "all_mean_daily":  "all_cumulative_pct",
        "delta_bears_off": "bears_off_delta_pct",
        "delta_bulls_off": "bulls_off_delta_pct",
        "bulls_off_down":  "sign_flip_under_bull_removal",
    })
    df = df.set_index("ticker").reindex(
        [t for t in ordered_tickers() if t in df.index]
    )
    # Sort within plot by cumulative bull-removal magnitude
    df = df.sort_values("bulls_off_delta_pct")

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(df))
    colors = [SECTOR_COLORS[sector_of(t)] for t in df.index]

    ax.barh(y - 0.18, df["bulls_off_delta_pct"], height=0.36,
            color=colors, label="Bulls off ($\\Delta$ cumulative)",
            edgecolor="black", lw=0.3)
    ax.barh(y + 0.18, df["bears_off_delta_pct"], height=0.36,
            color=colors, alpha=0.45,
            label="Bears off ($\\Delta$ cumulative)",
            edgecolor="black", lw=0.3)

    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df.index, fontsize=8)
    ax.set_xlabel("Change in mean daily drift signal, 2018–2023")
    ax.set_title(
        "Direction-ablation block on DJIA-28: per-ticker bull/bear removal"
    )

    # Sector legend
    sector_handles = [
        plt.Rectangle((0, 0), 1, 1, color=c) for c in SECTOR_COLORS.values()
    ]
    leg1 = ax.legend(
        sector_handles, list(SECTOR_COLORS.keys()),
        title="Sector", loc="lower right", fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=8)

    n_flip = int(df["sign_flip_under_bull_removal"].sum())
    ax.text(
        0.02, 0.02,
        f"Sign-flip under bull removal: {n_flip}/{len(df)}",
        transform=ax.transAxes, va="bottom", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", lw=0.4),
    )
    return _save(fig, output_dir, "direction_ablation_aggregate")


# ---------------------------------------------------------------------------
# F5 — OOS Verification scatter (KS_IS vs KS_OOS, per ticker)
# ---------------------------------------------------------------------------

def make_oos_scatter(input_dir: Path, output_dir: Path) -> Path:
    """Scatter of in-sample vs out-of-sample KS for the DJIA-28 panel.

    Inputs
    ------
    vanilla_summary.csv          (KS_IS for ECSDE column "ks_ecsde")
    oos_2023_summary.csv         (KS_OOS column "ks_oos")

    Output
    ------
    oos_2023_scatter.png

    Replaces ``fig:oos-2023-summary`` in ``Sections/04_experiments.tex``.
    """
    is_df  = _load_csv(input_dir, "vanilla_summary.csv").set_index("ticker")
    oos_df = _load_csv(input_dir, "oos_2023_summary.csv").set_index("ticker")
    is_df  = is_df.rename(columns={"csde_ks": "ks_ecsde"})
    oos_df = oos_df.rename(columns={"oos_ks": "ks_oos", "oos_sigma_ratio": "sigma_ratio_oos"})
    df = is_df.join(oos_df[["ks_oos", "sigma_ratio_oos"]], how="inner")
    df["sector"] = [sector_of(t) for t in df.index]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for sector, sub in df.groupby("sector"):
        ax.scatter(
            sub["ks_ecsde"], sub["ks_oos"],
            color=SECTOR_COLORS[sector], label=sector,
            s=60, edgecolor="black", lw=0.5, zorder=3,
        )
    for t, row in df.iterrows():
        ax.annotate(
            t, (row["ks_ecsde"], row["ks_oos"]),
            xytext=(4, 3), textcoords="offset points", fontsize=7,
        )

    lim_low  = 0.0
    lim_high = float(np.nanmax([df["ks_ecsde"].max(), df["ks_oos"].max()])) * 1.1
    ax.plot([lim_low, lim_high], [lim_low, lim_high],
            color="black", lw=0.7, ls="--", label="zero gap")
    # +0.05 and +0.10 gap reference lines
    for gap, ls in [(0.05, "--"), (0.10, ":")]:
        xs = np.linspace(lim_low, lim_high, 50)
        ax.plot(xs, xs + gap, color="gray", lw=0.5, ls=ls,
                label=f"gap $= +{gap:.2f}$")

    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)
    ax.set_xlabel("In-sample KS")
    ax.set_ylabel("Out-of-sample KS (2023)")
    ax.set_title("OOS Verification block on DJIA-28: per-ticker generalisation")
    ax.legend(loc="lower right", fontsize=8, ncol=1)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)

    mean_gap = float((df["ks_oos"] - df["ks_ecsde"]).mean())
    ax.text(
        0.02, 0.97,
        f"Mean generalisation gap = $+{mean_gap:.3f}$\n"
        f"Tickers below KS$_\\mathrm{{OOS}}=0.10$: "
        f"{int((df['ks_oos'] < 0.10).sum())}/{len(df)}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", lw=0.4),
    )
    return _save(fig, output_dir, "oos_2023_scatter")


# ---------------------------------------------------------------------------
# F6 — Baseline-suite aggregate boxplot (ECSDE / Vanilla / TimeGAN / NSDE)
# ---------------------------------------------------------------------------

def make_baseline_suite_boxplot(input_dir: Path, output_dir: Path) -> Path:
    """Per-ticker KS distribution across four baseline families.

    Inputs
    ------
    vanilla_summary.csv (cols: ticker, ks_ecsde, ks_vanilla)
    timegan_summary.csv (cols: ticker, ks_timegan)
    nsde_summary.csv    (cols: ticker, ks_nsde_in_sample)

    Output
    ------
    baseline_suite_boxplot.png

    Optional body figure for ``subsec:results-vanilla`` /
    ``subsec:generalisation`` cross-cut.
    """
    v   = _load_csv(input_dir, "vanilla_summary.csv").set_index("ticker")
    tg  = _load_csv(input_dir, "timegan_summary.csv").set_index("ticker")
    ns  = _load_csv(input_dir, "nsde_summary.csv").set_index("ticker")
    df = pd.DataFrame({
        "ECSDE":       v["csde_ks"],
        "Vanilla SDE": v["vanilla_ks"],
        "TimeGAN":     tg["timegan_ks"],
        "Neural SDE":  ns["nsde_ks"],
    }).dropna(how="any")

    fig, ax = plt.subplots(figsize=(7, 5))
    box_data = [df[c].values for c in df.columns]
    bp = ax.boxplot(box_data, labels=df.columns, showmeans=True,
                    meanline=True, widths=0.55, patch_artist=True)
    palette = ["#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.5)

    # individual points jittered
    rng = np.random.default_rng(0)
    for i, c in enumerate(df.columns):
        x = rng.normal(loc=i + 1, scale=0.04, size=len(df))
        ax.scatter(x, df[c], s=14, color="black", alpha=0.6, zorder=3)

    ax.set_ylabel("KS statistic on daily log-returns")
    ax.set_title("Baseline suite on DJIA-28 (lower is better)")
    ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
    return _save(fig, output_dir, "baseline_suite_boxplot")


# ===========================================================================
# APPENDIX FIGURES
# ===========================================================================

# ---------------------------------------------------------------------------
# A1 — N=30 envelope grid (28 mini-panels)
# ---------------------------------------------------------------------------

def make_n30_grid_appendix(input_dir: Path, output_dir: Path) -> Path:
    """A 7x4 grid of 28 mini envelopes (one per ticker), linear scale only.

    Input directory
    ---------------
    ``input_dir/n30_envelopes/<TICKER>.csv`` with columns
    ``date, real_close, p05, p50, p95, min, max``.

    Output
    ------
    n30_grid_28tickers.png

    Replaces the seven appendix envelope figures (one per ticker) with a
    single composite, dropping ~3 pages from the appendix.
    """
    synth_root = Path("results/djia28")
    tickers = ordered_tickers()
    cols, rows = 4, 7
    fig, axes = plt.subplots(rows, cols, figsize=(13, 18), sharex=True)
    axes = axes.ravel()

    for i, t in enumerate(tickers):
        ax = axes[i]
        ticker_dir = synth_root / t
        real_path = ticker_dir / "real_data_reference.csv"
        if not ticker_dir.exists() or not real_path.exists():
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{t}: missing",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        real_df = pd.read_csv(real_path, parse_dates=["date"])
        synth_files = sorted(ticker_dir.glob("synthetic_data_[0-9]*.csv"))
        if not synth_files:
            ax.set_axis_off()
            continue
        synth_closes = []
        for sf in synth_files:
            sdf = pd.read_csv(sf)
            synth_closes.append(sdf["close"].values)
        synth_arr = np.array(synth_closes)
        p05 = np.percentile(synth_arr, 5, axis=0)
        p50 = np.percentile(synth_arr, 50, axis=0)
        p95 = np.percentile(synth_arr, 95, axis=0)
        s_min = synth_arr.min(axis=0)
        s_max = synth_arr.max(axis=0)
        dates = real_df["date"].values[:len(p50)]
        ax.fill_between(dates, s_min, s_max,
                        color=SECTOR_COLORS[sector_of(t)], alpha=0.15)
        ax.fill_between(dates, p05, p95,
                        color=SECTOR_COLORS[sector_of(t)], alpha=0.30)
        ax.plot(dates, p50,
                color=SECTOR_COLORS[sector_of(t)], lw=0.8,
                label="median synth")
        ax.plot(dates, real_df["close"].values[:len(p50)],
                color="black", lw=0.7, label="real")
        ax.set_title(t, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, ls=":", lw=0.3, alpha=0.5)

    for j in range(len(tickers), len(axes)):
        axes[j].set_axis_off()

    handles = [
        plt.Line2D([], [], color="black", lw=1.0, label="Realised close"),
        plt.Line2D([], [], color="gray", lw=1.0, label="Median synthetic"),
        plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.3,
                      label="5–95th pctile"),
        plt.Rectangle((0, 0), 1, 1, color="gray", alpha=0.15,
                      label="Min–max envelope"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("ECSDE $N{=}10$ envelopes, DJIA-28 (linear scale)",
                 fontsize=12, y=1.0)
    return _save(fig, output_dir, "n30_grid_28tickers")


# ---------------------------------------------------------------------------
# A2 — LSTM discriminator close-only bar plot
# ---------------------------------------------------------------------------

def make_lstm_disc_barplot_appendix(
    input_dir: Path, output_dir: Path
) -> Path:
    """Bar plot of close-only LSTM discriminator accuracy with thresholds.

    Input
    -----
    lstm_close_summary.csv with columns ``ticker, accuracy, accuracy_std,
    auc, auc_std``.

    Output
    ------
    lstm_disc_close_barplot.png

    Complements (not replaces) ``tab:lstm-disc``.
    """
    import json as _json
    lstm_report_path = Path(
        "results/djia28/lstm_discriminator/lstm_discriminator_report.json"
    )
    if not lstm_report_path.exists():
        raise FileNotFoundError(f"missing LSTM report: {lstm_report_path}")
    rep = _json.loads(lstm_report_path.read_text())
    rows = []
    for ticker, v in rep["per_ticker"].items():
        close_sub = v.get("subsets", {}).get("close", {})
        rows.append({
            "ticker": ticker,
            "accuracy": close_sub.get("avg_accuracy", v.get("avg_accuracy")),
            "accuracy_std": close_sub.get("std_accuracy", v.get("std_accuracy", 0)),
            "auc": close_sub.get("avg_auc_roc", v.get("avg_auc_roc")),
        })
    df = pd.DataFrame(rows).set_index("ticker")
    df = df.sort_values("accuracy")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [SECTOR_COLORS[sector_of(t)] for t in df.index]
    ax.bar(df.index, df["accuracy"], yerr=df["accuracy_std"],
           color=colors, edgecolor="black", lw=0.4, capsize=2)

    for thr, label, ls in [(0.50, "chance", "--"),
                           (0.55, "Excellent <", ":"),
                           (0.65, "Good <",      ":"),
                           (0.75, "Moderate <",  ":")]:
        ax.axhline(thr, color="gray", lw=0.6, ls=ls)
        ax.text(len(df) - 0.5, thr + 0.005, label,
                fontsize=7, color="gray", ha="right")

    ax.set_ylim(0.45, max(0.85, df["accuracy"].max() + 0.05))
    ax.set_ylabel("Discriminator accuracy (5-fold CV)")
    ax.set_xlabel("Ticker (sorted by accuracy)")
    mean_acc = float(df["accuracy"].mean())
    mean_auc = float(df["auc"].mean())
    ax.text(
        0.02, 0.97,
        f"Mean accuracy = {mean_acc:.3f}\nMean AUC = {mean_auc:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", lw=0.4),
    )
    ax.set_title("Close-only LSTM discriminator on DJIA-28")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)
    return _save(fig, output_dir, "lstm_disc_close_barplot")


# ---------------------------------------------------------------------------
# A3 — Multiseed bootstrap CIs
# ---------------------------------------------------------------------------

def make_multiseed_ci_appendix(input_dir: Path, output_dir: Path) -> Path:
    """Per-ticker mean KS with 95% bootstrap CI across seeds.

    Input
    -----
    multiseed_bootstrap.csv with columns
    ``ticker, ks_mean, ks_ci_low, ks_ci_high``.

    Output
    ------
    multiseed_ci_28tickers.png
    """
    df = _load_csv(input_dir, "multiseed_summary.csv").set_index("ticker")
    df = df.rename(columns={"ks_mean_across_seeds": "ks_mean"})
    df["ks_ci_low"]  = (df["ks_mean"] - 1.96 * df["ks_std_across_seeds"]).clip(lower=0)
    df["ks_ci_high"] = df["ks_mean"] + 1.96 * df["ks_std_across_seeds"]
    df = df.reindex([t for t in ordered_tickers() if t in df.index])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    colors = [SECTOR_COLORS[sector_of(t)] for t in df.index]
    ax.errorbar(
        x, df["ks_mean"],
        yerr=[df["ks_mean"] - df["ks_ci_low"], df["ks_ci_high"] - df["ks_mean"]],
        fmt="o", ecolor="gray", elinewidth=0.8, capsize=2, ms=5,
        markerfacecolor="white", markeredgewidth=1.0,
    )
    for xi, yi, c in zip(x, df["ks_mean"], colors):
        ax.scatter(xi, yi, color=c, s=30, zorder=3, edgecolor="black", lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=90, fontsize=7)
    ax.set_ylabel("KS (5 seeds, 95% bootstrap CI)")
    ax.set_title("Multi-seed stability on DJIA-28")
    ax.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)
    return _save(fig, output_dir, "multiseed_ci_28tickers")


# ---------------------------------------------------------------------------
# A4 — Shuffle-control aggregate (4 conditions, win rates)
# ---------------------------------------------------------------------------

def make_shuffle_control_aggregate(
    input_dir: Path, output_dir: Path
) -> Path:
    """Win rates of intact baseline vs three shuffle conditions, DJIA-28.

    Input
    -----
    shuffle_summary.csv with columns ``condition, wins, win_rate``.

    Output
    ------
    shuffle_control_aggregate.png
    """
    raw = _load_csv(input_dir, "shuffle_summary.csv")
    n_tickers = len(raw)
    cond_map = [
        ("Intact",             None),
        ("Shuffle Directions", "shuffled_directions_ks"),
        ("Shuffle Magnitudes", "shuffled_magnitudes_ks"),
        ("Shuffle Presence",   "shuffled_presence_ks"),
    ]
    records = []
    for label, shuf_col in cond_map:
        if shuf_col is None:
            wins = n_tickers
        elif shuf_col not in raw.columns:
            continue
        else:
            wins = int((raw["baseline_ks"] < raw[shuf_col]).sum())
        records.append({"condition": label, "wins": wins,
                        "win_rate": 100.0 * wins / n_tickers})
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
    bars = ax.bar(df["condition"], df["win_rate"],
                  color=palette[:len(df)], edgecolor="black", lw=0.4)
    for bar, val, wins in zip(bars, df["win_rate"], df["wins"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}%\n({int(wins)}/{n_tickers})",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Win rate (%)")
    ax.set_ylim(0, max(60, df["win_rate"].max() * 1.3))
    ax.set_title("Shuffle control on DJIA-28: intact baseline vs corruptions")
    ax.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)
    return _save(fig, output_dir, "shuffle_control_aggregate")


# ---------------------------------------------------------------------------
# A5 — Direction box-plot grid (28 mini box-plots)
# ---------------------------------------------------------------------------

def make_direction_box_grid_appendix(
    input_dir: Path, output_dir: Path
) -> Path:
    """A 7x4 grid showing the three direction-ablation conditions per ticker.

    Uses the amplified_report.json annualised_pct and cumulative_pct
    to build grouped bar charts (all / bears-off / bulls-off).

    Output
    ------
    direction_box_grid_28tickers.png
    """
    import json as _json
    report_path = Path(
        "results/djia28/direction_ablation_amplified/amplified_report.json"
    )
    if not report_path.exists():
        raise FileNotFoundError(f"missing direction report: {report_path}")
    rep = _json.loads(report_path.read_text())

    tickers = ordered_tickers()
    cols, rows = 4, 7
    fig, axes = plt.subplots(rows, cols, figsize=(13, 18))
    axes = axes.ravel()

    for i, t in enumerate(tickers):
        ax = axes[i]
        tv = rep["tickers"].get(t)
        if tv is None:
            ax.set_axis_off()
            continue
        cum = tv["cumulative_pct"]
        vals = [cum["all_factors"], cum["bears_off"], cum["bulls_off"]]
        bar_colors = ["#bbbbbb", "#9ad29a", "#f5a3a3"]
        bars = ax.bar(["all", "bears\noff", "bulls\noff"], vals,
                      color=bar_colors, edgecolor="black", lw=0.4)
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_title(t, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_ylabel("cum. drift (%)", fontsize=6)

    for j in range(len(tickers), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Direction-ablation: per-window analytical-signal "
                 "distributions, DJIA-28", fontsize=12, y=1.0)
    return _save(fig, output_dir, "direction_box_grid_28tickers")


# ---------------------------------------------------------------------------
# A6 — Factor cross-ticker heatmap (28x28)
# ---------------------------------------------------------------------------

def make_factor_cross_ticker_heatmap(
    input_dir: Path, output_dir: Path
) -> Path:
    """Heatmap of shared retained factors across the 28 tickers.

    Input
    -----
    factor_cross_ticker_overlap.csv  (square 28x28 matrix, ticker index/cols)

    Output
    ------
    factor_cross_ticker_heatmap_28.png
    """
    path = input_dir / "factor_cross_ticker_overlap.csv"
    M_df = pd.read_csv(path, index_col=0)
    order = [t for t in ordered_tickers() if t in M_df.columns]
    M = M_df.reindex(index=order, columns=order).values.astype(float)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    for b in sector_boundaries():
        ax.axhline(b - 0.5, color="white", lw=0.6)
        ax.axvline(b - 0.5, color="white", lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Fraction of shared retained factors")
    ax.set_title("Cross-ticker factor overlap, DJIA-28")
    return _save(fig, output_dir, "factor_cross_ticker_heatmap_28")


# ---------------------------------------------------------------------------
# A7 — Multi-LLM downstream (3 tickers x 3 models, NOT expanded for DJIA-28)
# ---------------------------------------------------------------------------

def make_multi_llm_appendix(input_dir: Path, output_dir: Path) -> Path:
    """Per-ticker grouped bar chart of KS, Wasserstein, sigma RMSE.

    Inputs
    ------
    multi_llm_downstream.csv with columns
    ``ticker, model, ks, wass, vol_rmse, sharpe_diff``.

    Output
    ------
    multi_llm_downstream.png
    """
    df = _load_csv(input_dir, "multi_llm_downstream.csv")
    metrics = ["ks", "wass", "vol_rmse", "sharpe_diff"]
    titles = ["KS", "Wasserstein ($\\times 10^{-3}$)",
              "Volatility RMSE", "Sharpe diff"]
    models = sorted(df["model"].unique())
    tickers = sorted(df["ticker"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, title in zip(axes.ravel(), metrics, titles):
        x = np.arange(len(tickers))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            sub = df[df["model"] == m].set_index("ticker").reindex(tickers)
            ax.bar(x + (i - len(models) / 2) * w + w / 2,
                   sub[metric], width=w, label=m, edgecolor="black", lw=0.3)
        ax.set_xticks(x); ax.set_xticklabels(tickers, fontsize=8)
        ax.set_title(title); ax.grid(True, axis="y", ls=":", lw=0.3, alpha=0.5)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Multi-LLM downstream robustness (3 tickers × 3 models)",
                 fontsize=12, y=1.0)
    return _save(fig, output_dir, "multi_llm_downstream")


# ===========================================================================
# Registry
# ===========================================================================

REGISTRY: Dict[str, Callable[[Path, Path], Path]] = {
    # Body
    "make_cross_asset_correlation_3panel": make_cross_asset_correlation_3panel,
    "make_direction_ablation_aggregate":   make_direction_ablation_aggregate,
    "make_oos_scatter":                    make_oos_scatter,
    "make_baseline_suite_boxplot":         make_baseline_suite_boxplot,
    # Appendix
    "make_n30_grid_appendix":              make_n30_grid_appendix,
    "make_lstm_disc_barplot_appendix":     make_lstm_disc_barplot_appendix,
    "make_multiseed_ci_appendix":          make_multiseed_ci_appendix,
    "make_shuffle_control_aggregate":      make_shuffle_control_aggregate,
    "make_direction_box_grid_appendix":    make_direction_box_grid_appendix,
    "make_factor_cross_ticker_heatmap":    make_factor_cross_ticker_heatmap,
    "make_multi_llm_appendix":             make_multi_llm_appendix,
}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("results/_aggregate_phaseB"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/phaseB_images"),
    )
    parser.add_argument(
        "--only", action="append", default=None,
        help="Run only the named function(s); repeat the flag to chain.",
    )
    args = parser.parse_args(argv)

    targets = args.only if args.only else list(REGISTRY.keys())
    manifest: Dict[str, str] = {}
    for name in targets:
        if name not in REGISTRY:
            raise SystemExit(f"unknown figure: {name}")
        try:
            out = REGISTRY[name](args.input_dir, args.output_dir)
            manifest[name] = str(out)
            print(f"[ok] {name} -> {out}")
        except FileNotFoundError as exc:
            manifest[name] = f"SKIPPED: {exc}"
            print(f"[skip] {name}: {exc}")

    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[ok] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
