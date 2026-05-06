#!/usr/bin/env python3
"""Item 9 — Rolling KS by sub-period (LCSDE vs Vanilla SDE).

For each ticker and each of 4 sub-periods, computes the KS statistic
between the real log-returns and the synthetic log-returns produced by
LCSDE and (when available) by the Vanilla SDE. Plots side-by-side
heatmaps.

Sub-periods:
    pre_covid:        2018-01-01 → 2019-12-31
    covid:            2020-01-01 → 2020-12-31
    recovery:         2021-01-01 → 2022-12-31
    post_tightening:  2023-01-01 → 2023-12-31

Outputs:
    results/final_run_gpt4o_mini/rolling_ks/
        rolling_ks_report.json
        rolling_ks_heatmap.png (and .pdf)
        rolling_ks_grouped_bar.png (LCSDE only if Vanilla absent)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
RESULTS_DIR_NAME = "final_run_gpt4o_mini"

SUB_PERIODS = [
    ("pre_covid",         "2018-01-01", "2019-12-31"),
    ("covid",             "2020-01-01", "2020-12-31"),
    ("recovery",          "2021-01-01", "2022-12-31"),
    ("post_tightening",   "2023-01-01", "2023-12-31"),
]


def _slice_returns(df: pd.DataFrame, start: str, end: str) -> np.ndarray:
    sub = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
    close = sub["close"].astype(np.float64).clip(lower=1e-8).values
    if len(close) < 3:
        return np.array([])
    return np.diff(np.log(close))


def _pool_synth_returns(
    sample_files: list[Path], start: str, end: str
) -> np.ndarray:
    """Concatenate per-sample log-returns sliced to [start, end]."""
    pooled = []
    for sf in sample_files:
        sdf = pd.read_csv(sf)
        if "date" not in sdf.columns or "close" not in sdf.columns:
            continue
        sdf["date"] = pd.to_datetime(sdf["date"]).dt.strftime("%Y-%m-%d")
        rets = _slice_returns(sdf[["date", "close"]], start, end)
        if len(rets) > 1:
            pooled.append(rets)
    if not pooled:
        return np.array([])
    return np.concatenate(pooled)


def _list_lcsde_samples(ticker_dir: Path) -> list[Path]:
    return sorted(ticker_dir.glob("synthetic_data_[0-9].csv"))


def _list_vanilla_samples(vanilla_dir: Path, ticker: str) -> list[Path]:
    tdir = vanilla_dir / ticker
    if not tdir.exists():
        return []
    return sorted(tdir.glob("synthetic_data_*.csv"))


def _find_vanilla_dir(results_dir: Path) -> Path | None:
    """Locate per-ticker vanilla synthetic CSVs if they exist."""
    candidates = [
        results_dir / "ablation_vanilla_sde",
        PROJECT_ROOT / "results" / "ablation_vanilla_sde",
    ]
    for c in candidates:
        if not c.exists():
            continue
        for ticker in TICKERS:
            tdir = c / ticker
            if tdir.exists() and any(tdir.glob("synthetic_data_*.csv")):
                return c
    return None


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_NAME
    out_dir = results_dir / "rolling_ks"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_csv = results_dir / "training_data.csv"
    real_full = pd.read_csv(training_csv)
    real_full["date"] = pd.to_datetime(real_full["date"])

    vanilla_dir = _find_vanilla_dir(results_dir)
    if vanilla_dir is None:
        print("⚠ No per-ticker Vanilla SDE CSVs found — Vanilla column omitted.")
    else:
        print(f"✓ Using Vanilla SDE CSVs from {vanilla_dir}")

    period_names = [p[0] for p in SUB_PERIODS]
    lcsde_ks = pd.DataFrame(index=TICKERS, columns=period_names, dtype=float)
    vanilla_ks = pd.DataFrame(index=TICKERS, columns=period_names, dtype=float)
    report = {
        "sub_periods": [
            {"name": n, "start": s, "end": e} for n, s, e in SUB_PERIODS
        ],
        "tickers": TICKERS,
        "per_ticker": {},
        "vanilla_available": vanilla_dir is not None,
    }

    for ticker in TICKERS:
        real_t = (
            real_full[real_full["tic"] == ticker]
            .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
            .copy()
        )
        lcsde_files = _list_lcsde_samples(results_dir / ticker)
        vanilla_files = (
            _list_vanilla_samples(vanilla_dir, ticker)
            if vanilla_dir is not None else []
        )

        per_period = {}
        for name, start, end in SUB_PERIODS:
            real_ret = _slice_returns(real_t, start, end)
            entry = {"real_n": int(len(real_ret))}

            if lcsde_files and len(real_ret) > 1:
                lc_ret = _pool_synth_returns(lcsde_files, start, end)
                if len(lc_ret) > 1:
                    ks, p = stats.ks_2samp(real_ret, lc_ret)
                    lcsde_ks.loc[ticker, name] = float(ks)
                    entry["lcsde"] = {
                        "ks": float(ks), "pvalue": float(p),
                        "n": int(len(lc_ret)),
                    }

            if vanilla_files and len(real_ret) > 1:
                v_ret = _pool_synth_returns(vanilla_files, start, end)
                if len(v_ret) > 1:
                    ks_v, p_v = stats.ks_2samp(real_ret, v_ret)
                    vanilla_ks.loc[ticker, name] = float(ks_v)
                    entry["vanilla"] = {
                        "ks": float(ks_v), "pvalue": float(p_v),
                        "n": int(len(v_ret)),
                    }

            per_period[name] = entry

        report["per_ticker"][ticker] = per_period

    # ── Heatmaps ───────────────────────────────────────────────────
    has_vanilla = vanilla_dir is not None and vanilla_ks.notna().any().any()
    n_panels = 2 if has_vanilla else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels + 1, 5.5), squeeze=False,
    )
    cmap = "magma_r"
    common_kwargs = dict(
        vmin=0.0, vmax=0.15, cmap=cmap, annot=True, fmt=".3f",
        linewidths=0.4, linecolor="white",
        cbar=False, square=False,
    )

    sns.heatmap(lcsde_ks.astype(float), ax=axes[0, 0], **common_kwargs)
    axes[0, 0].set_title("LCSDE — KS statistic")
    axes[0, 0].set_xlabel("Sub-period")
    axes[0, 0].set_ylabel("Ticker")

    if has_vanilla:
        sns.heatmap(vanilla_ks.astype(float), ax=axes[0, 1], **common_kwargs)
        axes[0, 1].set_title("Vanilla SDE — KS statistic")
        axes[0, 1].set_xlabel("Sub-period")
        axes[0, 1].set_ylabel("")

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=0.15)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("KS statistic (lower = better)")

    fig.suptitle("Rolling KS — real vs synthetic log-returns")
    png = out_dir / "rolling_ks_heatmap.png"
    pdf = out_dir / "rolling_ks_heatmap.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # ── Grouped bar (one panel per model) ─────────────────────────
    n_panels_bar = 2 if has_vanilla else 1
    fig2, axes2 = plt.subplots(
        n_panels_bar, 1, figsize=(13, 4.5 * n_panels_bar),
        sharex=True, squeeze=False,
    )
    n_periods = len(period_names)
    bar_w = 0.8 / n_periods
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, n_periods))
    ind = np.arange(len(TICKERS))
    ymax = 0.0
    panel_specs = [("LCSDE", lcsde_ks, axes2[0, 0])]
    if has_vanilla:
        panel_specs.append(("Vanilla SDE", vanilla_ks, axes2[1, 0]))

    for label, mat, ax_b in panel_specs:
        for i, name in enumerate(period_names):
            vals = mat[name].astype(float).fillna(0.0).values
            ymax = max(ymax, float(np.nanmax(vals)))
            ax_b.bar(
                ind + (i - (n_periods - 1) / 2) * bar_w, vals, bar_w,
                label=name, color=palette[i],
            )
        ax_b.set_xticks(ind)
        ax_b.set_xticklabels(TICKERS)
        ax_b.set_ylabel("KS statistic")
        ax_b.set_title(f"{label} — KS by sub-period (lower = better)")
        ax_b.grid(axis="y", alpha=0.3)
        ax_b.legend(fontsize=8, ncol=4, frameon=False, loc="upper right")

    for spec in panel_specs:
        spec[2].set_ylim(0, max(0.15, ymax * 1.15))

    fig2.tight_layout()
    fig2.savefig(out_dir / "rolling_ks_grouped_bar.png", dpi=200)
    plt.close(fig2)

    # ── Save report ───────────────────────────────────────────────
    report["lcsde_ks_matrix"] = lcsde_ks.astype(float).round(6).where(
        lcsde_ks.notna(), None
    ).to_dict()
    if has_vanilla:
        report["vanilla_ks_matrix"] = vanilla_ks.astype(float).round(6).where(
            vanilla_ks.notna(), None
        ).to_dict()
    with open(out_dir / "rolling_ks_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Saved heatmap: {png}")
    print(f"✓ Saved report:  {out_dir / 'rolling_ks_report.json'}")


if __name__ == "__main__":
    main()
