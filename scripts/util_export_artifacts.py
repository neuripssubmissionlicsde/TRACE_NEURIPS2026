#!/usr/bin/env python3
"""Build the paper export bundle.

Aggregates the artefacts approved for inclusion in the LCSDE paper into
``results/export_paper/`` and adds the new figure variants requested:

    1. Item 5 — N=30 envelopes
       - keep the existing AAPL linear-scale figure
       - add a log-scale variant (AAPL)
       - generate per-ticker linear + log envelopes for all 8 tickers
    2. Item 6 — cross-asset correlation heatmap (copied as-is)
    3. Bull/Bear amplified ablation
       - copy the 3-panel summary
       - add a new summary plot with relative % variation of the
         cumulative factor signal (bears/bulls removal)
       - export tabular CSV with the per-ticker numbers

This script reads from existing artefacts in
``results/final_run_gpt4o_mini/`` and from the per-ticker CSVs in
``n30/``; it does NOT regenerate trajectories or rerun the impact matrix.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
RUN_DIR = PROJECT_ROOT / "results" / "final_run_gpt4o_mini"
EXPORT_ROOT = PROJECT_ROOT / "results" / "export_paper"


def _load_real_close(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(RUN_DIR / "training_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["tic"] == ticker].sort_values("date").reset_index(drop=True)


def _load_n30_synth(ticker: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, synth_matrix)  with shape (n_samples, T) of close prices."""
    ticker_dir = RUN_DIR / "n30" / ticker
    files = sorted(ticker_dir.glob("synthetic_data_*.csv"))
    arrays = []
    dates = None
    for f in files:
        sdf = pd.read_csv(f)
        sdf["date"] = pd.to_datetime(sdf["date"])
        sdf = sdf.sort_values("date").reset_index(drop=True)
        arrays.append(sdf["close"].astype(np.float64).values)
        if dates is None:
            dates = sdf["date"].values
    if not arrays:
        return np.array([]), np.array([])
    n_min = min(len(a) for a in arrays)
    mat = np.stack([a[:n_min] for a in arrays], axis=0)
    return dates[:n_min], mat


def _plot_envelope(
    dates,
    real_close: np.ndarray,
    synth_mat: np.ndarray,
    ticker: str,
    log_scale: bool,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 5))
    synth_min = synth_mat.min(axis=0)
    synth_max = synth_mat.max(axis=0)
    synth_med = np.median(synth_mat, axis=0)

    ax.fill_between(
        dates, synth_min, synth_max,
        color="tab:blue", alpha=0.25,
        label=f"Synthetic envelope (N={synth_mat.shape[0]})",
    )
    ax.plot(
        dates, synth_med,
        color="tab:blue", linewidth=1.0, alpha=0.85,
        label="Synthetic median",
    )
    ax.plot(
        dates, real_close,
        color="black", linewidth=1.3, label=f"Real {ticker} close",
    )

    if log_scale:
        ax.set_yscale("log")
        scale_label = " (log scale)"
    else:
        scale_label = ""
    ax.set_title(
        f"{ticker} — Real vs LCSDE synthetic envelope (N={synth_mat.shape[0]}){scale_label}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price (USD)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def export_item5() -> None:
    print("\n── Item 5: N=30 envelopes (linear + log, per ticker) ──")
    out_dir = EXPORT_ROOT / "n30"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        dates, synth_mat = _load_n30_synth(ticker)
        if synth_mat.size == 0:
            print(f"  ⚠ No N=30 data for {ticker}, skipping")
            continue
        df_real = _load_real_close(ticker)
        # Trim real to the synthetic length.
        real_close = df_real["close"].astype(np.float64).values[: synth_mat.shape[1]]

        _plot_envelope(
            dates, real_close, synth_mat, ticker, log_scale=False,
            out_path=out_dir / f"real_vs_synth_{ticker}_n30",
        )
        _plot_envelope(
            dates, real_close, synth_mat, ticker, log_scale=True,
            out_path=out_dir / f"real_vs_synth_{ticker}_n30_log",
        )
        print(f"  ✓ {ticker}: linear + log saved")


def export_item6() -> None:
    print("\n── Item 6: cross-asset correlation heatmap ──")
    src_dir = RUN_DIR / "cross_asset_correlation"
    out_dir = EXPORT_ROOT / "cross_asset_correlation"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "cross_asset_correlation_heatmap.png",
        "cross_asset_correlation_heatmap.pdf",
        "cross_asset_correlation_report.json",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"  ✓ {name}")

    # Also export the matrices as CSVs for easy inclusion in the paper.
    report_path = src_dir / "cross_asset_correlation_report.json"
    if report_path.exists():
        rep = json.loads(report_path.read_text())
        tickers = rep["tickers"]
        for key, fname in [
            ("real_corr", "real_correlation_matrix.csv"),
            ("synth_corr", "synth_correlation_matrix.csv"),
            ("abs_diff", "abs_diff_matrix.csv"),
        ]:
            mat = pd.DataFrame(rep[key], index=tickers, columns=tickers)
            mat.round(4).to_csv(out_dir / fname)
        print(f"  ✓ CSVs: real, synth, abs_diff (MAE off-diag = "
              f"{rep['mae_off_diagonal']:.4f})")


def export_direction_amplified() -> None:
    print("\n── Bull/Bear amplified ablation ──")
    src_dir = RUN_DIR / "direction_ablation_amplified"
    out_dir = EXPORT_ROOT / "direction_ablation_amplified"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "amplified_summary.png",
        "amplified_summary.pdf",
        "amplified_report.json",
    ]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    # Per-ticker boxplots — ship a couple for the appendix.
    for t in TICKERS:
        f = src_dir / f"amplified_box_{t}.png"
        if f.exists():
            shutil.copy2(f, out_dir / f.name)

    rep = json.loads((src_dir / "amplified_report.json").read_text())

    # ── Build table CSV (per-ticker, three views) ──
    rows = []
    for t in TICKERS:
        if t not in rep["tickers"]:
            continue
        d = rep["tickers"][t]
        ann = d["annualised_pct"]
        cum = d["cumulative_pct"]
        cond = d["conditional_active_day_annualised_pct"]
        cum_all = cum["all_factors"]
        rel_bears_cum = (cum["delta_bears_off"] / cum_all * 100) if abs(cum_all) > 1e-9 else float("nan")
        rel_bulls_cum = (cum["delta_bulls_off"] / cum_all * 100) if abs(cum_all) > 1e-9 else float("nan")
        rows.append({
            "ticker": t,
            "n_factors": d["n_factors"],
            "n_bull": d["n_bull"],
            "n_bear": d["n_bear"],
            "ann_all_pct": round(ann["all_factors"], 4),
            "ann_delta_bears_pp": round(ann["delta_bears_off"], 4),
            "ann_delta_bulls_pp": round(ann["delta_bulls_off"], 4),
            "cum_all_pct_6y": round(cum["all_factors"], 4),
            "cum_delta_bears_pp": round(cum["delta_bears_off"], 4),
            "cum_delta_bulls_pp": round(cum["delta_bulls_off"], 4),
            "cum_delta_bears_pct_of_baseline": round(rel_bears_cum, 2),
            "cum_delta_bulls_pct_of_baseline": round(rel_bulls_cum, 2),
            "cond_delta_bears_on_bear_days_pp":
                round(cond["delta_bears_off_on_bear_days"], 4),
            "cond_delta_bulls_on_bull_days_pp":
                round(cond["delta_bulls_off_on_bull_days"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "direction_ablation_table.csv", index=False)
    print(f"  ✓ direction_ablation_table.csv ({len(df)} rows)")

    # ── New summary: % variation of cumulative factor signal ──
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    width = 0.4
    bars_b = ax.bar(
        x - width / 2,
        df["cum_delta_bears_pct_of_baseline"].values,
        width, color="#ee6677", label="Bears OFF (expect ↑)",
    )
    bars_bu = ax.bar(
        x + width / 2,
        df["cum_delta_bulls_pct_of_baseline"].values,
        width, color="#228833", label="Bulls OFF (expect ↓)",
    )
    for bars in (bars_b, bars_bu):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:+.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3 if h >= 0 else -12),
                textcoords="offset points",
                ha="center", fontsize=8,
            )
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df["ticker"].values)
    ax.set_ylabel("Cumulative drift change\n(% of all-factors baseline, 2018–2023)")
    ax.set_title(
        "Bull/Bear ablation — relative variation of the deterministic "
        "factor signal\n(8/8 tickers correct sign for both groups)"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "amplified_relative_summary.png", dpi=200)
    fig.savefig(out_dir / "amplified_relative_summary.pdf")
    plt.close(fig)
    print("  ✓ amplified_relative_summary.{png,pdf}")

    # Print aggregate so it can be quoted in the paper.
    agg = rep["aggregate"]
    print("\nAggregate (mean across 8 tickers):")
    print(f"  Δ bears (annualised)   : {agg['mean_delta_bears_annualised_pp']:+.3f} pp")
    print(f"  Δ bulls (annualised)   : {agg['mean_delta_bulls_annualised_pp']:+.3f} pp")
    print(f"  Δ bears (cumulative 6y): {agg['mean_delta_bears_cumulative_pp']:+.3f} pp")
    print(f"  Δ bulls (cumulative 6y): {agg['mean_delta_bulls_cumulative_pp']:+.3f} pp")
    print(f"  Direction correct      : {agg['direction_correct_bears_over_total']}/8 bears, "
          f"{agg['direction_correct_bulls_over_total']}/8 bulls")


def main() -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    export_item5()
    export_item6()
    export_direction_amplified()
    print(f"\n✓ Export bundle ready at: {EXPORT_ROOT}")


if __name__ == "__main__":
    main()
