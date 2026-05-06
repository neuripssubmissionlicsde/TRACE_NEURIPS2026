#!/usr/bin/env python3
"""Out-of-Sample (OOS) evaluation — 2023 holdout.

Splits the 2018-01-01 -- 2023-12-31 sample into:
    * IS  (in-sample)      : 2018-01-01 -- 2022-12-31
    * OOS (out-of-sample)  : 2023-01-01 -- 2023-12-31

Pipeline state:
    * Phase 1 (factor activations): reused from cached analyses.json.
      The 2023 bi-weekly windows are already cached, so no LLM calls
      are issued.
    * Phase 2 (ImpactMatrix): reused from results/<run>/<TICKER>/.
      Phase 2 leakage is acknowledged in the paper.
    * Phase 3 (GJR-GARCH / jumps / regime / Cholesky): RECALIBRATED
      using only the 2018-2022 real returns; 2023 is held out from
      every parameter estimation in the SDE engine.

Outputs:
    results/export_paper/oos_2023/
        oos_report.json
        oos_summary.csv
        oos_<TICKER>.{png,pdf}
        oos_summary.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator
from sde_causal_generator.oos_validation import oos_validate
from scripts.ablation_causal_direction import (
    build_presence_matrix,
    load_analyses,
)

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN",
    "AMZN", "AXP", "BA", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "KO", "MCD", "MMM", "MRK", "NKE",
    "PG", "TRV", "UNH", "V",
]
DEFAULT_RUN_DIR = "djia28"
WINDOW_TRADING_DAYS = 63


def _log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.clip(close.astype(np.float64), 1e-8, None)))


def _split_dates(df: pd.DataFrame, oos_start: str):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    is_df = df[df["date"] < oos_start].sort_values("date").reset_index(drop=True)
    oos_df = df[df["date"] >= oos_start].sort_values("date").reset_index(drop=True)
    return is_df, oos_df


def _generate_oos(
    generator: ImpactDrivenGenerator,
    ticker: str,
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    oos_presence: np.ndarray,
    n_samples: int,
    seed: int,
    vol_target: str = "is",
) -> pd.DataFrame:
    """Generate N OOS trajectories with IS-only stochastic-engine calibration.

    ``vol_target`` rescales the close-return path so that the synthetic
    daily-return std matches a target. The drift, jumps and regime
    structure are preserved; only the magnitude is normalised.

    * ``"none"`` — no rescaling (raw engine output)
    * ``"is"``   — match the IS real close-return std (default)
    """
    np.random.seed(seed)
    feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in is_df.columns]

    is_prices = np.clip(
        is_df[feature_cols].values.astype(np.float64), 1e-8, None
    )
    is_log_ret = np.diff(np.log(is_prices), axis=0)
    is_volume = is_df["volume"].values.astype(np.float64)

    oos_dates = pd.to_datetime(oos_df["date"]).values
    n_oos = len(oos_dates)

    last_row = is_df.iloc[-1]
    initial_prices = np.array(
        [last_row.get(c, 100.0) for c in feature_cols], dtype=np.float32
    )

    synth_df = generator.generate_finrl_df(
        ticker=ticker,
        n_steps=n_oos,
        initial_prices=initial_prices,
        n_samples=n_samples,
        real_dates=oos_dates,
        start_date=str(oos_dates[0])[:10],
        real_log_returns=is_log_ret,
        real_volume=is_volume,
        historical_presence=oos_presence,
        real_prices=is_prices,
    )

    if vol_target == "none":
        return synth_df

    # Volatility scaling: per-sample, on close, IS-only target.
    close_idx = 3 if "close" in is_df.columns else -1
    is_close_ret = is_log_ret[:, close_idx]
    target_std = float(np.std(is_close_ret))
    if target_std <= 0:
        return synth_df

    out_frames = []
    for sid, sdf in synth_df.groupby("sample"):
        sdf = sdf.sort_values("date").copy()
        close = sdf["close"].astype(np.float64).values
        if len(close) < 3 or close[0] <= 0:
            out_frames.append(sdf); continue
        log_close = np.log(np.clip(close, 1e-8, None))
        ret = np.diff(log_close)
        cur_std = float(np.std(ret))
        if cur_std <= 1e-12:
            out_frames.append(sdf); continue
        # Centre-scale with mean preserved (drift is the causal signal).
        mean_ret = float(np.mean(ret))
        scaled_ret = (ret - mean_ret) * (target_std / cur_std) + mean_ret
        # Reconstruct the close path from scaled returns.
        new_log_close = np.concatenate([[log_close[0]], log_close[0] + np.cumsum(scaled_ret)])
        new_close = np.exp(new_log_close)
        ratio = new_close / close
        for col in ("open", "high", "low", "close"):
            if col in sdf.columns:
                sdf[col] = sdf[col].astype(np.float64).values * ratio
        out_frames.append(sdf)
    return pd.concat(out_frames, ignore_index=True)


def _summary_metrics(real_close: np.ndarray, synth_df: pd.DataFrame) -> dict:
    real_r = _log_returns(real_close)
    sample_ids = sorted(synth_df["sample"].unique())
    per_traj = []
    for sid in sample_ids:
        s = synth_df[synth_df["sample"] == sid].sort_values("date")
        sr = _log_returns(s["close"].values.astype(np.float64))
        n = min(len(real_r), len(sr))
        if n < 10:
            continue
        ks = float(ks_2samp(real_r[:n], sr[:n]).statistic)
        w1 = float(wasserstein_distance(real_r[:n], sr[:n]))
        sig = (float(np.std(sr[:n]) / np.std(real_r[:n]))
               if np.std(real_r[:n]) > 0 else float("nan"))
        per_traj.append({"sample": int(sid), "ks": ks,
                         "wasserstein": w1, "sigma_ratio": sig})
    return {
        "per_trajectory": per_traj,
        "ks_mean":          float(np.mean([m["ks"] for m in per_traj])),
        "ks_median":        float(np.median([m["ks"] for m in per_traj])),
        "wasserstein_mean": float(np.mean([m["wasserstein"] for m in per_traj])),
        "sigma_ratio_mean": float(np.mean([m["sigma_ratio"] for m in per_traj])),
    }


def _plot_ticker(ticker, oos_dates, real_close, synth_df, metrics, ax):
    closes = []
    for sid in sorted(synth_df["sample"].unique()):
        s = synth_df[synth_df["sample"] == sid].sort_values("date")
        closes.append(s["close"].values[:len(oos_dates)])
    closes = np.array(closes)
    lo = np.min(closes, axis=0)
    hi = np.max(closes, axis=0)
    median = np.median(closes, axis=0)
    ax.fill_between(oos_dates, lo, hi, color="tab:blue", alpha=0.18,
                    label=f"Synth min-max (N={closes.shape[0]})")
    ax.plot(oos_dates, median, color="tab:blue", linewidth=1.0, label="Synth median")
    ax.plot(oos_dates, real_close, color="black", linewidth=1.4, label="Real 2023")
    ax.set_title(
        f"{ticker} — OOS 2023  KS={metrics['ks_mean']:.3f}  "
        f"σ={metrics['sigma_ratio_mean']:.2f}",
        fontsize=10,
    )
    ax.set_ylabel("Close (USD)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--oos-start", default="2023-01-01")
    p.add_argument("--oos-end",   default="2023-12-31")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--out", default="results/oos_2023")
    p.add_argument("--vol-target", choices=["none", "is"], default="is",
                   help="Rescale synth close-return std to match IS real (default) or none.")
    args = p.parse_args()

    results_dir = PROJECT_ROOT / "results" / args.results_dir
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    training_csv = results_dir / "training_data.csv"
    full_df = pd.read_csv(training_csv)

    report = {
        "oos_start": args.oos_start,
        "oos_end": args.oos_end,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "phase2_leakage": True,
        "phase3_calibration": "IS_only_2018-2022",
        "vol_target": args.vol_target,
        "tickers": {},
    }
    summary_rows = []

    n = len(args.tickers)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    t0 = time.time()
    last_idx = -1
    for idx, ticker in enumerate(args.tickers):
        ax = axes_flat[idx]
        last_idx = idx
        print(f"\n── {ticker} ──")
        im_path = results_dir / ticker / "impact_matrix.json"
        if not im_path.exists():
            print(f"  skip: missing {im_path}")
            ax.set_visible(False)
            continue

        im = ImpactMatrix.load(str(im_path))
        analyses = load_analyses(ticker, results_dir)

        df_t = full_df[full_df["tic"] == ticker].copy()
        if df_t.empty:
            ax.set_visible(False)
            continue
        is_df, oos_df = _split_dates(df_t, args.oos_start)
        if len(oos_df) < 30 or len(is_df) < 100:
            print(f"  skip: insufficient ({len(is_df)} IS / {len(oos_df)} OOS)")
            ax.set_visible(False)
            continue

        full_presence = build_presence_matrix(analyses, df_t, im.factor_names)
        n_is = len(is_df)
        oos_presence = full_presence[n_is:n_is + len(oos_df)]

        gen = ImpactDrivenGenerator(im, window_trading_days=WINDOW_TRADING_DAYS)
        synth_df = _generate_oos(
            gen, ticker, is_df, oos_df, oos_presence,
            n_samples=args.n_samples, seed=args.seed,
            vol_target=args.vol_target,
        )

        real_close = oos_df["close"].astype(np.float64).values
        metrics = _summary_metrics(real_close, synth_df)

        try:
            oos_full = oos_validate(
                synth_df=synth_df, test_df=oos_df, train_df=is_df, price_col="close",
            )
        except Exception as exc:
            oos_full = {"error": str(exc)}

        report["tickers"][ticker] = {
            "summary": metrics,
            "oos_validate": oos_full,
            "n_is_days": int(n_is),
            "n_oos_days": int(len(oos_df)),
        }
        summary_rows.append({
            "ticker": ticker,
            "ks_mean": metrics["ks_mean"],
            "ks_median": metrics["ks_median"],
            "wasserstein_mean": metrics["wasserstein_mean"],
            "sigma_ratio_mean": metrics["sigma_ratio_mean"],
            "oos_ks_stat":   oos_full.get("oos_ks_stat"),
            "oos_ks_pval":   oos_full.get("oos_ks_pval"),
            "oos_vol_rmse":  oos_full.get("oos_vol_rmse"),
            "in_sample_ks":  oos_full.get("in_sample_ks"),
            "gen_gap_ks":    oos_full.get("generalisation_gap_ks"),
        })

        oos_dates = pd.to_datetime(oos_df["date"]).values
        _plot_ticker(ticker, oos_dates, real_close, synth_df, metrics, ax)

        fig_t, ax_t = plt.subplots(figsize=(9, 4.2))
        _plot_ticker(ticker, oos_dates, real_close, synth_df, metrics, ax_t)
        ax_t.legend(loc="best", fontsize=8, frameon=False)
        fig_t.tight_layout()
        fig_t.savefig(out_dir / f"oos_{ticker}.png", dpi=180)
        fig_t.savefig(out_dir / f"oos_{ticker}.pdf")
        plt.close(fig_t)

        gap = oos_full.get("generalisation_gap_ks")
        gap_s = f"{gap:+.4f}" if isinstance(gap, (int, float)) else "n/a"
        print(f"  KS={metrics['ks_mean']:.4f}  "
              f"W1={metrics['wasserstein_mean']:.4e}  "
              f"σ={metrics['sigma_ratio_mean']:.3f}  "
              f"gap={gap_s}")

    for j in range(last_idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"OOS 2023 — N={args.n_samples} synth trajectories per ticker  "
        f"(Phase 3 calibrated on 2018-2022 only)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "oos_summary.png", dpi=160)
    fig.savefig(out_dir / "oos_summary.pdf")
    plt.close(fig)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        report["aggregate"] = {
            "ks_mean":           float(summary_df["ks_mean"].mean()),
            "ks_median":         float(summary_df["ks_mean"].median()),
            "ks_max":            float(summary_df["ks_mean"].max()),
            "wasserstein_mean":  float(summary_df["wasserstein_mean"].mean()),
            "sigma_ratio_mean":  float(summary_df["sigma_ratio_mean"].mean()),
            "in_sample_ks_mean": float(summary_df["in_sample_ks"].mean()),
            "gen_gap_ks_mean":   float(summary_df["gen_gap_ks"].mean()),
            "n_below_threshold_0_10": int((summary_df["ks_mean"] < 0.10).sum()),
        }
        summary_df.to_csv(out_dir / "oos_summary.csv", index=False)

    with open(out_dir / "oos_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n✓ saved to {out_dir}  ({elapsed:.0f}s)")
    if "aggregate" in report:
        a = report["aggregate"]
        print(f"  Aggregate: KS={a['ks_mean']:.4f}  σ={a['sigma_ratio_mean']:.3f}  "
              f"gap_vs_IS={a['gen_gap_ks_mean']:+.4f}  "
              f"{a['n_below_threshold_0_10']}/{len(summary_rows)} below 0.10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
