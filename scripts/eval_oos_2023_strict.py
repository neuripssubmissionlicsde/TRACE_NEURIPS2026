#!/usr/bin/env python3
"""OOS 2023 — path 4: NO Phase 2 leakage and NO post-hoc vol scaling.

For each ticker:

    1. Filter cached ``analyses.json`` to windows ENDING strictly before
       ``--oos-start`` → IS-only factor analyses.
    2. Build the IS-only factor name list (union of factors that appear
       in IS windows). The frozen factor schema is reused: the OOS
       activations only consult these names; no new factor is created.
    3. Train a fresh ``TemporalFusionTransformerFIN`` on IS-only
       (factor_matrix, log_returns) → ``impact_matrix_is.json`` per
       ticker.
    4. Calibrate the SDE engine (GJR-GARCH/jumps/regime/microstructure)
       on a rolling window of the LAST ``--calib-days`` IS days only,
       so 2023 never enters parameter estimation.
    5. Generate N OOS trajectories conditioned on OOS activations
       (no post-hoc vol scaling).
    6. Score KS / Wasserstein-1 / sigma-ratio against real 2023.

Outputs:
    results/<run>/<TICKER>/impact_matrix_is.json   (new per ticker)
    results/export_paper/oos_2023_path4/
        oos_report.json   oos_summary.csv
        oos_<TICKER>.{png,pdf}   oos_summary.{png,pdf}
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
import torch
from scipy.stats import ks_2samp, wasserstein_distance

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix, WindowAnalysis
from sde_causal_generator.factor_impact_network import train_impact_network
from sde_causal_generator.generate import ImpactDrivenGenerator
from sde_causal_generator.oos_validation import oos_validate
from scripts.ablation_causal_direction import (
    build_presence_matrix,
    compute_llm_directions,
    load_analyses,
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
DEFAULT_RUN_DIR = "final_run_gpt4o_mini"
WINDOW_TRADING_DAYS = 63


# ── helpers ─────────────────────────────────────────────────────────


def _log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.clip(close.astype(np.float64), 1e-8, None)))


def _split_dates(df: pd.DataFrame, oos_start: str):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    is_df = df[df["date"] < oos_start].sort_values("date").reset_index(drop=True)
    oos_df = df[df["date"] >= oos_start].sort_values("date").reset_index(drop=True)
    return is_df, oos_df


def _filter_is_analyses(
    analyses: list[WindowAnalysis], oos_start: str,
) -> list[WindowAnalysis]:
    cutoff = pd.Timestamp(oos_start)
    return [a for a in analyses if pd.Timestamp(a.end_date) < cutoff]


def _is_factor_names(analyses_is: list[WindowAnalysis]) -> list[str]:
    """Stable, ordered union of factor names appearing in IS windows."""
    seen, ordered = set(), []
    for a in analyses_is:
        for f in a.factors:
            if f.name not in seen:
                seen.add(f.name); ordered.append(f.name)
    return ordered


def _retrain_phase2_is(
    ticker: str,
    is_df: pd.DataFrame,
    analyses_is: list[WindowAnalysis],
    factor_names_is: list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    n_lags: int,
    architecture: str,
    device: str,
    out_path: Path,
) -> ImpactMatrix:
    feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in is_df.columns]
    price_data = np.clip(
        is_df[feature_cols].values.astype(np.float64), 1e-8, None
    )
    presence_is = build_presence_matrix(analyses_is, is_df, factor_names_is)
    llm_dirs = compute_llm_directions(analyses_is, factor_names_is)

    print(f"  [Phase 2 IS retrain] T={presence_is.shape[0]} "
          f"K={presence_is.shape[1]} arch={architecture}")
    _, im_is, history = train_impact_network(
        factor_matrix=presence_is,
        price_data=price_data,
        factor_names=factor_names_is,
        feature_names=feature_cols,
        n_lags=n_lags,
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        val_fraction=0.15,
        device=device,
        patience=patience,
        architecture=architecture,
        llm_directions=llm_dirs,
    )
    im_is.save(str(out_path))
    print(f"  [Phase 2 IS retrain] saved {out_path} "
          f"(best_epoch={history['best_epoch']})")
    return im_is


def _generate_oos(
    generator: ImpactDrivenGenerator,
    ticker: str,
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    oos_presence: np.ndarray,
    n_samples: int,
    seed: int,
    calib_days: int,
    counterfactual: bool = False,
) -> pd.DataFrame:
    """Generate OOS trajectories with rolling-window IS-only calibration."""
    np.random.seed(seed)
    feature_cols = [c for c in ["open", "high", "low", "close", "volume"]
                    if c in is_df.columns]
    is_full = np.clip(
        is_df[feature_cols].values.astype(np.float64), 1e-8, None
    )

    # Last ``calib_days`` IS rows only — no 2023 leakage.
    if calib_days > 0 and len(is_full) > calib_days:
        is_calib = is_full[-calib_days:]
        is_volume = is_df["volume"].values.astype(np.float64)[-calib_days:]
    else:
        is_calib = is_full
        is_volume = is_df["volume"].values.astype(np.float64)

    is_log_ret = np.diff(np.log(is_calib), axis=0)
    oos_dates = pd.to_datetime(oos_df["date"]).values
    n_oos = len(oos_dates)

    last_row = is_df.iloc[-1]
    initial_prices = np.array(
        [last_row.get(c, 100.0) for c in feature_cols], dtype=np.float32
    )
    return generator.generate_finrl_df(
        ticker=ticker,
        n_steps=n_oos,
        initial_prices=initial_prices,
        n_samples=n_samples,
        real_dates=oos_dates,
        start_date=str(oos_dates[0])[:10],
        real_log_returns=is_log_ret,
        real_volume=is_volume,
        historical_presence=oos_presence,
        real_prices=is_calib,
        counterfactual=counterfactual,
    )


def _summary_metrics(real_close: np.ndarray, synth_df: pd.DataFrame) -> dict:
    real_r = _log_returns(real_close)
    per_traj = []
    for sid in sorted(synth_df["sample"].unique()):
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
    lo, hi, med = closes.min(0), closes.max(0), np.median(closes, 0)
    ax.fill_between(oos_dates, lo, hi, color="tab:blue", alpha=0.18,
                    label=f"Synth min-max (N={closes.shape[0]})")
    ax.plot(oos_dates, med, color="tab:blue", linewidth=1.0, label="Synth median")
    ax.plot(oos_dates, real_close, color="black", linewidth=1.4, label="Real 2023")
    ax.set_title(
        f"{ticker}  KS={metrics['ks_mean']:.3f}  "
        f"σ={metrics['sigma_ratio_mean']:.2f}", fontsize=10,
    )
    ax.set_ylabel("Close (USD)")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


# ── main ────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--oos-start", default="2023-01-01")
    p.add_argument("--oos-end",   default="2023-12-31")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--out", default="results/export_paper/oos_2023_path4")
    p.add_argument("--calib-days", type=int, default=504,
                   help="Rolling-window length for engine calibration (last K IS days).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--n-lags", type=int, default=10)
    p.add_argument("--architecture", default="tft")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--reuse-is-impact", action="store_true",
                   help="Skip Phase 2 retrain if impact_matrix_is.json already exists.")
    p.add_argument("--counterfactual", action="store_true",
                   help="Disable regime drift calibration (factor-driven drift only).")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[path-4] device={device} calib_days={args.calib_days}")

    results_dir = PROJECT_ROOT / "results" / args.results_dir
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(results_dir / "training_data.csv")

    report = {
        "oos_start": args.oos_start, "oos_end": args.oos_end,
        "n_samples": args.n_samples, "seed": args.seed,
        "phase2_leakage": False,
        "phase3_calibration": f"IS_only_last_{args.calib_days}_days",
        "vol_target": "none",
        "tickers": {},
    }
    summary_rows = []

    n = len(args.tickers); cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    t0 = time.time(); last_idx = -1
    for idx, ticker in enumerate(args.tickers):
        ax = axes_flat[idx]; last_idx = idx
        print(f"\n── {ticker} ──")
        df_t = full_df[full_df["tic"] == ticker].copy()
        if df_t.empty:
            ax.set_visible(False); continue
        is_df, oos_df = _split_dates(df_t, args.oos_start)
        if len(is_df) < 200 or len(oos_df) < 30:
            ax.set_visible(False); continue

        analyses_full = load_analyses(ticker, results_dir)
        analyses_is = _filter_is_analyses(analyses_full, args.oos_start)
        factor_names_is = _is_factor_names(analyses_is)
        if not factor_names_is:
            print(f"  skip: no IS factors"); ax.set_visible(False); continue

        # ── Phase 2 IS retrain ──────────────────────────────────────
        im_is_path = results_dir / ticker / "impact_matrix_is.json"
        if args.reuse_is_impact and im_is_path.exists():
            im_is = ImpactMatrix.load(str(im_is_path))
            print(f"  [Phase 2 IS retrain] reusing {im_is_path}")
        else:
            t_p2 = time.time()
            im_is = _retrain_phase2_is(
                ticker=ticker, is_df=is_df,
                analyses_is=analyses_is,
                factor_names_is=factor_names_is,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                n_lags=args.n_lags,
                architecture=args.architecture,
                device=device,
                out_path=im_is_path,
            )
            print(f"  [Phase 2] {time.time() - t_p2:.0f}s")

        # ── OOS presence on IS-only factor schema ──────────────────
        full_presence = build_presence_matrix(
            analyses_full, df_t, im_is.factor_names,
        )
        n_is = len(is_df)
        oos_presence = full_presence[n_is:n_is + len(oos_df)]

        # ── Generate OOS ───────────────────────────────────────────
        gen = ImpactDrivenGenerator(im_is, window_trading_days=WINDOW_TRADING_DAYS)
        synth_df = _generate_oos(
            gen, ticker, is_df, oos_df, oos_presence,
            n_samples=args.n_samples, seed=args.seed,
            calib_days=args.calib_days,
            counterfactual=args.counterfactual,
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
            "summary": metrics, "oos_validate": oos_full,
            "n_is_days": int(n_is), "n_oos_days": int(len(oos_df)),
            "n_factors_is": len(im_is.factor_names),
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
            "n_factors_is": len(im_is.factor_names),
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
              f"σ={metrics['sigma_ratio_mean']:.3f}  gap={gap_s}")

    for j in range(last_idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"OOS 2023 — N={args.n_samples}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "oos_summary.png", dpi=160)
    fig.savefig(out_dir / "oos_summary.pdf")
    plt.close(fig)

    if summary_rows:
        sd = pd.DataFrame(summary_rows)
        report["aggregate"] = {
            "ks_mean":           float(sd["ks_mean"].mean()),
            "ks_median":         float(sd["ks_mean"].median()),
            "ks_max":            float(sd["ks_mean"].max()),
            "wasserstein_mean":  float(sd["wasserstein_mean"].mean()),
            "sigma_ratio_mean":  float(sd["sigma_ratio_mean"].mean()),
            "in_sample_ks_mean": float(sd["in_sample_ks"].mean()),
            "gen_gap_ks_mean":   float(sd["gen_gap_ks"].mean()),
            "n_below_threshold_0_10": int((sd["ks_mean"] < 0.10).sum()),
        }
        sd.to_csv(out_dir / "oos_summary.csv", index=False)

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
