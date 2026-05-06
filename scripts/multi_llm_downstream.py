#!/usr/bin/env python3
"""
Multi-LLM Downstream Robustness Test (2.2)
=============================================

Runs the FULL pipeline (factor extraction → FIN training → generation →
evaluation) using 2-3 alternative LLMs on a subset of tickers, then
compares final downstream metrics to assess whether LLM choice
materially affects synthetic data quality.

This addresses the reviewer concern about LLM sensitivity.

Usage:
    python scripts/run_multi_llm_downstream.py
    python scripts/run_multi_llm_downstream.py --tickers AAPL MSFT JPM
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import yaml


# ── LLMs to compare ────────────────────────────────────────────────

MODELS = [
    ("GPT-4o-mini",       "openrouter/openai/gpt-4o-mini"),
    ("Claude-3.5-Haiku",  "openrouter/anthropic/claude-3.5-haiku"),
    ("Gemini-2.0-Flash",  "openrouter/google/gemini-2.0-flash-001"),
]


# ── Metric helpers ──────────────────────────────────────────────────


def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(nlags + 1)
    result = np.correlate(x, x, mode="full")
    result = result[len(x) - 1:]
    return result[:nlags + 1] / (var * len(x))


def compute_extended_metrics(real_rets: np.ndarray, synth_rets: np.ndarray) -> dict:
    from sde_causal_generator.benchmark import compute_distributional_metrics
    base = compute_distributional_metrics(real_rets, synth_rets)
    base["wasserstein"] = float(stats.wasserstein_distance(real_rets, synth_rets))

    real_acf = acf(real_rets)
    synth_acf = acf(synth_rets)
    base["acf_dist_returns"] = float(np.mean(np.abs(real_acf - synth_acf)))

    real_abs_acf = acf(np.abs(real_rets))
    synth_abs_acf = acf(np.abs(synth_rets))
    base["acf_dist_abs_returns"] = float(np.mean(np.abs(real_abs_acf - synth_abs_acf)))

    window = min(21, len(real_rets) // 10)
    if window > 3:
        real_vol = pd.Series(real_rets).rolling(window).std().dropna().values
        synth_vol = pd.Series(synth_rets).rolling(window).std().dropna().values
        min_len = min(len(real_vol), len(synth_vol))
        if min_len > 5:
            base["vol_rmse"] = float(np.sqrt(np.mean(
                (real_vol[:min_len] - synth_vol[:min_len]) ** 2
            )))
            corr = np.corrcoef(real_vol[:min_len], synth_vol[:min_len])[0, 1]
            base["vol_corr"] = float(corr) if np.isfinite(corr) else 0.0

    def _sharpe(r):
        if np.std(r) < 1e-10:
            return 0.0
        return float(np.mean(r) / np.std(r) * np.sqrt(252))

    def _max_dd(r):
        cum = np.cumsum(r)
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    base["sharpe_diff"] = abs(_sharpe(real_rets) - _sharpe(synth_rets))
    base["max_dd_diff"] = abs(_max_dd(real_rets) - _max_dd(synth_rets))

    return base


# ── Pipeline runner ─────────────────────────────────────────────────


def run_pipeline_for_model(
    model_name: str,
    model_id: str,
    ticker: str,
    base_config_path: str,
    output_base: Path,
) -> dict:
    """Run the full pipeline for one model + one ticker."""
    from sde_causal_generator.pipeline import CausalSDEPipeline, PipelineConfig

    with open(base_config_path) as f:
        raw_cfg = yaml.safe_load(f)

    # Override LLM model
    raw_cfg["llm_model"] = model_id
    model_slug = model_name.lower().replace(" ", "_").replace(".", "")
    run_dir = output_base / model_slug / ticker
    raw_cfg["output_dir"] = str(run_dir.relative_to(PROJECT_ROOT))
    raw_cfg["cache_dir"] = str(PROJECT_ROOT / "cache" / "multi_llm" / model_slug / ticker)
    raw_cfg["tickers"] = [ticker]

    # Disable heavy optional features for speed
    raw_cfg["quality_report"] = {"enabled": False}
    raw_cfg["neural_sde"] = {"enabled": False}

    # Write temp config
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_config = run_dir / "config_used.yaml"
    with open(tmp_config, "w") as f:
        yaml.dump(raw_cfg, f, default_flow_style=False)

    cfg = PipelineConfig.from_yaml(str(tmp_config))

    pipe = CausalSDEPipeline(config=cfg)

    # Load data
    training_csv = PROJECT_ROOT / "results" / "final_run_gpt4o_mini" / "training_data.csv"
    if not training_csv.exists():
        return {"error": f"training_data.csv not found at {training_csv}"}

    df = pd.read_csv(training_csv)
    df["date"] = pd.to_datetime(df["date"])
    ticker_df = df[df["tic"] == ticker].copy()

    if len(ticker_df) < 100:
        return {"error": f"Insufficient data for {ticker}"}

    # Rename columns for pipeline
    ticker_df = ticker_df.sort_values("date").reset_index(drop=True)

    t0 = time.time()
    try:
        report = pipe.run(df=ticker_df, ticker=ticker)
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t0}
    elapsed = time.time() - t0

    # Load synthetic data and compute metrics
    results_root = PROJECT_ROOT / "results"
    synth_csv = results_root / raw_cfg["output_dir"] / "synthetic_data_causal_sde.csv"
    if not synth_csv.exists():
        # Try ticker subdirectory pattern
        synth_csv = results_root / raw_cfg["output_dir"] / ticker / "synthetic_data_causal_sde.csv"
    if not synth_csv.exists():
        return {"error": f"No synthetic data at {synth_csv}", "report": report}

    synth_df = pd.read_csv(synth_csv)
    if "sample" in synth_df.columns:
        synth_s0 = synth_df[synth_df["sample"] == 0].sort_values("date")
    else:
        synth_s0 = synth_df.sort_values("date")

    real_close = ticker_df["close"].values
    real_rets = np.diff(np.log(np.clip(real_close, 1e-8, None)))

    synth_close = synth_s0["close"].values
    synth_rets = np.diff(np.log(np.clip(synth_close, 1e-8, None)))

    metrics = compute_extended_metrics(real_rets, synth_rets)

    # Count factors
    im_path = results_root / raw_cfg["output_dir"] / "impact_matrix.json"
    if not im_path.exists():
        im_path = results_root / raw_cfg["output_dir"] / ticker / "impact_matrix.json"
    n_factors = 0
    if im_path.exists():
        from sde_causal_generator.data_structures import ImpactMatrix
        im = ImpactMatrix.load(str(im_path))
        n_factors = len(im.factor_names)

    return {
        "model_name": model_name,
        "model_id": model_id,
        "ticker": ticker,
        "n_factors": n_factors,
        "elapsed_seconds": round(elapsed, 1),
        "metrics": metrics,
    }


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM Downstream Robustness Test")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "final_run.yaml"))
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "JPM"],
                        help="Subset of tickers to test (default: AAPL MSFT JPM)")
    args = parser.parse_args()

    output_base = PROJECT_ROOT / "results" / "multi_llm_downstream"
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MULTI-LLM DOWNSTREAM ROBUSTNESS TEST")
    print("=" * 70)
    print(f"  Models:  {[m[0] for m in MODELS]}")
    print(f"  Tickers: {args.tickers}")
    print(f"  Config:  {args.config}")
    print(f"  Output:  {output_base}")
    print("=" * 70)

    t_start = time.time()
    all_results = {}

    for model_name, model_id in MODELS:
        all_results[model_name] = {}
        print(f"\n{'═' * 60}")
        print(f"  MODEL: {model_name} ({model_id})")
        print(f"{'═' * 60}")

        for ticker in args.tickers:
            print(f"\n  ── {ticker} ──")
            result = run_pipeline_for_model(
                model_name=model_name,
                model_id=model_id,
                ticker=ticker,
                base_config_path=args.config,
                output_base=output_base,
            )
            all_results[model_name][ticker] = result

            if "error" in result:
                print(f"    ✗ Error: {result['error']}")
            else:
                m = result["metrics"]
                print(f"    ✓ {result['n_factors']} factors, "
                      f"KS={m.get('ks_statistic', 'N/A'):.4f}, "
                      f"WD={m.get('wasserstein', 'N/A'):.5f}, "
                      f"Sharpe_diff={m.get('sharpe_diff', 'N/A'):.4f} "
                      f"({result['elapsed_seconds']:.0f}s)")

    total_time = time.time() - t_start

    # ── Cross-model comparison ──────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CROSS-MODEL COMPARISON")
    print(f"{'=' * 70}")

    metric_keys = [
        "ks_statistic", "wasserstein", "acf_dist_returns",
        "acf_dist_abs_returns", "sharpe_diff", "max_dd_diff",
        "vol_rmse", "vol_corr",
    ]

    # Table header
    header = f"  {'Ticker':<8s} {'Metric':<22s}"
    for m_name, _ in MODELS:
        header += f" {m_name:>16s}"
    header += f" {'Spread':>10s}"
    print(header)
    print(f"  {'─' * (30 + 17 * len(MODELS) + 11)}")

    # Per-ticker, per-metric comparison
    max_spreads = {}
    for ticker in args.tickers:
        for key in metric_keys:
            vals = {}
            for m_name, _ in MODELS:
                r = all_results.get(m_name, {}).get(ticker, {})
                if "metrics" in r:
                    vals[m_name] = r["metrics"].get(key)

            valid_vals = [v for v in vals.values() if v is not None and np.isfinite(v)]
            if len(valid_vals) < 2:
                continue

            spread = max(valid_vals) - min(valid_vals)
            mean_val = np.mean(valid_vals)
            relative_spread = spread / max(abs(mean_val), 1e-10)

            print(f"  {ticker:<8s} {key:<22s}", end="")
            for m_name, _ in MODELS:
                v = vals.get(m_name)
                if v is not None:
                    print(f" {v:>16.4f}", end="")
                else:
                    print(f" {'N/A':>16s}", end="")
            print(f" {spread:>10.4f}")

            if key not in max_spreads or spread > max_spreads[key]:
                max_spreads[key] = spread

    # Summary statistics across models
    print(f"\n  {'─' * 50}")
    print(f"  AGGREGATE MODEL STABILITY")
    print(f"  {'─' * 50}")

    model_avg_metrics = {}
    for m_name, _ in MODELS:
        avg = {}
        for key in metric_keys:
            vals = []
            for ticker in args.tickers:
                r = all_results.get(m_name, {}).get(ticker, {})
                if "metrics" in r:
                    v = r["metrics"].get(key)
                    if v is not None and np.isfinite(v):
                        vals.append(v)
            if vals:
                avg[key] = float(np.mean(vals))
        model_avg_metrics[m_name] = avg

    print(f"  {'Metric':<22s}", end="")
    for m_name, _ in MODELS:
        print(f" {m_name:>16s}", end="")
    print(f" {'Max Spread':>12s}")
    print(f"  {'─' * (22 + 17 * len(MODELS) + 13)}")

    for key in metric_keys:
        print(f"  {key:<22s}", end="")
        vals = []
        for m_name, _ in MODELS:
            v = model_avg_metrics.get(m_name, {}).get(key)
            if v is not None:
                print(f" {v:>16.4f}", end="")
                vals.append(v)
            else:
                print(f" {'N/A':>16s}", end="")
        spread = max(vals) - min(vals) if len(vals) >= 2 else 0
        print(f" {spread:>12.4f}")

    # Determine overall verdict
    key_spreads = []
    for key in ["ks_statistic", "wasserstein", "sharpe_diff"]:
        for ticker in args.tickers:
            vals = []
            for m_name, _ in MODELS:
                r = all_results.get(m_name, {}).get(ticker, {})
                if "metrics" in r:
                    v = r["metrics"].get(key)
                    if v is not None and np.isfinite(v):
                        vals.append(v)
            if len(vals) >= 2:
                mean_v = np.mean(vals)
                if abs(mean_v) > 1e-10:
                    key_spreads.append((max(vals) - min(vals)) / abs(mean_v))

    avg_relative_spread = np.mean(key_spreads) if key_spreads else 1.0

    if avg_relative_spread < 0.15:
        verdict = ("LLM choice has MINIMAL impact on downstream metrics — "
                   "the pipeline is robust to LLM variation")
    elif avg_relative_spread < 0.30:
        verdict = ("LLM choice has MODERATE impact — differences exist but "
                   "main conclusions are qualitatively stable")
    else:
        verdict = ("LLM choice has SIGNIFICANT impact on downstream metrics — "
                   "results are sensitive to the extraction model")

    print(f"\n  Average relative spread: {avg_relative_spread:.3f}")
    print(f"  Verdict: {verdict}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "experiment": "multi_llm_downstream",
        "description": "Full pipeline comparison across LLMs to assess downstream robustness",
        "models": [{"name": n, "id": i} for n, i in MODELS],
        "tickers": args.tickers,
        "aggregate": {
            "model_avg_metrics": model_avg_metrics,
            "max_spreads": max_spreads,
            "avg_relative_spread": float(avg_relative_spread),
            "verdict": verdict,
        },
        "per_model": all_results,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = output_base / "multi_llm_downstream_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\n  ✓ Report: {report_path}")

    # ── Plot ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"GPT-4o-mini": "#1565C0", "Claude-3.5-Haiku": "#E65100",
                  "Gemini-2.0-Flash": "#2E7D32"}

        metrics_to_plot = [
            ("ks_statistic", "KS Statistic ↓"),
            ("wasserstein", "Wasserstein ↓"),
            ("sharpe_diff", "Sharpe Diff ↓"),
        ]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        x = np.arange(len(args.tickers))
        width = 0.25

        for ax, (key, title) in zip(axes, metrics_to_plot):
            for i, (m_name, _) in enumerate(MODELS):
                vals = []
                for ticker in args.tickers:
                    r = all_results.get(m_name, {}).get(ticker, {})
                    if "metrics" in r:
                        vals.append(r["metrics"].get(key, 0))
                    else:
                        vals.append(0)
                offset = (i - len(MODELS) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=m_name,
                       color=colors.get(m_name, "#333"), alpha=0.85)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(args.tickers)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Multi-LLM Downstream Robustness", fontsize=14, fontweight="bold")
        fig.tight_layout()
        plot_path = output_base / "multi_llm_comparison.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        fig.savefig(str(output_base / "multi_llm_comparison.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Comparison plot: {plot_path}")

    except Exception as e:
        print(f"  ⚠ Plotting failed: {e}")


if __name__ == "__main__":
    main()
