#!/usr/bin/env python3
"""
Ablation Study: Daily Windows Resolution
==========================================

Compares factor extraction quality and generation fidelity with and
without business-day-level (daily) extraction windows.

Design:
  Config A (baseline): windows = [2W, 1ME, 1QE]      (~39 windows/ticker)
  Config B (daily):    windows = [1BD, 2W, 1ME, 1QE]  (~300 windows/ticker)

Period: 2020-01-01 to 2020-12-31 (COVID crash + recovery — rich causal events)
Tickers: AAPL + JPM (tech + financial, 2 sectors)
Cost: ~2,950 LLM calls for daily variant vs ~390 for baseline

Comparison metrics:
  - Number of factors extracted & retained after scoring
  - Factor relevance scores (mean composite score)
  - Cross-window validation prune rate
  - KS statistic, Wasserstein, ACF distance of generated data
  - Kurtosis/skewness fidelity

Usage:
    # Step 1: Run baseline pipeline
    python scripts/run_pipeline.py configs/ablation_daily_baseline.yaml

    # Step 2: Run daily pipeline
    python scripts/run_pipeline.py configs/ablation_daily_with.yaml

    # Step 3: Compare results
    python scripts/ablation_daily_windows.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    results_root = PROJECT_ROOT / "results"
    baseline_dir = results_root / "ablation_daily" / "no_daily"
    daily_dir = results_root / "ablation_daily" / "with_daily"
    out_dir = results_root / "ablation_daily_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = ["AAPL", "JPM"]

    # Check both pipelines have run
    for label, d in [("baseline", baseline_dir), ("daily", daily_dir)]:
        if not d.exists():
            print(f"ERROR: {label} results not found at {d}")
            print(f"Run the pipeline first:")
            cfg = "ablation_daily_baseline" if label == "baseline" else "ablation_daily_with"
            print(f"  python scripts/run_pipeline.py configs/{cfg}.yaml")
            return

    print("═" * 70)
    print("  ABLATION: Daily Windows Resolution Comparison")
    print("═" * 70)
    print(f"  Tickers:    {tickers}")
    print(f"  Period:     2020 (COVID crash + recovery)")
    print(f"  Baseline:   windows = [2W, 1ME, 1QE]")
    print(f"  With-daily: windows = [1BD, 2W, 1ME, 1QE]")
    print("═" * 70)

    all_results = {}

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker}")
        print(f"{'─' * 50}")

        ticker_result = {}

        for label, result_dir in [("baseline", baseline_dir), ("daily", daily_dir)]:
            report_path = result_dir / ticker / "pipeline_report.json"
            if not report_path.exists():
                print(f"  ⚠ {label} pipeline_report.json not found for {ticker}")
                continue

            with open(report_path) as f:
                report = json.load(f)

            phases = report.get("phases", {})
            extraction = phases.get("extraction", {})
            evaluation = phases.get("evaluation", {})
            metrics = evaluation.get("metrics", {})

            n_factors = extraction.get("n_factors", 0)
            presence = extraction.get("presence_shape", [0, 0])

            # Vol ratio
            vr = None
            if metrics.get("synth_std") and metrics.get("real_std") and metrics["real_std"] > 0:
                vr = metrics["synth_std"] / metrics["real_std"]

            ticker_result[label] = {
                "n_factors": n_factors,
                "presence_days": presence[0] if len(presence) > 0 else 0,
                "ks_stat": metrics.get("ks_stat"),
                "ks_pval": metrics.get("ks_pval"),
                "wasserstein": metrics.get("wasserstein"),
                "kl_sym": metrics.get("kl_sym"),
                "acf_dist_returns": metrics.get("acf_dist_returns"),
                "acf_dist_abs_returns": metrics.get("acf_dist_abs_returns"),
                "vol_ratio": vr,
                "vol_rmse": metrics.get("vol_rmse"),
                "vol_corr": metrics.get("vol_corr"),
                "price_rmse": metrics.get("price_rmse"),
                "price_corr": metrics.get("price_corr"),
                "disc_score": metrics.get("disc_score"),
                "pred_score": metrics.get("pred_score"),
                "sharpe_diff": metrics.get("sharpe_diff"),
                "max_dd_diff": metrics.get("max_dd_diff"),
                "var95_diff": metrics.get("var95_diff"),
                "synth_kurt": metrics.get("synth_kurt"),
                "real_kurt": metrics.get("real_kurt"),
                "synth_skew": metrics.get("synth_skew"),
                "real_skew": metrics.get("real_skew"),
            }

            print(f"  {label:>10s}: {n_factors} factors, "
                  f"KS={metrics.get('ks_stat', 0):.4f}, "
                  f"Wass={metrics.get('wasserstein', 0):.4f}")

        all_results[ticker] = ticker_result

    # ── Comparison table ────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  COMPARISON TABLE")
    print(f"{'═' * 70}")

    # lower_better: True = lower is better, False = higher is better, None = closer to target
    metrics_info = [
        ("n_factors", "Unique factors", False),
        ("ks_stat", "KS statistic", True),
        ("ks_pval", "KS p-value", False),
        ("wasserstein", "Wasserstein", True),
        ("kl_sym", "KL symmetric", True),
        ("acf_dist_returns", "ACF dist (returns)", True),
        ("acf_dist_abs_returns", "ACF dist (|returns|)", True),
        ("vol_rmse", "Vol RMSE", True),
        ("vol_corr", "Vol correlation", False),
        ("price_rmse", "Price RMSE", True),
        ("price_corr", "Price correlation", False),
        ("disc_score", "Disc score", None),  # closer to 0.5 is better
        ("pred_score", "Pred score", True),
        ("sharpe_diff", "Sharpe diff", True),
        ("max_dd_diff", "Max DD diff", True),
        ("var95_diff", "VaR95 diff", True),
    ]

    header = f"  {'Metric':<25s} {'Baseline':>12s} {'Daily':>12s}  Winner"
    print(f"\n{header}")
    print(f"  {'─' * 65}")

    # Track wins per ticker and overall
    wins = {"baseline": 0, "daily": 0, "tie": 0}

    for ticker in tickers:
        print(f"\n  {ticker}:")
        b = all_results[ticker].get("baseline", {})
        d = all_results[ticker].get("daily", {})

        for key, name, lower_better in metrics_info:
            bv = b.get(key)
            dv = d.get(key)
            if bv is None or dv is None:
                print(f"    {name:<25s} {'?':>12s} {'?':>12s}")
                continue

            if lower_better is None:
                # Disc score: closer to 0.5 is better
                b_err = abs(bv - 0.5)
                d_err = abs(dv - 0.5)
                if abs(b_err - d_err) < 1e-6:
                    winner = "tie"
                elif d_err < b_err:
                    winner = "daily"
                else:
                    winner = "baseline"
            elif lower_better:
                if abs(bv - dv) < 1e-6:
                    winner = "tie"
                elif dv < bv:
                    winner = "daily"
                else:
                    winner = "baseline"
            else:
                if abs(bv - dv) < 1e-6:
                    winner = "tie"
                elif dv > bv:
                    winner = "daily"
                else:
                    winner = "baseline"

            wins[winner] += 1
            icon = "★" if winner == "daily" else ("·" if winner == "baseline" else "=")
            print(f"    {name:<25s} {bv:>12.4f} {dv:>12.4f}  {icon} {winner}")

        # Vol ratio (closer to 1.0 is better)
        bvr = b.get("vol_ratio")
        dvr = d.get("vol_ratio")
        if bvr is not None and dvr is not None:
            b_err = abs(bvr - 1.0)
            d_err = abs(dvr - 1.0)
            if abs(b_err - d_err) < 1e-6:
                winner = "tie"
            elif d_err < b_err:
                winner = "daily"
            else:
                winner = "baseline"
            wins[winner] += 1
            icon = "★" if winner == "daily" else ("·" if winner == "baseline" else "=")
            print(f"    {'Vol ratio (→1.0)':<25s} {bvr:>12.4f} {dvr:>12.4f}  {icon} {winner}")

        # Kurtosis fidelity (closer to real)
        for stat_name, s_key, r_key in [("Kurtosis match", "synth_kurt", "real_kurt"),
                                          ("Skewness match", "synth_skew", "real_skew")]:
            bs, br = b.get(s_key), b.get(r_key)
            ds, dr = d.get(s_key), d.get(r_key)
            if all(v is not None for v in [bs, br, ds, dr]):
                b_err = abs(bs - br)
                d_err = abs(ds - dr)
                winner = "daily" if d_err < b_err else "baseline"
                wins[winner] += 1
                icon = "★" if winner == "daily" else "·"
                print(f"    {stat_name:<25s} {b_err:>12.4f} {d_err:>12.4f}  {icon} {winner}")

    # ── Summary ─────────────────────────────────────────────────────
    total = wins["baseline"] + wins["daily"] + wins["tie"]
    print(f"\n{'═' * 70}")
    print("  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Total comparisons: {total}")
    print(f"  Baseline wins:     {wins['baseline']} ({wins['baseline']/total*100:.0f}%)")
    print(f"  Daily wins:        {wins['daily']} ({wins['daily']/total*100:.0f}%)")
    print(f"  Ties:              {wins['tie']}")

    if wins["daily"] > wins["baseline"]:
        verdict = "Daily windows IMPROVE generation quality"
    elif wins["daily"] < wins["baseline"]:
        verdict = "Daily windows do NOT clearly improve quality — baseline [2W, 1ME, 1QE] is optimal"
    else:
        verdict = "Daily windows show MIXED results — no clear advantage"

    # Cost analysis
    print(f"\n  Cost analysis:")
    print(f"    Baseline windows: ~37 per ticker (2W + 1ME + 1QE)")
    print(f"    Daily windows:    ~289 per ticker (1BD + 2W + 1ME + 1QE)")
    print(f"    Cost multiplier:  ~7.8x more LLM calls")
    print(f"\n  Verdict: {verdict}")
    print(f"{'═' * 70}")

    # ── Save report ─────────────────────────────────────────────────
    report = {
        "experiment": "ablation_daily_windows",
        "description": "Compare factor extraction and generation quality with/without daily windows",
        "period": "2020-01-01 to 2020-12-31",
        "tickers": tickers,
        "configs": {
            "baseline": "window_sizes: [2W, 1ME, 1QE]",
            "daily": "window_sizes: [1BD, 2W, 1ME, 1QE]",
        },
        "wins": wins,
        "verdict": verdict,
        "per_ticker": all_results,
    }

    report_path = out_dir / "daily_windows_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\n  ✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
