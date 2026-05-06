#!/usr/bin/env python3
"""
Ablation Study: Shuffled Factors — Negative Control
=====================================================

Tests whether the specific assignment of factors to time windows matters,
by randomly permuting factor directions and magnitudes across windows.

If the causal module is meaningful, shuffled factors should produce
WORSE synthetic data compared to the real (correctly assigned) factors.

Uses the already-trained ImpactMatrix and presence matrix — only re-runs
the generation phase (Phase 3).  No LLM calls required ($0 cost).

Three conditions:
  1. Baseline: original ImpactMatrix (from final run)
  2. Shuffled Directions: base_impact signs randomly permuted across factors
  3. Shuffled Magnitudes: base_impact magnitudes randomly reassigned across factors

Usage:
    python scripts/ablation_shuffle_control.py
    python scripts/ablation_shuffle_control.py --config configs/final_run.yaml
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

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator
from sde_causal_generator.benchmark import compute_distributional_metrics


# ── Shuffle helpers ─────────────────────────────────────────────────


def shuffle_directions(im: ImpactMatrix, seed: int = 42) -> ImpactMatrix:
    """Randomly flip the signs of base_impact rows (permute across factors)."""
    rng = np.random.RandomState(seed)
    new_impact = im.base_impact.copy()
    K = new_impact.shape[0]
    # Randomly permute which factor gets which sign pattern
    perm = rng.permutation(K)
    signs = np.sign(im.base_impact)
    new_signs = signs[perm]
    new_impact = np.abs(new_impact) * new_signs
    return ImpactMatrix(
        factor_names=list(im.factor_names),
        base_impact=new_impact,
        impact_std=im.impact_std.copy(),
        occurrence_prob=im.occurrence_prob.copy(),
        temporal_profile=im.temporal_profile.copy(),
        interaction_matrix=im.interaction_matrix.copy(),
        feature_names=list(im.feature_names),
        nonlinearity_scores=im.nonlinearity_scores.copy() if im.nonlinearity_scores is not None else None,
        response_curves=im.response_curves.copy() if im.response_curves is not None else None,
    )


def shuffle_magnitudes(im: ImpactMatrix, seed: int = 42) -> ImpactMatrix:
    """Randomly permute the magnitude (absolute value) of base_impact across factors."""
    rng = np.random.RandomState(seed)
    new_impact = im.base_impact.copy()
    K = new_impact.shape[0]
    perm = rng.permutation(K)
    magnitudes = np.abs(im.base_impact)
    new_magnitudes = magnitudes[perm]
    new_impact = new_magnitudes * np.sign(im.base_impact)
    return ImpactMatrix(
        factor_names=list(im.factor_names),
        base_impact=new_impact,
        impact_std=im.impact_std.copy(),
        occurrence_prob=im.occurrence_prob.copy(),
        temporal_profile=im.temporal_profile.copy(),
        interaction_matrix=im.interaction_matrix.copy(),
        feature_names=list(im.feature_names),
        nonlinearity_scores=im.nonlinearity_scores.copy() if im.nonlinearity_scores is not None else None,
        response_curves=im.response_curves.copy() if im.response_curves is not None else None,
    )


def shuffle_presence(presence: np.ndarray, seed: int = 42) -> np.ndarray:
    """Randomly permute the temporal assignment of factor activations."""
    rng = np.random.RandomState(seed)
    shuffled = presence.copy()
    K = shuffled.shape[1]
    for k in range(K):
        rng.shuffle(shuffled[:, k])
    return shuffled


# ── Metrics ─────────────────────────────────────────────────────────


def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(nlags + 1)
    result = np.correlate(x, x, mode="full")
    result = result[len(x) - 1:]
    return result[:nlags + 1] / (var * len(x))


def compute_extended_metrics(real_rets: np.ndarray, synth_rets: np.ndarray) -> dict:
    base = compute_distributional_metrics(real_rets, synth_rets)
    from scipy.stats import wasserstein_distance
    base["wasserstein"] = float(wasserstein_distance(real_rets, synth_rets))

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

    real_cum = np.cumsum(real_rets)
    synth_cum = np.cumsum(synth_rets)
    min_len = min(len(real_cum), len(synth_cum))
    if min_len > 5:
        corr = np.corrcoef(real_cum[:min_len], synth_cum[:min_len])[0, 1]
        base["price_corr"] = float(corr) if np.isfinite(corr) else 0.0

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


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Shuffled Factors Negative Control")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "final_run.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["AAPL"])
    output_name = cfg.get("output_dir", "final_run_gpt4o_mini")
    results_dir = PROJECT_ROOT / "results" / output_name
    n_samples = cfg.get("generation", {}).get("n_samples", 10)
    gen_seed = cfg.get("seed", 42)

    out_dir = PROJECT_ROOT / "results" / "ablation_shuffle_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["open", "high", "low", "close", "volume"]

    CONDITIONS = ["baseline", "shuffled_directions", "shuffled_magnitudes", "shuffled_presence"]

    print("=" * 70)
    print("  ABLATION: Shuffled Factors — Negative Control")
    print("=" * 70)
    print(f"  Tickers:    {tickers}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {out_dir}")
    print("=" * 70)

    t_start = time.time()
    all_results = {}

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker}")
        print(f"{'─' * 50}")

        ticker_dir = results_dir / ticker

        # Load impact matrix
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            print(f"  ⚠ impact_matrix.json not found for {ticker}, skipping")
            continue
        im = ImpactMatrix.load(str(im_path))
        K = len(im.factor_names)
        print(f"  ✓ Loaded ImpactMatrix: {K} factors")

        # Load analyses (for presence matrix)
        analyses_path = ticker_dir / "analyses.json"
        if not analyses_path.exists():
            print(f"  ⚠ analyses.json not found for {ticker}, skipping")
            continue
        with open(analyses_path) as f:
            analyses = json.load(f)

        # Load real data
        training_csv = results_dir / "training_data.csv"
        if not training_csv.exists():
            print(f"  ⚠ training_data.csv not found, skipping")
            continue
        df = pd.read_csv(training_csv)
        df["date"] = pd.to_datetime(df["date"])
        ticker_df = df[df["tic"] == ticker].sort_values("date").copy()

        if len(ticker_df) < 100:
            print(f"  ⚠ {ticker}: insufficient data ({len(ticker_df)} rows), skipping")
            continue

        feature_cols = [c for c in feature_names if c in ticker_df.columns]
        real_prices = ticker_df[feature_cols].values.astype(np.float64)
        real_prices = np.clip(real_prices, 1e-8, None)
        real_log_returns = np.diff(np.log(real_prices), axis=0)
        real_volume = ticker_df["volume"].values.astype(np.float64)
        real_dates = pd.to_datetime(ticker_df["date"]).values
        initial_prices = real_prices[0]
        n_steps = len(ticker_df)

        real_close = ticker_df["close"].values
        real_rets = np.diff(np.log(np.clip(real_close, 1e-8, None)))

        # Build historical presence for the full matrix
        dates = pd.to_datetime(ticker_df["date"]).values
        date_to_row = {d: i for i, d in enumerate(dates)}
        name_to_idx = {n: i for i, n in enumerate(im.factor_names)}

        full_presence = np.zeros((len(dates), K), dtype=np.float64)
        for w in analyses:
            ws = pd.Timestamp(w["start_date"])
            we = pd.Timestamp(w["end_date"])
            for f_dict in w.get("factors", []):
                col = name_to_idx.get(f_dict["name"])
                if col is None:
                    continue
                mag = abs(f_dict["magnitude"])
                for d in dates:
                    if ws <= pd.Timestamp(d) <= we:
                        row = date_to_row[d]
                        full_presence[row, col] = max(full_presence[row, col], mag)

        ticker_results = {"n_factors": K, "conditions": {}}

        for condition in CONDITIONS:
            print(f"\n  Condition: {condition}")

            if condition == "baseline":
                cond_im = im
                cond_presence = full_presence
            elif condition == "shuffled_directions":
                cond_im = shuffle_directions(im, seed=args.seed)
                cond_presence = full_presence
            elif condition == "shuffled_magnitudes":
                cond_im = shuffle_magnitudes(im, seed=args.seed)
                cond_presence = full_presence
            elif condition == "shuffled_presence":
                cond_im = im
                cond_presence = shuffle_presence(full_presence, seed=args.seed)
            else:
                continue

            gen = ImpactDrivenGenerator(cond_im, window_trading_days=63)
            prices = gen.generate_scenario(
                n_steps=n_steps,
                initial_prices=initial_prices,
                n_samples=n_samples,
                noise_scale=1.0,
                seed=gen_seed,
                real_log_returns=real_log_returns,
                real_volume=real_volume,
                historical_presence=cond_presence,
                real_prices=real_prices,
                real_dates=real_dates,
            )

            close_col = min(3, prices.shape[2] - 1)
            synth_close = prices[0, :, close_col]
            synth_rets = np.diff(np.log(np.clip(synth_close, 1e-8, None)))

            metrics = compute_extended_metrics(real_rets, synth_rets)

            # Multi-sample KS average
            ks_all = []
            for s in range(n_samples):
                s_close = prices[s, :, close_col]
                s_rets = np.diff(np.log(np.clip(s_close, 1e-8, None)))
                ks, _ = stats.ks_2samp(real_rets, s_rets)
                ks_all.append(ks)
            metrics["mean_ks_all_samples"] = float(np.mean(ks_all))

            ticker_results["conditions"][condition] = metrics
            print(f"    KS={metrics.get('ks_statistic', 'N/A'):.4f}  "
                  f"WD={metrics.get('wasserstein', 'N/A'):.5f}  "
                  f"Sharpe_diff={metrics.get('sharpe_diff', 'N/A'):.4f}")

        # Print comparison table
        print(f"\n  {'Metric':<25s}", end="")
        for c in CONDITIONS:
            print(f" {c:>18s}", end="")
        print()
        print(f"  {'─' * (25 + 19 * len(CONDITIONS))}")

        metrics_display = [
            ("ks_statistic", "KS statistic", True),
            ("wasserstein", "Wasserstein", True),
            ("acf_dist_returns", "ACF dist (ret)", True),
            ("acf_dist_abs_returns", "ACF dist (|ret|)", True),
            ("sharpe_diff", "Sharpe diff", True),
            ("max_dd_diff", "Max DD diff", True),
            ("vol_rmse", "Vol RMSE", True),
            ("vol_corr", "Vol correlation", False),
            ("price_corr", "Price correlation", False),
        ]

        wins = {c: 0 for c in CONDITIONS}
        for key, name, lower_better in metrics_display:
            print(f"  {name:<25s}", end="")
            vals = {}
            for c in CONDITIONS:
                v = ticker_results["conditions"][c].get(key)
                vals[c] = v
                if v is not None:
                    print(f" {v:>18.4f}", end="")
                else:
                    print(f" {'N/A':>18s}", end="")

            # Determine winner (baseline should be best if causal module works)
            baseline_v = vals.get("baseline")
            if baseline_v is not None:
                for c in CONDITIONS:
                    if c == "baseline":
                        continue
                    cv = vals.get(c)
                    if cv is None:
                        continue
                    if lower_better is None:
                        if abs(baseline_v - 1.0) <= abs(cv - 1.0):
                            wins["baseline"] += 1
                        else:
                            wins[c] += 1
                    elif lower_better:
                        if baseline_v <= cv:
                            wins["baseline"] += 1
                        else:
                            wins[c] += 1
                    else:
                        if baseline_v >= cv:
                            wins["baseline"] += 1
                        else:
                            wins[c] += 1
            print()

        ticker_results["wins"] = wins
        all_results[ticker] = ticker_results

    # Aggregate
    total_time = time.time() - t_start

    print(f"\n{'=' * 70}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    agg_wins = {c: 0 for c in CONDITIONS}
    for ticker_r in all_results.values():
        for c, w in ticker_r.get("wins", {}).items():
            agg_wins[c] += w

    total_comparisons = sum(agg_wins.values())
    for c in CONDITIONS:
        pct = agg_wins[c] / max(total_comparisons, 1) * 100
        print(f"  {c:<25s}: {agg_wins[c]:>4d} wins ({pct:.1f}%)")

    baseline_wins = agg_wins.get("baseline", 0)
    shuffle_wins = sum(v for k, v in agg_wins.items() if k != "baseline")

    if baseline_wins > shuffle_wins:
        verdict = ("Shuffled factors produce WORSE synthetic data — "
                   "the causal module captures meaningful temporal structure")
    elif baseline_wins < shuffle_wins:
        verdict = ("Shuffled factors produce COMPARABLE or BETTER data — "
                   "causal assignment may not be adding value beyond factor presence")
    else:
        verdict = "Mixed results — some benefit from correct causal assignment"

    print(f"\n  Verdict: {verdict}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")

    # Save report
    report = {
        "experiment": "ablation_shuffle_control",
        "description": "Negative control: shuffled factor directions/magnitudes/presence vs baseline",
        "conditions": CONDITIONS,
        "seed": args.seed,
        "tickers": list(all_results.keys()),
        "aggregate": {
            "wins": agg_wins,
            "total_comparisons": total_comparisons,
            "verdict": verdict,
        },
        "per_ticker": all_results,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = out_dir / "shuffle_control_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\n  ✓ Report saved to {report_path}")

    # Comparison bar plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tickers_list = list(all_results.keys())
        if not tickers_list:
            return

        x = np.arange(len(tickers_list))
        width = 0.18
        colors = {"baseline": "#1565C0", "shuffled_directions": "#E65100",
                  "shuffled_magnitudes": "#9E9E9E", "shuffled_presence": "#6A1B9A"}

        metrics_to_plot = [
            ("ks_statistic", "KS Statistic ↓"),
            ("wasserstein", "Wasserstein ↓"),
            ("sharpe_diff", "Sharpe Diff ↓"),
        ]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, (key, title) in zip(axes, metrics_to_plot):
            for i, cond in enumerate(CONDITIONS):
                vals = [all_results[t]["conditions"][cond].get(key, 0) for t in tickers_list]
                offset = (i - len(CONDITIONS) / 2 + 0.5) * width
                ax.bar(x + offset, vals, width, label=cond.replace("_", " ").title(),
                       color=colors.get(cond, "#333"), alpha=0.85)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(tickers_list)
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Shuffle Control — Negative Control Ablation", fontsize=14, fontweight="bold")
        fig.tight_layout()
        plot_path = out_dir / "shuffle_control_comparison.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        fig.savefig(str(out_dir / "shuffle_control_comparison.pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Comparison plot: {plot_path}")

    except Exception as e:
        print(f"  ⚠ Could not generate comparison plot: {e}")


if __name__ == "__main__":
    main()
