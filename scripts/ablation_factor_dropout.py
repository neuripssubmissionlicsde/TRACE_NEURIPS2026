#!/usr/bin/env python3
"""
Ablation Study: 50% Random Factor Dropout
==========================================

Tests the sensitivity of generated data quality to the number of active
causal factors by randomly deactivating 50% of them.

Uses the already-trained ImpactMatrix and presence matrix — only re-runs
the generation phase (Phase 3).  No LLM calls required.

Produces for each ticker:
  - Comparison metrics (KS, Wasserstein, ACF, Sharpe, etc.)
  - A causal-events plot with:
      • GREEN ▲ / RED ▼  =  active factors (kept)
      • GRAY ▲ / ▼       =  deactivated factors (dropped)
  - JSON report with full metrics and dropout map

Usage:
    python scripts/ablation_factor_dropout.py
    python scripts/ablation_factor_dropout.py --config configs/causal_sde_config.yaml
    python scripts/ablation_factor_dropout.py --seed 123  # different dropout draw
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


# ── Helpers ─────────────────────────────────────────────────────────


def make_dropout_impact_matrix(
    im: ImpactMatrix,
    keep_mask: np.ndarray,
) -> ImpactMatrix:
    """Create a new ImpactMatrix keeping only factors where keep_mask is True."""
    idx = np.where(keep_mask)[0]
    return ImpactMatrix(
        factor_names=[im.factor_names[i] for i in idx],
        base_impact=im.base_impact[idx, :].copy(),
        impact_std=im.impact_std[idx, :].copy(),
        occurrence_prob=im.occurrence_prob[idx].copy(),
        temporal_profile=im.temporal_profile[idx, :].copy(),
        interaction_matrix=im.interaction_matrix[np.ix_(idx, idx)].copy(),
        feature_names=list(im.feature_names),
        nonlinearity_scores=(
            im.nonlinearity_scores[idx].copy()
            if im.nonlinearity_scores is not None else None
        ),
        response_curves=(
            im.response_curves[idx, :, :].copy()
            if im.response_curves is not None else None
        ),
    )


def acf(x: np.ndarray, nlags: int = 20) -> np.ndarray:
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(nlags + 1)
    result = np.correlate(x, x, mode="full")
    result = result[len(x) - 1 :]
    return result[: nlags + 1] / (var * len(x))


def compute_extended_metrics(
    real_rets: np.ndarray, synth_rets: np.ndarray,
) -> dict:
    """Compute full set of comparison metrics."""
    base = compute_distributional_metrics(real_rets, synth_rets)

    # Wasserstein
    from scipy.stats import wasserstein_distance
    base["wasserstein"] = float(wasserstein_distance(real_rets, synth_rets))

    # ACF distances
    real_acf = acf(real_rets)
    synth_acf = acf(synth_rets)
    base["acf_dist_returns"] = float(np.mean(np.abs(real_acf - synth_acf)))

    real_abs_acf = acf(np.abs(real_rets))
    synth_abs_acf = acf(np.abs(synth_rets))
    base["acf_dist_abs_returns"] = float(np.mean(np.abs(real_abs_acf - synth_abs_acf)))

    # Rolling vol comparison
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

    # Price correlation
    real_cum = np.cumsum(real_rets)
    synth_cum = np.cumsum(synth_rets)
    min_len = min(len(real_cum), len(synth_cum))
    if min_len > 5:
        corr = np.corrcoef(real_cum[:min_len], synth_cum[:min_len])[0, 1]
        base["price_corr"] = float(corr) if np.isfinite(corr) else 0.0

    # Sharpe & MaxDD diff
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


# ── Plotting ────────────────────────────────────────────────────────


def plot_factor_dropout_map(
    real_df: pd.DataFrame,
    analyses: list[dict],
    keep_mask: np.ndarray,
    factor_names: list[str],
    ticker: str,
    output_dir: str,
    top_n: int = 30,
    min_magnitude: float = 0.08,
) -> None:
    """Plot real close price annotated with active (color) and dropped (gray) factors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    kept_set = {factor_names[i] for i in range(len(factor_names)) if keep_mask[i]}
    dropped_set = {factor_names[i] for i in range(len(factor_names)) if not keep_mask[i]}

    # Build event list from analyses
    events = []
    for w in analyses:
        mid_date = pd.Timestamp(w["start_date"]) + (
            pd.Timestamp(w["end_date"]) - pd.Timestamp(w["start_date"])
        ) / 2
        for f in w.get("factors", []):
            name = f["name"]
            if any(skip in name.lower() for skip in ("noise", "residual")):
                continue
            mag = abs(f["magnitude"])
            if mag < min_magnitude:
                continue
            # Determine if this factor survived scoring and is in our matrix
            is_kept = name in kept_set
            is_dropped = name in dropped_set
            if not is_kept and not is_dropped:
                continue  # Factor was pruned by scoring — not in ImpactMatrix
            events.append({
                "date": mid_date,
                "name": name.replace("_", " ").title(),
                "magnitude": mag,
                "direction": f["direction"],
                "window_start": w["start_date"],
                "window_end": w["end_date"],
                "active": is_kept,
            })

    # Dedup: keep highest-magnitude occurrence per name
    best: dict[str, dict] = {}
    for ev in events:
        key = ev["name"]
        if key not in best or ev["magnitude"] > best[key]["magnitude"]:
            best[key] = ev
    events = sorted(best.values(), key=lambda x: x["magnitude"], reverse=True)

    # Split into active and dropped, take top_n/2 from each for readability
    active_events = [e for e in events if e["active"]][:top_n // 2 + 5]
    dropped_events = [e for e in events if not e["active"]][:top_n // 2 + 5]
    plot_events = active_events + dropped_events
    plot_events.sort(key=lambda x: x["date"])

    # Prepare real data
    real = real_df.copy()
    real["date"] = pd.to_datetime(real["date"])
    real = real.sort_values("date")

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(real["date"], real["close"], color="#1565C0", linewidth=1.2,
            alpha=0.9, label="Real Close")

    y_lo, y_hi = real["close"].min(), real["close"].max()
    y_rng = y_hi - y_lo
    n_events = len(plot_events)
    stagger_k = max(5, min(n_events // 3, 12))
    ann_fontsize = max(4.5, 6.5 - 0.02 * max(0, n_events - 40))

    for i, ev in enumerate(plot_events):
        idx = (real["date"] - ev["date"]).abs().idxmin()
        price_at = real.loc[idx, "close"]
        event_date = real.loc[idx, "date"]
        ws = pd.Timestamp(ev["window_start"])
        we = pd.Timestamp(ev["window_end"])

        if ev["active"]:
            # Active: normal color scheme
            if ev["direction"] > 0.15:
                color, marker = "#2E7D32", "^"
            elif ev["direction"] < -0.15:
                color, marker = "#C62828", "v"
            else:
                color, marker = "#757575", "o"
            alpha_line = 0.6
            alpha_marker = 0.85
            alpha_text = 0.8
        else:
            # Dropped: gray with reduced alpha
            color = "#9E9E9E"
            marker = "^" if ev["direction"] > 0.15 else (
                "v" if ev["direction"] < -0.15 else "o"
            )
            alpha_line = 0.3
            alpha_marker = 0.45
            alpha_text = 0.5

        # Extent line
        ax.hlines(y=price_at, xmin=ws, xmax=we, colors=color,
                  linewidth=1.8 if ev["active"] else 1.0,
                  alpha=alpha_line, zorder=4,
                  linestyles="solid" if ev["active"] else "dashed")
        cap_h = y_rng * 0.012
        for cap_x in (ws, we):
            ax.vlines(x=cap_x, ymin=price_at - cap_h, ymax=price_at + cap_h,
                      colors=color, linewidth=1.4 if ev["active"] else 0.8,
                      alpha=alpha_line, zorder=4)

        # Marker
        size = (30 + ev["magnitude"] * 200) * (1.0 if ev["active"] else 0.6)
        ax.scatter(event_date, price_at, s=size, color=color, zorder=5,
                   alpha=alpha_marker, marker=marker, edgecolors="white",
                   linewidth=0.6)

        # Annotation
        offset_sign = 1 if (i % 2 == 0) else -1
        text_y = price_at + offset_sign * y_rng * (0.06 + 0.03 * (i % stagger_k))
        text_y = max(y_lo - y_rng * 0.05, min(y_hi + y_rng * 0.15, text_y))

        label = ev["name"]
        if not ev["active"]:
            label = f"✗ {label}"

        ax.annotate(
            label, xy=(event_date, price_at), xytext=(event_date, text_y),
            fontsize=ann_fontsize, color=color, fontweight="bold",
            ha="center", va="bottom" if offset_sign > 0 else "top",
            arrowprops=dict(arrowstyle="-", color=color,
                            alpha=0.4 if ev["active"] else 0.2, linewidth=0.5),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=alpha_text, linewidth=0.5),
        )

    # Legend
    legend_handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32",
               markersize=8, label="Active — Bullish"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#C62828",
               markersize=8, label="Active — Bearish"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9E9E9E",
               markersize=8, label="Dropped (50% dropout)"),
        Line2D([0], [0], color="#1565C0", linewidth=2, label="Real Close"),
        Line2D([0, 1], [0, 0], color="#9E9E9E", linewidth=1.2,
               linestyle="dashed", label="Dropped Window"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.9)

    n_kept = int(keep_mask.sum())
    n_total = len(keep_mask)
    ax.set_title(
        f"{ticker} — Factor Dropout Ablation "
        f"({n_kept}/{n_total} factors active, "
        f"{n_total - n_kept} dropped)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Close Price ($)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--", which="both")

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"factor_dropout_{ticker}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Dropout map saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="50% Factor Dropout Ablation")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "causal_sde_config.yaml"))
    parser.add_argument("--seed", type=int, default=42, help="Seed for dropout draw")
    parser.add_argument("--dropout-frac", type=float, default=0.5, help="Fraction of factors to drop")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["AAPL"])
    output_name = cfg.get("output_dir", "multi_asset_2018_2023")
    results_dir = PROJECT_ROOT / "results" / output_name
    n_samples = cfg.get("generation", {}).get("n_samples", 10)
    gen_seed = cfg.get("seed", 42)

    out_dir = PROJECT_ROOT / "results" / "ablation_factor_dropout"
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = ["open", "high", "low", "close", "volume"]

    print("═" * 70)
    print(f"  ABLATION: {args.dropout_frac:.0%} Random Factor Dropout")
    print("═" * 70)
    print(f"  Tickers:      {tickers}")
    print(f"  Dropout frac: {args.dropout_frac:.0%}")
    print(f"  Dropout seed: {args.seed}")
    print(f"  Gen samples:  {n_samples}")
    print(f"  Output:       {out_dir}")
    print("═" * 70)

    t_start = time.time()
    all_results = {}

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker}")
        print(f"{'─' * 50}")

        ticker_dir = results_dir / ticker

        # ── Load impact matrix ──────────────────────────────────────
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            print(f"  ⚠ impact_matrix.json not found for {ticker}, skipping")
            continue
        im = ImpactMatrix.load(str(im_path))
        K = len(im.factor_names)
        print(f"  ✓ Loaded ImpactMatrix: {K} factors")

        # ── Load analyses (for plotting) ────────────────────────────
        analyses_path = ticker_dir / "analyses.json"
        analyses = []
        if analyses_path.exists():
            with open(analyses_path) as f:
                analyses = json.load(f)

        # ── Load real data ──────────────────────────────────────────
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

        # ── Load Causal SDE baseline ────────────────────────────────
        csde_csv = ticker_dir / "synthetic_data_causal_sde.csv"
        if not csde_csv.exists():
            print(f"  ⚠ No Causal SDE results for {ticker}, skipping")
            continue
        csde_df = pd.read_csv(csde_csv)
        if "sample" in csde_df.columns:
            csde_s0 = csde_df[csde_df["sample"] == 0].sort_values("date")
        else:
            csde_s0 = csde_df.sort_values("date")
        csde_close = csde_s0["close"].values
        csde_rets = np.diff(np.log(np.clip(csde_close, 1e-8, None)))
        print(f"  ✓ Causal SDE baseline loaded ({len(csde_s0)} rows)")

        # ── Build historical presence for the FULL matrix ───────────
        # We need the presence matrix to pass to generator.
        # Rebuild it from analyses matching the full ImpactMatrix factors.
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

        # ── Random dropout ──────────────────────────────────────────
        rng = np.random.RandomState(args.seed)
        keep_mask = np.ones(K, dtype=bool)
        n_drop = int(K * args.dropout_frac)
        drop_indices = rng.choice(K, size=n_drop, replace=False)
        keep_mask[drop_indices] = False

        kept_names = [im.factor_names[i] for i in range(K) if keep_mask[i]]
        dropped_names = [im.factor_names[i] for i in range(K) if not keep_mask[i]]
        print(f"  Dropout: {n_drop}/{K} factors deactivated ({args.dropout_frac:.0%})")
        print(f"  Kept:    {len(kept_names)} factors")

        # ── Build dropout ImpactMatrix & presence ───────────────────
        dropout_im = make_dropout_impact_matrix(im, keep_mask)
        keep_indices = np.where(keep_mask)[0]
        dropout_presence = full_presence[:, keep_indices]

        # ── Generate with dropout ───────────────────────────────────
        print(f"  Generating dropout SDE ({len(kept_names)} factors)...")
        dropout_gen = ImpactDrivenGenerator(dropout_im, window_trading_days=63)
        dropout_prices = dropout_gen.generate_scenario(
            n_steps=n_steps,
            initial_prices=initial_prices,
            n_samples=n_samples,
            noise_scale=1.0,
            seed=gen_seed,
            real_log_returns=real_log_returns,
            real_volume=real_volume,
            historical_presence=dropout_presence,
            real_prices=real_prices,
            real_dates=real_dates,
        )

        close_col = min(3, dropout_prices.shape[2] - 1)
        dropout_close = dropout_prices[0, :, close_col]
        dropout_rets = np.diff(np.log(np.clip(dropout_close, 1e-8, None)))
        print(f"  ✓ Dropout SDE generated ({n_steps} steps × {n_samples} samples)")

        # ── Compute metrics ─────────────────────────────────────────
        csde_metrics = compute_extended_metrics(real_rets, csde_rets)
        dropout_metrics = compute_extended_metrics(real_rets, dropout_rets)

        # Multi-sample KS average
        dropout_ks_all = []
        for s in range(n_samples):
            s_close = dropout_prices[s, :, close_col]
            s_rets = np.diff(np.log(np.clip(s_close, 1e-8, None)))
            ks, _ = stats.ks_2samp(real_rets, s_rets)
            dropout_ks_all.append(ks)
        dropout_metrics["mean_ks_all_samples"] = float(np.mean(dropout_ks_all))

        ticker_results = {
            "n_factors_full": K,
            "n_factors_kept": int(keep_mask.sum()),
            "n_factors_dropped": int(n_drop),
            "dropout_fraction": args.dropout_frac,
            "dropped_factors": dropped_names,
            "kept_factors": kept_names,
            "causal_sde": csde_metrics,
            "dropout_sde": dropout_metrics,
        }
        all_results[ticker] = ticker_results

        # ── Print comparison ────────────────────────────────────────
        metrics_display = [
            ("ks_statistic", "KS statistic", True),
            ("wasserstein", "Wasserstein", True),
            ("acf_dist_returns", "ACF dist (ret)", True),
            ("acf_dist_abs_returns", "ACF dist (|ret|)", True),
            ("vol_rmse", "Vol RMSE", True),
            ("vol_corr", "Vol correlation", False),
            ("price_corr", "Price correlation", False),
            ("sharpe_diff", "Sharpe diff", True),
            ("max_dd_diff", "Max DD diff", True),
            ("volatility_ratio", "Vol ratio (→1)", None),
            ("kurtosis_diff", "Kurtosis diff", True),
            ("skewness_diff", "Skewness diff", True),
        ]

        print(f"\n  {'Metric':<25s} {'Full CSDE':>12s} {'50% Dropout':>12s} {'Winner':>10s}")
        print(f"  {'─' * 62}")

        ticker_wins = {"full": 0, "dropout": 0}
        for key, name, lower_better in metrics_display:
            cv = csde_metrics.get(key)
            dv = dropout_metrics.get(key)
            if cv is None or dv is None:
                continue

            if lower_better is None:
                # Closer to 1.0
                c_err = abs(cv - 1.0)
                d_err = abs(dv - 1.0)
                winner = "Full" if c_err <= d_err else "Dropout"
            elif lower_better:
                winner = "Full" if cv <= dv else "Dropout"
            else:
                winner = "Full" if cv >= dv else "Dropout"

            if winner == "Full":
                ticker_wins["full"] += 1
            else:
                ticker_wins["dropout"] += 1

            icon = "·" if winner == "Full" else "★"
            print(f"  {name:<25s} {cv:>12.4f} {dv:>12.4f}  {icon} {winner}")

        ticker_results["wins_full"] = ticker_wins["full"]
        ticker_results["wins_dropout"] = ticker_wins["dropout"]

        # ── Plot dropout map ────────────────────────────────────────
        ticker_out = out_dir / ticker
        ticker_out.mkdir(parents=True, exist_ok=True)

        plot_factor_dropout_map(
            real_df=ticker_df,
            analyses=analyses,
            keep_mask=keep_mask,
            factor_names=im.factor_names,
            ticker=ticker,
            output_dir=str(ticker_out),
        )

        # ── Save per-ticker JSON ────────────────────────────────────
        ticker_report_path = ticker_out / "dropout_results.json"
        with open(ticker_report_path, "w") as f:
            json.dump(ticker_results, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
        print(f"  ✓ Results saved: {ticker_report_path}")

    # ── Aggregate ───────────────────────────────────────────────────
    total_time = time.time() - t_start

    print(f"\n{'═' * 70}")
    print("  AGGREGATE COMPARISON")
    print(f"{'═' * 70}")

    total_full = sum(r.get("wins_full", 0) for r in all_results.values())
    total_dropout = sum(r.get("wins_dropout", 0) for r in all_results.values())
    total = total_full + total_dropout

    print(f"  Full CSDE wins:       {total_full}/{total}")
    print(f"  50% Dropout wins:     {total_dropout}/{total}")

    if total_full > total_dropout:
        verdict = (
            f"Removing {args.dropout_frac:.0%} of causal factors DEGRADES quality — "
            "factors carry meaningful causal signal"
        )
    elif total_full < total_dropout:
        verdict = (
            f"Removing {args.dropout_frac:.0%} of factors does NOT degrade quality — "
            "pipeline may have redundant factors"
        )
    else:
        verdict = "Mixed results — factor dropout shows marginal impact"

    print(f"  Verdict: {verdict}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'═' * 70}")

    # ── Save global report ──────────────────────────────────────────
    report = {
        "experiment": "ablation_factor_dropout",
        "description": f"Full CSDE ({K} factors) vs {args.dropout_frac:.0%} random dropout",
        "dropout_fraction": args.dropout_frac,
        "dropout_seed": args.seed,
        "tickers": list(all_results.keys()),
        "aggregate": {
            "full_wins": total_full,
            "dropout_wins": total_dropout,
            "total_comparisons": total,
            "verdict": verdict,
        },
        "per_ticker": all_results,
        "total_time_seconds": round(total_time, 1),
    }

    report_path = out_dir / "factor_dropout_ablation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.integer, np.floating)) else str(o))
    print(f"\n  ✓ Report saved to {report_path}")

    # ── Comparison bar plot ─────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tickers_list = list(all_results.keys())
        if not tickers_list:
            return

        x = np.arange(len(tickers_list))
        width = 0.3

        metrics_to_plot = [
            ("ks_statistic", "KS Statistic ↓", True),
            ("wasserstein", "Wasserstein ↓", True),
            ("sharpe_diff", "Sharpe Diff ↓", True),
        ]

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, (key, title, lower_better) in zip(axes, metrics_to_plot):
            full_vals = [all_results[t]["causal_sde"].get(key, 0) for t in tickers_list]
            drop_vals = [all_results[t]["dropout_sde"].get(key, 0) for t in tickers_list]

            bars1 = ax.bar(x - width / 2, full_vals, width, label="Full CSDE",
                           color="#1565C0", alpha=0.85)
            bars2 = ax.bar(x + width / 2, drop_vals, width, label="50% Dropout",
                           color="#9E9E9E", alpha=0.85)

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(tickers_list)
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Factor Dropout Ablation ({args.dropout_frac:.0%} dropped)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        plot_path = out_dir / "factor_dropout_comparison.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Comparison plot: {plot_path}")

    except Exception as e:
        print(f"  ⚠ Could not generate comparison plot: {e}")


if __name__ == "__main__":
    main()
