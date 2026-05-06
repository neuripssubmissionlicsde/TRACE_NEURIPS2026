#!/usr/bin/env python3
"""
Generate Synthetic Datasets for FinRL Benchmark: Full vs 50% Dropout
=====================================================================

Produces two sets of synthetic OHLCV CSVs using the already-trained
ImpactMatrix and presence matrix (no LLM calls, no re-training):

  Condition A (full):    All K causal factors active
  Condition B (dropout): 50% of factors randomly deactivated

Each condition generates 1 synthetic trajectory per ticker, saved in
the standard FinRL-compatible format (date, tic, open, high, low,
close, volume).  A combined multi-asset CSV is also produced for
direct ingestion by FinRL.

Additionally, a dropout-map PNG is generated per ticker showing which
factors were kept (colour) and which were dropped (gray).

Usage:
    python scripts/generate_dropout_benchmark.py
    python scripts/generate_dropout_benchmark.py --seed 42
    python scripts/generate_dropout_benchmark.py --dropout-frac 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.generate import ImpactDrivenGenerator


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


def build_presence_from_analyses(
    analyses: list[dict],
    factor_names: list[str],
    dates: np.ndarray,
) -> np.ndarray:
    """Rebuild presence matrix (T, K) from analyses JSON."""
    date_to_row = {d: i for i, d in enumerate(dates)}
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    K = len(factor_names)
    presence = np.zeros((len(dates), K), dtype=np.float64)

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
                    presence[row, col] = max(presence[row, col], mag)
    return presence


def generate_synthetic_csv(
    im: ImpactMatrix,
    presence: np.ndarray,
    ticker_df: pd.DataFrame,
    ticker: str,
    seed: int,
    feature_names: list[str],
) -> pd.DataFrame:
    """Generate 1 synthetic trajectory and return as FinRL-format DataFrame."""
    feature_cols = [c for c in feature_names if c in ticker_df.columns]
    real_prices = ticker_df[feature_cols].values.astype(np.float64)
    real_prices = np.clip(real_prices, 1e-8, None)
    real_log_returns = np.diff(np.log(real_prices), axis=0)
    real_volume = ticker_df["volume"].values.astype(np.float64)
    real_dates = pd.to_datetime(ticker_df["date"]).values
    initial_prices = real_prices[0]
    n_steps = len(ticker_df)

    gen = ImpactDrivenGenerator(im, window_trading_days=63)
    prices = gen.generate_scenario(
        n_steps=n_steps,
        initial_prices=initial_prices,
        n_samples=1,
        noise_scale=1.0,
        seed=seed,
        real_log_returns=real_log_returns,
        real_volume=real_volume,
        historical_presence=presence,
        real_prices=real_prices,
        real_dates=real_dates,
    )

    # prices shape: (1, n_steps, M)
    synth = prices[0]  # (n_steps, M)
    dates = pd.to_datetime(ticker_df["date"]).values

    df = pd.DataFrame({
        "date": dates,
        "tic": ticker,
        "open": synth[:, 0],
        "high": synth[:, 1],
        "low": synth[:, 2],
        "close": synth[:, 3],
        "volume": synth[:, 4] if synth.shape[1] > 4 else real_volume,
    })
    return df


# ── Plotting ────────────────────────────────────────────────────────


def plot_dropout_map(
    real_df: pd.DataFrame,
    analyses: list[dict],
    keep_mask: np.ndarray,
    factor_names: list[str],
    ticker: str,
    output_path: str,
    top_n: int = 30,
    min_magnitude: float = 0.08,
) -> None:
    """Plot real price with active (colour) and dropped (gray) factors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D

    kept_set = {factor_names[i] for i in range(len(factor_names)) if keep_mask[i]}
    dropped_set = {factor_names[i] for i in range(len(factor_names)) if not keep_mask[i]}

    # Build event list
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
            is_kept = name in kept_set
            is_dropped = name in dropped_set
            if not is_kept and not is_dropped:
                continue
            events.append({
                "date": mid_date,
                "name": name.replace("_", " ").title(),
                "magnitude": mag,
                "direction": f["direction"],
                "window_start": w["start_date"],
                "window_end": w["end_date"],
                "active": is_kept,
            })

    # Dedup by name
    best: dict[str, dict] = {}
    for ev in events:
        key = ev["name"]
        if key not in best or ev["magnitude"] > best[key]["magnitude"]:
            best[key] = ev
    events = sorted(best.values(), key=lambda x: x["magnitude"], reverse=True)

    active_events = [e for e in events if e["active"]][:top_n // 2 + 5]
    dropped_events = [e for e in events if not e["active"]][:top_n // 2 + 5]
    plot_events = sorted(active_events + dropped_events, key=lambda x: x["date"])

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
            if ev["direction"] > 0.15:
                color, marker = "#2E7D32", "^"
            elif ev["direction"] < -0.15:
                color, marker = "#C62828", "v"
            else:
                color, marker = "#757575", "o"
            alpha_line, alpha_marker, alpha_text = 0.6, 0.85, 0.8
            lw_line, lw_cap, ls = 1.8, 1.4, "solid"
        else:
            color = "#9E9E9E"
            marker = "^" if ev["direction"] > 0.15 else (
                "v" if ev["direction"] < -0.15 else "o")
            alpha_line, alpha_marker, alpha_text = 0.3, 0.45, 0.5
            lw_line, lw_cap, ls = 1.0, 0.8, "dashed"

        ax.hlines(y=price_at, xmin=ws, xmax=we, colors=color,
                  linewidth=lw_line, alpha=alpha_line, zorder=4, linestyles=ls)
        cap_h = y_rng * 0.012
        for cap_x in (ws, we):
            ax.vlines(x=cap_x, ymin=price_at - cap_h, ymax=price_at + cap_h,
                      colors=color, linewidth=lw_cap, alpha=alpha_line, zorder=4)

        size = (30 + ev["magnitude"] * 200) * (1.0 if ev["active"] else 0.6)
        ax.scatter(event_date, price_at, s=size, color=color, zorder=5,
                   alpha=alpha_marker, marker=marker, edgecolors="white",
                   linewidth=0.6)

        offset_sign = 1 if (i % 2 == 0) else -1
        text_y = price_at + offset_sign * y_rng * (0.06 + 0.03 * (i % stagger_k))
        text_y = max(y_lo - y_rng * 0.05, min(y_hi + y_rng * 0.15, text_y))

        label = f"✗ {ev['name']}" if not ev["active"] else ev["name"]
        ax.annotate(
            label, xy=(event_date, price_at), xytext=(event_date, text_y),
            fontsize=ann_fontsize, color=color, fontweight="bold",
            ha="center", va="bottom" if offset_sign > 0 else "top",
            arrowprops=dict(arrowstyle="-", color=color,
                            alpha=0.4 if ev["active"] else 0.2, linewidth=0.5),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=alpha_text, linewidth=0.5),
        )

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
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9)

    n_kept = int(keep_mask.sum())
    n_total = len(keep_mask)
    ax.set_title(
        f"{ticker} — Factor Dropout Map "
        f"({n_kept}/{n_total} active, {n_total - n_kept} dropped)",
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
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets: Full vs 50% Factor Dropout")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "causal_sde_config.yaml"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout-frac", type=float, default=0.5)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tickers = cfg.get("tickers", ["AAPL"])
    output_name = cfg.get("output_dir", "multi_asset_2018_2023")
    results_dir = PROJECT_ROOT / "results" / output_name
    gen_seed = cfg.get("seed", 42)

    out_dir = PROJECT_ROOT / "results" / "dropout_benchmark"
    full_dir = out_dir / "full"
    drop_dir = out_dir / "dropout"
    full_dir.mkdir(parents=True, exist_ok=True)
    drop_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    feature_names = ["open", "high", "low", "close", "volume"]

    print("═" * 70)
    print("  SYNTHETIC DATA GENERATION: Full vs Factor Dropout")
    print("═" * 70)
    print(f"  Tickers:       {tickers}")
    print(f"  Dropout frac:  {args.dropout_frac:.0%}")
    print(f"  Dropout seed:  {args.seed}")
    print(f"  Gen seed:      {gen_seed}")
    print(f"  Output:        {out_dir}")
    print("═" * 70)

    t_start = time.time()
    all_full_dfs = []
    all_drop_dfs = []
    report = {
        "experiment": "dropout_benchmark",
        "dropout_fraction": args.dropout_frac,
        "dropout_seed": args.seed,
        "generation_seed": gen_seed,
        "tickers": {},
    }

    for ticker in tickers:
        print(f"\n{'─' * 50}")
        print(f"  {ticker}")
        print(f"{'─' * 50}")

        ticker_dir = results_dir / ticker

        # ── Load impact matrix ──────────────────────────────────────
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            print(f"  ⚠ impact_matrix.json not found, skipping")
            continue
        im = ImpactMatrix.load(str(im_path))
        K = len(im.factor_names)
        print(f"  ✓ ImpactMatrix: {K} factors")

        # ── Load analyses ───────────────────────────────────────────
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
            print(f"  ⚠ insufficient data ({len(ticker_df)} rows), skipping")
            continue

        dates = pd.to_datetime(ticker_df["date"]).values
        print(f"  ✓ Real data: {len(ticker_df)} rows")

        # ── Build presence matrix ───────────────────────────────────
        full_presence = build_presence_from_analyses(
            analyses, im.factor_names, dates,
        )

        # ── Condition A: Full (0% dropout) ──────────────────────────
        print(f"  Generating FULL ({K} factors)...")
        full_df = generate_synthetic_csv(
            im=im,
            presence=full_presence,
            ticker_df=ticker_df,
            ticker=ticker,
            seed=gen_seed,
            feature_names=feature_names,
        )
        full_csv = full_dir / f"synthetic_{ticker}.csv"
        full_df.to_csv(full_csv, index=False)
        all_full_dfs.append(full_df)
        print(f"  ✓ Full CSV: {full_csv} ({len(full_df)} rows)")

        # ── Condition B: 50% Dropout ────────────────────────────────
        rng = np.random.RandomState(args.seed)
        keep_mask = np.ones(K, dtype=bool)
        n_drop = int(K * args.dropout_frac)
        drop_indices = rng.choice(K, size=n_drop, replace=False)
        keep_mask[drop_indices] = False
        n_kept = int(keep_mask.sum())

        dropout_im = make_dropout_impact_matrix(im, keep_mask)
        keep_indices = np.where(keep_mask)[0]
        dropout_presence = full_presence[:, keep_indices]

        dropped_names = [im.factor_names[i] for i in range(K) if not keep_mask[i]]
        kept_names = [im.factor_names[i] for i in range(K) if keep_mask[i]]

        print(f"  Generating DROPOUT ({n_kept}/{K} factors, {n_drop} dropped)...")
        drop_df = generate_synthetic_csv(
            im=dropout_im,
            presence=dropout_presence,
            ticker_df=ticker_df,
            ticker=ticker,
            seed=gen_seed,
            feature_names=feature_names,
        )
        drop_csv = drop_dir / f"synthetic_{ticker}.csv"
        drop_df.to_csv(drop_csv, index=False)
        all_drop_dfs.append(drop_df)
        print(f"  ✓ Dropout CSV: {drop_csv} ({len(drop_df)} rows)")

        # ── Dropout map plot ────────────────────────────────────────
        plot_path = out_dir / "plots" / f"factor_dropout_{ticker}.png"
        plot_dropout_map(
            real_df=ticker_df,
            analyses=analyses,
            keep_mask=keep_mask,
            factor_names=im.factor_names,
            ticker=ticker,
            output_path=str(plot_path),
        )
        print(f"  ✓ Dropout map: {plot_path}")

        # ── Report entry ────────────────────────────────────────────
        report["tickers"][ticker] = {
            "n_factors_total": K,
            "n_factors_kept": n_kept,
            "n_factors_dropped": n_drop,
            "dropped_factors": dropped_names,
            "kept_factors": kept_names,
            "n_rows": len(ticker_df),
        }

    # ── Combined multi-asset CSVs ───────────────────────────────────
    if all_full_dfs:
        combined_full = pd.concat(all_full_dfs, ignore_index=True)
        combined_full = combined_full.sort_values(["date", "tic"]).reset_index(drop=True)
        combined_full_path = full_dir / "synthetic_all_tickers.csv"
        combined_full.to_csv(combined_full_path, index=False)
        print(f"\n  ✓ Combined FULL CSV: {combined_full_path} "
              f"({len(combined_full)} rows, "
              f"{combined_full['tic'].nunique()} tickers)")

    if all_drop_dfs:
        combined_drop = pd.concat(all_drop_dfs, ignore_index=True)
        combined_drop = combined_drop.sort_values(["date", "tic"]).reset_index(drop=True)
        combined_drop_path = drop_dir / "synthetic_all_tickers.csv"
        combined_drop.to_csv(combined_drop_path, index=False)
        print(f"  ✓ Combined DROPOUT CSV: {combined_drop_path} "
              f"({len(combined_drop)} rows, "
              f"{combined_drop['tic'].nunique()} tickers)")

    # ── Save report ─────────────────────────────────────────────────
    total_time = time.time() - t_start
    report["total_time_seconds"] = round(total_time, 1)
    report_path = out_dir / "dropout_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'═' * 70}")
    print("  DONE")
    print(f"{'═' * 70}")
    print(f"  Output structure:")
    print(f"    {out_dir}/")
    print(f"    ├── full/")
    print(f"    │   ├── synthetic_AAPL.csv       (per-ticker)")
    print(f"    │   └── synthetic_all_tickers.csv (combined for FinRL)")
    print(f"    ├── dropout/")
    print(f"    │   ├── synthetic_AAPL.csv")
    print(f"    │   └── synthetic_all_tickers.csv")
    print(f"    ├── plots/")
    print(f"    │   └── factor_dropout_AAPL.png  (dropout map)")
    print(f"    └── dropout_benchmark_report.json")
    print(f"  Time: {total_time:.1f}s")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
