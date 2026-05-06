#!/usr/bin/env python3
"""Amplified-view metrics for the bull/bear direction ablation.

Reads the existing trained ImpactMatrix + WindowAnalysis artefacts and
re-frames the deterministic drift ablation under four complementary
views, *without modifying any underlying numbers*:

    1. Annualised drift (baseline, what the existing report shows)
    2. Cumulative drift over the full 6-year horizon
    3. Active-day conditional drift — restricts the average to dates
       where the disabled factor group is actually firing, removing
       the dilution from inactive days
    4. Relative change vs the all-factors baseline (% of baseline)

Plus per-window box-plots of the drift contribution under each
condition (showing the variability the mean hides).

Outputs:
    results/final_run_gpt4o_mini/direction_ablation_amplified/
        amplified_report.json
        amplified_summary.png  (and .pdf)
        amplified_box_<TICKER>.png  (one per ticker)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix
from scripts.ablation_causal_direction import (
    build_presence_matrix,
    compute_analytical_signal,
    load_analyses,
)

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN", "AMZN", "AXP", "BA", "CRM", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "KO", "MCD", "MMM", "MRK", "NKE", "PG", "TRV", "UNH", "V"]
RESULTS_DIR_NAME = "djia28"
TRADING_DAYS_PER_YEAR = 252


def _load_real(ticker: str, training_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(training_csv)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[df["tic"] == ticker].sort_values("date").reset_index(drop=True)


def _bull_bear_masks(im: ImpactMatrix, eps: float = 1e-6):
    close_idx = min(3, im.base_impact.shape[1] - 1)
    impact_close = im.base_impact[:, close_idx]
    bull = impact_close > eps
    bear = impact_close < -eps
    return bull, bear


def _zero_columns(presence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = presence.copy()
    out[:, mask] = 0.0
    return out


def _conditional_mean(signal: np.ndarray, active_mask: np.ndarray) -> float:
    if not active_mask.any():
        return float("nan")
    return float(signal[active_mask].mean())


def _per_window_means(signal: np.ndarray, daily_dates, analyses):
    daily_dates = pd.to_datetime(daily_dates)
    out = []
    for a in sorted(analyses, key=lambda x: pd.Timestamp(x.start_date)):
        m = (daily_dates >= pd.Timestamp(a.start_date)) & (
            daily_dates <= pd.Timestamp(a.end_date)
        )
        if m.any():
            out.append(float(signal[m].mean()))
    return np.array(out)


def main() -> None:
    results_dir = PROJECT_ROOT / "results" / RESULTS_DIR_NAME
    out_dir = results_dir / "direction_ablation_amplified"
    out_dir.mkdir(parents=True, exist_ok=True)
    training_csv = results_dir / "training_data.csv"

    report = {"tickers": {}, "n_trading_days_per_year": TRADING_DAYS_PER_YEAR}

    # Aggregate matrices for the summary figure (in pp, annualised).
    agg_rows = []

    for ticker in TICKERS:
        ticker_dir = results_dir / ticker
        im_path = ticker_dir / "impact_matrix.json"
        if not im_path.exists():
            continue

        im = ImpactMatrix.load(str(im_path))
        analyses = load_analyses(ticker, results_dir)
        df_real = _load_real(ticker, training_csv)
        if df_real.empty:
            continue
        presence = build_presence_matrix(analyses, df_real, im.factor_names)
        daily_dates = pd.to_datetime(sorted(df_real["date"].unique())).values

        bull_mask, bear_mask = _bull_bear_masks(im)
        n_bull, n_bear = int(bull_mask.sum()), int(bear_mask.sum())

        signal_all = compute_analytical_signal(im, presence)
        signal_no_bears = compute_analytical_signal(
            im, _zero_columns(presence, bear_mask)
        )
        signal_no_bulls = compute_analytical_signal(
            im, _zero_columns(presence, bull_mask)
        )

        # Days where any bull / bear factor is active (presence > 0).
        bull_active = (presence[:, bull_mask] > 0).any(axis=1) if n_bull else (
            np.zeros(len(presence), dtype=bool)
        )
        bear_active = (presence[:, bear_mask] > 0).any(axis=1) if n_bear else (
            np.zeros(len(presence), dtype=bool)
        )

        # ── (1) annualised mean ──
        ann = lambda s: float(s.mean()) * TRADING_DAYS_PER_YEAR
        # ── (2) cumulative ──
        cum = lambda s: float(s.sum())
        # ── (3) conditional on activation ──
        ann_cond = lambda s, m: _conditional_mean(s, m) * TRADING_DAYS_PER_YEAR

        ann_all = ann(signal_all)
        ann_nb = ann(signal_no_bears)
        ann_nbu = ann(signal_no_bulls)

        cum_all = cum(signal_all)
        cum_nb = cum(signal_no_bears)
        cum_nbu = cum(signal_no_bulls)

        # Conditional: how much does removing bears change drift on the
        # days bears are active? (similarly for bulls).
        ann_all_on_bear = ann_cond(signal_all, bear_active)
        ann_nb_on_bear = ann_cond(signal_no_bears, bear_active)
        ann_all_on_bull = ann_cond(signal_all, bull_active)
        ann_nbu_on_bull = ann_cond(signal_no_bulls, bull_active)

        # Relative change vs baseline (in % of baseline).
        def rel(a, b):
            return float((b - a) / abs(a) * 100) if abs(a) > 1e-12 else float("nan")

        # Per-window distribution.
        win_all = _per_window_means(signal_all, daily_dates, analyses)
        win_nb = _per_window_means(signal_no_bears, daily_dates, analyses)
        win_nbu = _per_window_means(signal_no_bulls, daily_dates, analyses)

        ticker_report = {
            "n_factors": int(len(im.factor_names)),
            "n_bull": n_bull,
            "n_bear": n_bear,
            "fraction_days_bull_active": float(bull_active.mean()),
            "fraction_days_bear_active": float(bear_active.mean()),
            "annualised_pct": {
                "all_factors": ann_all * 100,
                "bears_off": ann_nb * 100,
                "bulls_off": ann_nbu * 100,
                "delta_bears_off": (ann_nb - ann_all) * 100,
                "delta_bulls_off": (ann_nbu - ann_all) * 100,
            },
            "cumulative_pct": {
                "all_factors": cum_all * 100,
                "bears_off": cum_nb * 100,
                "bulls_off": cum_nbu * 100,
                "delta_bears_off": (cum_nb - cum_all) * 100,
                "delta_bulls_off": (cum_nbu - cum_all) * 100,
            },
            "conditional_active_day_annualised_pct": {
                "all_on_bear_days": ann_all_on_bear * 100,
                "bears_off_on_bear_days": ann_nb_on_bear * 100,
                "delta_bears_off_on_bear_days": (
                    ann_nb_on_bear - ann_all_on_bear
                ) * 100,
                "all_on_bull_days": ann_all_on_bull * 100,
                "bulls_off_on_bull_days": ann_nbu_on_bull * 100,
                "delta_bulls_off_on_bull_days": (
                    ann_nbu_on_bull - ann_all_on_bull
                ) * 100,
            },
            "relative_change_pct_of_baseline": {
                "bears_off": rel(ann_all, ann_nb),
                "bulls_off": rel(ann_all, ann_nbu),
            },
        }
        report["tickers"][ticker] = ticker_report

        agg_rows.append({
            "ticker": ticker,
            "delta_bears_ann_pct": (ann_nb - ann_all) * 100,
            "delta_bulls_ann_pct": (ann_nbu - ann_all) * 100,
            "delta_bears_cum_pct": (cum_nb - cum_all) * 100,
            "delta_bulls_cum_pct": (cum_nbu - cum_all) * 100,
            "delta_bears_cond_pct":
                (ann_nb_on_bear - ann_all_on_bear) * 100,
            "delta_bulls_cond_pct":
                (ann_nbu_on_bull - ann_all_on_bull) * 100,
        })

        # ── per-ticker box plot of window-level drift contributions ──
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data = [
            win_all * TRADING_DAYS_PER_YEAR * 100,
            win_nb * TRADING_DAYS_PER_YEAR * 100,
            win_nbu * TRADING_DAYS_PER_YEAR * 100,
        ]
        bp = ax.boxplot(
            data, labels=["All factors", "Bears OFF", "Bulls OFF"],
            patch_artist=True, widths=0.55,
        )
        for patch, color in zip(
            bp["boxes"], ["#4477aa", "#ee6677", "#228833"]
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_ylabel("Annualised drift per window (%)")
        ax.set_title(
            f"{ticker} — per-window drift distribution  "
            f"(K={len(im.factor_names)}, bull={n_bull}, bear={n_bear})"
        )
        ax.axhline(0, color="grey", linewidth=0.5, linestyle=":")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"amplified_box_{ticker}.png", dpi=200)
        plt.close(fig)

        print(
            f"  ✓ {ticker}: "
            f"Δbears ann={ticker_report['annualised_pct']['delta_bears_off']:+.3f}%  "
            f"cum={ticker_report['cumulative_pct']['delta_bears_off']:+.2f}%  "
            f"cond={ticker_report['conditional_active_day_annualised_pct']['delta_bears_off_on_bear_days']:+.2f}%"
        )

    # ── Aggregate summary figure: 3 panels ───────────────────────
    df_agg = pd.DataFrame(agg_rows).set_index("ticker")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panel_specs = [
        (
            "Annualised drift Δ (pp)",
            "delta_bears_ann_pct", "delta_bulls_ann_pct",
        ),
        (
            "Cumulative drift Δ over 6y (pp)",
            "delta_bears_cum_pct", "delta_bulls_cum_pct",
        ),
        (
            "Conditional Δ on active days (pp)",
            "delta_bears_cond_pct", "delta_bulls_cond_pct",
        ),
    ]
    width = 0.38
    x = np.arange(len(df_agg))
    for ax, (title, c_b, c_bu) in zip(axes, panel_specs):
        ax.bar(
            x - width / 2, df_agg[c_b].values, width,
            color="#ee6677", label="Bears OFF (expect ↑)",
        )
        ax.bar(
            x + width / 2, df_agg[c_bu].values, width,
            color="#228833", label="Bulls OFF (expect ↓)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(df_agg.index, rotation=30)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=9, frameon=False)

    fig.suptitle(
        "Bull/Bear direction ablation — three complementary views",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "amplified_summary.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "amplified_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    report["aggregate"] = {
        "mean_delta_bears_annualised_pp": float(df_agg["delta_bears_ann_pct"].mean()),
        "mean_delta_bulls_annualised_pp": float(df_agg["delta_bulls_ann_pct"].mean()),
        "mean_delta_bears_cumulative_pp": float(df_agg["delta_bears_cum_pct"].mean()),
        "mean_delta_bulls_cumulative_pp": float(df_agg["delta_bulls_cum_pct"].mean()),
        "mean_delta_bears_conditional_pp": float(df_agg["delta_bears_cond_pct"].mean()),
        "mean_delta_bulls_conditional_pp": float(df_agg["delta_bulls_cond_pct"].mean()),
        "direction_correct_bears_over_total": int(
            (df_agg["delta_bears_ann_pct"] > 0).sum()
        ),
        "direction_correct_bulls_over_total": int(
            (df_agg["delta_bulls_ann_pct"] < 0).sum()
        ),
        "n_tickers": int(len(df_agg)),
    }

    with open(out_dir / "amplified_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Saved summary + per-ticker boxplots in {out_dir}")
    print("\nAggregate (mean across tickers, in percentage points):")
    for k, v in report["aggregate"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
