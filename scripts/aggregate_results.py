# -*- coding: utf-8 -*-
"""
Aggregate DJIA-28 Results
=========================

Consolidates per-baseline reports (JSON + CSV) into a single aggregate
directory with cross-baseline summary tables suitable for the manuscript.

Usage:
    python scripts/aggregate_results.py [--phase main]

Outputs go to results/_aggregate/ (or _aggregate_phaseA when --phase A).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"  ⚠ missing: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def df_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  ⚠ missing: {path}")
        return None
    return pd.read_csv(path)


# ── Per-baseline extractors ─────────────────────────────────────────────


def extract_vanilla(report_path: Path) -> Optional[pd.DataFrame]:
    rep = load_json(report_path)
    if rep is None:
        return None
    rows = []
    for ticker, t in rep.get("per_ticker", {}).items():
        c = t.get("causal_sde", {})
        v = t.get("vanilla_sde", {})
        rows.append({
            "ticker": ticker,
            "csde_ks": c.get("ks_statistic"),
            "vanilla_ks": v.get("ks_statistic"),
            "csde_w1": c.get("wasserstein"),
            "vanilla_w1": v.get("wasserstein"),
            "csde_vol_ratio": c.get("volatility_ratio"),
            "vanilla_vol_ratio": v.get("volatility_ratio"),
        })
    return pd.DataFrame(rows)


def extract_direction(report_path: Path) -> Optional[pd.DataFrame]:
    rep = load_json(report_path)
    if rep is None:
        return None
    rows = []
    for ticker, t in rep.get("per_ticker", {}).items():
        a = t.get("analytical_signal", {})
        d_bears = t.get("generated_delta_bears_off")
        d_bulls = t.get("generated_delta_bulls_off")
        all_d = a.get("all_factors", {}).get("mean_daily")
        boff = a.get("bears_off", {}).get("mean_daily")
        bulloff = a.get("bulls_off", {}).get("mean_daily")
        bears_up = bool(a.get("bears_off_increases_signal"))
        bulls_down = bool(a.get("bulls_off_decreases_signal"))
        rows.append({
            "ticker": ticker,
            "n_factors": t.get("n_factors"),
            "n_bull": t.get("n_bull"),
            "n_bear": t.get("n_bear"),
            "all_mean_daily": all_d,
            "bears_off_mean_daily": boff,
            "bulls_off_mean_daily": bulloff,
            "delta_bears_off": d_bears,
            "delta_bulls_off": d_bulls,
            "bears_off_up": int(bears_up),
            "bulls_off_down": int(bulls_down),
        })
    return pd.DataFrame(rows)


def extract_shuffle(report_path: Path) -> Optional[pd.DataFrame]:
    rep = load_json(report_path)
    if rep is None:
        return None
    rows = []
    for ticker, t in rep.get("per_ticker", {}).items():
        cond = t.get("conditions", {})
        rows.append({
            "ticker": ticker,
            "baseline_ks": cond.get("baseline", {}).get("ks_statistic"),
            "shuffled_directions_ks": cond.get("shuffled_directions", {}).get("ks_statistic"),
            "shuffled_magnitudes_ks": cond.get("shuffled_magnitudes", {}).get("ks_statistic"),
            "shuffled_presence_ks": cond.get("shuffled_presence", {}).get("ks_statistic"),
            "baseline_w1": cond.get("baseline", {}).get("wasserstein"),
            "shuffled_directions_w1": cond.get("shuffled_directions", {}).get("wasserstein"),
        })
    return pd.DataFrame(rows)


def extract_nsde(csv_path: Path) -> Optional[pd.DataFrame]:
    df = df_or_none(csv_path)
    if df is None:
        return None
    df = df.rename(columns={
        "ks_mean": "nsde_ks", "wasserstein_mean": "nsde_w1",
        "sigma_ratio_mean": "nsde_sigma_ratio",
    })
    return df[["ticker", "nsde_ks", "nsde_w1", "nsde_sigma_ratio"]]


def extract_oos(csv_path: Path) -> Optional[pd.DataFrame]:
    df = df_or_none(csv_path)
    if df is None:
        return None
    df = df.rename(columns={
        "ks_mean": "oos_ks", "wasserstein_mean": "oos_w1",
        "sigma_ratio_mean": "oos_sigma_ratio",
    })
    keep = [c for c in ["ticker", "oos_ks", "oos_w1", "oos_sigma_ratio"] if c in df.columns]
    return df[keep]


def extract_multiseed(report_path: Path, ci_csv: Path) -> Optional[pd.DataFrame]:
    rep = load_json(report_path)
    if rep is None:
        return None
    import numpy as np
    rows = []
    for ticker, t in rep.get("per_ticker", {}).items():
        ks = [m.get("ks_statistic") for m in t.get("per_seed_metrics", []) if m.get("ks_statistic") is not None]
        if not ks:
            continue
        rows.append({
            "ticker": ticker,
            "ks_mean_across_seeds": float(np.mean(ks)),
            "ks_std_across_seeds": float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0,
            "n_seeds": len(ks),
        })
    return pd.DataFrame(rows)


def extract_lstm(report_path: Path) -> Optional[pd.DataFrame]:
    rep = load_json(report_path)
    if rep is None:
        return None
    rows = []
    pt = rep.get("per_ticker", rep.get("results", {}))
    if isinstance(pt, dict):
        for ticker, t in pt.items():
            rows.append({
                "ticker": ticker,
                "lstm_accuracy": t.get("avg_accuracy", t.get("accuracy_mean", t.get("accuracy"))),
                "lstm_auc": t.get("avg_auc_roc", t.get("auc_mean", t.get("auc"))),
                "lstm_f1": t.get("avg_f1", t.get("f1_mean", t.get("f1"))),
            })
    return pd.DataFrame(rows) if rows else None


def extract_timegan(report_path: Path) -> Optional[pd.DataFrame]:
    rep = load_json(report_path)
    if rep is None:
        return None
    rows = []
    comp = rep.get("comparison", {})
    tstr = rep.get("tstr", {})
    for r in comp.get("per_ticker", []):
        ticker = r.get("ticker")
        tstr_t = tstr.get(ticker, {})
        csde_tstr_acc = (tstr_t.get("causal_sde", {})
                         .get("direction", {}).get("tstr", {}).get("accuracy"))
        tgan_tstr_acc = (tstr_t.get("timegan", {})
                         .get("direction", {}).get("tstr", {}).get("accuracy"))
        rows.append({
            "ticker": ticker,
            "timegan_ks": r.get("timegan_ks_stat"),
            "csde_ks_b": r.get("causal_sde_ks_stat"),
            "timegan_vol_ratio": r.get("timegan_vol_ratio"),
            "csde_vol_ratio_b": r.get("causal_sde_vol_ratio"),
            "csde_tstr_acc": csde_tstr_acc,
            "timegan_tstr_acc": tgan_tstr_acc,
        })
    return pd.DataFrame(rows) if rows else None


# ── Main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="B", choices=["A", "B"])
    args = parser.parse_args()
    phase = args.phase

    out_dir = RESULTS / ("_aggregate" if phase == "B" else "_aggregate_phaseA")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[aggregate] writing to {out_dir}")

    suffix = "_phaseA" if phase == "A" else ""
    # For the main experiment, the latest run is in the un-suffixed directories.
    # For phase A, use the *_phaseA snapshots.

    sources = {
        "vanilla": (
            extract_vanilla,
            (RESULTS / f"ablation_vanilla_sde{suffix}" / "vanilla_sde_ablation_report.json",),
        ),
        "direction": (
            extract_direction,
            (RESULTS / f"ablation_causal_direction{suffix}" / "causal_direction_ablation_report.json",),
        ),
        "shuffle": (
            extract_shuffle,
            (RESULTS / f"ablation_shuffle_control{suffix}" / "shuffle_control_report.json",),
        ),
        "multiseed": (
            extract_multiseed,
            (RESULTS / f"multiseed_bootstrap{suffix}" / "multiseed_bootstrap_report.json",
             RESULTS / f"multiseed_bootstrap{suffix}" / "bootstrap_ci.csv"),
        ),
        "nsde": (
            extract_nsde,
            (RESULTS / ("neural_sde" if phase == "B" else "neural_sde_baseline_phaseA") / "neural_sde_summary.csv",),
        ),
        "oos_2023": (
            extract_oos,
            (RESULTS / ("oos_2023" if phase == "B" else "oos_2023_phaseA") / "oos_summary.csv",),
        ),
        "lstm": (
            extract_lstm,
            (RESULTS / f"djia28" / "lstm_discriminator" / "lstm_discriminator_report.json",) if phase == "B"
            else (RESULTS / "final_run_gpt4o_mini" / "lstm_discriminator" / "lstm_discriminator_report.json",),
        ),
        "timegan": (
            extract_timegan,
            (RESULTS / (f"benchmark" if phase == "B" else "benchmark_phaseA") / "benchmark_report.json",),
        ),
    }

    for name, (fn, paths) in sources.items():
        try:
            df = fn(*paths)
        except Exception as e:
            print(f"  ⚠ {name}: extractor failed: {e}")
            continue
        if df is None or len(df) == 0:
            print(f"  ⚠ {name}: no data")
            continue
        out_csv = out_dir / f"{name}_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"  ✓ {name}: {len(df)} rows → {out_csv}")

    # Combined per-ticker view (best-effort)
    combined = None
    for f in out_dir.glob("*_summary.csv"):
        try:
            df = pd.read_csv(f)
            if "ticker" not in df.columns:
                continue
            combined = df if combined is None else combined.merge(df, on="ticker", how="outer")
        except Exception as e:
            print(f"  ⚠ merge failed for {f.name}: {e}")
    if combined is not None:
        combined_path = out_dir / "all_baselines_per_ticker.csv"
        combined.to_csv(combined_path, index=False)
        print(f"  ✓ combined per-ticker → {combined_path}  ({len(combined)} tickers)")

    # Headline numbers
    headline = {}
    vp = out_dir / "vanilla_summary.csv"
    if vp.exists():
        df = pd.read_csv(vp)
        headline["csde_ks_mean"] = float(df["csde_ks"].mean())
        headline["vanilla_ks_mean"] = float(df["vanilla_ks"].mean())
        headline["csde_wins_ks"] = int((df["csde_ks"] < df["vanilla_ks"]).sum())
        headline["n_tickers"] = int(len(df))
    np_ = out_dir / "nsde_summary.csv"
    if np_.exists():
        df = pd.read_csv(np_)
        headline["nsde_ks_mean"] = float(df["nsde_ks"].mean())
    op = out_dir / "oos_2023_summary.csv"
    if op.exists():
        df = pd.read_csv(op)
        headline["oos_ks_mean"] = float(df["oos_ks"].mean())
    mp = out_dir / "multiseed_summary.csv"
    if mp.exists():
        df = pd.read_csv(mp)
        if "ks_mean_across_seeds" in df.columns:
            headline["multiseed_ks_mean"] = float(df["ks_mean_across_seeds"].mean())
            headline["multiseed_ks_std_avg"] = float(df["ks_std_across_seeds"].mean())
    sp = out_dir / "shuffle_summary.csv"
    if sp.exists():
        df = pd.read_csv(sp)
        headline["shuffle_baseline_ks_mean"] = float(df["baseline_ks"].mean())
        headline["shuffle_directions_ks_mean"] = float(df["shuffled_directions_ks"].mean())
    dp = out_dir / "direction_summary.csv"
    if dp.exists():
        df = pd.read_csv(dp)
        headline["direction_bears_off_up"] = int(df["bears_off_up"].sum())
        headline["direction_bulls_off_down"] = int(df["bulls_off_down"].sum())
        headline["direction_n"] = int(len(df))

    tp = out_dir / "timegan_summary.csv"
    if tp.exists():
        df = pd.read_csv(tp)
        if "csde_tstr_acc" in df.columns and df["csde_tstr_acc"].notna().any():
            from scipy import stats as _stats
            import numpy as _np
            ca = df["csde_tstr_acc"].dropna().values
            ta = df["timegan_tstr_acc"].dropna().values
            headline["csde_tstr_acc_mean"] = float(_np.mean(ca))
            headline["timegan_tstr_acc_mean"] = float(_np.mean(ta))
            if len(ca) == len(ta) and len(ca) > 1:
                _t, _p = _stats.ttest_rel(ca, ta)
                headline["tstr_ttest_t"] = float(_t)
                headline["tstr_ttest_p"] = float(_p)

    headline_path = out_dir / "headline_numbers.json"
    with open(headline_path, "w") as f:
        json.dump(headline, f, indent=2)
    print(f"  ✓ headline → {headline_path}")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
