#!/usr/bin/env python3
"""Neural SDE baseline benchmark.

Trains one ConditionalNeuralSDE per ticker on the cached factor matrix
+ real log-returns (2018-2023), then samples N OOS-style trajectories
and reports per-ticker KS / Wasserstein / sigma-ratio against real
returns. Mirrors the reporting format of ``run_full_benchmark.py`` so
the Neural SDE row drops directly into the comparison table.

Two modes:
    --mode train  : train + checkpoint each ticker
    --mode eval   : load checkpoints, sample, score
    --mode all    : train then eval (default)

Outputs:
    results/export_paper/neural_sde_baseline/
        neural_sde_metrics.json
        neural_sde_summary.csv
        per ticker .pt checkpoint (in results/<run>/<TICKER>/neural_sde.pt)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sde_causal_generator.data_structures import ImpactMatrix
from sde_causal_generator.neural_sde import (
    NeuralSDEModel,
    sample_neural_sde,
    train_neural_sde,
)
from scripts.ablation_causal_direction import (
    build_presence_matrix,
    load_analyses,
)

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN"]
DEFAULT_RUN_DIR = "final_run_gpt4o_mini"


def _log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.clip(close.astype(np.float64), 1e-8, None)))


def _prepare_inputs(ticker: str, results_dir: Path):
    """Load real prices + factor presence for a ticker."""
    im = ImpactMatrix.load(str(results_dir / ticker / "impact_matrix.json"))
    analyses = load_analyses(ticker, results_dir)

    full_df = pd.read_csv(results_dir / "training_data.csv")
    df_t = full_df[full_df["tic"] == ticker].copy().sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in ["open", "high", "low", "close"] if c in df_t.columns]
    prices = np.clip(df_t[feature_cols].values.astype(np.float64), 1e-8, None)
    log_ret = np.diff(np.log(prices), axis=0)

    presence = build_presence_matrix(analyses, df_t, im.factor_names)
    # Align: log_ret has T rows (T+1 prices); presence has T+1 rows.
    return im, df_t, prices, log_ret, presence


def cmd_train(args: argparse.Namespace) -> dict:
    results_dir = PROJECT_ROOT / "results" / args.results_dir
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[train] device={device}")
    out_records = []
    for ticker in args.tickers:
        print(f"\n── {ticker} ──")
        t0 = time.time()
        try:
            im, _, _, log_ret, presence = _prepare_inputs(ticker, results_dir)
        except FileNotFoundError as e:
            print(f"  skip: {e}")
            continue

        model, history = train_neural_sde(
            log_returns=log_ret,
            factor_matrix=presence,
            n_price_features=4,
            hidden=args.hidden,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            device=device,
        )
        ckpt_path = results_dir / ticker / "neural_sde.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "state_dim": model.state_dim,
            "factor_dim": model.factor_dim,
            "hidden": args.hidden,
            "factor_names": im.factor_names,
            "history_best_epoch": history["best_epoch"],
            "history_best_val": min(history["val_loss"]),
        }, ckpt_path)
        elapsed = time.time() - t0
        print(f"  saved {ckpt_path}  ({elapsed:.0f}s, "
              f"best_epoch={history['best_epoch']}, "
              f"best_val={min(history['val_loss']):.4f})")
        out_records.append({
            "ticker": ticker, "elapsed_s": elapsed,
            "best_epoch": history["best_epoch"],
            "best_val": float(min(history["val_loss"])),
        })
    return {"train_records": out_records}


def cmd_eval(args: argparse.Namespace) -> dict:
    results_dir = PROJECT_ROOT / "results" / args.results_dir
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[eval] device={device}")

    rows = []
    for ticker in args.tickers:
        ckpt_path = results_dir / ticker / "neural_sde.pt"
        if not ckpt_path.exists():
            print(f"  {ticker}: no checkpoint")
            rows.append({"ticker": ticker, "status": "no_checkpoint"})
            continue

        try:
            im, df_t, prices, log_ret, presence = _prepare_inputs(ticker, results_dir)
        except FileNotFoundError as e:
            rows.append({"ticker": ticker, "status": f"missing_inputs: {e}"})
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = NeuralSDEModel(
            ckpt["state_dim"], ckpt["factor_dim"], ckpt.get("hidden", 64)
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # Sample over the full historical horizon (matches CSDE eval).
        T = log_ret.shape[0]
        n_steps = T
        # factor_schedule: tile presence to n_samples
        K = presence.shape[1]
        np.random.seed(args.seed)
        factor_schedule = np.tile(presence[:n_steps, :], (args.n_samples, 1, 1))

        synth_prices = sample_neural_sde(
            model=model,
            n_steps=n_steps,
            n_samples=args.n_samples,
            initial_prices=prices[0, :],
            factor_schedule=factor_schedule,
            device=device,
        )

        # Score per sample on close
        close_idx = 3
        real_close = prices[:n_steps + 1, close_idx]
        real_r = _log_returns(real_close)
        per_traj = []
        for s in range(synth_prices.shape[0]):
            synth_close = synth_prices[s, :, close_idx]
            sr = _log_returns(synth_close)
            n = min(len(real_r), len(sr))
            ks = float(ks_2samp(real_r[:n], sr[:n]).statistic)
            w1 = float(wasserstein_distance(real_r[:n], sr[:n]))
            sig = (float(np.std(sr[:n]) / np.std(real_r[:n]))
                   if np.std(real_r[:n]) > 0 else float("nan"))
            per_traj.append({"ks": ks, "wasserstein": w1, "sigma_ratio": sig})

        ks_mean = float(np.mean([m["ks"] for m in per_traj]))
        w1_mean = float(np.mean([m["wasserstein"] for m in per_traj]))
        sig_mean = float(np.mean([m["sigma_ratio"] for m in per_traj]))

        rows.append({
            "ticker": ticker,
            "status": "ok",
            "ks_mean": ks_mean,
            "wasserstein_mean": w1_mean,
            "sigma_ratio_mean": sig_mean,
            "ks_min": float(np.min([m["ks"] for m in per_traj])),
            "ks_max": float(np.max([m["ks"] for m in per_traj])),
        })
        print(f"  {ticker}: KS={ks_mean:.4f}  W1={w1_mean:.4e}  σ={sig_mean:.3f}")

    summary = {
        "n_tickers": len(args.tickers),
        "n_evaluated": int(sum(r.get("status") == "ok" for r in rows)),
        "rows": rows,
    }
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if ok_rows:
        ks_arr = np.array([r["ks_mean"] for r in ok_rows])
        summary["aggregate"] = {
            "ks_mean": float(ks_arr.mean()),
            "ks_median": float(np.median(ks_arr)),
            "ks_max": float(ks_arr.max()),
            "wasserstein_mean": float(np.mean([r["wasserstein_mean"] for r in ok_rows])),
            "sigma_ratio_mean": float(np.mean([r["sigma_ratio_mean"] for r in ok_rows])),
            "n_below_threshold_0_10": int((ks_arr < 0.10).sum()),
        }
        pd.DataFrame(ok_rows).to_csv(out_dir / "neural_sde_summary.csv", index=False)

    out_file = out_dir / "neural_sde_metrics.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[eval] saved {out_file}")
    if "aggregate" in summary:
        a = summary["aggregate"]
        print(f"  Aggregate: KS={a['ks_mean']:.4f}  σ={a['sigma_ratio_mean']:.3f}  "
              f"{a['n_below_threshold_0_10']}/{summary['n_evaluated']} below 0.10")
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    p.add_argument("--results-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA available.")
    p.add_argument("--out", default="results/export_paper/neural_sde_baseline")
    args = p.parse_args()

    if args.mode in ("train", "all"):
        cmd_train(args)
    if args.mode in ("eval", "all"):
        cmd_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
