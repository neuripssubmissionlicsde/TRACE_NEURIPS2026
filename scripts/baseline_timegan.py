#!/usr/bin/env python3
"""
Full Benchmark: Causal SDE vs TimeGAN
======================================

Runs the complete comparison pipeline:

1. Verifies Causal SDE results exist (from prior generation run).
2. Trains TimeGAN baseline on the same training data.
3. Runs TSTR (Train Synthetic, Test Real) for both methods.
4. Computes distributional metrics.
5. Generates comparison report + visualisations.

All configuration is driven by the YAML file — no CLI flags needed.

Usage:
    python scripts/run_full_benchmark.py

    # Custom config path
    python scripts/run_full_benchmark.py configs/my_config.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Default config path
DEFAULT_CONFIG = str(PROJECT_ROOT / "configs" / "causal_sde_config.yaml")


def main(config_path: str | None = None):
    path = config_path or DEFAULT_CONFIG

    from sde_causal_generator.benchmark import BenchmarkPipeline, BenchmarkConfig

    # Load config — everything comes from YAML
    cfg = BenchmarkConfig.from_yaml(path)

    # Run benchmark
    pipeline = BenchmarkPipeline(cfg)
    report = pipeline.run()
    pipeline.save_report()

    # Print summary
    comparison = report.get("comparison", {})
    per_ticker = comparison.get("per_ticker", [])

    if per_ticker:
        print("\n" + "═" * 70)
        print("  COMPARISON SUMMARY")
        print("═" * 70)
        print(f"  {'Ticker':>6s}  {'CSDE TSTR Acc':>13s}  {'TGAN TSTR Acc':>13s}  "
              f"{'CSDE KS':>8s}  {'TGAN KS':>8s}")
        print("  " + "─" * 66)
        for row in per_ticker:
            csde_acc = row.get("causal_sde_dir_tstr_acc")
            tgan_acc = row.get("timegan_dir_tstr_acc")
            csde_ks = row.get("causal_sde_ks_stat")
            tgan_ks = row.get("timegan_ks_stat")
            print(
                f"  {row['ticker']:>6s}  "
                f"{_fmt(csde_acc):>13s}  {_fmt(tgan_acc):>13s}  "
                f"{_fmt(csde_ks):>8s}  {_fmt(tgan_ks):>8s}"
            )
        print("═" * 70)


def _fmt(val, decimals=4):
    """Format a value or '—' if None."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


if __name__ == "__main__":
    # Optional: pass config path as first positional argument
    _path = sys.argv[1] if len(sys.argv) > 1 else None
    main(config_path=_path)
