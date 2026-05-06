#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Experiment Orchestrator — TRACE NeurIPS 2026.

Executes ALL experiments in sequence using configs/djia28.yaml:

  1. Main pipeline:          run_pipeline.py              (Phase 1–3 for 28 DJIA tickers)
  2. TSTR Benchmark:         baseline_timegan.py          (Causal SDE vs TimeGAN)
  3. Ablation A (Vanilla):   ablation_vanilla_sde.py      (no factors)
  4. Ablation B (Direction): ablation_causal_direction.py (bulls off / bears off)
  5. Cross-LLM Concordance:  multi_llm_concordance.py     (4 models, Price+RAG)
  6. Factor Statistics:       eval_factor_statistics.py    (term analysis)
  7. Results Organization:    copy/link artefacts into publication_artefacts/

Estimated cost: ~$15–25 (LLM API calls)
Estimated time: 2–6 hours (depending on API rate limits)

Usage:
    python scripts/reproduce_all.py

    # Skip already-completed steps:
    python scripts/reproduce_all.py --skip 1,2

    # Run only specific steps:
    python scripts/reproduce_all.py --only 5,6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_PATH = PROJECT_ROOT / "configs" / "djia28.yaml"
RESULTS_DIR = PROJECT_ROOT / "results" / "djia28"
ARTEFACTS_DIR = RESULTS_DIR / "publication_artefacts"

TICKERS = [
    "AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "CAT", "AMGN",
    "AMZN", "AXP", "BA", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "IBM", "KO", "MCD", "MMM", "MRK", "NKE",
    "PG", "TRV", "UNH", "V",
]

# ══════════════════════════════════════════════════════════════════════
# Step definitions
# ══════════════════════════════════════════════════════════════════════

STEPS = [
    {
        "id": 1,
        "name": "Main Pipeline (Phase 1–3)",
        "description": "Generate synthetic data for 28 DJIA tickers",
        "command": [
            sys.executable,
            str(SCRIPTS_DIR / "run_pipeline.py"),
            str(CONFIG_PATH),
        ],
    },
    {
        "id": 2,
        "name": "TSTR Benchmark (Causal SDE vs TimeGAN)",
        "description": "Train TimeGAN baseline and compare via TSTR",
        "command": [
            sys.executable,
            str(SCRIPTS_DIR / "baseline_timegan.py"),
            str(CONFIG_PATH),
        ],
    },
    {
        "id": 3,
        "name": "Ablation A: Vanilla SDE (no factors)",
        "description": "Generate with identical SDE engine but K=0 factors",
        "command": [
            sys.executable,
            str(SCRIPTS_DIR / "ablation_vanilla_sde.py"),
            str(CONFIG_PATH),
        ],
    },
    {
        "id": 4,
        "name": "Ablation B: Causal Direction",
        "description": "Selectively disable bull/bear factors",
        "command": [
            sys.executable,
            str(SCRIPTS_DIR / "ablation_causal_direction.py"),
            "--results-dir", "djia28",
        ],
    },
    {
        "id": 5,
        "name": "Cross-LLM Concordance Heatmap",
        "description": "4 models × 5 samples/window × 20 windows (Price+RAG)",
        "command": [
            sys.executable,
            str(SCRIPTS_DIR / "multi_llm_concordance.py"),
            "--ticker", "AAPL",
            "--n-samples", "5",
            "--n-windows", "20",
            "--output-dir", str(RESULTS_DIR / "concordance"),
        ],
    },
    {
        "id": 6,
        "name": "Factor Term Statistics",
        "description": "Analyse extracted factor terms, categories, overlap",
        "command": [
            sys.executable,
            str(SCRIPTS_DIR / "eval_factor_statistics.py"),
            "--results-dir", str(RESULTS_DIR),
            "--output-dir", str(RESULTS_DIR / "factor_statistics"),
        ],
    },
    {
        "id": 7,
        "name": "Organise Publication Artefacts",
        "description": "Copy results into a single clean folder",
        "function": "organise_artefacts",
    },
]


# ══════════════════════════════════════════════════════════════════════
# Artefact organisation
# ══════════════════════════════════════════════════════════════════════

def organise_artefacts():
    """Copy all key artefacts into publication_artefacts/."""
    print(f"\n  Organising artefacts into {ARTEFACTS_DIR}/")
    os.makedirs(ARTEFACTS_DIR, exist_ok=True)

    # ── 1. Config ───────────────────────────────────────────────────
    cfg_dst = ARTEFACTS_DIR / "config"
    os.makedirs(cfg_dst, exist_ok=True)
    shutil.copy2(CONFIG_PATH, cfg_dst / "final_run.yaml")
    if (RESULTS_DIR / "config_used.yaml").exists():
        shutil.copy2(RESULTS_DIR / "config_used.yaml", cfg_dst / "config_used.yaml")

    # ── 2. Per-ticker factors (analyses.json) ───────────────────────
    factors_dst = ARTEFACTS_DIR / "extracted_factors"
    os.makedirs(factors_dst, exist_ok=True)
    for ticker in TICKERS:
        src = RESULTS_DIR / ticker / "analyses.json"
        if src.exists():
            shutil.copy2(src, factors_dst / f"{ticker}_analyses.json")

    # ── 3. Per-ticker impact matrices ───────────────────────────────
    impact_dst = ARTEFACTS_DIR / "impact_matrices"
    os.makedirs(impact_dst, exist_ok=True)
    for ticker in TICKERS:
        src = RESULTS_DIR / ticker / "impact_matrix.json"
        if src.exists():
            shutil.copy2(src, impact_dst / f"{ticker}_impact_matrix.json")

    # ── 4. Per-ticker synthetic data ────────────────────────────────
    synth_dst = ARTEFACTS_DIR / "synthetic_data"
    os.makedirs(synth_dst, exist_ok=True)
    for ticker in TICKERS:
        src = RESULTS_DIR / ticker / "synthetic_data_causal_sde.csv"
        if src.exists():
            shutil.copy2(src, synth_dst / f"{ticker}_synthetic.csv")

    # ── 5. Per-ticker pipeline reports ──────────────────────────────
    reports_dst = ARTEFACTS_DIR / "pipeline_reports"
    os.makedirs(reports_dst, exist_ok=True)
    for ticker in TICKERS:
        src = RESULTS_DIR / ticker / "pipeline_report.json"
        if src.exists():
            shutil.copy2(src, reports_dst / f"{ticker}_pipeline_report.json")

    # ── 6. Per-ticker evaluation reports ────────────────────────────
    eval_dst = ARTEFACTS_DIR / "evaluation"
    os.makedirs(eval_dst, exist_ok=True)
    for ticker in TICKERS:
        eval_dir = RESULTS_DIR / ticker / "evaluation"
        if eval_dir.exists():
            ticker_eval = eval_dst / ticker
            os.makedirs(ticker_eval, exist_ok=True)
            for f in eval_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, ticker_eval / f.name)

    # ── 7. Cross-asset correlation ──────────────────────────────────
    src = RESULTS_DIR / "cross_asset_correlation.json"
    if src.exists():
        shutil.copy2(src, ARTEFACTS_DIR / "cross_asset_correlation.json")

    # ── 8. Generation summary ───────────────────────────────────────
    src = RESULTS_DIR / "generation_summary.json"
    if src.exists():
        shutil.copy2(src, ARTEFACTS_DIR / "generation_summary.json")

    # ── 9. Training data ────────────────────────────────────────────
    src = RESULTS_DIR / "training_data.csv"
    if src.exists():
        shutil.copy2(src, ARTEFACTS_DIR / "training_data.csv")

    # ── 10. Benchmark results ───────────────────────────────────────
    bench_src = PROJECT_ROOT / "results" / "benchmark"
    if bench_src.exists():
        bench_dst = ARTEFACTS_DIR / "benchmark"
        if bench_dst.exists():
            shutil.rmtree(bench_dst)
        shutil.copytree(bench_src, bench_dst)

    # ── 11. Ablation results ────────────────────────────────────────
    for ablation in ["ablation_vanilla_sde", "ablation_causal_direction"]:
        src = PROJECT_ROOT / "results" / ablation
        if src.exists():
            dst = ARTEFACTS_DIR / ablation
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # ── 12. Concordance results ─────────────────────────────────────
    conc_src = RESULTS_DIR / "concordance"
    if conc_src.exists():
        conc_dst = ARTEFACTS_DIR / "concordance"
        if conc_dst.exists():
            shutil.rmtree(conc_dst)
        shutil.copytree(conc_src, conc_dst)

    # ── 13. Factor statistics ───────────────────────────────────────
    fstats_src = RESULTS_DIR / "factor_statistics"
    if fstats_src.exists():
        fstats_dst = ARTEFACTS_DIR / "factor_statistics"
        if fstats_dst.exists():
            shutil.rmtree(fstats_dst)
        shutil.copytree(fstats_src, fstats_dst)

    # ── 14. Create README ───────────────────────────────────────────
    readme = ARTEFACTS_DIR / "README.md"
    with open(readme, "w") as f:
        f.write(_generate_readme())

    print(f"  ✓ Artefacts organised: {ARTEFACTS_DIR}/")
    _print_artefact_tree(ARTEFACTS_DIR)


def _generate_readme() -> str:
    return """\
# Publication Artefacts — LICSDE Final Run

## Overview

These artefacts correspond to the final experimental run of the
**LLM-Informed Causal Synthetic Data Engine (LICSDE)** paper.

**Primary LLM**: GPT-4o-mini (via OpenRouter)
**Extraction Windows**: Bi-weekly (2W)
**LLM Samples per Window**: 5
**Period**: 2018-01-01 to 2023-12-31
**Tickers**: AAPL, MSFT, JPM, XOM, JNJ, WMT, CAT, AMGN
**Seed**: 42

## Directory Structure

```
publication_artefacts/
├── README.md                     # This file
├── config/                       # Configuration files used
│   ├── final_run.yaml            # Experiment config
│   └── config_used.yaml          # Config as loaded by pipeline
├── training_data.csv             # Real OHLCV data (Yahoo Finance)
├── generation_summary.json       # Timing and summary statistics
├── cross_asset_correlation.json  # Estimated correlation matrix
├── extracted_factors/            # LLM-extracted causal factors
│   └── <TICKER>_analyses.json    # Per-window factor attributions
├── impact_matrices/              # Learned factor impact matrices
│   └── <TICKER>_impact_matrix.json
├── synthetic_data/               # Generated synthetic OHLCV data
│   └── <TICKER>_synthetic.csv    # 10 samples per ticker
├── pipeline_reports/             # Per-ticker pipeline execution reports
│   └── <TICKER>_pipeline_report.json
├── evaluation/                   # Quality metrics per ticker
│   └── <TICKER>/
│       └── quality_report.json   # KS, Wasserstein, moments, etc.
├── benchmark/                    # TSTR comparison (Causal SDE vs TimeGAN)
├── ablation_vanilla_sde/         # Ablation A: no factors (K=0)
├── ablation_causal_direction/    # Ablation B: bull/bear direction test
├── concordance/                  # Cross-LLM concordance heatmap
│   ├── concordance_results.json
│   └── heatmap_concordance.png
└── factor_statistics/            # Factor term analysis
    ├── factor_statistics.json
    ├── all_factors_raw.csv
    ├── factor_term_frequency.csv
    ├── factor_name_frequency.csv
    └── *.png                     # Visualisations
```

## Reproducing Results

```bash
# 1. Install dependencies
pip install -e .

# 2. Set API keys
export OPENROUTER_API_KEY="your-key"
export FRED_API_KEY="your-key"

# 3. Run all experiments
python scripts/reproduce_all.py

# Or run individual steps:
python scripts/run_pipeline.py --config configs/djia28.yaml
python scripts/baseline_timegan.py configs/djia28.yaml
python scripts/ablation_vanilla_sde.py
python scripts/ablation_causal_direction.py
python scripts/multi_llm_concordance.py
python scripts/eval_factor_statistics.py
```

## Data Sources

- **Market data**: Yahoo Finance (via yfinance)
- **News (RAG)**: Wikipedia Current Events, FRED, SEC EDGAR
- **LLM**: OpenAI GPT-4o-mini (via OpenRouter)
"""


def _print_artefact_tree(path: Path, prefix: str = "  ", max_depth: int = 2):
    """Print a simple directory tree."""
    if not path.is_dir():
        return
    items = sorted(path.iterdir())
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        if item.is_dir():
            n_files = sum(1 for _ in item.rglob("*") if _.is_file())
            print(f"{prefix}{connector}{item.name}/ ({n_files} files)")
            if max_depth > 1:
                ext = "    " if is_last else "│   "
                _print_artefact_tree(item, prefix + ext, max_depth - 1)
        else:
            size = item.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"
            print(f"{prefix}{connector}{item.name} ({size_str})")


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

def run_step(step: dict, log_dir: Path) -> dict:
    """Run a single experiment step."""
    step_id = step["id"]
    name = step["name"]

    print(f"\n{'═' * 70}")
    print(f"  STEP {step_id}: {name}")
    print(f"  {step['description']}")
    print(f"{'═' * 70}\n")

    t_start = time.time()
    result = {"step": step_id, "name": name, "status": "unknown"}

    if "function" in step:
        # Python function call
        try:
            func = globals()[step["function"]]
            func()
            result["status"] = "success"
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"\n  ✗ STEP {step_id} FAILED: {e}")
    else:
        # Subprocess
        log_path = log_dir / f"step_{step_id}.log"
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    step["command"],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600 * 6,  # 6 hour timeout
                )
                log_file.write(proc.stdout or "")

            if proc.returncode == 0:
                result["status"] = "success"
            else:
                result["status"] = "failed"
                result["returncode"] = proc.returncode
                # Print last 20 lines of output on failure
                lines = (proc.stdout or "").strip().split("\n")
                last = "\n".join(lines[-20:])
                print(f"\n  ✗ STEP {step_id} FAILED (rc={proc.returncode})")
                print(f"  Last output:\n{last}")

            # Also print to console
            print(proc.stdout or "")

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            print(f"\n  ✗ STEP {step_id} TIMED OUT (6h)")
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"\n  ✗ STEP {step_id} FAILED: {e}")

    elapsed = time.time() - t_start
    result["elapsed_seconds"] = round(elapsed, 1)
    result["elapsed_human"] = _format_duration(elapsed)

    status_icon = "✓" if result["status"] == "success" else "✗"
    print(f"\n  {status_icon} Step {step_id} {result['status']} "
          f"({result['elapsed_human']})")

    return result


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(
        description="Master Experiment Orchestrator — Final Publication Run")
    parser.add_argument("--skip", default="",
                        help="Comma-separated step IDs to skip (e.g. '1,2')")
    parser.add_argument("--only", default="",
                        help="Comma-separated step IDs to run exclusively")
    args = parser.parse_args()

    skip_ids = set(int(x) for x in args.skip.split(",") if x.strip())
    only_ids = set(int(x) for x in args.only.split(",") if x.strip())

    # Setup logging
    log_dir = RESULTS_DIR / "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("═" * 70)
    print("  LICSDE — FINAL PUBLICATION RUN")
    print("═" * 70)
    print(f"  Timestamp:  {timestamp}")
    print(f"  Config:     {CONFIG_PATH}")
    print(f"  Output:     {RESULTS_DIR}")
    print(f"  Steps:      {len(STEPS)}")
    if skip_ids:
        print(f"  Skipping:   {skip_ids}")
    if only_ids:
        print(f"  Only:       {only_ids}")
    print("═" * 70)

    t_global = time.time()
    all_results = []

    for step in STEPS:
        sid = step["id"]

        if only_ids and sid not in only_ids:
            print(f"\n  ⊘ Skipping step {sid} (not in --only)")
            continue
        if sid in skip_ids:
            print(f"\n  ⊘ Skipping step {sid} (--skip)")
            continue

        result = run_step(step, log_dir)
        all_results.append(result)

        # Save progress after each step
        progress = {
            "timestamp": timestamp,
            "steps": all_results,
            "total_elapsed": round(time.time() - t_global, 1),
        }
        progress_path = RESULTS_DIR / "experiment_progress.json"
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(progress_path, "w") as f:
            json.dump(progress, f, indent=2)

    total_elapsed = time.time() - t_global

    # ── Final report ────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("  FINAL REPORT")
    print(f"{'═' * 70}")
    for r in all_results:
        icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {icon} Step {r['step']}: {r['name']} — "
              f"{r['status']} ({r['elapsed_human']})")
    print(f"\n  Total elapsed: {_format_duration(total_elapsed)}")
    print(f"  Results:       {RESULTS_DIR}/")
    if ARTEFACTS_DIR.exists():
        print(f"  Artefacts:     {ARTEFACTS_DIR}/")
    print(f"{'═' * 70}\n")

    # Exit with error if any step failed
    if any(r["status"] != "success" for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
