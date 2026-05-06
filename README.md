# TRACE — LICSDE NeurIPS 2026 Reproducibility Snapshot

> **LLM-Informed Causal Synthetic Data Engine (LICSDE).**
> Anonymous code release for double-blind review.

This repository contains the code and **pre-computed causal-factor cache**
needed to reproduce every experimental table and figure in the NeurIPS 2026
paper *"TRACE: Trajectory-aware Reproduction of Asset Dynamics via
LLM-Informed Causal SDEs"*.

The cache (`cache/djia28/`) ships **4,340 LLM-extracted factor
windows across 28 DJIA constituents (2018-2023)**, so reviewers can
regenerate every result **without an LLM API key** and without paying any
inference cost. The pipeline transparently short-circuits all LLM calls
when a cache hit is found
(see [`sde_causal_generator/factor_extractor.py`](sde_causal_generator/factor_extractor.py),
lines 397–401).

---

## 1. Setup

The snapshot is tested on **Linux x86_64 + Python 3.12**. CPU-only works for
all evaluation scripts; a CUDA GPU is needed only for the optional Neural-SDE
baseline.

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate licsde
pip install -e .
```

### Option B — venv + pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.snapshot.txt
pip install -e .
```

### Verifying the install

```bash
python -c "import sde_causal_generator, numpy, torch, pandas; print('ok')"
pytest tests/ -q             # ~1 minute on a laptop
```

---

## 2. Reproducing the paper

All headline numbers and figures come from a single configuration:
[`configs/djia28.yaml`](configs/djia28.yaml)
(28 DJIA tickers, 2018-01-01 → 2023-12-31, 10 synthetic samples / ticker).

### 2.1 Quick path — only re-render figures from cached aggregates

If you only want to regenerate the paper figures from the *already-aggregated*
metrics shipped in `results/_aggregate/` (provided as a separate data
release; see Section 5), run:

```bash
python scripts/render_figures.py \
       --aggregate-dir results/_aggregate \
       --out-dir       results/figures
```

This produces all six headline + appendix figures listed in the table below.

### 2.2 Full reproduction — from cached factors

This regenerates **synthetic trajectories, all baselines, and all figures**
starting only from the factor cache (no LLM calls, no internet needed).

```bash
# (a) Synthetic data + ImpactMatrix for all 28 tickers
python scripts/run_pipeline.py \
       --config configs/djia28.yaml

# (b) Baselines and ablations
python scripts/ablation_vanilla_sde.py         --config configs/djia28.yaml
python scripts/ablation_shuffle_control.py     --config configs/djia28.yaml
python scripts/ablation_causal_direction.py
python scripts/ablation_direction_amplified.py
python scripts/eval_lstm_discriminator.py       --config configs/djia28.yaml
python scripts/eval_oos_2023.py
python scripts/eval_multiseed_bootstrap.py      --config configs/djia28.yaml
python scripts/eval_n30_trajectories.py

# (c) Aggregate everything into a single CSV / JSON
python scripts/aggregate_results.py

# (d) Render every paper figure
python scripts/render_figures.py
```

End-to-end (a)–(d) takes roughly **2-3 hours on a 16-core CPU** with the
factor cache hot.

### 2.3 Optional — re-extract factors with your own LLM

If you wish to verify the factor extraction itself, set
`OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) and delete the cache:

```bash
export OPENROUTER_API_KEY=...
rm -rf cache/djia28/
python scripts/run_pipeline.py --config configs/djia28.yaml
```

The default model is `GPT-4o-mini`. See
[`configs/djia28.yaml`](configs/djia28.yaml) section
`llm:` to switch providers.

---

## 3. Figure → script map

Every figure in the paper is reproducible from a single script:

| #  | Figure (label)             | Output file                                              | Script                                                                                  |
|----|----------------------------|----------------------------------------------------------|-----------------------------------------------------------------------------------------|
| 1  | `fig:baseline-suite-box`   | `baseline_suite_boxplot.png`                             | [scripts/render_figures.py](scripts/render_figures.py) → `make_baseline_suite_boxplot()` |
| 2  | `fig:oos-2023-summary`     | `oos_2023_scatter.png`                                   | `render_figures.py` → `make_oos_scatter()`                                       |
| 3  | `fig:causal-events`        | `factors_over_time.png` *(rename of)* `generation_factors_AAPL.png` | Pipeline auto-emits via `evaluate_data._plot_factor_boxes` during run (a)               |
| 4  | `fig:cross-asset-corr`     | `cross_asset_correlation_3panel.png`                     | `render_figures.py` → `make_cross_asset_correlation_3panel()`                    |
| 5  | `fig:n30-aapl`             | `real_vs_synth_AAPL_n30.png`                             | [scripts/eval_n30_trajectories.py](scripts/eval_n30_trajectories.py)            |
| 6  | `fig:n30-grid`             | `n30_grid_28tickers.png`                                 | `render_figures.py` → `make_n30_grid_appendix()`                                 |
| 7  | `fig:direction-amplified`  | `amplified_summary.png`                                  | [scripts/ablation_direction_amplified.py](scripts/ablation_direction_amplified.py)                |
| 8  | `fig:direction-boxes`      | `direction_box_grid_28tickers.png`                       | `render_figures.py` → `make_direction_box_grid_appendix()`                       |
| 9  | `fig:multi-llm`            | `multi_llm_comparison.png`                               | [scripts/multi_llm_downstream.py](scripts/multi_llm_downstream.py)              |
| 10 | `fig:multiseed-boxplot`    | `multiseed_boxplot.png`                                  | [scripts/eval_multiseed_bootstrap.py](scripts/eval_multiseed_bootstrap.py)                |
| 11 | `fig:multiseed-ci`         | `multiseed_ci_28tickers.png`                             | `render_figures.py` → `make_multiseed_ci_appendix()`                             |

---

## 4. Repository layout

```
sde_causal_generator/        # Main library
├── pipeline.py              # End-to-end orchestrator
├── factor_extractor.py      # LLM + cache short-circuit (line 397)
├── factor_impact_network.py # FIN training (TFT-style)
├── generate.py              # Impact-driven analytical generator
├── neural_sde.py            # Neural-SDE baseline
├── tstr_benchmark.py        # TSTR / TRTR (uses np.trapezoid; needs numpy>=2)
├── timegan_baseline.py      # TimeGAN wrapper (sources NOT shipped; TSTR cached)
├── evaluate_data.py         # All evaluation plots (incl. factor schedule)
└── ...
scripts/                     # Reproducibility CLI scripts (see Section 3)\n  run_pipeline.py            # Full pipeline (Phases 1–3)\n  reproduce_all.py           # Master orchestrator\n  ablation_*.py              # Ablation studies\n  eval_*.py                  # Evaluation scripts\n  baseline_*.py              # Baseline comparisons\n  multi_llm_*.py             # Cross-LLM experiments\n  render_figures.py          # Paper figure renderer\n  aggregate_results.py       # CSV/JSON aggregation
configs/
├── djia28.yaml    # Master config used in the paper
├── djia28_tstr.yaml
└── ...
cache/
└── djia28/        # 4,340 cached factor JSONs (28 tickers, ~21 MB)
tests/                       # pytest unit tests
```

### Cache format

Each file under `cache/djia28/<TICKER>/llm_factors/*.json` follows:

```json
{
  "start_date": "2018-01-02",
  "end_date":   "2018-04-04",
  "window_type": "quarterly",
  "price_change_pct": -1.43,
  "factors": [
    {
      "name": "iphone_demand_softening",
      "category": "company_specific",
      "direction": "bearish",
      "magnitude": 0.62,
      "persistence": "transient",
      "description": "..."
    }
  ],
  "raw_llm_response": "..."
}
```

---

## 5. What is and is not shipped

| Artifact                               | In repo? | Where                                  |
|----------------------------------------|----------|----------------------------------------|
| Source code (library + scripts)        | ✅       | `sde_causal_generator/`, `scripts/`    |
| Configs used in the paper              | ✅       | `configs/`                             |
| LLM-extracted factor cache (28 × 6 yr) | ✅       | `cache/djia28/` (~21 MB)     |
| RAG news cache                          | ❌       | Regenerable from FRED + Wikipedia      |
| Raw OHLCV                               | ❌       | Auto-downloaded via `yfinance`         |
| TimeGAN sources                         | ❌       | TSTR/TRTR results pre-summarised in paper; full TimeGAN code is third-party (FinRL) and not redistributed |
| Pre-trained ImpactMatrix `.npz`         | ❌       | Regenerable in <1 h from cached factors|


---

## 6. Hardware / runtime

| Stage                                      | Time (16-core CPU) | GPU needed? |
|--------------------------------------------|--------------------|-------------|
| Factor extraction (cache hit)              | < 1 min            | no          |
| ImpactMatrix training (28 tickers)         | ~30 min            | optional    |
| Synthetic generation (10 samples × 28)     | ~5 min             | no          |
| All baselines                              | ~45 min            | no          |
| Neural-SDE baseline                        | ~1 h (GPU: 10 min) | recommended |
| Aggregate + render figures                 | ~2 min             | no          |

Total disk footprint after a full run: **~1.5 GB** (synthetic CSVs + plots).

---

## 7. License

Released under the [MIT License](LICENSE).

---

## 8. Anonymous contact

This is a double-blind submission. Please raise reproducibility issues via
the anonymous repository's issue tracker. We will respond after the
review period.
