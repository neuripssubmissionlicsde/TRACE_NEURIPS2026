# -*- coding: utf-8 -*-
"""
Full Benchmark Pipeline: Causal SDE vs TimeGAN
================================================

Orchestrates the complete comparison pipeline:

1. Load / verify Causal SDE results (from prior ``generate_causal_sde_data.py`` run).
2. Train TimeGAN baseline on the same training data.
3. Generate synthetic data with TimeGAN.
4. Run TSTR benchmark for both methods.
5. Compute distributional metrics (KS, volatility, kurtosis).
6. Generate comparative report & visualisations.

Usage::

    from sde_causal_generator.benchmark import BenchmarkPipeline, BenchmarkConfig

    cfg = BenchmarkConfig.from_yaml("configs/causal_sde_config.yaml")
    bench = BenchmarkPipeline(cfg)
    report = bench.run()
    bench.save_report("results/benchmark/")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkConfig:
    """Configuration for the full benchmark pipeline."""

    # ── Data ─────────────────────────────────────────────────────────
    tickers: List[str] = field(default_factory=lambda: ["AAPL"])
    train_start: str = "2018-01-01"
    train_end: str = "2023-12-31"
    test_ratio: float = 0.2           # fraction of real data for test split
    features: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    # ── Causal SDE ───────────────────────────────────────────────────
    causal_sde_dir: str = ""          # path to existing Causal SDE results
    n_samples: int = 10

    # ── TimeGAN ──────────────────────────────────────────────────────
    timegan_seq_len: int = 24
    timegan_hidden_dim: int = 24
    timegan_embedding_epochs: int = 600
    timegan_supervised_epochs: int = 600
    timegan_joint_epochs: int = 600
    timegan_batch_size: int = 128
    timegan_device: str = "auto"

    # ── TSTR ─────────────────────────────────────────────────────────
    tstr_n_lags: int = 10
    tstr_epochs: int = 100
    tstr_n_runs: int = 3
    tstr_vol_window: int = 20

    # ── Output / execution ───────────────────────────────────────────
    output_dir: str = "results/benchmark"
    seed: int = 42
    skip_timegan: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load from the main pipeline config YAML, extracting relevant fields."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        output_name = raw.get("output_dir", "multi_asset_2018_2023")
        project_root = Path(path).resolve().parent.parent

        gen = raw.get("generation", {})
        timegan_cfg = raw.get("timegan", {})
        tstr_cfg = raw.get("tstr", {})
        bench_cfg = raw.get("benchmark", {})

        return cls(
            tickers=raw.get("tickers", ["AAPL"]),
            train_start=raw.get("train_start", "2018-01-01"),
            train_end=raw.get("train_end", "2023-12-31"),
            test_ratio=bench_cfg.get("test_ratio", 0.2),
            features=["open", "high", "low", "close", "volume"],
            causal_sde_dir=str(project_root / "results" / output_name),
            n_samples=gen.get("n_samples", 10),
            timegan_seq_len=timegan_cfg.get("seq_len", 24),
            timegan_hidden_dim=timegan_cfg.get("hidden_dim", 24),
            timegan_embedding_epochs=timegan_cfg.get("embedding_epochs", 600),
            timegan_supervised_epochs=timegan_cfg.get("supervised_epochs", 600),
            timegan_joint_epochs=timegan_cfg.get("joint_epochs", 600),
            timegan_batch_size=timegan_cfg.get("batch_size", 128),
            timegan_device=timegan_cfg.get("device", "auto"),
            tstr_n_lags=tstr_cfg.get("n_lags", 10),
            tstr_epochs=tstr_cfg.get("epochs", 100),
            tstr_n_runs=tstr_cfg.get("n_runs", 3),
            tstr_vol_window=tstr_cfg.get("vol_window", 20),
            output_dir=bench_cfg.get("output_dir", str(project_root / "results" / "benchmark")),
            seed=raw.get("seed", 42),
            skip_timegan=bench_cfg.get("skip_timegan", False),
        )


# ══════════════════════════════════════════════════════════════════════
# Distributional Metrics
# ══════════════════════════════════════════════════════════════════════


def compute_distributional_metrics(
    real_returns: np.ndarray,
    synth_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Compute distributional comparison metrics between real and synthetic returns.

    Returns:
        Dictionary with KS statistic, p-value, volatility ratio,
        kurtosis difference, and skewness difference.
    """
    ks_stat, ks_pval = stats.ks_2samp(real_returns, synth_returns)

    real_vol = float(np.std(real_returns))
    synth_vol = float(np.std(synth_returns))
    vol_ratio = synth_vol / max(real_vol, 1e-10)

    real_kurt = float(stats.kurtosis(real_returns))
    synth_kurt = float(stats.kurtosis(synth_returns))

    real_skew = float(stats.skew(real_returns))
    synth_skew = float(stats.skew(synth_returns))

    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "real_volatility": real_vol,
        "synth_volatility": synth_vol,
        "volatility_ratio": vol_ratio,
        "real_kurtosis": real_kurt,
        "synth_kurtosis": synth_kurt,
        "kurtosis_diff": abs(synth_kurt - real_kurt),
        "real_skewness": real_skew,
        "synth_skewness": synth_skew,
        "skewness_diff": abs(synth_skew - real_skew),
    }


# ══════════════════════════════════════════════════════════════════════
# Benchmark Pipeline
# ══════════════════════════════════════════════════════════════════════


class BenchmarkPipeline:
    """
    Full benchmark pipeline: Causal SDE vs TimeGAN.

    Produces a comprehensive comparison report with:
    - TSTR scores (direction accuracy, volatility R²)
    - Distributional metrics (KS, volatility, kurtosis)
    - Visualisations (return distributions, TSTR comparison bar charts)
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._report: Dict[str, Any] = {}
        np.random.seed(config.seed)

    def run(
        self,
        training_df: Optional[pd.DataFrame] = None,
        skip_timegan: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full benchmark pipeline.

        Args:
            training_df: Real OHLCV training data. If ``None``, loads from
                ``<causal_sde_dir>/training_data.csv``.
            skip_timegan: If ``True``, skip TimeGAN training. Defaults to
                ``config.skip_timegan`` when ``None``.

        Returns:
            Complete benchmark report dictionary.
        """
        if skip_timegan is None:
            skip_timegan = self.config.skip_timegan
        import torch
        torch.manual_seed(self.config.seed)

        t_start = time.time()
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        print("═" * 60)
        print("  BENCHMARK: Causal SDE vs TimeGAN")
        print("═" * 60)
        print(f"  Tickers: {cfg.tickers}")
        print(f"  Period:  {cfg.train_start} → {cfg.train_end}")
        print(f"  Output:  {cfg.output_dir}")
        print("═" * 60)

        # ── 1. Load real training data ──────────────────────────────
        if training_df is None:
            training_csv = os.path.join(cfg.causal_sde_dir, "training_data.csv")
            if not os.path.exists(training_csv):
                raise FileNotFoundError(
                    f"Training data not found: {training_csv}. "
                    "Run generate_causal_sde_data.py first."
                )
            training_df = pd.read_csv(training_csv)
            training_df["date"] = pd.to_datetime(training_df["date"])

        # ── 2. Train/test split ─────────────────────────────────────
        real_train, real_test = self._split_train_test(training_df)

        print(f"\n  Real data split:")
        print(f"    Train: {len(real_train)} rows")
        print(f"    Test:  {len(real_test)} rows")

        # ── 3. Load Causal SDE results ──────────────────────────────
        print(f"\n{'─' * 50}")
        print("  Loading Causal SDE synthetic data...")
        csde_synth = self._load_causal_sde_results()

        # ── 4. Train TimeGAN & generate ─────────────────────────────
        if not skip_timegan:
            print(f"\n{'─' * 50}")
            print("  Training TimeGAN baseline...")
            tgan_synth = self._run_timegan(real_train)
        else:
            tgan_synth = self._load_timegan_results()

        # ── 5. TSTR evaluation ──────────────────────────────────────
        print(f"\n{'─' * 50}")
        print("  Running TSTR benchmark...")

        from sde_causal_generator.tstr_benchmark import TSTRBenchmark, TSTRConfig

        tstr_cfg = TSTRConfig(
            n_lags=cfg.tstr_n_lags,
            epochs=cfg.tstr_epochs,
            n_runs=cfg.tstr_n_runs,
            vol_window=cfg.tstr_vol_window,
        )
        tstr_bench = TSTRBenchmark(tstr_cfg)

        tstr_results: Dict[str, Any] = {}

        for ticker in cfg.tickers:
            print(f"\n  TSTR: {ticker}")

            csde_df = csde_synth.get(ticker)
            tgan_df = tgan_synth.get(ticker)

            ticker_results: Dict[str, Any] = {}

            if csde_df is not None and len(csde_df) > 0:
                try:
                    ticker_results["causal_sde"] = tstr_bench.evaluate(
                        real_train, real_test, csde_df, ticker
                    )
                except Exception as e:
                    print(f"    ⚠ Causal SDE TSTR failed for {ticker}: {e}")

            if tgan_df is not None and len(tgan_df) > 0:
                try:
                    ticker_results["timegan"] = tstr_bench.evaluate(
                        real_train, real_test, tgan_df, ticker
                    )
                except Exception as e:
                    print(f"    ⚠ TimeGAN TSTR failed for {ticker}: {e}")

            tstr_results[ticker] = ticker_results

        # ── 6. Distributional metrics ───────────────────────────────
        print(f"\n{'─' * 50}")
        print("  Computing distributional metrics...")

        dist_results = self._compute_all_distributional(
            training_df, csde_synth, tgan_synth
        )

        # ── 7. Compile report ───────────────────────────────────────
        total_time = time.time() - t_start

        self._report = {
            "config": {
                "tickers": cfg.tickers,
                "train_period": f"{cfg.train_start} → {cfg.train_end}",
                "test_ratio": cfg.test_ratio,
                "n_samples": cfg.n_samples,
                "seed": cfg.seed,
            },
            "tstr": tstr_results,
            "distributional": dist_results,
            "comparison": self._build_comparison_table(tstr_results, dist_results),
            "total_time_seconds": round(total_time, 1),
        }

        # ── 8. Visualisations ──────────────────────────────────────
        print(f"\n{'─' * 50}")
        print("  Generating visualisations...")
        self._plot_comparison(tstr_results, dist_results)

        print(f"\n{'═' * 60}")
        print(f"  BENCHMARK COMPLETE — {total_time:.1f}s")
        print(f"  Results: {cfg.output_dir}/")
        print(f"{'═' * 60}\n")

        return self._report

    # ────────────────────────────────────────────────────────────────
    # Data loading
    # ────────────────────────────────────────────────────────────────

    def _split_train_test(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split real data into train/test by date."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        for ticker in self.config.tickers:
            if ticker not in df["tic"].values:
                continue

        # Sort by date
        dates = sorted(df["date"].unique())
        split_idx = int(len(dates) * (1 - self.config.test_ratio))
        split_date = dates[split_idx]

        train = df[df["date"] < split_date].copy()
        test = df[df["date"] >= split_date].copy()

        return train, test

    def _load_causal_sde_results(self) -> Dict[str, pd.DataFrame]:
        """Load Causal SDE synthetic data from the results directory."""
        results = {}
        cfg = self.config

        for ticker in cfg.tickers:
            merged_path = os.path.join(
                cfg.causal_sde_dir, ticker, "synthetic_data_causal_sde.csv"
            )
            first_sample_path = os.path.join(
                cfg.causal_sde_dir, ticker, "synthetic_data_0.csv"
            )

            if os.path.exists(merged_path):
                df = pd.read_csv(merged_path)
                # Use only sample 0 for TSTR (single trajectory)
                if "sample" in df.columns:
                    df = df[df["sample"] == 0].copy()
                df["tic"] = ticker
                results[ticker] = df
                print(f"    ✓ {ticker}: loaded {len(df)} rows (Causal SDE)")
            elif os.path.exists(first_sample_path):
                df = pd.read_csv(first_sample_path)
                df["tic"] = ticker
                results[ticker] = df
                print(f"    ✓ {ticker}: loaded {len(df)} rows (sample 0)")
            else:
                print(f"    ⚠ {ticker}: no Causal SDE results found")

        return results

    def _run_timegan(
        self, real_train: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Train TimeGAN and generate synthetic data."""
        from sde_causal_generator.timegan_baseline import (
            TimeGANBaseline,
            TimeGANBaselineConfig,
        )

        cfg = self.config

        tg_cfg = TimeGANBaselineConfig(
            seq_len=cfg.timegan_seq_len,
            hidden_dim=cfg.timegan_hidden_dim,
            embedding_epochs=cfg.timegan_embedding_epochs,
            supervised_epochs=cfg.timegan_supervised_epochs,
            joint_epochs=cfg.timegan_joint_epochs,
            batch_size=cfg.timegan_batch_size,
            device=cfg.timegan_device,
            features=cfg.features,
        )

        baseline = TimeGANBaseline(tg_cfg)
        baseline.train(real_train, tickers=cfg.tickers)

        # Determine trading days from real data
        # Use first ticker to get the count
        sample_ticker = cfg.tickers[0]
        ticker_data = real_train[real_train["tic"] == sample_ticker]
        num_days = len(ticker_data)
        start_date = str(ticker_data["date"].min())[:10]

        results = baseline.generate_all(
            n_samples=cfg.n_samples,
            num_trading_days=num_days,
            start_date=start_date,
        )

        # Save TimeGAN results
        timegan_dir = os.path.join(cfg.output_dir, "timegan")
        baseline.save_results(timegan_dir)
        baseline.save_checkpoint(os.path.join(timegan_dir, "checkpoints"))

        # Convert to single-sample DataFrames for TSTR
        merged = {}
        for ticker, dfs in results.items():
            if dfs:
                merged[ticker] = dfs[0]  # use first sample for TSTR

        return merged

    def _load_timegan_results(self) -> Dict[str, pd.DataFrame]:
        """Load previously generated TimeGAN results."""
        results = {}
        timegan_dir = os.path.join(self.config.output_dir, "timegan")

        for ticker in self.config.tickers:
            path = os.path.join(timegan_dir, ticker, "synthetic_data_0.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df["tic"] = ticker
                results[ticker] = df
                print(f"    ✓ {ticker}: loaded TimeGAN results")
            else:
                print(f"    ⚠ {ticker}: no TimeGAN results found")

        return results

    # ────────────────────────────────────────────────────────────────
    # Distributional metrics
    # ────────────────────────────────────────────────────────────────

    def _compute_all_distributional(
        self,
        real_df: pd.DataFrame,
        csde_synth: Dict[str, pd.DataFrame],
        tgan_synth: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Compute distributional metrics for all tickers and methods."""
        results = {}

        for ticker in self.config.tickers:
            real_ticker = real_df[real_df["tic"] == ticker]
            if len(real_ticker) == 0:
                continue

            real_rets = np.diff(np.log(np.clip(
                real_ticker.sort_values("date")["close"].values, 1e-8, None
            )))

            ticker_results: Dict[str, Any] = {}

            csde_df = csde_synth.get(ticker)
            if csde_df is not None and "close" in csde_df.columns and len(csde_df) > 1:
                csde_rets = np.diff(np.log(np.clip(
                    csde_df.sort_values("date")["close"].values, 1e-8, None
                )))
                ticker_results["causal_sde"] = compute_distributional_metrics(
                    real_rets, csde_rets
                )

            tgan_df = tgan_synth.get(ticker)
            if tgan_df is not None and "close" in tgan_df.columns and len(tgan_df) > 1:
                tgan_rets = np.diff(np.log(np.clip(
                    tgan_df.sort_values("date")["close"].values, 1e-8, None
                )))
                ticker_results["timegan"] = compute_distributional_metrics(
                    real_rets, tgan_rets
                )

            results[ticker] = ticker_results

        return results

    # ────────────────────────────────────────────────────────────────
    # Comparison table
    # ────────────────────────────────────────────────────────────────

    def _build_comparison_table(
        self,
        tstr_results: Dict[str, Any],
        dist_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a summary comparison table."""
        rows = []

        for ticker in self.config.tickers:
            tstr = tstr_results.get(ticker, {})
            dist = dist_results.get(ticker, {})

            row: Dict[str, Any] = {"ticker": ticker}

            # TSTR direction accuracy
            for method in ["causal_sde", "timegan"]:
                prefix = method
                m = tstr.get(method, {})
                dir_tstr = m.get("direction", {}).get("tstr", {})
                dir_trtr = m.get("direction", {}).get("trtr", {})

                row[f"{prefix}_dir_tstr_acc"] = dir_tstr.get("accuracy", None)
                row[f"{prefix}_dir_trtr_acc"] = dir_trtr.get("accuracy", None)

                # Volatility R²
                vol_tstr = m.get("volatility", {}).get("tstr", {})
                vol_trtr = m.get("volatility", {}).get("trtr", {})
                row[f"{prefix}_vol_tstr_r2"] = vol_tstr.get("r2", None)
                row[f"{prefix}_vol_trtr_r2"] = vol_trtr.get("r2", None)

                # Distributional
                d = dist.get(method, {})
                row[f"{prefix}_ks_stat"] = d.get("ks_statistic", None)
                row[f"{prefix}_vol_ratio"] = d.get("volatility_ratio", None)
                row[f"{prefix}_kurt_diff"] = d.get("kurtosis_diff", None)

            rows.append(row)

        return {
            "per_ticker": rows,
            "columns": [
                "ticker",
                "causal_sde_dir_tstr_acc", "causal_sde_dir_trtr_acc",
                "timegan_dir_tstr_acc", "timegan_dir_trtr_acc",
                "causal_sde_vol_tstr_r2", "causal_sde_vol_trtr_r2",
                "timegan_vol_tstr_r2", "timegan_vol_trtr_r2",
                "causal_sde_ks_stat", "timegan_ks_stat",
                "causal_sde_vol_ratio", "timegan_vol_ratio",
                "causal_sde_kurt_diff", "timegan_kurt_diff",
            ],
        }

    # ────────────────────────────────────────────────────────────────
    # Visualisations
    # ────────────────────────────────────────────────────────────────

    def _plot_comparison(
        self,
        tstr_results: Dict[str, Any],
        dist_results: Dict[str, Any],
    ) -> None:
        """Generate comparison plots."""
        cfg = self.config
        out = cfg.output_dir

        # Collect data for bar charts
        tickers = []
        csde_acc = []
        tgan_acc = []
        csde_ks = []
        tgan_ks = []

        for ticker in cfg.tickers:
            tstr = tstr_results.get(ticker, {})
            dist = dist_results.get(ticker, {})

            csde_tstr = tstr.get("causal_sde", {}).get("direction", {}).get("tstr", {})
            tgan_tstr = tstr.get("timegan", {}).get("direction", {}).get("tstr", {})

            if csde_tstr.get("accuracy") is not None or tgan_tstr.get("accuracy") is not None:
                tickers.append(ticker)
                csde_acc.append(csde_tstr.get("accuracy", 0))
                tgan_acc.append(tgan_tstr.get("accuracy", 0))
                csde_ks.append(dist.get("causal_sde", {}).get("ks_statistic", 0))
                tgan_ks.append(dist.get("timegan", {}).get("ks_statistic", 0))

        if not tickers:
            print("    No data to plot.")
            return

        # ── Bar chart: TSTR direction accuracy ──────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(tickers))
        width = 0.35

        ax = axes[0]
        bars1 = ax.bar(x - width / 2, csde_acc, width, label="Causal SDE", color="#2196F3")
        bars2 = ax.bar(x + width / 2, tgan_acc, width, label="TimeGAN", color="#FF5722")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("TSTR Accuracy")
        ax.set_title("TSTR: Direction Classification Accuracy")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.7, label="Random")
        ax.legend()
        ax.set_ylim(0.3, 0.8)

        # ── Bar chart: KS statistic ────────────────────────────────
        ax = axes[1]
        ax.bar(x - width / 2, csde_ks, width, label="Causal SDE", color="#2196F3")
        ax.bar(x + width / 2, tgan_ks, width, label="TimeGAN", color="#FF5722")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("KS Statistic")
        ax.set_title("Return Distribution: KS Distance from Real")
        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(out, "benchmark_comparison.png"), dpi=150)

        # ── PDF report ──────────────────────────────────────────────
        with PdfPages(os.path.join(out, "benchmark_comparison.pdf")) as pdf:
            pdf.savefig(fig)
        plt.close(fig)

        print(f"    ✓ Plots saved to {out}/")

    # ────────────────────────────────────────────────────────────────
    # I/O
    # ────────────────────────────────────────────────────────────────

    def save_report(self, path: Optional[str] = None) -> str:
        """Save the benchmark report to JSON."""
        if path is None:
            path = os.path.join(self.config.output_dir, "benchmark_report.json")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._report, f, indent=2, default=_json_default)

        print(f"  ✓ Benchmark report saved to {path}")
        return path

    @property
    def report(self) -> Dict[str, Any]:
        """Return the benchmark report."""
        return self._report


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
