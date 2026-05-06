# -*- coding: utf-8 -*-
"""
TimeGAN Baseline Adapter
=========================

Integrates the existing TimeGAN implementation (from ``emergent-finrl``) as a
baseline synthetic data generator, producing outputs in the same CSV format as
the Causal SDE pipeline for direct comparison.

Usage::

    from sde_causal_generator.timegan_baseline import TimeGANBaseline, TimeGANBaselineConfig

    cfg = TimeGANBaselineConfig(seq_len=24, hidden_dim=24)
    baseline = TimeGANBaseline(cfg)
    baseline.train(training_df, tickers=["AAPL", "MSFT"])
    results = baseline.generate_all(n_samples=10, num_trading_days=252)
    baseline.save_results("results/timegan_baseline/")
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Resolve the TimeGAN import path ─────────────────────────────────
# Optional dependency. If a vendored FinRL/TimeGAN tree is present at
# <repo>/emergent-finrl/finrl/meta/data_generators/timegan/, we expose
# the trainer; otherwise TimeGAN baselines are simply unavailable
# (the rest of the pipeline still works). The reproducibility snapshot
# does not ship TimeGAN sources — TSTR/TRTR results are pre-cached.

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

_TIMEGAN_CANDIDATES = [
    _PROJECT_ROOT / "emergent-finrl",
]

_timegan_available = False
for _candidate in _TIMEGAN_CANDIDATES:
    if (_candidate / "finrl" / "meta" / "data_generators" / "timegan").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        _timegan_available = True
        break

if _timegan_available:
    from finrl.meta.data_generators.timegan.timegan_trainer import (
        TimeGANTrainer,
        TimeGANConfig,
    )
    from finrl.meta.data_generators.timegan.data_utils import (
        prepare_timegan_input,
        postprocess_timegan_output,
        TimeGANMetadata,
    )
    from finrl.meta.data_generators.timegan.validation import (
        discriminative_score,
        predictive_score,
    )
else:
    TimeGANTrainer = None  # type: ignore[assignment,misc]
    TimeGANConfig = None  # type: ignore[assignment,misc]


def _check_timegan_available() -> None:
    """Raise ImportError with a helpful message if TimeGAN is not found."""
    if not _timegan_available:
        raise ImportError(
            "TimeGAN implementation not found. Expected at: "
            f"{_PROJECT_ROOT / 'emergent-finrl' / 'finrl' / 'meta' / 'data_generators' / 'timegan'}. "
            "Clone or symlink the emergent-finrl repository into the project root."
        )


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TimeGANBaselineConfig:
    """Configuration for the TimeGAN baseline."""

    # Architecture
    seq_len: int = 24
    hidden_dim: int = 24
    num_layers: int = 3
    module_type: str = "gru"

    # Training epochs per phase
    embedding_epochs: int = 600
    supervised_epochs: int = 600
    joint_epochs: int = 600

    # Optimiser
    learning_rate: float = 1e-3
    batch_size: int = 128

    # Joint-phase loss weights
    gamma: float = 1.0
    tail_loss_weight: float = 10.0

    # Activations & noise
    latent_activation: str = "tanh"
    output_activation: str = "sigmoid"
    noise_type: str = "gaussian"

    # Features to use from the OHLCV DataFrame
    features: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "volume"]
    )

    # Device
    device: str = "auto"

    # Logging
    print_every: int = 200

    def to_timegan_config(self) -> "TimeGANConfig":
        """Convert to the internal TimeGANConfig."""
        _check_timegan_available()
        return TimeGANConfig(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            module_type=self.module_type,
            embedding_epochs=self.embedding_epochs,
            supervised_epochs=self.supervised_epochs,
            joint_epochs=self.joint_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            gamma=self.gamma,
            tail_loss_weight=self.tail_loss_weight,
            latent_activation=self.latent_activation,
            output_activation=self.output_activation,
            noise_type=self.noise_type,
            device=self.device,
            print_every=self.print_every,
        )


# ══════════════════════════════════════════════════════════════════════
# TimeGAN Baseline
# ══════════════════════════════════════════════════════════════════════


class TimeGANBaseline:
    """
    Trains TimeGAN per-ticker and generates synthetic data in the same
    CSV format as the Causal SDE pipeline.

    This enables direct TSTR and distributional comparison between
    Causal SDE and TimeGAN outputs.
    """

    def __init__(self, config: Optional[TimeGANBaselineConfig] = None):
        _check_timegan_available()
        self.config = config or TimeGANBaselineConfig()

        # Per-ticker trainers and metadata
        self._trainers: Dict[str, TimeGANTrainer] = {}
        self._metadata: Dict[str, TimeGANMetadata] = {}
        self._training_times: Dict[str, float] = {}

        # Generated results (populated by generate_all)
        self._results: Dict[str, List[pd.DataFrame]] = {}

    # ────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """
        Train a TimeGAN model per ticker.

        Args:
            df: OHLCV DataFrame with columns [date, tic, open, high, low, close, volume].
            tickers: Subset of tickers to train. ``None`` trains all tickers in ``df``.

        Returns:
            Dictionary mapping ticker → training history.
        """
        _check_timegan_available()

        if tickers is None:
            tickers = sorted(df["tic"].unique().tolist())

        histories: Dict[str, dict] = {}

        for ticker in tickers:
            ticker_df = df[df["tic"] == ticker].copy()
            if len(ticker_df) < self.config.seq_len + 10:
                print(f"  ⚠ {ticker}: insufficient data ({len(ticker_df)} rows), skipping")
                continue

            print(f"\n{'─' * 50}")
            print(f"  TimeGAN Training: {ticker} ({len(ticker_df)} rows)")
            print(f"{'─' * 50}")

            t0 = time.time()

            # Prepare 3D input
            data_3d, metadata = prepare_timegan_input(
                ticker_df,
                tickers=[ticker],
                features=self.config.features,
                seq_len=self.config.seq_len,
            )
            self._metadata[ticker] = metadata

            # Build & train
            tg_config = self.config.to_timegan_config()
            trainer = TimeGANTrainer(tg_config)
            trainer.train(data_3d)

            elapsed = time.time() - t0
            self._trainers[ticker] = trainer
            self._training_times[ticker] = elapsed
            histories[ticker] = trainer.training_history

            print(f"  ✓ {ticker} trained in {elapsed:.1f}s")

        return histories

    def train_single(
        self,
        df: pd.DataFrame,
        ticker: str,
    ) -> dict:
        """Train TimeGAN for a single ticker. Returns training history."""
        return self.train(df, tickers=[ticker]).get(ticker, {})

    # ────────────────────────────────────────────────────────────────
    # Generation
    # ────────────────────────────────────────────────────────────────

    def generate(
        self,
        ticker: str,
        n_samples: int = 10,
        num_trading_days: int = 252,
        start_date: str = "2018-01-02",
    ) -> List[pd.DataFrame]:
        """
        Generate synthetic data for a single ticker.

        Args:
            ticker: Ticker symbol (must have been trained).
            n_samples: Number of independent synthetic trajectories.
            num_trading_days: Length of each trajectory.
            start_date: Start date for the generated calendar.

        Returns:
            List of DataFrames, one per sample, matching Causal SDE format.
        """
        if ticker not in self._trainers:
            raise ValueError(
                f"Ticker {ticker!r} not trained. Available: {list(self._trainers)}"
            )

        trainer = self._trainers[ticker]
        metadata = self._metadata[ticker]

        all_dfs: List[pd.DataFrame] = []

        for sample_i in range(n_samples):
            # Generate enough windows
            n_windows_needed = max(
                50, (num_trading_days // self.config.seq_len) * 2
            )
            gen_array = trainer.generate(n_windows_needed)

            # Post-process into DataFrame
            sample_df = postprocess_timegan_output(
                gen_array,
                metadata,
                num_trading_days=num_trading_days,
                start_date=start_date,
                sample_id=sample_i,
            )

            # Filter to only this ticker (postprocess may output multiple)
            sample_df = sample_df[sample_df["tic"] == ticker].copy()
            all_dfs.append(sample_df)

        self._results[ticker] = all_dfs
        return all_dfs

    def generate_all(
        self,
        n_samples: int = 10,
        num_trading_days: int = 252,
        start_date: str = "2018-01-02",
    ) -> Dict[str, List[pd.DataFrame]]:
        """Generate synthetic data for all trained tickers."""
        results = {}
        for ticker in self._trainers:
            results[ticker] = self.generate(
                ticker, n_samples, num_trading_days, start_date
            )
        self._results = results
        return results

    # ────────────────────────────────────────────────────────────────
    # I/O
    # ────────────────────────────────────────────────────────────────

    def save_results(self, output_dir: str) -> Dict[str, str]:
        """
        Save generated synthetic data to CSV files matching the Causal SDE layout.

        Creates:
            ``<output_dir>/<TICKER>/synthetic_data_timegan.csv``   (merged)
            ``<output_dir>/<TICKER>/synthetic_data_<i>.csv``       (per-sample)
            ``<output_dir>/<TICKER>/timegan_report.json``          (training info)
        """
        saved_paths: Dict[str, str] = {}

        for ticker, dfs in self._results.items():
            ticker_dir = os.path.join(output_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)

            # Save per-sample CSVs
            for i, sample_df in enumerate(dfs):
                path = os.path.join(ticker_dir, f"synthetic_data_{i}.csv")
                sample_df.to_csv(path, index=False)

            # Save merged CSV
            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                merged_path = os.path.join(ticker_dir, "synthetic_data_timegan.csv")
                merged.to_csv(merged_path, index=False)
                saved_paths[ticker] = merged_path

            # Save report
            report = {
                "ticker": ticker,
                "method": "timegan",
                "n_samples": len(dfs),
                "training_time_seconds": self._training_times.get(ticker),
                "config": {
                    "seq_len": self.config.seq_len,
                    "hidden_dim": self.config.hidden_dim,
                    "embedding_epochs": self.config.embedding_epochs,
                    "supervised_epochs": self.config.supervised_epochs,
                    "joint_epochs": self.config.joint_epochs,
                },
            }
            if ticker in self._trainers:
                report["training_history"] = self._trainers[ticker].training_history

            report_path = os.path.join(ticker_dir, "timegan_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        return saved_paths

    def save_checkpoint(self, output_dir: str) -> None:
        """Save TimeGAN model checkpoints and metadata for all tickers."""
        import pickle
        os.makedirs(output_dir, exist_ok=True)
        for ticker, trainer in self._trainers.items():
            ckpt_path = os.path.join(output_dir, f"timegan_{ticker}.pt")
            trainer.save_checkpoint(ckpt_path)

            # Save metadata alongside (needed for generation)
            if ticker in self._metadata:
                meta_path = os.path.join(output_dir, f"timegan_{ticker}_meta.pkl")
                with open(meta_path, "wb") as f:
                    pickle.dump(self._metadata[ticker], f)

    def load_checkpoint(self, output_dir: str, tickers: List[str]) -> None:
        """Load TimeGAN model checkpoints and metadata for specified tickers."""
        import pickle
        _check_timegan_available()
        for ticker in tickers:
            ckpt_path = os.path.join(output_dir, f"timegan_{ticker}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  ⚠ Checkpoint not found for {ticker}: {ckpt_path}")
                continue
            trainer = TimeGANTrainer()
            trainer.load_checkpoint(ckpt_path)
            self._trainers[ticker] = trainer

            # Load metadata if available
            meta_path = os.path.join(output_dir, f"timegan_{ticker}_meta.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self._metadata[ticker] = pickle.load(f)

    # ────────────────────────────────────────────────────────────────
    # Validation metrics
    # ────────────────────────────────────────────────────────────────

    def compute_quality_metrics(
        self,
        real_df: pd.DataFrame,
        ticker: str,
        disc_epochs: int = 200,
        pred_epochs: int = 200,
        n_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Compute discriminative and predictive scores for a ticker.

        Args:
            real_df: Real OHLCV DataFrame.
            ticker: Ticker to evaluate.
            disc_epochs: Epochs for the discriminative classifier.
            pred_epochs: Epochs for the predictive model.
            n_runs: Number of independent evaluation runs.

        Returns:
            Dictionary with ``discriminative_score`` and ``predictive_score_mae``.
        """
        _check_timegan_available()

        if ticker not in self._trainers or ticker not in self._metadata:
            raise ValueError(f"Ticker {ticker!r} not trained.")

        metadata = self._metadata[ticker]

        # Prepare real data windows
        real_3d, _ = prepare_timegan_input(
            real_df[real_df["tic"] == ticker],
            tickers=[ticker],
            features=self.config.features,
            seq_len=self.config.seq_len,
        )

        # Generate synthetic windows
        trainer = self._trainers[ticker]
        synth_3d = trainer.generate(len(real_3d))

        disc = discriminative_score(
            real_3d, synth_3d,
            epochs=disc_epochs,
            n_runs=n_runs,
        )
        pred = predictive_score(
            real_3d, synth_3d,
            epochs=pred_epochs,
            n_runs=n_runs,
        )

        return {
            "discriminative_score": disc,
            "predictive_score_mae": pred,
        }

    @property
    def trained_tickers(self) -> List[str]:
        """Return list of trained ticker symbols."""
        return list(self._trainers.keys())
