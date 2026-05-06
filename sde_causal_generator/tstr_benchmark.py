# -*- coding: utf-8 -*-
"""
Train on Synthetic, Test on Real (TSTR) Benchmark
===================================================

Evaluates synthetic data quality by training predictive models on synthetic
data and testing on held-out real data.  Compares against the **TRTR**
(Train Real, Test Real) upper bound.

Implements two downstream tasks:

1. **Return Direction Classification** — binary (up / down) from lagged features.
2. **Volatility Forecasting** — predict realised volatility from lagged features.

The TSTR score is reported as a ratio TSTR / TRTR (closer to 1.0 = better).

Usage::

    from sde_causal_generator.tstr_benchmark import TSTRBenchmark, TSTRConfig

    bench = TSTRBenchmark(TSTRConfig(n_lags=10, epochs=50))
    results = bench.evaluate(
        real_train_df=train_df,
        real_test_df=test_df,
        synthetic_df=synth_df,
        ticker="AAPL",
    )
    print(results)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TSTRConfig:
    """Configuration for the TSTR benchmark."""

    # Feature engineering
    n_lags: int = 10                    # number of lagged returns as features
    vol_window: int = 20                # realised vol estimation window
    features: List[str] = field(
        default_factory=lambda: ["close"]
    )

    # Model architecture
    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 15                  # early stopping patience

    # Evaluation
    n_runs: int = 3                     # independent runs to average
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ══════════════════════════════════════════════════════════════════════
# Feature engineering
# ══════════════════════════════════════════════════════════════════════


def _compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Log-returns from a 1D price series."""
    return np.diff(np.log(np.clip(prices, 1e-8, None)))


def _build_lagged_features(
    returns: np.ndarray,
    n_lags: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y_direction, y_volatility) from a return series.

    X[t] = [r_{t-n_lags}, ..., r_{t-1}]  (lagged returns)
    y_dir[t] = 1 if r_t > 0 else 0       (next-day direction)

    Returns:
        X: (N, n_lags) lagged returns
        y_dir: (N,) binary labels {0, 1}
    """
    N = len(returns) - n_lags
    if N <= 0:
        return np.zeros((0, n_lags)), np.zeros(0)

    X = np.zeros((N, n_lags), dtype=np.float32)
    for i in range(N):
        X[i] = returns[i : i + n_lags]

    y_dir = (returns[n_lags:] > 0).astype(np.float32)
    return X, y_dir


def _build_vol_features(
    returns: np.ndarray,
    n_lags: int,
    vol_window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y_vol) for the volatility forecasting task.

    X[t] = [r_{t-n_lags}, ..., r_{t-1}]
    y_vol[t] = std(r_{t:t+vol_window})   (forward realised volatility)

    Returns:
        X: (N, n_lags)
        y_vol: (N,) realised volatility targets
    """
    N = len(returns) - n_lags - vol_window + 1
    if N <= 0:
        return np.zeros((0, n_lags)), np.zeros(0)

    X = np.zeros((N, n_lags), dtype=np.float32)
    y_vol = np.zeros(N, dtype=np.float32)

    for i in range(N):
        X[i] = returns[i : i + n_lags]
        fwd = returns[i + n_lags : i + n_lags + vol_window]
        y_vol[i] = fwd.std()

    return X, y_vol


def _extract_returns_from_df(
    df: pd.DataFrame,
    ticker: str,
    feature: str = "close",
) -> np.ndarray:
    """Extract log-returns from a DataFrame for a specific ticker."""
    ticker_df = df[df["tic"] == ticker].sort_values("date") if "tic" in df.columns else df.sort_values("date")
    prices = ticker_df[feature].values.astype(np.float64)
    return _compute_log_returns(prices)


# ══════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════


class _GRUClassifier(nn.Module):
    """GRU-based binary classifier for return direction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1) OR (batch, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        logit = self.fc(out[:, -1, :])
        return logit


class _GRURegressor(nn.Module):
    """GRU-based regressor for volatility forecasting."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()  # volatility is non-negative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        pred = self.relu(self.fc(out[:, -1, :]))
        return pred


# ══════════════════════════════════════════════════════════════════════
# Training & Evaluation helpers
# ══════════════════════════════════════════════════════════════════════


def _train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TSTRConfig,
) -> _GRUClassifier:
    """Train a GRU classifier on (X, y) data."""
    device = config.resolve_device()
    model = _GRUClassifier(1, config.hidden_dim, config.num_layers, config.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Reshape for GRU: (N, n_lags) → (N, n_lags, 1)
    X_t = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            logit = model(xb)
            loss = criterion(logit, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    return model


def _eval_classifier(
    model: _GRUClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a classifier and return metrics."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
        logits = model(X_t).cpu().numpy().flatten()
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        preds = (probs > 0.5).astype(float)

    accuracy = float(np.mean(preds == y_test))
    # F1 score (manual to avoid sklearn dependency)
    tp = float(np.sum((preds == 1) & (y_test == 1)))
    fp = float(np.sum((preds == 1) & (y_test == 0)))
    fn = float(np.sum((preds == 0) & (y_test == 1)))
    precision = tp / max(tp + fp, 1e-10)
    recall = tp / max(tp + fn, 1e-10)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    # AUC-ROC (simple trapezoidal from sorted predictions)
    auc = _compute_auc(y_test, probs)

    return {"accuracy": accuracy, "f1": f1, "auc_roc": auc}


def _compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Simple AUC-ROC computation without sklearn."""
    # Sort by predicted score descending
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Cumulative sum of true positives
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    # TPR, FPR
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])

    # Trapezoidal integration
    auc = float(np.trapezoid(tpr, fpr))
    return auc


def _train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: TSTRConfig,
) -> _GRURegressor:
    """Train a GRU regressor for volatility forecasting."""
    device = config.resolve_device()
    model = _GRURegressor(1, config.hidden_dim, config.num_layers, config.dropout).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
    y_t = torch.FloatTensor(y_train).unsqueeze(-1).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    return model


def _eval_regressor(
    model: _GRURegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate a regressor and return metrics."""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
        preds = model(X_t).cpu().numpy().flatten()

    mse = float(np.mean((preds - y_test) ** 2))
    mae = float(np.mean(np.abs(preds - y_test)))

    # R² score
    ss_res = np.sum((y_test - preds) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    return {"mse": mse, "mae": mae, "r2": r2}


# ══════════════════════════════════════════════════════════════════════
# TSTR Benchmark
# ══════════════════════════════════════════════════════════════════════


class TSTRBenchmark:
    """
    Train on Synthetic, Test on Real (TSTR) benchmark.

    Evaluates synthetic data quality through:
    - **TRTR**: Train Real → Test Real (upper bound / oracle)
    - **TSTR**: Train Synthetic → Test Real (quality measure)
    - **TSTR Ratio**: TSTR_metric / TRTR_metric (closer to 1.0 = better)
    """

    def __init__(self, config: Optional[TSTRConfig] = None):
        self.config = config or TSTRConfig()

    def evaluate(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        ticker: str,
        feature: str = "close",
    ) -> Dict[str, Any]:
        """
        Run full TSTR evaluation for one ticker.

        Args:
            real_train_df: Real training data (OHLCV format).
            real_test_df: Real test/holdout data (OHLCV format).
            synthetic_df: Synthetic data to evaluate (OHLCV format).
            ticker: Ticker symbol.
            feature: Price column to use for returns.

        Returns:
            Dictionary with TRTR, TSTR results and ratios per task.
        """
        cfg = self.config

        # Extract returns
        real_train_rets = _extract_returns_from_df(real_train_df, ticker, feature)
        real_test_rets = _extract_returns_from_df(real_test_df, ticker, feature)
        synth_rets = _extract_returns_from_df(synthetic_df, ticker, feature)

        results: Dict[str, Any] = {
            "ticker": ticker,
            "feature": feature,
            "n_lags": cfg.n_lags,
            "n_runs": cfg.n_runs,
        }

        # ── Task 1: Return Direction Classification ─────────────────
        results["direction"] = self._run_direction_task(
            real_train_rets, real_test_rets, synth_rets
        )

        # ── Task 2: Volatility Forecasting ──────────────────────────
        results["volatility"] = self._run_volatility_task(
            real_train_rets, real_test_rets, synth_rets
        )

        # ── Summary ─────────────────────────────────────────────────
        results["summary"] = self._compute_summary(results)

        return results

    def evaluate_multi_ticker(
        self,
        real_train_df: pd.DataFrame,
        real_test_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        tickers: List[str],
        feature: str = "close",
    ) -> Dict[str, Any]:
        """Run TSTR evaluation for multiple tickers and aggregate."""
        per_ticker = {}
        for ticker in tickers:
            try:
                per_ticker[ticker] = self.evaluate(
                    real_train_df, real_test_df, synthetic_df, ticker, feature
                )
            except Exception as e:
                print(f"  ⚠ TSTR failed for {ticker}: {e}")
                continue

        # Aggregate
        agg = self._aggregate_results(per_ticker)
        return {
            "per_ticker": per_ticker,
            "aggregate": agg,
        }

    # ────────────────────────────────────────────────────────────────
    # Tasks
    # ────────────────────────────────────────────────────────────────

    def _run_direction_task(
        self,
        real_train_rets: np.ndarray,
        real_test_rets: np.ndarray,
        synth_rets: np.ndarray,
    ) -> Dict[str, Any]:
        """Run the return direction classification task."""
        cfg = self.config
        device = cfg.resolve_device()

        # Build features
        X_real_train, y_real_train = _build_lagged_features(real_train_rets, cfg.n_lags)
        X_real_test, y_real_test = _build_lagged_features(real_test_rets, cfg.n_lags)
        X_synth, y_synth = _build_lagged_features(synth_rets, cfg.n_lags)

        if len(X_real_test) == 0 or len(X_synth) == 0:
            return {"error": "insufficient data for direction task"}

        trtr_runs = []
        tstr_runs = []

        for run in range(cfg.n_runs):
            np.random.seed(run * 42)
            torch.manual_seed(run * 42)

            # TRTR: train on real, test on real
            model_trtr = _train_classifier(X_real_train, y_real_train, cfg)
            metrics_trtr = _eval_classifier(model_trtr, X_real_test, y_real_test, device)
            trtr_runs.append(metrics_trtr)

            # TSTR: train on synthetic, test on real
            model_tstr = _train_classifier(X_synth, y_synth, cfg)
            metrics_tstr = _eval_classifier(model_tstr, X_real_test, y_real_test, device)
            tstr_runs.append(metrics_tstr)

        # Average across runs
        trtr_avg = _average_metrics(trtr_runs)
        tstr_avg = _average_metrics(tstr_runs)

        # Compute ratios
        ratios = {}
        for key in trtr_avg:
            trtr_val = trtr_avg[key]
            tstr_val = tstr_avg[key]
            if abs(trtr_val) > 1e-10:
                ratios[f"{key}_ratio"] = tstr_val / trtr_val
            else:
                ratios[f"{key}_ratio"] = 0.0

        return {
            "trtr": trtr_avg,
            "tstr": tstr_avg,
            "ratios": ratios,
            "n_train_real": len(X_real_train),
            "n_test": len(X_real_test),
            "n_train_synth": len(X_synth),
        }

    def _run_volatility_task(
        self,
        real_train_rets: np.ndarray,
        real_test_rets: np.ndarray,
        synth_rets: np.ndarray,
    ) -> Dict[str, Any]:
        """Run the volatility forecasting task."""
        cfg = self.config
        device = cfg.resolve_device()

        # Build features
        X_real_train, y_real_train = _build_vol_features(
            real_train_rets, cfg.n_lags, cfg.vol_window
        )
        X_real_test, y_real_test = _build_vol_features(
            real_test_rets, cfg.n_lags, cfg.vol_window
        )
        X_synth, y_synth = _build_vol_features(
            synth_rets, cfg.n_lags, cfg.vol_window
        )

        if len(X_real_test) == 0 or len(X_synth) == 0:
            return {"error": "insufficient data for volatility task"}

        trtr_runs = []
        tstr_runs = []

        for run in range(cfg.n_runs):
            np.random.seed(run * 42 + 1000)
            torch.manual_seed(run * 42 + 1000)

            # TRTR
            model_trtr = _train_regressor(X_real_train, y_real_train, cfg)
            metrics_trtr = _eval_regressor(model_trtr, X_real_test, y_real_test, device)
            trtr_runs.append(metrics_trtr)

            # TSTR
            model_tstr = _train_regressor(X_synth, y_synth, cfg)
            metrics_tstr = _eval_regressor(model_tstr, X_real_test, y_real_test, device)
            tstr_runs.append(metrics_tstr)

        trtr_avg = _average_metrics(trtr_runs)
        tstr_avg = _average_metrics(tstr_runs)

        # For regression, ratios are on R² and inverted MSE
        ratios = {}
        for key in ["r2"]:
            trtr_val = trtr_avg[key]
            tstr_val = tstr_avg[key]
            if abs(trtr_val) > 1e-10:
                ratios[f"{key}_ratio"] = tstr_val / trtr_val
            else:
                ratios[f"{key}_ratio"] = 0.0

        # MSE ratio (lower is better, so invert: TRTR/TSTR)
        if tstr_avg.get("mse", 0) > 1e-15:
            ratios["mse_ratio"] = trtr_avg["mse"] / tstr_avg["mse"]
        else:
            ratios["mse_ratio"] = 0.0

        return {
            "trtr": trtr_avg,
            "tstr": tstr_avg,
            "ratios": ratios,
            "n_train_real": len(X_real_train),
            "n_test": len(X_real_test),
            "n_train_synth": len(X_synth),
        }

    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute summary TSTR score across tasks."""
        scores = []

        dir_res = results.get("direction", {})
        if "ratios" in dir_res:
            acc_ratio = dir_res["ratios"].get("accuracy_ratio", 0)
            scores.append(acc_ratio)

        vol_res = results.get("volatility", {})
        if "ratios" in vol_res:
            r2_ratio = vol_res["ratios"].get("r2_ratio", 0)
            scores.append(r2_ratio)

        avg_tstr_ratio = float(np.mean(scores)) if scores else 0.0

        return {
            "avg_tstr_ratio": avg_tstr_ratio,
            "direction_accuracy_ratio": dir_res.get("ratios", {}).get("accuracy_ratio", 0),
            "volatility_r2_ratio": vol_res.get("ratios", {}).get("r2_ratio", 0),
        }

    def _aggregate_results(
        self, per_ticker: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate results across tickers."""
        all_dir_ratios = []
        all_vol_ratios = []

        for ticker, res in per_ticker.items():
            summary = res.get("summary", {})
            dr = summary.get("direction_accuracy_ratio", None)
            vr = summary.get("volatility_r2_ratio", None)
            if dr is not None:
                all_dir_ratios.append(dr)
            if vr is not None:
                all_vol_ratios.append(vr)

        return {
            "mean_direction_accuracy_ratio": float(np.mean(all_dir_ratios)) if all_dir_ratios else 0.0,
            "mean_volatility_r2_ratio": float(np.mean(all_vol_ratios)) if all_vol_ratios else 0.0,
            "n_tickers": len(per_ticker),
        }

    # ────────────────────────────────────────────────────────────────
    # I/O
    # ────────────────────────────────────────────────────────────────

    def save_results(
        self, results: Dict[str, Any], output_path: str
    ) -> None:
        """Save TSTR results to JSON."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"  ✓ TSTR results saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════


def _average_metrics(runs: List[Dict[str, float]]) -> Dict[str, float]:
    """Average metric dictionaries across runs."""
    if not runs:
        return {}
    keys = runs[0].keys()
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
