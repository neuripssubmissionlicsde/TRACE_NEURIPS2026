# -*- coding: utf-8 -*-
"""
LLM-based Causal Factor Extraction.

Extracts market-moving factors from historical price data using an LLM
grounded by a temporal news RAG, producing structured factor analyses
and a binary factor-presence matrix.

Prompt Engineering
------------------
Two prompt variants are used:

* **RAG-aware** (preferred): the LLM receives only contemporaneous news
  articles and must anchor its analysis in them — mitigates hindsight
  bias and narrative fallacy.
* **Fallback**: used when no news context is available.  The LLM is
  instructed to avoid future knowledge, but has no enforcement.
"""

from __future__ import annotations

import json
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_structures import (
    FACTOR_CATEGORIES,
    CausalFactor,
    WindowAnalysis,
)
from .news_rag import NewsRAGManager


# ══════════════════════════════════════════════════════════════════════
# Prompt templates
# ══════════════════════════════════════════════════════════════════════

_RAG_PROMPT = """\
You are a senior financial market analyst performing a CONTEMPORANEOUS
causal attribution analysis.  Your goal is to identify the specific
factors that CAUSED price movements — not factors that merely coincided.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

PRICE DATA:
- Opening price: {open_price}
- Closing price: {close_price}
- High: {high_price}
- Low: {low_price}
- Price change: {price_change_pct:.2f}%
- Average daily volume: {avg_volume}
{intra_window_stats}

{news_context}

ANALYSIS METHODOLOGY — Think step-by-step:
1. First, identify the DOMINANT REGIME during this window (bull/bear/sideways).
2. Examine the intra-window statistics: high volatility, drawdowns, or weekly
   reversals suggest multiple competing factors, not a single driver.
3. For each candidate factor, ask: "Would the price have moved THIS WAY
   and THIS MUCH without this factor?"  If not, it is causal.
4. Separate structural factors (e.g., Fed policy) from transient catalysts
   (e.g., a single earnings miss).  Mark persistence accordingly.
5. Attribute unexplained variance honestly to "market_noise".

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the news articles provided above.
2. If no relevant news explains the price movement, attribute it to
   "market_noise" with high magnitude.
3. Do NOT reference any events that are NOT in the provided news context.
4. Do NOT use knowledge of what happened AFTER {end_date}.
5. For each factor, cite the article number [N] that supports it.
6. Keep magnitude and confidence SEPARATE: magnitude = how much market
   impact; confidence = how certain you are of the causal link.
7. NAMING RULE: Factor names must be SPECIFIC CAUSAL EVENTS, not generic
   descriptions of indicator movements.  BAD: "rising_treasury_yields",
   "declining_vix", "stable_unemployment_rate".  GOOD: "fed_tightening_cycle_2022",
   "post_covid_vix_normalization", "labor_market_recovery_q3".
   Always tie the name to the CAUSE or EVENT, not the indicator reading.
   If yields rose because of a specific event, name the event, not the yield move.

FEW-SHOT EXAMPLES (for format reference only — do NOT copy these factors):

Example 1 — Bearish window with clear cause:
[
  {{"name": "fed_rate_hike_75bps", "category": "macroeconomic",
    "direction": -0.85, "magnitude": 0.55, "persistence": "medium",
    "confidence": 0.95, "source_articles": [2, 5],
    "description": "Fed raised rates by 75bps [2], markets sold off as tighter monetary conditions reduced equity valuations [5]."}},
  {{"name": "tech_earnings_miss", "category": "company_specific",
    "direction": -0.6, "magnitude": 0.25, "persistence": "transient",
    "confidence": 0.80, "source_articles": [7],
    "description": "Major tech companies missed revenue estimates [7], adding sector-wide selling pressure."}},
  {{"name": "market_noise", "category": "market_sentiment",
    "direction": 0.0, "magnitude": 0.20, "persistence": "transient",
    "confidence": 1.0, "source_articles": [],
    "description": "Residual variance not attributable to identified factors."}}
]

Example 2 — Sideways window with mixed signals:
[
  {{"name": "stimulus_package", "category": "macroeconomic",
    "direction": 0.7, "magnitude": 0.30, "persistence": "long",
    "confidence": 0.85, "source_articles": [1, 3],
    "description": "Congress passed $1.9T fiscal stimulus [1], supporting consumer spending expectations [3]."}},
  {{"name": "supply_chain_disruption", "category": "sector_specific",
    "direction": -0.5, "magnitude": 0.25, "persistence": "medium",
    "confidence": 0.70, "source_articles": [4],
    "description": "Global shipping delays raised input costs [4], offsetting positive fiscal sentiment."}},
  {{"name": "market_noise", "category": "market_sentiment",
    "direction": 0.0, "magnitude": 0.45, "persistence": "transient",
    "confidence": 1.0, "source_articles": [],
    "description": "Large unexplained component — price was range-bound with no dominant driver."}}
]

NOW RESPOND for {ticker} ({start_date} to {end_date}).

Respond with a JSON array of factors.  Each factor must have:
- "name": short snake_case identifier (e.g., "fed_rate_hike")
- "category": one of {categories}
- "direction": float from -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float from 0.0 to 1.0
- "persistence": "transient" | "short" | "medium" | "long"
- "confidence": float 0.0 to 1.0
- "source_articles": list of article numbers that support this factor
- "description": one-sentence explanation citing specific evidence

The magnitudes across all factors MUST sum to 1.0.
Include a "market_noise" factor for truly unexplained random variance.

IMPORTANT — DISTINGUISH NOISE FROM CONSOLIDATION (DISC-3):
- "market_noise" = purely stochastic, no identifiable cause, direction=0.0.
- If the market is SIDEWAYS or RANGE-BOUND due to opposing forces
  balancing out, do NOT dump that into "market_noise".  Instead, name a
  "consolidation" or "range_bound" factor with category="market_sentiment",
  direction near 0.0, and describe the competing forces.  This is
  qualitatively different from noise: it signals low-volatility regimes
  with mean-reverting dynamics.

Respond ONLY with valid JSON.  No markdown, no commentary.
"""

_FALLBACK_PROMPT = """\
You are a financial market analyst.  You are given real historical
price data for {ticker} during the period {start_date} to {end_date}.

Price data summary:
- Opening price: {open_price}
- Closing price: {close_price}
- High: {high_price}
- Low: {low_price}
- Price change: {price_change_pct:.2f}%
- Average daily volume: {avg_volume}

IMPORTANT CONSTRAINTS:
1. Only cite factors that were KNOWN and OBSERVABLE during this period.
   Do NOT use hindsight from future events.
2. Distinguish between factors that CAUSED movement vs. those that
   merely COINCIDED with it.
3. If the movement was likely random noise, say so explicitly.
4. Be honest about uncertainty in your attributions.

Respond with a JSON array of factors.  Each factor must have:
- "name": short snake_case identifier
- "category": one of {categories}
- "direction": float from -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float from 0.0 to 1.0
- "persistence": "transient" | "short" | "medium" | "long"
- "confidence": float 0.0 to 1.0
- "description": one-sentence explanation

The magnitudes across all factors should sum to approximately 1.0.
Include a "noise_residual" factor for truly unexplained random variance.
If the market is sideways or range-bound, name a separate "consolidation"
factor — do NOT collapse it into noise.

Respond ONLY with valid JSON.  No markdown, no commentary.
"""


# ══════════════════════════════════════════════════════════════════════
# Extractor
# ══════════════════════════════════════════════════════════════════════


class LLMFactorExtractor:
    """
    Extracts causal factors from historical price data using an LLM.

    * Supports optional RAG grounding (``news_rag`` parameter).
    * Runs ``n_samples`` LLM calls per window and averages factor
      magnitudes for robustness.
    * Results are cached to disk so repeated runs are free.

    Parameters
    ----------
    llm_model : str
        LiteLLM-compatible model identifier.
    api_key : str | None
        API key; falls back to ``OPENROUTER_API_KEY`` env var.
    cache_dir : str
        Directory for JSON-cached window analyses.
    temperature : float
        LLM sampling temperature (low → consistent).
    n_samples : int
        Number of LLM calls per window (averaged).
    news_rag : NewsRAGManager | None
        If provided, retrieves contemporaneous news context.
    """

    def __init__(
        self,
        llm_model: str = "openrouter/anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        cache_dir: str = "cache/llm_factors",
        temperature: float = 0.1,
        n_samples: int = 3,
        news_rag: Optional[NewsRAGManager] = None,
        max_articles: int = 50,
        llm_provider: str = "openrouter",
        ollama_base_url: str = "",
    ):
        self.llm_model = llm_model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.n_samples = n_samples
        self.news_rag = news_rag
        self.max_articles = max_articles
        self.llm_provider = llm_provider
        self.ollama_base_url = ollama_base_url.rstrip("/") if ollama_base_url else ""
        os.makedirs(cache_dir, exist_ok=True)

        # ── prompt version hash (included in cache key) ─────────────
        self._prompt_hash = hashlib.md5(
            (_RAG_PROMPT + _FALLBACK_PROMPT).encode()
        ).hexdigest()[:8]

        # ── LLM consistency tracking ────────────────────────────────
        self._consistency_stats: List[Dict] = []

    # ── LLM call ────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, _max_retries: int = 4) -> str:
        """Call the LLM API via LiteLLM (default) or Ollama.

        Retries on transient errors (Timeout, RateLimitError, APIError)
        with exponential backoff: 5s → 10s → 20s → 40s.
        """
        if self.llm_provider == "ollama":
            return self._call_ollama(prompt)

        import litellm  # type: ignore[import-untyped]

        litellm.suppress_debug_info = True

        for attempt in range(_max_retries + 1):
            try:
                response = litellm.completion(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    api_key=self.api_key,
                    max_tokens=4096,
                )
                return response.choices[0].message.content
            except ImportError:
                raise ImportError("Install litellm:  pip install litellm")
            except (
                litellm.exceptions.Timeout,
                litellm.exceptions.RateLimitError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.APIError,
                litellm.exceptions.InternalServerError,
            ) as e:
                if attempt == _max_retries:
                    raise
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
                print(
                    f"      ⟳ Retry {attempt + 1}/{_max_retries} after "
                    f"{type(e).__name__} — waiting {wait}s …"
                )
                import time
                time.sleep(wait)

    def _call_ollama(self, prompt: str) -> str:
        """Call a self-hosted Ollama instance."""
        import requests  # stdlib-adjacent, always available

        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 4096,
            },
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["response"]

    # ── parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_factors(raw_json: str) -> List[CausalFactor]:
        """Parse LLM JSON response into ``CausalFactor`` objects.

        Includes a sanitization step that fixes common LLM JSON issues
        (trailing commas, single quotes, unquoted keys) before parsing.
        """
        try:
            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            # ── JSON sanitization ───────────────────────────────────
            import re
            # Remove trailing commas before } or ]
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            # Replace single-quoted strings with double-quoted
            # (only when it looks like JSON keys/values)
            cleaned = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', cleaned)
            cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)

            factors_data = json.loads(cleaned)
            factors: List[CausalFactor] = []
            for f in factors_data:
                # BUG-C6: keep magnitude and confidence separate.
                # Multiplying them violates the sum-to-1 invariant
                # that the prompt asks the LLM to obey.  Confidence
                # is preserved in the description for downstream use.
                magnitude = float(f["magnitude"])
                confidence = f.get("confidence", 1.0)
                desc = f.get("description", "")
                if confidence < 1.0:
                    desc = f"[conf={confidence:.2f}] {desc}"
                factors.append(
                    CausalFactor(
                        name=f["name"],
                        category=f.get("category", "unknown"),
                        direction=float(f["direction"]),
                        magnitude=magnitude,
                        persistence=f.get("persistence", "medium"),
                        description=desc,
                    )
                )
            return factors
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"      WARNING: Failed to parse LLM response: {e}")
            return []

    # ── caching ─────────────────────────────────────────────────────

    def _cache_key(self, ticker: str, start: str, end: str) -> str:
        # BUG-C7: include hyperparams so changing temperature/n_samples/
        # max_articles invalidates stale cache entries.
        raw = (
            f"{ticker}_{start}_{end}_{self.llm_model}"
            f"_t{self.temperature}_ns{self.n_samples}"
            f"_ma{self.max_articles}_{self._prompt_hash}"
        )
        return hashlib.md5(raw.encode()).hexdigest()

    def _save_cache(self, path: str, analysis: WindowAnalysis):
        with open(path, "w") as fh:
            json.dump(analysis.to_dict(), fh, indent=2)

    def _load_cache(self, path: str) -> WindowAnalysis:
        with open(path, "r") as fh:
            return WindowAnalysis.from_dict(json.load(fh))

    # ── single window ──────────────────────────────────────────────

    def extract_window(
        self,
        df: pd.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> WindowAnalysis:
        """
        Extract causal factors for a single time window.

        Runs ``n_samples`` LLM calls and averages factor magnitudes
        for robustness.  Results are cached.
        """
        cache_key = self._cache_key(ticker, start_date, end_date)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        # ── prepare price summary ───────────────────────────────────
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        window_df = df[mask].sort_values("date")

        if window_df.empty:
            return WindowAnalysis(
                start_date=start_date,
                end_date=end_date,
                window_type="unknown",
                price_change_pct=0.0,
            )

        open_price = window_df.iloc[0]["open"]
        close_price = window_df.iloc[-1]["close"]
        price_change = ((close_price - open_price) / open_price) * 100

        # ── compute intra-window statistics ─────────────────────────
        intra_stats_lines = []
        closes = window_df["close"].values.astype(float)
        if len(closes) > 2:
            log_ret = np.diff(np.log(closes))
            log_ret = log_ret[np.isfinite(log_ret)]
            if len(log_ret) > 1:
                realized_vol = float(np.std(log_ret) * np.sqrt(252)) * 100
                max_drawdown = float(
                    (np.maximum.accumulate(closes) - closes).max()
                    / max(np.maximum.accumulate(closes).max(), 1e-10)
                    * 100
                )
                # Weekly returns (every 5 trading days)
                weekly_rets = []
                for wi in range(0, len(closes) - 5, 5):
                    wr = (closes[wi + 5] - closes[wi]) / closes[wi] * 100
                    weekly_rets.append(wr)
                intra_stats_lines.append("")
                intra_stats_lines.append("INTRA-WINDOW STATISTICS:")
                intra_stats_lines.append(
                    f"- Annualised volatility: {realized_vol:.1f}%"
                )
                intra_stats_lines.append(
                    f"- Max intra-window drawdown: {max_drawdown:.1f}%"
                )
                intra_stats_lines.append(
                    f"- Trading days in window: {len(closes)}"
                )
                if weekly_rets:
                    up_weeks = sum(1 for w in weekly_rets if w > 0)
                    dn_weeks = len(weekly_rets) - up_weeks
                    intra_stats_lines.append(
                        f"- Weekly returns: {up_weeks} up / {dn_weeks} down "
                        f"(best: {max(weekly_rets):+.1f}%, "
                        f"worst: {min(weekly_rets):+.1f}%)"
                    )
        intra_window_stats = "\n".join(intra_stats_lines)

        # ── build prompt ────────────────────────────────────────────
        news_context = ""
        if self.news_rag is not None:
            news_context = self.news_rag.get_context(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                max_articles=self.max_articles,
            )

        fmt_kwargs = dict(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            open_price=f"{open_price:.2f}",
            close_price=f"{close_price:.2f}",
            high_price=f"{window_df['high'].max():.2f}",
            low_price=f"{window_df['low'].min():.2f}",
            price_change_pct=price_change,
            avg_volume=f"{window_df['volume'].mean():,.0f}",
            intra_window_stats=intra_window_stats,
            categories=FACTOR_CATEGORIES,
        )

        use_rag = (
            news_context
            and "[No news articles found" not in news_context
        )
        if use_rag:
            fmt_kwargs["news_context"] = news_context
            prompt = _RAG_PROMPT.format(**fmt_kwargs)
        else:
            prompt = _FALLBACK_PROMPT.format(**fmt_kwargs)

        # ── multiple LLM samples (parallel) ──────────────────────
        all_factors: Dict[str, List[CausalFactor]] = {}
        raw_responses: List[str] = []
        parsed_per_sample: List[List[CausalFactor]] = []

        with ThreadPoolExecutor(max_workers=self.n_samples) as pool:
            futures = [
                pool.submit(self._call_llm, prompt)
                for _ in range(self.n_samples)
            ]
            for fut in futures:
                raw = fut.result()
                raw_responses.append(raw)
                factors = self._parse_factors(raw)
                parsed_per_sample.append(factors)
                for f in factors:
                    all_factors.setdefault(f.name, []).append(f)

        # ── LLM consistency tracking ──────────────────────────────
        n_parsed_ok = sum(1 for s in parsed_per_sample if len(s) > 0)
        factor_sets = [set(f.name for f in s) for s in parsed_per_sample if len(s) > 0]
        if len(factor_sets) >= 2:
            # Pairwise Jaccard similarity
            jaccards = []
            for i in range(len(factor_sets)):
                for j in range(i + 1, len(factor_sets)):
                    inter = len(factor_sets[i] & factor_sets[j])
                    union = len(factor_sets[i] | factor_sets[j])
                    jaccards.append(inter / union if union > 0 else 0.0)
            avg_jaccard = float(np.mean(jaccards))
            # Direction agreement for shared factors
            dir_agreements = []
            for name, flist in all_factors.items():
                if len(flist) >= 2:
                    dirs = [1 if f.direction > 0 else (-1 if f.direction < 0 else 0) for f in flist]
                    dir_agreements.append(1.0 if len(set(dirs)) == 1 else 0.0)
            avg_dir_agree = float(np.mean(dir_agreements)) if dir_agreements else 0.0
            # Magnitude CV for shared factors
            mag_cvs = []
            for name, flist in all_factors.items():
                if len(flist) >= 2:
                    mags = [f.magnitude for f in flist]
                    mean_m = np.mean(mags)
                    if mean_m > 0:
                        mag_cvs.append(float(np.std(mags) / mean_m))
            avg_mag_cv = float(np.mean(mag_cvs)) if mag_cvs else 0.0
        else:
            avg_jaccard = 0.0
            avg_dir_agree = 0.0
            avg_mag_cv = 0.0

        self._consistency_stats.append({
            "window": f"{start_date}→{end_date}",
            "n_samples": self.n_samples,
            "n_parsed_ok": n_parsed_ok,
            "parse_rate": n_parsed_ok / self.n_samples,
            "n_unique_factors": len(all_factors),
            "jaccard_similarity": avg_jaccard,
            "direction_agreement": avg_dir_agree,
            "magnitude_cv": avg_mag_cv,
        })

        # ── average across samples ──────────────────────────────────
        averaged: List[CausalFactor] = []
        for name, factor_list in all_factors.items():
            if len(factor_list) < self.n_samples / 2:
                continue  # appeared in fewer than half the samples
            avg_mag = float(np.mean([f.magnitude for f in factor_list]))
            avg_dir = float(np.mean([f.direction for f in factor_list]))
            averaged.append(
                CausalFactor(
                    name=name,
                    category=factor_list[0].category,
                    direction=avg_dir,
                    magnitude=avg_mag,
                    persistence=factor_list[0].persistence,
                    description=factor_list[0].description,
                )
            )

        # re-normalize magnitudes
        total_mag = sum(f.magnitude for f in averaged)
        if total_mag > 0:
            for f in averaged:
                f.magnitude /= total_mag

        analysis = WindowAnalysis(
            start_date=start_date,
            end_date=end_date,
            window_type=self._infer_window_type(start_date, end_date),
            price_change_pct=price_change,
            factors=averaged,
            raw_llm_response=raw_responses[0] if raw_responses else "",
        )

        self._save_cache(cache_path, analysis)
        return analysis

    # ── all windows ─────────────────────────────────────────────────

    def extract_all_windows(
        self,
        df: pd.DataFrame,
        ticker: str,
        window_sizes: Optional[List[str]] = None,
        max_workers: int = 4,
    ) -> List[WindowAnalysis]:
        """
        Extract factors across multiple time-window granularities.

        Parameters
        ----------
        df : pd.DataFrame
            Price DataFrame (date, open, high, low, close, volume).
        ticker : str
            Stock ticker.
        window_sizes : list[str] | None
            Pandas offset aliases.  Default ``["1ME", "1QE"]``.
        max_workers : int
            Max concurrent window extractions (default 4).

        Returns
        -------
        list[WindowAnalysis]
        """
        if window_sizes is None:
            window_sizes = ["1ME", "1QE"]

        dates = sorted(pd.to_datetime(df["date"].unique()))

        # ── pre-compute all (start, end) pairs for progress bar ─────
        window_pairs: List[Tuple[str, str]] = []
        for ws in window_sizes:
            if ws == "1BD":
                # True business-day daily: every trading day
                for d in dates:
                    s = d.strftime("%Y-%m-%d")
                    window_pairs.append((s, s))
            elif ws == "1D":
                sampled = dates[::5]
                for d in sampled:
                    s = d.strftime("%Y-%m-%d")
                    window_pairs.append((s, s))
            else:
                periods = pd.date_range(
                    start=dates[0], end=dates[-1], freq=ws
                )
                for i in range(len(periods) - 1):
                    s = periods[i].strftime("%Y-%m-%d")
                    e = periods[i + 1].strftime("%Y-%m-%d")
                    window_pairs.append((s, e))

        # ── parallel extraction with progress bar ───────────────────
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            pbar = tqdm(
                total=len(window_pairs),
                desc="    Extracting factors",
                unit="win",
                ncols=90,
            )
        except ImportError:
            pbar = None

        # Maintain original ordering via index
        results: List[Optional[WindowAnalysis]] = [None] * len(window_pairs)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.extract_window, df, ticker, s, e): idx
                for idx, (s, e) in enumerate(window_pairs)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # ── print LLM consistency report ────────────────────────────
        self._print_consistency_report()

        return [r for r in results if r is not None]

    # ── LLM consistency report ──────────────────────────────────────

    def _print_consistency_report(self) -> None:
        """Print a summary of LLM consistency metrics across all windows."""
        stats = self._consistency_stats
        if not stats:
            return

        n_windows = len(stats)
        parse_rates = [s["parse_rate"] for s in stats]
        jaccards = [s["jaccard_similarity"] for s in stats]
        dir_agrees = [s["direction_agreement"] for s in stats]
        mag_cvs = [s["magnitude_cv"] for s in stats]

        print(f"\n{'─' * 60}")
        print(f"  LLM CONSISTENCY REPORT  ({n_windows} windows)")
        print(f"{'─' * 60}")
        print(f"  Parse success rate:      {np.mean(parse_rates)*100:.1f}%"
              f"  (min {np.min(parse_rates)*100:.0f}%,"
              f" max {np.max(parse_rates)*100:.0f}%)")
        print(f"  Factor Jaccard (avg):    {np.mean(jaccards):.3f}"
              f"  (std {np.std(jaccards):.3f})")
        print(f"  Direction agreement:     {np.mean(dir_agrees)*100:.1f}%"
              f"  (std {np.std(dir_agrees)*100:.1f}%)")
        print(f"  Magnitude CV (avg):      {np.mean(mag_cvs):.3f}"
              f"  (std {np.std(mag_cvs):.3f})")
        print()

        # Interpretation guide
        j = np.mean(jaccards)
        d = np.mean(dir_agrees)
        m = np.mean(mag_cvs)
        verdicts = []
        if j >= 0.6:
            verdicts.append("Factor identity: GOOD (≥0.6 Jaccard)")
        elif j >= 0.4:
            verdicts.append("Factor identity: MODERATE (0.4-0.6 Jaccard)")
        else:
            verdicts.append("Factor identity: LOW (<0.4 Jaccard) — consider reducing temperature")
        if d >= 0.85:
            verdicts.append("Direction consensus: STRONG (≥85%)")
        elif d >= 0.70:
            verdicts.append("Direction consensus: MODERATE (70-85%)")
        else:
            verdicts.append("Direction consensus: WEAK (<70%) — factors direction is noisy")
        if m <= 0.3:
            verdicts.append("Magnitude stability: GOOD (CV≤0.3)")
        elif m <= 0.6:
            verdicts.append("Magnitude stability: MODERATE (CV 0.3-0.6)")
        else:
            verdicts.append("Magnitude stability: POOR (CV>0.6) — magnitude estimates vary widely")

        for v in verdicts:
            print(f"  → {v}")
        print(f"{'─' * 60}\n")

    # ── embedding matrix ────────────────────────────────────────────

    @staticmethod
    def build_factor_embedding(
        analyses: List[WindowAnalysis],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build the factor embedding matrix from window analyses.

        Returns
        -------
        embedding : ndarray, shape (n_windows, n_factors)
            Signed magnitude (direction × magnitude) per (window, factor).
        factor_names : list[str]
        window_labels : list[str]
            ``"start|end"`` label per window.
        """
        all_names: set = set()
        for a in analyses:
            for f in a.factors:
                all_names.add(f.name)
        factor_names = sorted(all_names)
        name_to_idx = {n: i for i, n in enumerate(factor_names)}

        n_windows = len(analyses)
        n_factors = len(factor_names)
        embedding = np.zeros((n_windows, n_factors), dtype=np.float32)
        window_labels: List[str] = []

        for w_idx, analysis in enumerate(analyses):
            window_labels.append(
                f"{analysis.start_date}|{analysis.end_date}"
            )
            for f in analysis.factors:
                f_idx = name_to_idx[f.name]
                embedding[w_idx, f_idx] = f.direction * f.magnitude

        return embedding, factor_names, window_labels

    # ── presence matrix (impact-weighted) ─────────────────────────────

    @staticmethod
    def build_presence_matrix(
        analyses: List[WindowAnalysis],
        df: pd.DataFrame,
        threshold: float = 0.0,
    ) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
        """
        Build the impact-weighted factor-presence matrix F ∈ [0,1]^(T×K).

        DISC-2: Instead of collapsing magnitude to binary {0,1}, preserves
        the LLM-assigned magnitude so that rare-but-intense events (e.g.
        COVID, flash crashes) are not treated identically to frequent-but-
        mild factors (e.g. market_noise).  The FIN/TFT architecture
        already multiplies ``factor_presence * embeddings``, so continuous
        values in [0,1] propagate naturally as intensity weights.

        Expands window-level factors to daily resolution by assigning
        each day within a window the factor's magnitude (if the factor
        was identified for that window with magnitude > ``threshold``).
        When windows overlap, the maximum magnitude is kept.

        Parameters
        ----------
        analyses : list[WindowAnalysis]
            Output of ``extract_all_windows``.
        df : pd.DataFrame
            Price DataFrame with a ``date`` column.
        threshold : float
            Minimum magnitude to consider a factor present.

        Returns
        -------
        presence : ndarray, shape (T, K)
            Impact-weighted presence matrix in [0.0, 1.0].
        factor_names : list[str]
        dates : pd.DatetimeIndex
        """
        all_names: set = set()
        for a in analyses:
            for f in a.factors:
                if f.magnitude > threshold:
                    all_names.add(f.name)

        factor_names = sorted(all_names)
        name_to_idx = {n: i for i, n in enumerate(factor_names)}

        dates = pd.to_datetime(sorted(df["date"].unique()))
        T = len(dates)
        K = len(factor_names)
        presence = np.zeros((T, K), dtype=np.float32)

        date_to_row = {d: i for i, d in enumerate(dates)}

        for analysis in analyses:
            start = pd.Timestamp(analysis.start_date)
            end = pd.Timestamp(analysis.end_date)
            for f in analysis.factors:
                if f.magnitude <= threshold:
                    continue
                if f.name not in name_to_idx:
                    continue
                col = name_to_idx[f.name]
                for d in dates:
                    if start <= d <= end:
                        row = date_to_row[d]
                        # DISC-2: keep the maximum magnitude across
                        # overlapping windows (not binary 1.0).
                        presence[row, col] = max(
                            presence[row, col], f.magnitude
                        )

        return presence, factor_names, dates

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _infer_window_type(start: str, end: str) -> str:
        delta = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if delta <= 1:
            return "daily"
        elif delta <= 35:
            return "monthly"
        else:
            return "quarterly"
