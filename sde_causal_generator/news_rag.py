# -*- coding: utf-8 -*-
"""
Temporal News RAG for LLM Factor Extraction.

Provides time-bounded retrieval of news articles to ground LLM
causal factor extraction in contemporaneous information, mitigating
hindsight bias and narrative fallacy.

Supported vector-store backend: ChromaDB (local, persistent).

News sources (ingestors):
    - Wikipedia Current Events  (curated daily world-event summaries)
    - FRED                      (macroeconomic indicator releases)
    - SEC EDGAR                 (8-K / 10-K / 10-Q filings)
    - Custom CSV / JSON         (user-provided corpora)
"""

from __future__ import annotations

import io
import json
import os
import re
import time as _time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .data_structures import NewsArticle


# ══════════════════════════════════════════════════════════════════════
# Vector Store
# ══════════════════════════════════════════════════════════════════════

class TemporalNewsStore:
    """
    ChromaDB-backed vector store with **strict temporal filtering**.

    Design principle: retrieval *never* returns articles published
    after the query ``end_date``, preventing information leakage.
    """

    def __init__(
        self,
        persist_dir: str = "cache/news_vectorstore",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        os.makedirs(persist_dir, exist_ok=True)

        self._embedder = None
        self._collection = None
        self._init_backend()

    # ── backend ─────────────────────────────────────────────────────

    def _init_backend(self):
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                name="news_temporal",
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            raise ImportError(
                "chromadb is required for the news RAG store. "
                "Install it with:  pip install chromadb"
            )

    def _get_embedder(self):
        if self._embedder is None:
            try:
                import logging
                from sentence_transformers import SentenceTransformer

                # Suppress noisy "UNEXPECTED key" warnings from the
                # BertModel weight-loading report (position_ids).
                _st_logger = logging.getLogger("sentence_transformers.SentenceTransformer")
                _prev = _st_logger.level
                _st_logger.setLevel(logging.ERROR)

                self._embedder = SentenceTransformer(
                    self.embedding_model_name
                )

                _st_logger.setLevel(_prev)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with:  pip install sentence-transformers"
                )
        return self._embedder

    # ── ingest ──────────────────────────────────────────────────────

    def ingest_articles(
        self, articles: List[NewsArticle], batch_size: int = 100
    ):
        """Ingest news articles into the vector store."""
        if not articles:
            return

        print(f"    Ingesting {len(articles)} articles …")
        embedder = self._get_embedder()

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            texts = [f"{a.title}. {a.content[:500]}" for a in batch]
            ids = [a.article_id for a in batch]
            metadatas = [
                {
                    "published_date": a.published_date,
                    "published_date_epoch": int(
                        pd.Timestamp(a.published_date).timestamp()
                    ),
                    "source": a.source,
                    "tickers": ",".join(a.tickers_mentioned),
                    "categories": ",".join(a.categories),
                    "title": a.title,
                }
                for a in batch
            ]

            embeddings = embedder.encode(texts).tolist()
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        print(f"    ✓ Ingested {len(articles)} articles")

    # ── retrieve ────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        ticker: str,
        start_date: str,
        end_date: str,
        max_results: int = 20,
        include_general: bool = True,
    ) -> List[Dict]:
        """
        Retrieve news with **strict temporal bounds**.

        Only returns articles published on or before ``end_date``.
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        embedder = self._get_embedder()
        query_embedding = embedder.encode([query]).tolist()

        # ChromaDB $gte/$lte require numeric operands → use epoch ints
        start_epoch = int(pd.Timestamp(start_date).timestamp())
        end_epoch = int(pd.Timestamp(end_date).timestamp())

        where_filter = {
            "$and": [
                {"published_date_epoch": {"$gte": start_epoch}},
                {"published_date_epoch": {"$lte": end_epoch}},
            ]
        }

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=max_results * 3,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        articles: List[Dict] = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                tickers_in_article = meta.get("tickers", "").split(",")
                is_ticker_relevant = ticker in tickers_in_article
                is_general = not tickers_in_article or tickers_in_article == [
                    ""
                ]

                if is_ticker_relevant or (include_general and is_general):
                    articles.append(
                        {
                            "title": meta.get("title", ""),
                            "content": doc,
                            "date": meta["published_date"],
                            "source": meta.get("source", ""),
                            "relevance": 1.0 - dist,
                            "ticker_specific": is_ticker_relevant,
                        }
                    )

        articles.sort(
            key=lambda x: (x["ticker_specific"], x["relevance"]),
            reverse=True,
        )
        return articles[:max_results]

    # ── stats ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        if self._collection is not None:
            return {
                "backend": "chroma",
                "total_articles": self._collection.count(),
            }
        return {"backend": "chroma", "total_articles": 0}


# ══════════════════════════════════════════════════════════════════════
# News Source Ingestors
# ══════════════════════════════════════════════════════════════════════


class WikipediaCurrentEventsIngestor:
    """
    Ingest Wikipedia's 'Portal:Current events' entries.

    Human-curated daily summaries of world events — high quality,
    temporally precise, and freely available (2004–present).

    Uses the MediaWiki batch API (``action=query``) to fetch up to
    50 day-pages per request, avoiding 429 rate-limit errors.
    """

    # Polite delay (seconds) between Wikipedia API requests.
    _REQUEST_DELAY: float = 1.0
    _API_URL = "https://en.wikipedia.org/w/api.php"
    _USER_AGENT = (
        "emergent-finrl-research/0.1 "
        "(https://github.com/emergent-finrl; research-only)"
    )

    @staticmethod
    def _api_get(params: dict, max_retries: int = 3) -> dict:
        """Call the MediaWiki API with back-off on 429."""
        params.setdefault("format", "json")
        qs = urllib.parse.urlencode(params, doseq=True)
        url = f"{WikipediaCurrentEventsIngestor._API_URL}?{qs}"

        _time.sleep(WikipediaCurrentEventsIngestor._REQUEST_DELAY)
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": WikipediaCurrentEventsIngestor._USER_AGENT,
                    },
                )
                resp = urllib.request.urlopen(req, timeout=30)
                return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as he:
                if he.code == 429:
                    wait = 2 ** (attempt + 1)
                    _time.sleep(wait)
                else:
                    raise
        return {}

    @staticmethod
    def _parse_day_wikitext(wikitext: str, iso_date: str) -> List[NewsArticle]:
        """Extract event articles from a single day's raw wikitext."""
        articles: List[NewsArticle] = []

        # Each top-level bullet (*) or sub-bullet (**) is an event
        for line in wikitext.split("\n"):
            line = line.strip()
            if not line.startswith("*"):
                continue

            # Strip leading *, **, etc.
            clean = re.sub(r"^\*+\s*", "", line)
            # Remove wikilinks: [[Target|Display]] → Display, [[Target]] → Target
            clean = re.sub(
                r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", clean
            )
            # Remove bold/italic markup
            clean = re.sub(r"'{2,}", "", clean)
            # Remove external links: [https://... (Source)] → (Source)
            clean = re.sub(r"\[https?://\S+\s*", "", clean)
            clean = clean.strip().rstrip("]")

            if len(clean) > 25:
                articles.append(
                    NewsArticle(
                        title=f"World Event: {clean[:80]}",
                        content=clean,
                        source="wikipedia_current_events",
                        published_date=iso_date,
                        categories=["geopolitical", "world_events"],
                    )
                )
        return articles

    @staticmethod
    def fetch_month(
        year: int, month: int, max_retries: int = 3
    ) -> List[NewsArticle]:
        """Fetch all days in a month using the MediaWiki batch API."""
        import calendar

        n_days = calendar.monthrange(year, month)[1]
        # Build page titles for each day
        all_titles = [
            f"Portal:Current events/{year}_{datetime(year, month, 1).strftime('%B')}_{d}"
            for d in range(1, n_days + 1)
        ]

        articles: List[NewsArticle] = []

        # MediaWiki API allows up to 50 titles per request
        for batch_start in range(0, len(all_titles), 50):
            batch_titles = all_titles[batch_start : batch_start + 50]
            titles_str = "|".join(batch_titles)

            data = WikipediaCurrentEventsIngestor._api_get(
                {
                    "action": "query",
                    "titles": titles_str,
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                },
                max_retries=max_retries,
            )

            pages = data.get("query", {}).get("pages", {})
            for _pid, page in pages.items():
                title = page.get("title", "")
                # Skip missing pages (e.g., future dates)
                if "missing" in page:
                    continue

                revisions = page.get("revisions", [])
                if not revisions:
                    continue
                wikitext = (
                    revisions[0]
                    .get("slots", {})
                    .get("main", {})
                    .get("*", "")
                )
                if not wikitext:
                    continue

                # Extract date from title: "Portal:Current events/2020 March 11"
                match = re.search(
                    r"(\d{4})[\s_](\w+)[\s_](\d+)$", title
                )
                if not match:
                    continue
                try:
                    iso_date = datetime(
                        int(match.group(1)),
                        datetime.strptime(match.group(2), "%B").month,
                        int(match.group(3)),
                    ).strftime("%Y-%m-%d")
                except (ValueError, KeyError):
                    continue

                day_articles = WikipediaCurrentEventsIngestor._parse_day_wikitext(
                    wikitext, iso_date
                )
                articles.extend(day_articles)

        return articles

    @staticmethod
    def fetch_range(start_date: str, end_date: str) -> List[NewsArticle]:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        all_articles: List[NewsArticle] = []
        current = start.replace(day=1)

        total_months = (
            (end.year - start.year) * 12 + end.month - start.month + 1
        )
        fetched = 0

        while current <= end:
            fetched += 1
            month_label = current.strftime("%B %Y")
            monthly = WikipediaCurrentEventsIngestor.fetch_month(
                current.year, current.month
            )
            filtered = [
                a
                for a in monthly
                if start_date <= a.published_date <= end_date
            ]
            all_articles.extend(filtered)

            if fetched % 12 == 0 or fetched == total_months:
                print(
                    f"      Progress: {fetched}/{total_months} months, "
                    f"{len(all_articles)} events so far"
                )

            current += pd.DateOffset(months=1)

        return all_articles


class FREDIngestor:
    """
    Ingest macroeconomic data releases from FRED as news events.

    Captures release dates of key indicators with their values.
    Requires the ``fredapi`` package and ``FRED_API_KEY`` env var.
    """

    KEY_SERIES = {
        "FEDFUNDS": "Federal Funds Effective Rate",
        "CPIAUCSL": "Consumer Price Index",
        "UNRATE": "Unemployment Rate",
        "GDP": "Gross Domestic Product",
        "INDPRO": "Industrial Production Index",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "VIXCLS": "VIX Volatility Index",
        "DGS10": "10-Year Treasury Rate",
    }

    @staticmethod
    def fetch_releases(
        start_date: str,
        end_date: str,
        fred_api_key: Optional[str] = None,
    ) -> List[NewsArticle]:
        api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        if not api_key:
            print(
                "      WARNING: No FRED_API_KEY set. Skipping FRED ingest."
            )
            return []

        articles: List[NewsArticle] = []
        try:
            from fredapi import Fred  # type: ignore[import-untyped]

            fred = Fred(api_key=api_key)

            for series_id, description in FREDIngestor.KEY_SERIES.items():
                try:
                    data = fred.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date,
                    )
                    for date, value in data.items():
                        if pd.notna(value):
                            articles.append(
                                NewsArticle(
                                    title=f"FRED Release: {description}",
                                    content=(
                                        f"{description} ({series_id}) "
                                        f"reported at {value:.4f} "
                                        f"on {date.strftime('%Y-%m-%d')}."
                                    ),
                                    source="fred",
                                    published_date=date.strftime("%Y-%m-%d"),
                                    categories=["macroeconomic"],
                                )
                            )
                except Exception as e:
                    print(f"      FRED series {series_id} failed: {e}")

        except ImportError:
            print("      Install fredapi: pip install fredapi")

        return articles


class SECEdgarIngestor:
    """
    Ingest SEC EDGAR filings (10-K, 10-Q, 8-K) as news events.

    8-K filings capture material events (earnings, M&A, management
    changes) in near-real-time.
    """

    @staticmethod
    def fetch_filings(
        ticker: str,
        start_date: str,
        end_date: str,
        max_filings: int = 50,
    ) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{ticker}%22&dateRange=custom"
            f"&startdt={start_date}&enddt={end_date}"
            f"&forms=8-K,10-K,10-Q"
            f"&from=0&size={max_filings}"
        )
        headers = {"User-Agent": "emergent-finrl research@example.com"}

        try:
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req, timeout=30)
            data = json.loads(response.read().decode())

            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                filing_date = source.get("file_date", "")
                form_type = source.get("form_type", "")
                display_names = source.get("display_names", [])
                entity = display_names[0] if display_names else ticker

                articles.append(
                    NewsArticle(
                        title=f"SEC {form_type}: {entity}",
                        content=(
                            f"{entity} filed {form_type} with the SEC "
                            f"on {filing_date}."
                        ),
                        source="sec_edgar",
                        published_date=filing_date,
                        tickers_mentioned=[ticker],
                        categories=["company_specific", "regulatory"],
                    )
                )
        except Exception as e:
            print(f"      SEC EDGAR fetch failed for {ticker}: {e}")

        return articles


# ══════════════════════════════════════════════════════════════════════
# RAG Manager (unified interface)
# ══════════════════════════════════════════════════════════════════════


class NewsRAGManager:
    """
    Manages news ingestion from multiple sources and provides a
    unified retrieval interface with strict temporal bounds.

    Usage::

        rag = NewsRAGManager(persist_dir="cache/news")
        rag.build_corpus(ticker="AAPL",
                         start_date="2005-01-01",
                         end_date="2006-12-31")
        context = rag.get_context(ticker="AAPL",
                                  start_date="2006-03-01",
                                  end_date="2006-03-31")
    """

    def __init__(
        self,
        persist_dir: str = "cache/news_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.store = TemporalNewsStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )
        self.corpus_built = False

    # ── build ───────────────────────────────────────────────────────

    def build_corpus(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        sources: Optional[List[str]] = None,
        fred_api_key: Optional[str] = None,
        cache_mode: str = "reuse",
    ):
        """Build the news corpus from multiple sources.

        Default sources: ``["wikipedia", "fred", "sec_edgar"]``.

        Parameters
        ----------
        cache_mode : str
            ``"reuse"``  – load articles from disk cache if available,
            only download what is missing.
            ``"refresh"`` – delete existing cache and re-download
            everything from scratch.
        """
        if sources is None:
            sources = ["wikipedia", "fred", "sec_edgar"]

        cache_dir = Path(self.store.persist_dir) / "article_cache"

        # ── refresh mode: wipe caches ──────────────────────────────
        if cache_mode == "refresh" and cache_dir.exists():
            print("    Cache mode = refresh → deleting old article cache …")
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_articles: List[NewsArticle] = []
        print(
            f"  > Building news corpus for {ticker} "
            f"({start_date} to {end_date}) …"
        )

        for src in sources:
            cache_file = (
                cache_dir
                / f"{src}_{ticker}_{start_date}_{end_date}.json"
            )

            # ── try loading from cache ─────────────────────────────
            if cache_mode == "reuse" and cache_file.exists():
                print(f"    Loading cached {src} articles from {cache_file.name} …")
                with open(cache_file, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                arts = [NewsArticle.from_dict(d) for d in cached]
                print(f"      → {len(arts)} articles (cached)")
                all_articles.extend(arts)
                continue

            # ── download from source ───────────────────────────────
            arts: List[NewsArticle] = []
            if src == "wikipedia":
                print("    Fetching Wikipedia Current Events …")
                arts = WikipediaCurrentEventsIngestor.fetch_range(
                    start_date, end_date
                )
                print(f"      → {len(arts)} events")
            elif src == "fred":
                print("    Fetching FRED releases …")
                arts = FREDIngestor.fetch_releases(
                    start_date, end_date, fred_api_key
                )
                print(f"      → {len(arts)} releases")
            elif src == "sec_edgar":
                print("    Fetching SEC EDGAR filings …")
                arts = SECEdgarIngestor.fetch_filings(
                    ticker, start_date, end_date
                )
                print(f"      → {len(arts)} filings")
            else:
                print(f"    ⚠ Unknown source '{src}', skipping.")
                continue

            # ── save to cache ──────────────────────────────────────
            if arts:
                with open(cache_file, "w", encoding="utf-8") as fh:
                    json.dump(
                        [a.to_dict() for a in arts], fh,
                        ensure_ascii=False,
                    )
                print(f"      Cached → {cache_file.name}")

            all_articles.extend(arts)

        if all_articles:
            self.store.ingest_articles(all_articles)
            self.corpus_built = True

        stats = self.store.get_stats()
        print(f"    ✓ Corpus ready: {stats['total_articles']} total articles")

    # ── retrieve ────────────────────────────────────────────────────

    def get_context(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
        query_template: Optional[str] = None,
        content_limit: int = 500,
    ) -> str:
        """Retrieve and format news context for a time window.

        Returns a formatted string ready for LLM prompt injection,
        containing **only** articles published within
        ``[start_date, end_date]``.

        Parameters
        ----------
        content_limit : int
            Maximum characters per article content in the output.
        """
        if query_template is None:
            query_template = (
                f"Market moving events affecting {ticker} stock price "
                f"macroeconomic policy interest rates earnings"
            )

        articles = self.store.retrieve(
            query=query_template,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_results=max_articles,
        )

        if not articles:
            return (
                f"[No news articles found for {ticker} "
                f"between {start_date} and {end_date}]"
            )

        lines = [
            f"=== NEWS CONTEXT FOR {ticker} "
            f"({start_date} to {end_date}) ===",
            f"Total articles retrieved: {len(articles)}",
            "",
        ]

        specific = [a for a in articles if a.get("ticker_specific")]
        general = [a for a in articles if not a.get("ticker_specific")]

        if specific:
            lines.append(f"── {ticker}-SPECIFIC NEWS ──")
            for i, a in enumerate(specific, 1):
                lines.append(
                    f"[{i}] ({a['date']}) [{a['source']}] {a['title']}"
                )
                lines.append(f"    {a['content'][:content_limit]}")
                lines.append("")

        if general:
            lines.append("── MACRO/GEOPOLITICAL CONTEXT ──")
            for i, a in enumerate(general, 1):
                lines.append(
                    f"[{i}] ({a['date']}) [{a['source']}] {a['title']}"
                )
                lines.append(f"    {a['content'][:content_limit]}")
                lines.append("")

        lines.append("=== END NEWS CONTEXT ===")
        return "\n".join(lines)

    # ── custom corpus ───────────────────────────────────────────────

    def ingest_custom_csv(
        self,
        csv_path: str,
        date_col: str = "date",
        title_col: str = "title",
        content_col: str = "content",
        ticker_col: Optional[str] = "ticker",
        source_name: str = "custom",
    ):
        """Ingest a custom CSV of news articles."""
        df = pd.read_csv(csv_path)
        articles: List[NewsArticle] = []
        for _, row in df.iterrows():
            tickers: List[str] = []
            if ticker_col and ticker_col in df.columns:
                tickers = [str(row[ticker_col])]

            articles.append(
                NewsArticle(
                    title=str(row[title_col]),
                    content=str(row.get(content_col, "")),
                    source=source_name,
                    published_date=str(row[date_col]),
                    tickers_mentioned=tickers,
                )
            )

        self.store.ingest_articles(articles)
        print(f"    ✓ Ingested {len(articles)} articles from {csv_path}")
