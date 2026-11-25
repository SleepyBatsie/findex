"""findex: Full-text search engine package."""

from findex.corpus import Document, iter_corpus, iter_documents, load_corpus
from findex.crawler import (
    DEFAULT_USER_AGENT,
    CrawledPage,
    CrawlStats,
    FetchResult,
    RobotsCache,
    canonicalize_url,
    crawl,
    crawl_to_jsonl,
    extract_text_and_links,
    fetch,
)
from findex.index import (
    BuildMetrics,
    DocMeta,
    ExecutorName,
    Index,
    InvertedIndex,
    Posting,
    build_index,
    build_partial,
    chunk_paths,
    merge,
)
from findex.parser import parse_query
from findex.ranking import BM25Scorer, Scorer, ScorerName, TfIdfScorer, create_scorer
from findex.search import EngineName, SearchResult, search_ast, search_boolean, search_ranked
from findex.stats import IndexStats, compute_stats
from findex.storage import load_index, open_index, save_index
from findex.tokenize import iter_tokens, tokenize

__version__ = "0.6.0"

__all__ = [
    "BM25Scorer",
    "BuildMetrics",
    "CrawlStats",
    "CrawledPage",
    "DEFAULT_USER_AGENT",
    "DocMeta",
    "Document",
    "EngineName",
    "ExecutorName",
    "FetchResult",
    "Index",
    "IndexStats",
    "InvertedIndex",
    "Posting",
    "RobotsCache",
    "Scorer",
    "ScorerName",
    "SearchResult",
    "TfIdfScorer",
    "__version__",
    "build_index",
    "build_partial",
    "canonicalize_url",
    "chunk_paths",
    "compute_stats",
    "crawl",
    "crawl_to_jsonl",
    "create_scorer",
    "extract_text_and_links",
    "fetch",
    "iter_corpus",
    "iter_documents",
    "iter_tokens",
    "load_corpus",
    "load_index",
    "merge",
    "open_index",
    "parse_query",
    "save_index",
    "search_ast",
    "search_boolean",
    "search_ranked",
    "tokenize",
]
