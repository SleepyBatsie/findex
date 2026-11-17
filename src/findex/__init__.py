"""findex: Full-text search engine package."""

from findex.corpus import Document, iter_corpus, load_corpus
from findex.index import DocMeta, Index, InvertedIndex, Posting
from findex.parser import parse_query
from findex.ranking import BM25Scorer, Scorer, ScorerName, TfIdfScorer, create_scorer
from findex.search import EngineName, SearchResult, search_ast, search_boolean, search_ranked
from findex.stats import IndexStats, compute_stats
from findex.storage import load_index, open_index, save_index
from findex.tokenize import iter_tokens, tokenize

__version__ = "0.4.0"

__all__ = [
    "BM25Scorer",
    "DocMeta",
    "Document",
    "EngineName",
    "Index",
    "IndexStats",
    "InvertedIndex",
    "Posting",
    "Scorer",
    "ScorerName",
    "SearchResult",
    "TfIdfScorer",
    "__version__",
    "compute_stats",
    "create_scorer",
    "iter_corpus",
    "iter_tokens",
    "load_corpus",
    "load_index",
    "open_index",
    "parse_query",
    "save_index",
    "search_ast",
    "search_boolean",
    "search_ranked",
    "tokenize",
]