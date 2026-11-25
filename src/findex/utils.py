"""Timing decorator, LRU query cache, snippet generator, and IR evaluation metrics."""

import logging
import re
import time
from collections.abc import Callable, Sequence
from functools import lru_cache, wraps
from typing import ParamSpec, TypeVar

from findex.index import InvertedIndex
from findex.search import SearchResult

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def timed(func: Callable[P, R]) -> Callable[P, R]:
    """Measure function execution time and log it at DEBUG level."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        logger.debug("%s executed in %.6f s", func.__name__, elapsed)
        return result

    return wrapper


SearchFunc = Callable[[InvertedIndex, str], Sequence[SearchResult] | Sequence[str]]


class QueryCache:
    """LRU query cache wrapping search functions with structured logging."""

    def __init__(self, maxsize: int = 128) -> None:
        self._maxsize = maxsize
        self._index_registry: dict[int, InvertedIndex] = {}
        self._fn_registry: dict[str, SearchFunc] = {}

        @lru_cache(maxsize=maxsize)
        def _cached_search(query: str, index_id: int, fn_name: str) -> tuple[str, ...]:
            idx = self._index_registry.get(index_id)
            fn = self._fn_registry.get(fn_name)
            if idx is None or fn is None:
                return ()
            results = fn(idx, query)
            doc_ids: list[str] = []
            for item in results:
                if isinstance(item, SearchResult):
                    doc_ids.append(item.doc_id)
                else:
                    doc_ids.append(item)
            return tuple(doc_ids)

        self._cached_search = _cached_search

    def get_or_execute(
        self,
        query: str,
        index: InvertedIndex,
        search_fn: SearchFunc,
    ) -> list[str]:
        """Return cached doc_ids or execute search_fn and populate cache."""
        index_id = id(index)
        fn_name = search_fn.__name__

        self._index_registry[index_id] = index
        self._fn_registry[fn_name] = search_fn

        info_before = self._cached_search.cache_info()
        res_tuple = self._cached_search(query, index_id, fn_name)
        info_after = self._cached_search.cache_info()

        if info_after.hits > info_before.hits:
            logger.debug("Cache hit for query %r", query)
        else:
            logger.debug("Cache miss for query %r", query)

        return list(res_tuple)


def generate_snippet(text: str, query_terms: list[str], window: int = 80) -> str:
    """Generate a context window snippet around the earliest matching term and highlight terms."""
    if not text or not query_terms:
        return text[: window * 2]

    text_lower = text.lower()
    clean_terms = [term.lower() for term in query_terms if term]

    first_pos = -1
    for term in clean_terms:
        pos = text_lower.find(term)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    if first_pos == -1:
        start = 0
        end = min(len(text), window * 2)
    else:
        start = max(0, first_pos - window)
        end = min(len(text), first_pos + window)

    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."

    for term in clean_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        snippet = pattern.sub(lambda m: f"**{m.group(0)}**", snippet)

    return snippet


def calculate_precision_at_k(
    actual_doc_ids: list[str],
    relevant_doc_ids: set[str],
    k: int = 5,
) -> float:
    """Compute Precision@k metric."""
    if k <= 0:
        return 0.0
    top_k = actual_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_retrieved / k
