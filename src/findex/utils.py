# src/findex/utils.py
import time
from functools import wraps, lru_cache
import re
from typing import Callable, Any, List, Set, Tuple, Dict


def timed(func: Callable) -> Callable:
    """
    Декоратор для вимірювання часу виконання функцій (build, load, search).
    Використовує functools.wraps для збереження метаданих оригінальної функції.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start_time
        print(f"[TIMED] {func.__name__} executed in {elapsed:.6f} s")
        return result
    return wrapper


class QueryCache:
    """Кеш для запитів на основі @lru_cache з логуванням cache hit."""

    def __init__(self, maxsize: int = 128):
        self._maxsize = maxsize
        self._index_registry: Dict[int, Any] = {}
        self._fn_registry: Dict[str, Callable] = {}

        @lru_cache(maxsize=maxsize)
        def _cached_search(query: str, index_id: int, fn_name: str) -> Tuple[str, ...]:
            idx = self._index_registry.get(index_id)
            fn = self._fn_registry.get(fn_name)
            if idx is None or fn is None:
                return ()
            results = fn(idx, query)
            if results and hasattr(results[0], "doc_id"):
                return tuple(r.doc_id for r in results)
            return tuple(results)

        self._cached_search = _cached_search

    def get_or_execute(self, query: str, index: Any, search_fn: Callable) -> List[str]:
        index_id = id(index)
        search_fn_name = search_fn.__name__

        self._index_registry[index_id] = index
        self._fn_registry[search_fn_name] = search_fn

        info_before = self._cached_search.cache_info()
        res_tuple = self._cached_search(query, index_id, search_fn_name)
        info_after = self._cached_search.cache_info()

        if info_after.hits > info_before.hits:
            print(f"[CACHE HIT] Query '{query}' served from LRU cache")

        return list(res_tuple)


def generate_snippet(text: str, query_terms: List[str], window: int = 80) -> str:
    """
    Генерує сніпет навколо найкращого входження слів із запиту (вікно ±80 символів)
    та підсвічує знайдені терми.
    """
    if not text or not query_terms:
        return text[: window * 2]

    text_lower = text.lower()
    clean_terms = [t.lower() for t in query_terms if t]

    first_pos = -1
    for term in clean_terms:
        pos = text_lower.find(term)
        if pos != -1:
            if first_pos == -1 or pos < first_pos:
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


def calculate_precision_at_k(actual_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 5) -> float:
    """Обчислює метрику Precision@k."""
    top_k = actual_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_retrieved / k