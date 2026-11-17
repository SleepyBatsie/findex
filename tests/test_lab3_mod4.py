"""Tests for @timed decorator, QueryCache, snippets, and Precision@5."""

import logging

import pytest

from findex.corpus import Document
from findex.index import Index
from findex.ranking import BM25Scorer, TfIdfScorer
from findex.search import search_ranked
from findex.utils import QueryCache, calculate_precision_at_k, generate_snippet, timed


def test_timed_decorator(caplog: pytest.LogCaptureFixture) -> None:
    @timed
    def mock_build_index(docs_count: int) -> int:
        """Mock build function."""
        return docs_count * 2

    assert mock_build_index.__name__ == "mock_build_index"
    assert mock_build_index.__doc__ == "Mock build function."

    with caplog.at_level(logging.DEBUG, logger="findex.utils"):
        result = mock_build_index(5)

    assert result == 10
    assert "mock_build_index executed in" in caplog.text


def test_lru_cache_query_hits(caplog: pytest.LogCaptureFixture) -> None:
    docs = [
        Document("doc1", None, "python async await programming"),
        Document("doc2", None, "python concurrency event loop"),
    ]
    idx = Index()
    idx.build(docs)

    cache = QueryCache(maxsize=10)

    with caplog.at_level(logging.DEBUG, logger="findex.utils"):
        res1 = cache.get_or_execute("python", idx, search_ranked)
        assert "doc1" in res1
        assert "Cache miss for query 'python'" in caplog.text

        caplog.clear()
        res2 = cache.get_or_execute("python", idx, search_ranked)
        assert res1 == res2
        assert "Cache hit for query 'python'" in caplog.text


def test_snippet_generation_and_highlighting() -> None:
    text = (
        "In computer programming, asynchronous I/O is a form of input/output processing "
        "that allows other processing to continue before the transmission has finished. "
        "Python supports async and await keywords for structured asynchronous code."
    )
    snippet = generate_snippet(text, ["python", "async"], window=40)
    assert "**Python**" in snippet or "**python**" in snippet or "**async**" in snippet
    assert len(snippet) <= 180


def test_precision_at_5_comparison() -> None:
    corpus = [
        Document("doc1", None, "python tutorial for beginners learning code"),
        Document("doc2", None, "advanced python async await event loop performance"),
        Document("doc3", None, "java spring boot microservices architecture"),
        Document("doc4", None, "python data science pandas numpy machine learning"),
        Document("doc5", None, "rust programming language systems memory safety"),
        Document("doc6", None, "python web development django flask api"),
        Document("doc7", None, "javascript react nodejs frontend web development"),
        Document("doc8", None, "python algorithms data structures search indexing"),
        Document("doc9", None, "go language concurrency goroutines channels"),
        Document("doc10", None, "python testing pytest unittest code quality"),
    ]

    idx = Index()
    idx.build(corpus)

    annotated_queries = [
        ("python", {"doc1", "doc2", "doc4", "doc6", "doc8", "doc10"}),
        ("web development", {"doc6", "doc7"}),
        ("async concurrency", {"doc2", "doc9"}),
        ("machine learning data", {"doc4"}),
        ("testing code", {"doc10", "doc1"}),
        ("java spring", {"doc3"}),
        ("memory safety rust", {"doc5"}),
        ("react frontend", {"doc7"}),
        ("search indexing algorithms", {"doc8"}),
        ("microservices api", {"doc3", "doc6"}),
    ]

    p5_tfidf: list[float] = []
    p5_bm25: list[float] = []

    for query, relevant in annotated_queries:
        res_tfidf = [r.doc_id for r in search_ranked(idx, query, scorer=TfIdfScorer(), top_k=5)]
        res_bm25 = [r.doc_id for r in search_ranked(idx, query, scorer=BM25Scorer(), top_k=5)]
        p5_tfidf.append(calculate_precision_at_k(res_tfidf, relevant, k=5))
        p5_bm25.append(calculate_precision_at_k(res_bm25, relevant, k=5))

    assert (sum(p5_bm25) / len(p5_bm25)) >= (sum(p5_tfidf) / len(p5_tfidf))