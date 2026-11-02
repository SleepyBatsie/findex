# tests/test_lab3_mod4.py
import pytest
from findex.corpus import Document
from findex.index import Index
from findex.ranking import TfIdfScorer, BM25Scorer
from findex.search import search_ranked
from findex.utils import timed, QueryCache, generate_snippet, calculate_precision_at_k


def test_timed_decorator(capsys):
    """Перевірка роботи декоратора @timed та збереження wraps."""
    @timed
    def mock_build_index(docs_count: int):
        """Mock build function."""
        return docs_count * 2

    assert mock_build_index.__name__ == "mock_build_index"
    assert mock_build_index.__doc__ == "Mock build function."

    result = mock_build_index(5)
    assert result == 10

    captured = capsys.readouterr()
    assert "[TIMED] mock_build_index executed in" in captured.out


def test_lru_cache_query_hits(capsys):
    """Перевірка кешування повторних запитів через QueryCache та логування hit."""
    docs = [
        Document("doc1", None, "python async await programming"),
        Document("doc2", None, "python concurrency event loop"),
    ]
    idx = Index()
    idx.build(docs)

    cache = QueryCache(maxsize=10)

    # Перший запит (Cache Miss)
    res1 = cache.get_or_execute("python", idx, search_ranked)
    assert "doc1" in res1

    captured = capsys.readouterr()
    assert "[CACHE HIT]" not in captured.out

    # Повторний запит (Cache Hit)
    res2 = cache.get_or_execute("python", idx, search_ranked)
    assert res1 == res2

    captured_hit = capsys.readouterr()
    assert "[CACHE HIT] Query 'python' served from LRU cache" in captured_hit.out


def test_snippet_generation_and_highlighting():
    """Перевірка формування вікна ±80 символів та підсвічування термінів."""
    text = (
        "In computer programming, asynchronous I/O is a form of input/output processing "
        "that allows other processing to continue before the transmission has finished. "
        "Python supports async and await keywords for structured asynchronous code."
    )
    terms = ["python", "async"]

    snippet = generate_snippet(text, terms, window=40)

    assert "**Python**" in snippet or "**python**" in snippet or "**async**" in snippet
    assert len(snippet) <= 40 * 2 + 100


def test_precision_at_5_comparison():
    """
    Оцінка якості 10 тестових запитів з розміткою релевантності:
    Порівняння Precision@5 для TF-IDF та BM25.
    """
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

    p5_tfidf_scores = []
    p5_bm25_scores = []

    for query, relevant in annotated_queries:
        res_tfidf = [r.doc_id for r in search_ranked(idx, query, scorer=TfIdfScorer(), top_k=5)]
        res_bm25 = [r.doc_id for r in search_ranked(idx, query, scorer=BM25Scorer(), top_k=5)]

        p5_tfidf_scores.append(calculate_precision_at_k(res_tfidf, relevant, k=5))
        p5_bm25_scores.append(calculate_precision_at_k(res_bm25, relevant, k=5))

    avg_p5_tfidf = sum(p5_tfidf_scores) / len(p5_tfidf_scores)
    avg_p5_bm25 = sum(p5_bm25_scores) / len(p5_bm25_scores)

    assert avg_p5_bm25 >= avg_p5_tfidf
    assert avg_p5_bm25 > 0.0