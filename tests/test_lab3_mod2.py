"""Tests for ranking scorers (TF-IDF, BM25) and sanity checks."""

from findex.corpus import Document
from findex.index import Index
from findex.ranking import BM25Scorer, Scorer, TfIdfScorer
from findex.search import search_ranked


def test_scorer_protocol() -> None:
    """Перевірка дотримання протоколу Scorer (Duck Typing)."""
    assert isinstance(TfIdfScorer(), Scorer)
    assert isinstance(BM25Scorer(), Scorer)


def test_sanity_check_1_rare_vs_frequent() -> None:
    """Sanity Check 1: Рідкісний термін має давати вищий бал, ніж частіший."""
    docs = [
        Document("rare_doc", None, "quantum physics"),
        Document("common_doc_1", None, "physics science"),
        Document("common_doc_2", None, "physics nature"),
        Document("common_doc_3", None, "physics world"),
    ]
    idx = Index()
    idx.build(docs)

    res_rare = search_ranked(idx, "quantum", scorer=BM25Scorer())
    res_common = search_ranked(idx, "physics", scorer=BM25Scorer())

    assert res_rare[0].score > res_common[0].score


def test_sanity_check_2_bm25_repetition_saturation() -> None:
    """Sanity Check 2: 20-те повторення терміна дає менший приріст балів (насичення BM25)."""
    scorer = BM25Scorer()

    score_1 = scorer.score(tf=1, df=1, num_docs=10, doc_len=100, avg_doc_len=100)
    score_2 = scorer.score(tf=2, df=1, num_docs=10, doc_len=100, avg_doc_len=100)
    gain_initial = score_2 - score_1

    score_19 = scorer.score(tf=19, df=1, num_docs=10, doc_len=100, avg_doc_len=100)
    score_20 = scorer.score(tf=20, df=1, num_docs=10, doc_len=100, avg_doc_len=100)
    gain_late = score_20 - score_19

    assert gain_initial > gain_late * 5


def test_sanity_check_3_length_normalization() -> None:
    """Sanity Check 3: Короткий документ з 1 входженням вище за довгий з 1 входженням."""
    docs = [
        Document("short_doc", None, "python"),
        Document("long_doc", None, "python " + "other " * 100),
    ]
    idx = Index()
    idx.build(docs)

    results = search_ranked(idx, "python", scorer=BM25Scorer())

    assert len(results) == 2
    assert results[0].doc_id == "short_doc"
    assert results[0].score > results[1].score
