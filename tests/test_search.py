"""Unit tests for boolean engines (merge/set) and ranking scorers (BM25/TF-IDF)."""

import pytest

from findex.corpus import Document
from findex.index import InvertedIndex
from findex.ranking import BM25Scorer, Scorer, TfIdfScorer, create_scorer
from findex.search import search_ast, search_boolean, search_ranked


def test_boolean_search_merge_and_set() -> None:
    docs = [
        Document("doc1", None, "apple banana orange"),
        Document("doc2", None, "apple banana"),
        Document("doc3", None, "apple orange"),
    ]
    idx = InvertedIndex()
    idx.build(docs)

    res_merge = search_boolean(idx, "apple banana", engine="merge")
    res_set = search_boolean(idx, "apple banana", engine="set")

    assert len(res_merge) == 2
    assert [r.doc_id for r in res_merge] == ["doc1", "doc2"]
    assert res_merge == res_set


@pytest.mark.parametrize("scorer_name", ["bm25", "tfidf"])
def test_scorers_conform_to_protocol(scorer_name: str) -> None:
    scorer = create_scorer(scorer_name)  # type: ignore[arg-type]  # parametrize passes str literals
    assert isinstance(scorer, Scorer)


def test_sanity_check_rare_term_beats_frequent() -> None:
    docs = [
        Document("rare_doc", None, "quantum physics"),
        Document("common_1", None, "physics science"),
        Document("common_2", None, "physics nature"),
        Document("common_3", None, "physics world"),
    ]
    idx = InvertedIndex()
    idx.build(docs)

    rare_hits = search_ranked(idx, "quantum", scorer=BM25Scorer())
    common_hits = search_ranked(idx, "physics", scorer=BM25Scorer())
    assert rare_hits[0].score > common_hits[0].score


def test_sanity_check_bm25_saturation() -> None:
    scorer = BM25Scorer()
    gain_early = scorer.score(tf=2, df=1, num_docs=10, doc_len=100, avg_doc_len=100) - scorer.score(
        tf=1, df=1, num_docs=10, doc_len=100, avg_doc_len=100
    )
    gain_late = scorer.score(tf=20, df=1, num_docs=10, doc_len=100, avg_doc_len=100) - scorer.score(
        tf=19, df=1, num_docs=10, doc_len=100, avg_doc_len=100
    )
    assert gain_early > gain_late * 5


def test_sanity_check_length_normalization() -> None:
    docs = [
        Document("short", None, "python"),
        Document("long", None, "python " + "filler " * 80),
    ]
    idx = InvertedIndex()
    idx.build(docs)

    hits = search_ranked(idx, "python", scorer=BM25Scorer())
    assert [h.doc_id for h in hits] == ["short", "long"]
    assert hits[0].score > hits[1].score


def test_tfidf_and_ast_ranked_search(built_index: InvertedIndex) -> None:
    tfidf_hits = search_ranked(built_index, "python", scorer=TfIdfScorer(), top_k=2)
    assert len(tfidf_hits) == 2

    ast_hits = search_ast(built_index, 'python AND "event loop" NOT runtime')
    assert [h.doc_id for h in ast_hits] == ["doc1"]
