"""Property-based tests using Hypothesis."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.corpus import Document
from findex.index import InvertedIndex
from findex.search import search_boolean
from findex.storage import load_index, save_index

_WORD_STRATEGY = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
    min_size=1,
    max_size=10,
)

_DOCS_STRATEGY = st.lists(
    st.tuples(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=12),
        st.lists(_WORD_STRATEGY, min_size=0, max_size=15),
    ),
    min_size=0,
    max_size=15,
    unique_by=lambda pair: pair[0],
)


def _build_from_pairs(pairs: list[tuple[str, list[str]]]) -> InvertedIndex:
    docs = [Document(doc_id=doc_id, path=None, text=" ".join(words)) for doc_id, words in pairs]
    idx = InvertedIndex()
    idx.build(docs)
    return idx


@given(pairs=_DOCS_STRATEGY)
@settings(max_examples=50)
def test_property_save_load_roundtrip(pairs: list[tuple[str, list[str]]]) -> None:
    """Property 1: load_index(save_index(index)) reproduces the exact same index."""
    idx = _build_from_pairs(pairs)
    with tempfile.TemporaryDirectory() as tmp_dir:
        saved = save_index(idx, Path(tmp_dir) / "prop_index.json")
        restored = load_index(saved)
        assert restored == idx


@given(pairs=_DOCS_STRATEGY)
@settings(max_examples=50)
def test_property_postings_always_sorted(pairs: list[tuple[str, list[str]]]) -> None:
    """Property 2: Every term's postings are strictly sorted by doc_id and positions."""
    idx = _build_from_pairs(pairs)
    for term in idx:
        postings = idx.get_postings(term)
        doc_ids = [p.doc_id for p in postings]
        assert doc_ids == sorted(doc_ids)
        for posting in postings:
            assert posting.positions == sorted(posting.positions)


@given(pairs=_DOCS_STRATEGY, query_words=st.lists(_WORD_STRATEGY, min_size=1, max_size=4))
@settings(max_examples=50)
def test_property_merge_matches_set_operations(
    pairs: list[tuple[str, list[str]]],
    query_words: list[str],
) -> None:
    """Property 3: Sorted postings merge matches Python set intersection."""
    idx = _build_from_pairs(pairs)
    query = " ".join(query_words)
    merge_hits = search_boolean(idx, query, engine="merge")
    set_hits = search_boolean(idx, query, engine="set")
    assert merge_hits == set_hits


@pytest.mark.slow
@given(pairs=_DOCS_STRATEGY)
@settings(max_examples=250)
def test_property_large_scale_invariant(pairs: list[tuple[str, list[str]]]) -> None:
    """Extended property stress check marked as slow."""
    idx = _build_from_pairs(pairs)
    assert idx.num_docs == len(pairs)
    for term in idx:
        assert idx.df(term) == len(idx[term])
