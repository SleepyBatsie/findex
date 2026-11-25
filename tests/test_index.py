"""Unit tests for InvertedIndex mapping protocol and metadata."""

from findex.corpus import Document
from findex.index import InvertedIndex, Posting


def test_inverted_index_build_and_postings() -> None:
    docs = [
        Document("doc1", None, "hello world"),
        Document("doc2", None, "hello python world"),
    ]
    idx = InvertedIndex()
    idx.build(docs)

    assert "hello" in idx.index
    assert len(idx.get_postings("hello")) == 2
    assert idx.get_postings("python") == [Posting("doc2", 1)]


def test_index_mapping_dunder_methods(built_index: InvertedIndex) -> None:
    assert "python" in built_index
    assert "nonexistent" not in built_index
    assert built_index.num_docs == 5
    assert len(built_index) > 0
    assert list(built_index) == list(built_index.index.keys())
    assert "doc1" in built_index["python"]
    assert repr(built_index) == f"<Index terms={len(built_index)} docs=5>"


def test_index_unicode_and_doc_lengths(built_index: InvertedIndex) -> None:
    assert "привіт" in built_index
    assert built_index.df("привіт") == 1
    assert built_index.df("python") == 3
    assert built_index.doc_len("doc1") == 6
    assert built_index.doc_len("missing") == 0
    assert built_index.avg_doc_length > 0.0
