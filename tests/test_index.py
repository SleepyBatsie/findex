from findex.corpus import Document
from findex.index import InvertedIndex, Posting


def test_inverted_index_build():
    docs = [
        Document("doc1", None, "hello world"),
        Document("doc2", None, "hello python world"),
    ]
    idx = InvertedIndex()
    idx.build(docs)

    assert "hello" in idx.index
    assert len(idx.get_postings("hello")) == 2
    assert idx.get_postings("python") == [Posting("doc2", 1)]