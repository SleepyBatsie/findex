from findex.corpus import Document
from findex.index import InvertedIndex
from findex.search import search_boolean


def test_boolean_search_merge_and_set():
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
    assert [r.doc_id for r in res_set] == ["doc1", "doc2"]