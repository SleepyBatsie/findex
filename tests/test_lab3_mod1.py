from findex.corpus import Document
from findex.index import Index
from findex.storage import open_index, save_index


def test_index_dunder_and_properties(tmp_path):
    docs = [
        Document("doc1", None, "python code async await"),
        Document("doc2", None, "python event loop"),
    ]

    idx = Index()
    idx.build(docs)

    # 1. dunder len: 6 унікальних слів (python, code, async, await, event, loop)
    assert len(idx) == 6

    # 2. dunder in
    assert "python" in idx
    assert "java" not in idx

    # 3. dunder getitem
    assert "doc1" in idx["python"]
    assert "doc2" in idx["python"]

    # 4. repr
    assert repr(idx) == f"<Index terms={len(idx)} docs=2>"

    # 5. num_docs & avg_doc_length (cached_property)
    assert idx.num_docs == 2
    assert idx.avg_doc_length == 3.5  # (4 + 3) / 2

    # 6. df (document frequency)
    assert idx.df("python") == 2
    assert idx.df("async") == 1

    # 7. context manager open_index
    file_path = tmp_path / "test_index.bin"
    save_index(idx, file_path)

    with open_index(file_path) as loaded_idx:
        assert loaded_idx.num_docs == 2
        assert "python" in loaded_idx
