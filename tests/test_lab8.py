from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.corpus import Document
from findex.index import InvertedIndex
from findex.search import search_ranked, search_ranked_python
from findex.semantic import (
    SemanticIndex,
    chunk_text,
    l2_normalize,
    reciprocal_rank_fusion,
)
from findex.storage import save_index
from findex.web.app import create_app
from findex.web.config import Settings


def _sample_docs() -> list[Document]:
    return [
        Document("d1.txt", None, "python asyncio event loop taskgroup concurrency"),
        Document("d2.txt", None, "okapi bm25 ranking saturation length normalization"),
        Document("d3.txt", None, "numpy contiguous int32 arrays ufunc vectorization"),
        Document("d4.txt", None, "fastapi pydantic openapi validation search server"),
    ]


def test_numpy_postings_dtype_and_equivalence() -> None:
    idx = InvertedIndex()
    idx.build(_sample_docs())

    assert idx.doc_lengths_np.dtype == np.int32
    doc_arr, tf_arr = idx.postings_np["python"]
    assert doc_arr.dtype == np.int32
    assert tf_arr.dtype == np.int32

    for scorer in ("bm25", "tfidf"):
        for q in ("python asyncio", "bm25 ranking", "numpy arrays", "nonexistent"):
            py_hits = search_ranked_python(idx, q, scorer=scorer, top_k=5)
            np_hits = search_ranked(idx, q, scorer=scorer, top_k=5)
            assert [h.doc_id for h in py_hits] == [h.doc_id for h in np_hits]
            for a, b in zip(py_hits, np_hits, strict=True):
                assert abs(a.score - b.score) < 1e-9


def test_chunking_and_semantic_rrf_offline(tmp_path: Path) -> None:
    chunks = chunk_text(" ".join(f"word{i}" for i in range(250)), chunk_words=100, overlap_words=20)
    assert len(chunks) == 3

    # Synthetic L2-normalized 4-dim vectors (no network model needed for unit test)
    vecs = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # d1 chunk 1
                [0.2, 0.9, 0.0, 0.0],  # d1 chunk 2 (higher match for query [0, 1, 0, 0])
                [0.0, 0.0, 1.0, 0.0],  # d2 chunk 1
                [0.0, 0.5, 0.5, 0.0],  # d3 chunk 1
            ],
            dtype=np.float32,
        )
    )
    sem = SemanticIndex(
        embeddings=vecs,
        chunk_doc_ids=["d1.txt", "d1.txt", "d2.txt", "d3.txt"],
        model_name="synthetic-test",
    )
    saved = sem.save(tmp_path / "embeddings.npy")
    loaded = SemanticIndex.load(saved)

    q_vec = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    sem_hits = loaded.search_by_vector(q_vec, top_k=3)
    assert sem_hits[0].doc_id == "d1.txt"
    assert abs(sem_hits[0].score - float(vecs[1, 1])) < 1e-6

    idx = InvertedIndex()
    idx.build(_sample_docs())
    lex_hits = search_ranked(idx, "bm25 ranking", top_k=3)
    fused = reciprocal_rank_fusion(lex_hits, sem_hits, rrf_k=60, top_k=3)
    assert {h.doc_id for h in fused} >= {"d1.txt", "d2.txt"}


def test_web_api_semantic_and_hybrid_modes(tmp_path: Path, monkeypatch) -> None:
    docs = _sample_docs()
    idx = InvertedIndex()
    idx.build(docs)
    idx_path = save_index(idx, tmp_path / "index.json")

    vecs = l2_normalize(
        np.array(
            [
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.1],
                [0.1, 0.0, 0.0, 0.9],
            ],
            dtype=np.float32,
        )
    )
    sem = SemanticIndex(
        embeddings=vecs,
        chunk_doc_ids=["d1.txt", "d2.txt", "d3.txt", "d4.txt"],
        model_name="mock-model",
    )
    emb_path = sem.save(tmp_path / "embeddings.npy")

    monkeypatch.setattr(
        "findex.semantic.encode_texts",
        lambda texts, model_name="mock-model": l2_normalize(
            np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
        ),
    )

    cfg = Settings(index_path=idx_path, embeddings_path=emb_path)
    app = create_app(settings_override=cfg)
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["semantic_loaded"] is True

        r_sem = client.get("/search", params={"q": "fast matrix math", "mode": "semantic", "k": 2})
        assert r_sem.status_code == 200
        body_sem = r_sem.json()
        assert body_sem["mode"] == "semantic"
        assert body_sem["results"][0]["doc_id"] == "d3.txt"

        r_hyb = client.get("/search", params={"q": "python", "mode": "hybrid", "k": 3})
        assert r_hyb.status_code == 200
        assert r_hyb.json()["mode"] == "hybrid"


def test_cli_search_modes(tmp_path: Path, monkeypatch) -> None:
    idx = InvertedIndex()
    idx.build(_sample_docs())
    idx_path = save_index(idx, tmp_path / "index.json")

    vecs = l2_normalize(np.eye(4, dtype=np.float32))
    sem = SemanticIndex(
        embeddings=vecs,
        chunk_doc_ids=["d1.txt", "d2.txt", "d3.txt", "d4.txt"],
        model_name="mock-model",
    )
    emb_path = sem.save(tmp_path / "embeddings.npy")

    monkeypatch.setattr(
        "findex.semantic.encode_texts",
        lambda texts, model_name="mock-model": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    )

    runner = CliRunner()
    res = runner.invoke(
        cli_app,
        [
            "search",
            str(idx_path),
            "relevance",
            "--mode",
            "hybrid",
            "--embeddings",
            str(emb_path),
            "--json",
        ],
    )
    assert res.exit_code == 0
    assert "d2.txt" in res.stdout


def test_benchmark_search_ranked(benchmark) -> None:
    docs = [
        Document(
            doc_id=f"doc_{i:04d}.txt",
            path=None,
            text=f"python web framework search engine ranking bm25 numpy index {i} " * 20,
        )
        for i in range(500)
    ]
    idx = InvertedIndex()
    idx.build(docs)

    hits = benchmark(search_ranked, idx, "python web framework", scorer="bm25", top_k=10)
    assert len(hits) == 10
