# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""FastAPI web API & Jinja2 UI tests using TestClient and dependency_overrides."""

from pathlib import Path

from fastapi.testclient import TestClient

from findex.corpus import Document
from findex.index import InvertedIndex
from findex.web.app import create_app, get_optional_index, require_index
from findex.web.config import Settings


def _build_in_memory_index(tmp_path: Path) -> InvertedIndex:
    doc1_path = tmp_path / "async_guide.txt"
    doc2_path = tmp_path / "bm25_notes.txt"
    doc3_path = tmp_path / "docker_deploy.txt"

    doc1_path.write_text(
        "Python asyncio event loop and TaskGroup handle concurrent network requests.",
        encoding="utf-8",
    )
    doc2_path.write_text(
        "BM25 ranking scores rare terms higher than frequent terms in search.",
        encoding="utf-8",
    )
    doc3_path.write_text(
        "Python FastAPI and Uvicorn run inside a non-root Docker container.",
        encoding="utf-8",
    )

    docs = [
        Document("docs/async_guide.txt", doc1_path, doc1_path.read_text(encoding="utf-8")),
        Document("docs/bm25_notes.txt", doc2_path, doc2_path.read_text(encoding="utf-8")),
        Document("docs/docker_deploy.txt", doc3_path, doc3_path.read_text(encoding="utf-8")),
    ]
    idx = InvertedIndex()
    idx.build(docs)
    return idx


def test_health_returns_503_when_index_missing_and_503_on_search(tmp_path: Path) -> None:
    missing_path = tmp_path / "nonexistent_index.json"
    app = create_app(Settings(index_path=missing_path))

    with TestClient(app) as client:
        res_health = client.get("/health")
        assert res_health.status_code == 503
        body = res_health.json()
        assert body["status"] == "unavailable"
        assert body["index_loaded"] is False

        res_search = client.get("/search", params={"q": "python"})
        assert res_search.status_code == 503


def test_api_search_pagination_validation_docs_and_stats_with_overrides(tmp_path: Path) -> None:
    idx = _build_in_memory_index(tmp_path)
    app = create_app(Settings(index_path=tmp_path / "unused.json"))
    app.dependency_overrides[require_index] = lambda: idx
    app.dependency_overrides[get_optional_index] = lambda: idx

    with TestClient(app) as client:
        res_health = client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["num_docs"] == 3

        res_p1 = client.get("/search", params={"q": "python", "k": 1, "page": 1, "scorer": "bm25"})
        assert res_p1.status_code == 200
        p1 = res_p1.json()
        assert p1["total_hits"] == 2
        assert p1["total_pages"] == 2
        assert p1["has_prev"] is False
        assert p1["has_next"] is True
        assert len(p1["results"]) == 1
        assert p1["results"][0]["rank"] == 1

        res_p2 = client.get("/search", params={"q": "python", "k": 1, "page": 2, "scorer": "tfidf"})
        assert res_p2.status_code == 200
        p2 = res_p2.json()
        assert p2["has_prev"] is True
        assert p2["has_next"] is False
        assert p2["results"][0]["rank"] == 2

        assert client.get("/search", params={"q": ""}).status_code == 422
        assert client.get("/search", params={"q": "   "}).status_code == 422
        assert client.get("/search", params={"q": "python", "k": 0}).status_code == 422
        assert client.get("/search", params={"q": "python", "page": 0}).status_code == 422
        assert client.get("/search", params={"q": "python", "scorer": "unknown"}).status_code == 422

        res_doc = client.get("/docs/docs/async_guide.txt")
        assert res_doc.status_code == 200
        assert "TaskGroup" in res_doc.json()["text"]

        res_missing = client.get("/docs/docs/does_not_exist.txt")
        assert res_missing.status_code == 404

        res_stats = client.get("/stats", params={"top": 5})
        assert res_stats.status_code == 200
        stats_json = res_stats.json()
        assert stats_json["num_docs"] == 3
        assert len(stats_json["top_terms"]) <= 5


def test_jinja_web_ui_renders_highlights_pagination_and_doc_viewer(tmp_path: Path) -> None:
    idx = _build_in_memory_index(tmp_path)
    app = create_app(Settings(index_path=tmp_path / "unused.json"))
    app.dependency_overrides[require_index] = lambda: idx
    app.dependency_overrides[get_optional_index] = lambda: idx

    with TestClient(app) as client:
        res_home = client.get(
            "/",
            params={
                "q": "asyncio",
                "k": 10,
                "page": 1,
                "doc": "docs/async_guide.txt",
            },
        )
        assert res_home.status_code == 200
        html = res_home.text
        assert "<mark>asyncio</mark>" in html.lower()
        assert "docs/async_guide.txt" in html
        assert "TaskGroup" in html


def test_unhandled_exception_returns_clean_500_without_traceback(tmp_path: Path) -> None:
    app = create_app(Settings(index_path=tmp_path / "unused.json"))

    def _boom() -> InvertedIndex:
        raise RuntimeError("Secret database password leaked in stacktrace")

    app.dependency_overrides[require_index] = _boom

    with TestClient(app, raise_server_exceptions=False) as client:
        res = client.get("/search", params={"q": "python"})
        assert res.status_code == 500
        body = res.json()
        assert body["detail"] == "Internal Server Error"
        assert "request_id" in body
        assert "Secret database password" not in res.text
        assert "Traceback" not in res.text
