"""Shared pytest fixtures for findex test suite."""

from pathlib import Path

import pytest

from findex.corpus import Document, load_corpus
from findex.index import InvertedIndex
from findex.storage import save_index


@pytest.fixture
def sample_docs() -> list[Document]:
    """Small deterministic corpus with English and Ukrainian documents."""
    return [
        Document("doc1", None, "python async await event loop concurrency"),
        Document("doc2", None, "python data science pandas numpy"),
        Document("doc3", None, "java spring boot microservices"),
        Document("doc4", None, "привіт світ пошукова система індекс"),
        Document("doc5", None, "python event loop fast runtime"),
    ]


@pytest.fixture
def built_index(sample_docs: list[Document]) -> InvertedIndex:
    """Pre-built InvertedIndex over sample_docs."""
    idx = InvertedIndex()
    idx.build(sample_docs)
    return idx


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Temporary directory populated with sample text files."""
    cdir = tmp_path / "corpus"
    cdir.mkdir()
    (cdir / "alpha.txt").write_text("Python async await event loop performance", encoding="utf-8")
    (cdir / "beta.md").write_text("Java spring boot microservices architecture", encoding="utf-8")
    (cdir / "gamma.txt").write_text("Пошуковий рушій на Python з індексом BM25", encoding="utf-8")
    return cdir


@pytest.fixture
def saved_index_path(tmp_path: Path, corpus_dir: Path) -> Path:
    """Serialized index JSON created from corpus_dir."""
    docs = load_corpus(corpus_dir)
    idx = InvertedIndex()
    idx.build(docs)
    return save_index(idx, tmp_path / "index.json")