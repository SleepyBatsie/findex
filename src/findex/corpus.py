"""Corpus discovery and streaming document readers (including JSONL from findex crawl)."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".rst", ".html", ".htm", ".jsonl"})


@dataclass(frozen=True)
class Document:
    """Represents a single text document in the corpus."""

    doc_id: str
    path: Path | None
    text: str


def discover_files(root: Path | str) -> list[Path]:
    """Return deterministically sorted list of indexable files under root."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Corpus path not found: {root_path}")

    if root_path.is_file():
        return [root_path]

    files = [
        p
        for p in root_path.rglob("*")
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.relative_to(root_path).as_posix())


def read_document(file_path: Path, base_dir: Path | None = None) -> Document:
    """Read a single plain-text document from disk."""
    if base_dir is not None:
        try:
            doc_id = file_path.relative_to(base_dir).as_posix()
        except ValueError:
            doc_id = file_path.name
    else:
        doc_id = file_path.name

    text = file_path.read_text(encoding="utf-8", errors="replace")
    return Document(doc_id=doc_id, path=file_path, text=text)


def read_documents(file_path: Path, base_dir: Path | None = None) -> Iterator[Document]:
    """
    Yield one or more Document objects from a file.
    Streams .jsonl files line-by-line without loading the entire file into memory.
    """
    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                obj = cast(dict[str, object], json.loads(line))
                doc_id = str(obj.get("doc_id") or obj.get("url") or f"{file_path.name}:{line_no}")
                text = str(obj.get("text", ""))
                yield Document(doc_id=doc_id, path=file_path, text=text)
        return

    yield read_document(file_path, base_dir=base_dir)


def iter_documents(corpus_path: Path | str) -> Iterator[Document]:
    """Stream all Document objects from a directory, text file, or JSONL file."""
    root_path = Path(corpus_path)
    base_dir = root_path if root_path.is_dir() else root_path.parent
    for file_path in discover_files(root_path):
        yield from read_documents(file_path, base_dir=base_dir)


iter_corpus = iter_documents


def load_corpus(corpus_path: Path | str) -> list[Document]:
    """Load all documents from corpus_path into a sorted list."""
    return sorted(iter_documents(corpus_path), key=lambda d: d.doc_id)
