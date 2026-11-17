"""Document representation and corpus loading utilities."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".rst", ".py", ".log"})


@dataclass(frozen=True)
class Document:
    """Represents a document to be indexed."""

    doc_id: str
    path: Path | str | None = None
    text: str = ""


def discover_files(root: Path | str) -> list[Path]:
    """Return a deterministically sorted list of supported text files under root."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {root_path}")
    if root_path.is_file():
        return [root_path]

    files = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    sorted_files = sorted(files, key=lambda p: p.as_posix())
    logger.debug("Discovered %d supported files in %s", len(sorted_files), root_path)
    return sorted_files


def read_document(file_path: Path, base_dir: Path | None = None) -> Document:
    """Read a UTF-8 text file into a Document instance."""
    rel_path = file_path.relative_to(base_dir) if base_dir is not None else file_path
    doc_id = rel_path.as_posix()
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return Document(doc_id=doc_id, path=file_path, text=text)


def iter_corpus(root: Path | str) -> Iterator[Document]:
    """Lazily yield Document instances from a file or directory."""
    root_path = Path(root)
    base_dir = root_path if root_path.is_dir() else root_path.parent
    for file_path in discover_files(root_path):
        yield read_document(file_path, base_dir=base_dir)


def load_corpus(root: Path | str) -> list[Document]:
    """Load all supported documents from a file or directory."""
    docs = list(iter_corpus(root))
    logger.info("Loaded %d documents from %s", len(docs), root)
    return docs