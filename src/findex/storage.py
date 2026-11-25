"""Index persistence (save, load, context manager) using UTF-8 JSON."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from findex.index import DocMeta, InvertedIndex

logger = logging.getLogger(__name__)


def save_index(index: InvertedIndex, path: Path | str) -> Path:
    """Serialize an InvertedIndex to disk and return the target Path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": 1,
        "with_positions": index.with_positions,
        "doc_lengths": dict(sorted(index.doc_lengths.items())),
        "doc_meta": {
            doc_id: {
                "doc_id": meta.doc_id,
                "length": meta.length,
                "path": meta.path,
            }
            for doc_id, meta in sorted(index.doc_meta.items())
        },
        "index": {
            term: {doc_id: list(positions) for doc_id, positions in sorted(postings.items())}
            for term, postings in sorted(index.index.items())
        },
    }

    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved index (%d terms, %d docs) to %s", len(index), index.num_docs, target)
    return target


def load_index(path: Path | str) -> InvertedIndex:
    """Deserialize an InvertedIndex from disk."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Index file not found: {source}")
    if not source.is_file():
        raise ValueError(f"Index path is not a file: {source}")

    try:
        raw_text = source.read_text(encoding="utf-8")
        loaded_obj: object = json.loads(raw_text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Corrupted index file {source}: {exc}") from exc

    if not isinstance(loaded_obj, dict):
        raise ValueError(f"Invalid index schema in file: {source}")

    data = cast(dict[str, object], loaded_obj)
    if "index" not in data or "doc_meta" not in data:
        raise ValueError(f"Invalid index schema in file: {source}")

    with_positions = bool(data.get("with_positions", True))
    idx = InvertedIndex(with_positions=with_positions)

    raw_lengths = data.get("doc_lengths")
    if isinstance(raw_lengths, dict):
        lengths_dict = cast(dict[object, object], raw_lengths)
        idx.doc_lengths = {str(k): int(v) for k, v in lengths_dict.items() if isinstance(v, int)}

    raw_meta = data.get("doc_meta")
    if isinstance(raw_meta, dict):
        meta_map = cast(dict[object, object], raw_meta)
        for doc_id_obj, meta_val in sorted(meta_map.items(), key=lambda item: str(item[0])):
            if isinstance(meta_val, dict):
                meta_dict = cast(dict[str, object], meta_val)
                doc_id = str(meta_dict.get("doc_id", doc_id_obj))
                length_val = meta_dict.get("length", 0)
                length = length_val if isinstance(length_val, int) else 0
                path_val = meta_dict.get("path")
                idx.doc_meta[doc_id] = DocMeta(
                    doc_id=doc_id,
                    length=length,
                    path=str(path_val) if path_val is not None else None,
                )

    raw_index = data.get("index")
    if isinstance(raw_index, dict):
        index_map = cast(dict[object, object], raw_index)
        for term_obj, postings_val in sorted(index_map.items(), key=lambda item: str(item[0])):
            if isinstance(postings_val, dict):
                postings_map = cast(dict[object, object], postings_val)
                term_postings: dict[str, list[int]] = {}
                for doc_id_obj, pos_val in sorted(
                    postings_map.items(), key=lambda item: str(item[0])
                ):
                    if isinstance(pos_val, list):
                        pos_list = cast(list[object], pos_val)
                        term_postings[str(doc_id_obj)] = [p for p in pos_list if isinstance(p, int)]
                idx.index[str(term_obj)] = term_postings

    logger.info("Loaded index (%d terms, %d docs) from %s", len(idx), idx.num_docs, source)
    return idx


@contextmanager
def open_index(path: Path | str) -> Generator[InvertedIndex, None, None]:
    """Context manager yielding a loaded InvertedIndex."""
    idx = load_index(path)
    yield idx
