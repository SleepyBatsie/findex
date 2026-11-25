"""Positional inverted index with NumPy int32 postings and parallel merge pipeline."""

import logging
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from findex.corpus import Document, discover_files, read_documents
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

ExecutorName = Literal["serial", "threads", "processes"]


def _empty_positions() -> list[int]:
    return []


@dataclass
class Posting:
    """Entry in a postings list storing document ID, term frequency, and positions."""

    doc_id: str
    tf: int = 0
    positions: list[int] = field(default_factory=_empty_positions)

    def __post_init__(self) -> None:
        if self.positions and self.tf == 0:
            self.tf = len(self.positions)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Posting):
            return NotImplemented
        if self.doc_id != other.doc_id or self.tf != other.tf:
            return False
        if self.positions and other.positions:
            return self.positions == other.positions
        return True


@dataclass(frozen=True)
class DocMeta:
    """Metadata for an indexed document."""

    doc_id: str
    length: int
    path: str | None = None


@dataclass(frozen=True)
class BuildMetrics:
    """Execution timing breakdown for index construction."""

    total_seconds: float
    map_seconds: float
    merge_seconds: float
    num_chunks: int
    workers: int
    executor: ExecutorName


class InvertedIndex(Mapping[str, dict[str, list[int]]]):
    """
    Positional inverted index implementing collections.abc.Mapping.
    Maintains dict postings for compatibility and compiled NumPy int32 arrays for fast scoring.
    """

    __slots__ = (
        "_index",
        "_doc_lengths",
        "doc_meta",
        "with_positions",
        "_np_dirty",
        "_doc_id_list",
        "_doc_lengths_np",
        "_postings_np",
        "__dict__",
    )

    def __init__(self, with_positions: bool = True) -> None:
        self._index: dict[str, dict[str, list[int]]] = {}
        self._doc_lengths: dict[str, int] = {}
        self.doc_meta: dict[str, DocMeta] = {}
        self.with_positions = with_positions
        self._np_dirty: bool = True
        self._doc_id_list: list[str] = []
        self._doc_lengths_np: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self._postings_np: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}

    def _invalidate_caches(self) -> None:
        self._np_dirty = True
        if "avg_doc_length" in self.__dict__:
            del self.__dict__["avg_doc_length"]

    def compile_numpy(self) -> None:
        """Compile postings and document lengths into contiguous NumPy int32 arrays."""
        sorted_doc_ids = sorted(self._doc_lengths.keys())
        self._doc_id_list = sorted_doc_ids
        doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(sorted_doc_ids)}

        self._doc_lengths_np = np.array(
            [self._doc_lengths[doc_id] for doc_id in sorted_doc_ids],
            dtype=np.int32,
        )

        compiled_postings: dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]] = {}
        for term, postings in self._index.items():
            # Sort by integer doc index (which matches sorted doc_id order)
            pairs = sorted(
                (doc_to_idx[doc_id], len(positions))
                for doc_id, positions in postings.items()
                if doc_id in doc_to_idx
            )
            if pairs:
                doc_arr = np.array([p[0] for p in pairs], dtype=np.int32)
                tf_arr = np.array([p[1] for p in pairs], dtype=np.int32)
            else:
                doc_arr = np.empty(0, dtype=np.int32)
                tf_arr = np.empty(0, dtype=np.int32)
            compiled_postings[term] = (doc_arr, tf_arr)

        self._postings_np = compiled_postings
        self._np_dirty = False

    @property
    def doc_id_list(self) -> list[str]:
        if self._np_dirty:
            self.compile_numpy()
        return self._doc_id_list

    @property
    def doc_lengths_np(self) -> NDArray[np.int32]:
        if self._np_dirty:
            self.compile_numpy()
        return self._doc_lengths_np

    @property
    def postings_np(self) -> dict[str, tuple[NDArray[np.int32], NDArray[np.int32]]]:
        if self._np_dirty:
            self.compile_numpy()
        return self._postings_np

    @property
    def doc_metas(self) -> dict[str, DocMeta]:
        """Compatibility alias for doc_meta."""
        return self.doc_meta

    @doc_metas.setter
    def doc_metas(self, value: dict[str, DocMeta]) -> None:
        self.doc_meta = value

    @property
    def doc_lengths(self) -> dict[str, int]:
        """Mapping from doc_id to document token length."""
        return self._doc_lengths

    @doc_lengths.setter
    def doc_lengths(self, value: dict[str, int]) -> None:
        self._doc_lengths = value
        self._invalidate_caches()

    @property
    def index(self) -> dict[str, dict[str, list[int]]]:
        """Raw mapping from term to {doc_id: [positions]}."""
        return self._index

    @index.setter
    def index(self, value: dict[str, dict[str, list[int]]]) -> None:
        self._index = value
        self._invalidate_caches()

    def build(self, docs: Iterable[Document]) -> None:
        """Build the inverted index from an iterable of documents."""
        self._index.clear()
        self._doc_lengths.clear()
        self.doc_meta.clear()
        self._invalidate_caches()

        sorted_docs = sorted(docs, key=lambda d: d.doc_id)
        for doc in sorted_docs:
            tokens = tokenize(doc.text)
            doc_len = len(tokens)
            self._doc_lengths[doc.doc_id] = doc_len
            path_str = str(doc.path) if doc.path is not None else None
            self.doc_meta[doc.doc_id] = DocMeta(
                doc_id=doc.doc_id,
                length=doc_len,
                path=path_str,
            )

            for pos, term in enumerate(tokens):
                term_postings = self._index.setdefault(term, {})
                doc_positions = term_postings.setdefault(doc.doc_id, [])
                if self.with_positions:
                    doc_positions.append(pos)
                else:
                    doc_positions.append(0)

        self.compile_numpy()
        logger.debug("Built index with %d terms across %d documents", len(self), self.num_docs)

    def get_postings(self, term: str) -> list[Posting]:
        """Return sorted list of Posting entries for a given term."""
        normalized = tokenize(term)
        if not normalized:
            return []
        clean_term = normalized[0]
        if clean_term not in self._index:
            return []
        return [
            Posting(doc_id=doc_id, tf=len(positions), positions=list(positions))
            for doc_id, positions in sorted(self._index[clean_term].items())
        ]

    # --- Mapping Protocol (Dunder Methods) ---

    def __getitem__(self, term: str) -> dict[str, list[int]]:
        return self._index[term]

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __contains__(self, term: object) -> bool:
        return term in self._index

    def __repr__(self) -> str:
        return f"<Index terms={len(self)} docs={self.num_docs}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InvertedIndex):
            return NotImplemented
        return (
            self._index == other._index
            and self._doc_lengths == other._doc_lengths
            and self.doc_meta == other.doc_meta
            and self.with_positions == other.with_positions
        )

    # --- Properties & Helpers ---

    @property
    def num_docs(self) -> int:
        """Total number of indexed documents."""
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        """Average token length across all indexed documents."""
        if not self._doc_lengths:
            return 0.0
        if not self._np_dirty and self._doc_lengths_np.size > 0:
            return float(np.mean(self._doc_lengths_np))
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def doc_len(self, doc_id: str) -> int:
        """Return token length of doc_id (0 if absent)."""
        return self._doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        """Return document frequency for term."""
        return len(self._index[term]) if term in self._index else 0


Index = InvertedIndex


def chunk_paths(paths: Sequence[Path], num_chunks: int) -> list[list[Path]]:
    """Split file paths into contiguous, non-overlapping ranges for parallel workers."""
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if not paths:
        return []
    total = len(paths)
    actual_chunks = min(num_chunks, total)
    base_size, remainder = divmod(total, actual_chunks)
    chunks: list[list[Path]] = []
    start = 0
    for i in range(actual_chunks):
        end = start + base_size + (1 if i < remainder else 0)
        chunks.append(list(paths[start:end]))
        start = end
    return chunks


def build_partial(
    paths: Sequence[Path | str],
    base_dir: Path | str | None = None,
    with_positions: bool = True,
) -> InvertedIndex:
    """
    Build a partial InvertedIndex from a sequence of file paths.
    Top-level module function safe for ProcessPoolExecutor ('spawn').
    """
    resolved_base = Path(base_dir) if base_dir is not None else None
    docs = [doc for p in paths for doc in read_documents(Path(p), base_dir=resolved_base)]
    partial = InvertedIndex(with_positions=with_positions)
    partial.build(docs)
    return partial


def merge(partials: Iterable[InvertedIndex]) -> InvertedIndex:
    """
    Merge multiple partial InvertedIndex instances into a single unified InvertedIndex.
    Preserves deterministic sorting of doc_ids, postings, and metadata.
    """
    partial_list = list(partials)
    if not partial_list:
        return InvertedIndex()

    with_positions = partial_list[0].with_positions
    merged = InvertedIndex(with_positions=with_positions)

    raw_index: dict[str, dict[str, list[int]]] = {}
    raw_lengths: dict[str, int] = {}
    raw_meta: dict[str, DocMeta] = {}

    for partial in partial_list:
        if partial.with_positions != with_positions:
            raise ValueError("Cannot merge indexes with different with_positions settings")

        for doc_id, meta in partial.doc_meta.items():
            if doc_id in raw_meta:
                raise ValueError(f"Duplicate doc_id encountered during merge: {doc_id}")
            raw_meta[doc_id] = meta
            raw_lengths[doc_id] = partial.doc_len(doc_id)

        for term, postings in partial.index.items():
            target_postings = raw_index.setdefault(term, {})
            for doc_id, positions in postings.items():
                target_postings[doc_id] = list(positions)

    merged.doc_meta = dict(sorted(raw_meta.items()))
    merged.doc_lengths = dict(sorted(raw_lengths.items()))
    merged.index = {
        term: dict(sorted(postings.items())) for term, postings in sorted(raw_index.items())
    }
    merged.compile_numpy()
    return merged


def build_index(
    corpus_dir: Path | str,
    *,
    workers: int = 1,
    executor: ExecutorName = "processes",
    with_positions: bool = True,
    on_chunk_done: Callable[[int], None] | None = None,
    metrics_out: list[BuildMetrics] | None = None,
) -> InvertedIndex:
    """
    Build an InvertedIndex using serial loop, ThreadPoolExecutor, or ProcessPoolExecutor
    via top-level build_partial and merge.
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if executor not in ("serial", "threads", "processes"):
        raise ValueError(f"Unsupported executor: {executor}")

    t_start = time.perf_counter()
    root_path = Path(corpus_dir)
    base_dir = root_path if root_path.is_dir() else root_path.parent
    files = discover_files(root_path)

    if not files:
        empty_idx = InvertedIndex(with_positions=with_positions)
        if metrics_out is not None:
            elapsed = time.perf_counter() - t_start
            metrics_out.append(
                BuildMetrics(
                    total_seconds=elapsed,
                    map_seconds=0.0,
                    merge_seconds=0.0,
                    num_chunks=0,
                    workers=workers,
                    executor=executor,
                )
            )
        return empty_idx

    chunks = chunk_paths(files, workers)
    t_map_start = time.perf_counter()
    partials: list[InvertedIndex] = []

    if executor == "serial":
        for chunk in chunks:
            partials.append(build_partial(chunk, base_dir, with_positions))
            if on_chunk_done is not None:
                on_chunk_done(len(chunk))
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_size = {
                pool.submit(build_partial, chunk, base_dir, with_positions): len(chunk)
                for chunk in chunks
            }
            try:
                for fut in as_completed(future_to_size):
                    partials.append(fut.result())
                    if on_chunk_done is not None:
                        on_chunk_done(future_to_size[fut])
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_size = {
                pool.submit(build_partial, chunk, base_dir, with_positions): len(chunk)
                for chunk in chunks
            }
            try:
                for fut in as_completed(future_to_size):
                    partials.append(fut.result())
                    if on_chunk_done is not None:
                        on_chunk_done(future_to_size[fut])
            except BaseException:
                pool.shutdown(wait=True, cancel_futures=True)
                raise

    t_merge_start = time.perf_counter()
    merged = merge(partials)
    t_end = time.perf_counter()

    if metrics_out is not None:
        metrics_out.append(
            BuildMetrics(
                total_seconds=t_end - t_start,
                map_seconds=t_merge_start - t_map_start,
                merge_seconds=t_end - t_merge_start,
                num_chunks=len(chunks),
                workers=workers,
                executor=executor,
            )
        )

    logger.info(
        "Built index (%d terms, %d docs) via %s (workers=%d) in %.4fs (merge=%.4fs)",
        len(merged),
        merged.num_docs,
        executor,
        workers,
        t_end - t_start,
        t_end - t_merge_start,
    )
    return merged
