"""Positional inverted index implementing collections.abc.Mapping."""

import logging
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property

from findex.corpus import Document
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)


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


class InvertedIndex(Mapping[str, dict[str, list[int]]]):
    """
    Positional inverted index implementing collections.abc.Mapping.
    Guarantees sorted postings by doc_id and ascending positions.
    """

    __slots__ = ("_index", "_doc_lengths", "doc_meta", "with_positions", "__dict__")

    def __init__(self, with_positions: bool = True) -> None:
        self._index: dict[str, dict[str, list[int]]] = {}
        self._doc_lengths: dict[str, int] = {}
        self.doc_meta: dict[str, DocMeta] = {}
        self.with_positions = with_positions

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

    @property
    def index(self) -> dict[str, dict[str, list[int]]]:
        """Raw mapping from term to {doc_id: [positions]}."""
        return self._index

    @index.setter
    def index(self, value: dict[str, dict[str, list[int]]]) -> None:
        self._index = value

    def build(self, docs: Iterable[Document]) -> None:
        """Build the inverted index from an iterable of documents."""
        self._index.clear()
        self._doc_lengths.clear()
        self.doc_meta.clear()

        if "avg_doc_length" in self.__dict__:
            del self.__dict__["avg_doc_length"]

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

        logger.info("Built index with %d terms across %d documents", len(self), self.num_docs)

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
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def doc_len(self, doc_id: str) -> int:
        """Return token length of doc_id (0 if absent)."""
        return self._doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        """Return document frequency for term."""
        return len(self._index[term]) if term in self._index else 0


Index = InvertedIndex