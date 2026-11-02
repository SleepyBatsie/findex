# src/findex/index.py
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterator, List, Any, Optional
from findex.corpus import Document


@dataclass
class Posting:
    """Запис у postings list для збереження інформації про документ та позиції."""
    doc_id: str
    tf: int = 0
    positions: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.tf, list):
            self.positions = self.tf
            self.tf = len(self.positions)
        elif isinstance(self.positions, int):
            self.tf = self.positions
            self.positions = []
        elif isinstance(self.positions, list) and self.positions and not self.tf:
            self.tf = len(self.positions)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Posting):
            if self.doc_id != other.doc_id:
                return False
            if self.tf != other.tf:
                return False
            if self.positions and other.positions:
                return self.positions == other.positions
            return True
        return False


@dataclass
class DocMeta:
    """Метадані заіндексованого документа."""
    doc_id: str
    length: int
    path: Optional[str] = None


class InvertedIndex(Mapping):
    """
    Позиційний інвертований індекс, що реалізує collections.abc.Mapping.
    Дозволяє доступ через dunder-методи: len(idx), 'term' in idx, idx['term'].
    """

    __slots__ = ("_index", "_doc_lengths", "doc_meta", "with_positions", "__dict__")

    def __init__(self, with_positions: bool = True) -> None:
        self._index: Dict[str, Dict[str, List[int]]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self.doc_meta: Dict[str, DocMeta] = {}
        self.with_positions = with_positions

    @property
    def doc_metas(self) -> Dict[str, DocMeta]:
        """Аліас для сумісності доступу через doc_metas."""
        return self.doc_meta

    @doc_metas.setter
    def doc_metas(self, value: Dict[str, DocMeta]) -> None:
        self.doc_meta = value

    @property
    def index(self) -> Dict[str, Dict[str, List[int]]]:
        return self._index

    @index.setter
    def index(self, value: Dict[str, Dict[str, List[int]]]) -> None:
        self._index = value

    def build(self, docs: List[Document]) -> None:
        """Побудова індексу з очищенням кешу avg_doc_length."""
        self._index.clear()
        self._doc_lengths.clear()
        self.doc_meta.clear()

        if "avg_doc_length" in self.__dict__:
            del self.__dict__["avg_doc_length"]

        for doc in docs:
            tokens = doc.text.lower().split()
            doc_len = len(tokens)
            self._doc_lengths[doc.doc_id] = doc_len
            self.doc_meta[doc.doc_id] = DocMeta(doc_id=doc.doc_id, length=doc_len, path=getattr(doc, "path", None))

            for pos, term in enumerate(tokens):
                if term not in self._index:
                    self._index[term] = {}
                if doc.doc_id not in self._index[term]:
                    self._index[term][doc.doc_id] = []
                self._index[term][doc.doc_id].append(pos)

    def get_postings(self, term: str) -> List[Posting]:
        """Повертає список Posting для заданого терміна (сумісність із тестами Лаб 1-2)."""
        term_clean = term.lower()
        if term_clean not in self._index:
            return []
        return [
            Posting(doc_id=doc_id, tf=len(positions), positions=positions)
            for doc_id, positions in self._index[term_clean].items()
        ]

    # --- Mapping Protocol (Dunder Methods) ---

    def __getitem__(self, term: str) -> Dict[str, List[int]]:
        return self._index[term]

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __contains__(self, term: object) -> bool:
        return term in self._index

    def __repr__(self) -> str:
        return f"<Index terms={len(self)} docs={self.num_docs}>"

    # --- Properties & Methods ---

    @property
    def num_docs(self) -> int:
        return len(self.doc_meta)

    @cached_property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def doc_len(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def df(self, term: str) -> int:
        """Document Frequency."""
        return len(self._index[term]) if term in self._index else 0


Index = InvertedIndex