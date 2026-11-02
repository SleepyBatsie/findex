import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    """Протокол (duck typing) для моделей ранжування."""

    def score(
        self,
        tf: int,
        df: int,
        num_docs: int,
        doc_len: int,
        avg_doc_len: float,
    ) -> float:
        ...


class TfIdfScorer:
    """Модель ранжування TF-IDF."""

    def score(
        self,
        tf: int,
        df: int,
        num_docs: int,
        doc_len: int,
        avg_doc_len: float,
    ) -> float:
        if tf == 0 or df == 0 or num_docs == 0:
            return 0.0
        # IDF
        idf = math.log((num_docs + 1) / (df + 0.5)) + 1.0
        return tf * idf


class BM25Scorer:
    """
    Модель ранжування BM25.
    Параметри за замовчуванням: k1 = 1.5 (насичення TF), b = 0.75 (нормалізація довжини).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        tf: int,
        df: int,
        num_docs: int,
        doc_len: int,
        avg_doc_len: float,
    ) -> float:
        if tf == 0 or df == 0 or num_docs == 0:
            return 0.0

        # BM25 IDF
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        
        # Нормалізація довжини документа
        len_norm = 1.0 - self.b + self.b * (doc_len / avg_doc_len if avg_doc_len > 0 else 1.0)
        
        # Насичення TF
        tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)

        return idf * tf_component