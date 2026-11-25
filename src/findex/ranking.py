"""Ranking scorers (TF-IDF and Okapi BM25) with scalar and vectorized NumPy implementations."""

import math
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

ScorerName = Literal["bm25", "tfidf"]


@runtime_checkable
class Scorer(Protocol):
    """Protocol implemented by all ranking scorers."""

    def score(
        self,
        tf: int,
        df: int,
        num_docs: int,
        doc_len: int,
        avg_doc_len: float,
    ) -> float: ...

    def score_array(
        self,
        tfs: NDArray[np.int32],
        df: int,
        num_docs: int,
        doc_lens: NDArray[np.int32],
        avg_doc_len: float,
    ) -> NDArray[np.float64]: ...


class TfIdfScorer:
    """Sublinear/raw TF-IDF ranking model with scalar and NumPy ufunc scoring."""

    def score(
        self,
        tf: int,
        df: int,
        num_docs: int,
        doc_len: int,
        avg_doc_len: float,
    ) -> float:
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0
        idf = math.log((num_docs + 1.0) / (df + 0.5)) + 1.0
        return float(tf) * idf

    def score_array(
        self,
        tfs: NDArray[np.int32],
        df: int,
        num_docs: int,
        doc_lens: NDArray[np.int32],
        avg_doc_len: float,
    ) -> NDArray[np.float64]:
        if df <= 0 or num_docs <= 0 or tfs.size == 0:
            return np.zeros(tfs.shape, dtype=np.float64)
        idf = math.log((num_docs + 1.0) / (df + 0.5)) + 1.0
        return tfs.astype(np.float64) * idf


class BM25Scorer:
    """Okapi BM25 ranking model with scalar and vectorized NumPy ufunc scoring."""

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
        if tf <= 0 or df <= 0 or num_docs <= 0:
            return 0.0

        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        norm_ratio = (doc_len / avg_doc_len) if avg_doc_len > 0.0 else 1.0
        len_norm = 1.0 - self.b + self.b * norm_ratio
        tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
        return idf * tf_component

    def score_array(
        self,
        tfs: NDArray[np.int32],
        df: int,
        num_docs: int,
        doc_lens: NDArray[np.int32],
        avg_doc_len: float,
    ) -> NDArray[np.float64]:
        if df <= 0 or num_docs <= 0 or tfs.size == 0:
            return np.zeros(tfs.shape, dtype=np.float64)

        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_f = tfs.astype(np.float64)
        if avg_doc_len > 0.0:
            norm_ratio = doc_lens.astype(np.float64) / avg_doc_len
        else:
            norm_ratio = np.ones(doc_lens.shape, dtype=np.float64)

        len_norm = (1.0 - self.b) + self.b * norm_ratio
        tf_component = (tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * len_norm)
        return idf * tf_component


def create_scorer(name: ScorerName) -> Scorer:
    """Factory returning a Scorer implementation by Literal name."""
    if name == "bm25":
        return BM25Scorer()
    if name == "tfidf":
        return TfIdfScorer()
    raise ValueError(f"Unsupported scorer: {name}")
