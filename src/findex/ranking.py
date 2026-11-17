"""Ranking protocols and scoring algorithms (TF-IDF and BM25)."""

import math
from typing import Literal, Protocol, runtime_checkable

ScorerName = Literal["bm25", "tfidf"]


@runtime_checkable
class Scorer(Protocol):
    """Structural typing protocol for ranking models."""

    def score(
        self,
        tf: int,
        df: int,
        num_docs: int,
        doc_len: int,
        avg_doc_len: float,
    ) -> float: ...


class TfIdfScorer:
    """TF-IDF ranking model."""

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


class BM25Scorer:
    """Okapi BM25 ranking model with configurable k1 and b parameters."""

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


def create_scorer(name: ScorerName) -> Scorer:
    """Factory returning a Scorer implementation by Literal name."""
    if name == "bm25":
        return BM25Scorer()
    if name == "tfidf":
        return TfIdfScorer()
    raise ValueError(f"Unsupported scorer: {name}")