"""Corpus and index statistics computation."""

from dataclasses import dataclass

from findex.index import InvertedIndex


@dataclass(frozen=True)
class IndexStats:
    """Summary statistics of an InvertedIndex."""

    num_docs: int
    num_terms: int
    total_tokens: int
    avg_doc_length: float
    top_terms: list[tuple[str, int]]


def compute_stats(index: InvertedIndex, top_n: int = 10) -> IndexStats:
    """Compute summary statistics and top frequent terms for an InvertedIndex."""
    total_tokens = sum(index.doc_lengths.values())
    term_frequencies: list[tuple[str, int]] = []
    for term, postings in index.items():
        total_tf = sum(len(positions) for positions in postings.values())
        term_frequencies.append((term, total_tf))

    term_frequencies.sort(key=lambda item: (-item[1], item[0]))
    top_terms = term_frequencies[: max(0, top_n)]

    return IndexStats(
        num_docs=index.num_docs,
        num_terms=len(index),
        total_tokens=total_tokens,
        avg_doc_length=index.avg_doc_length,
        top_terms=top_terms,
    )
