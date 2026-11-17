"""Boolean and ranked search functions."""

import heapq
import logging
from dataclasses import dataclass, field
from typing import Literal

from findex.index import InvertedIndex
from findex.parser import parse_query
from findex.ranking import BM25Scorer, Scorer, ScorerName, create_scorer
from findex.tokenize import tokenize

logger = logging.getLogger(__name__)

EngineName = Literal["merge", "set"]


@dataclass(order=True)
class SearchResult:
    """Ranked or boolean search hit ordered by descending score."""

    score: float = 0.0
    doc_id: str = field(default="", compare=False)


def _intersect_sorted(left: list[str], right: list[str]) -> list[str]:
    """Two-pointer intersection of two strictly sorted doc_id lists."""
    result: list[str] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            result.append(left[i])
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return result


def search_boolean(
    index: InvertedIndex,
    query: str,
    engine: EngineName = "merge",
) -> list[SearchResult]:
    """Execute conjunctive (AND) boolean search using either sorted merge or set intersection."""
    terms = tokenize(query)
    if not terms:
        return []

    if engine == "set":
        matching_docs: set[str] | None = None
        for term in terms:
            if term not in index:
                return []
            term_docs = set(index[term].keys())
            if matching_docs is None:
                matching_docs = term_docs
            else:
                matching_docs &= term_docs
        if not matching_docs:
            return []
        return [SearchResult(doc_id=doc_id) for doc_id in sorted(matching_docs)]

    if engine == "merge":
        merged_docs: list[str] | None = None
        for term in terms:
            postings = index.get_postings(term)
            if not postings:
                return []
            doc_ids = [posting.doc_id for posting in postings]
            if merged_docs is None:
                merged_docs = doc_ids
            else:
                merged_docs = _intersect_sorted(merged_docs, doc_ids)
                if not merged_docs:
                    return []
        if not merged_docs:
            return []
        return [SearchResult(doc_id=doc_id) for doc_id in merged_docs]

    raise ValueError(f"Unsupported boolean engine: {engine}")


def _resolve_scorer(scorer: Scorer | ScorerName | None) -> Scorer:
    if scorer is None:
        return BM25Scorer()
    if isinstance(scorer, str):
        return create_scorer(scorer)
    return scorer


def search_ranked(
    index: InvertedIndex,
    query: str,
    scorer: Scorer | ScorerName | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """Rank documents matching query terms and return top_k results via heapq.nlargest."""
    active_scorer = _resolve_scorer(scorer)
    terms = tokenize(query)
    if not terms or top_k <= 0:
        return []

    doc_scores: dict[str, float] = {}
    num_docs = index.num_docs
    avg_len = index.avg_doc_length

    for term in terms:
        if term not in index:
            continue
        postings = index[term]
        df = index.df(term)

        for doc_id, positions in postings.items():
            tf = len(positions)
            doc_len = index.doc_len(doc_id)
            term_score = active_scorer.score(
                tf=tf,
                df=df,
                num_docs=num_docs,
                doc_len=doc_len,
                avg_doc_len=avg_len,
            )
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

    results = [SearchResult(score=score, doc_id=doc_id) for doc_id, score in doc_scores.items()]
    ranked = heapq.nlargest(top_k, results)
    logger.debug("Ranked search for %r returned %d hits", query, len(ranked))
    return ranked


def search_ast(
    index: InvertedIndex,
    query: str,
    scorer: Scorer | ScorerName | None = None,
    top_k: int = 10,
) -> list[SearchResult]:
    """Evaluate a boolean/phrase AST query and rank the matching documents."""
    ast_root = parse_query(query)
    allowed_docs = ast_root.evaluate(index)
    if not allowed_docs or top_k <= 0:
        return []

    active_scorer = _resolve_scorer(scorer)
    terms = [
        tok
        for tok in tokenize(query)
        if tok not in {"and", "or", "not"}
    ]

    num_docs = index.num_docs
    avg_len = index.avg_doc_length
    doc_scores: dict[str, float] = {doc_id: 0.0 for doc_id in allowed_docs}

    for term in terms:
        if term not in index:
            continue
        postings = index[term]
        df = index.df(term)
        for doc_id in allowed_docs:
            if doc_id in postings:
                tf = len(postings[doc_id])
                doc_len = index.doc_len(doc_id)
                doc_scores[doc_id] += active_scorer.score(
                    tf=tf,
                    df=df,
                    num_docs=num_docs,
                    doc_len=doc_len,
                    avg_doc_len=avg_len,
                )

    results = [SearchResult(score=score, doc_id=doc_id) for doc_id, score in doc_scores.items()]
    return heapq.nlargest(top_k, results)