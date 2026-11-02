# src/findex/search.py
from dataclasses import dataclass, field
import heapq
from typing import List, Optional, Dict
from findex.index import InvertedIndex
from findex.ranking import Scorer, BM25Scorer
from findex.storage import load_index


@dataclass(order=True)
class SearchResult:
    """
    Результат пошуку. order=True дозволяє heapq.nlargest
    порівнювати об'єкти за першим полем (score).
    """
    score: float = 0.0
    doc_id: str = field(default="", compare=False)


def search_boolean(index: InvertedIndex, query: str, engine: str = "merge") -> List[SearchResult]:
    """Булевий пошук з поверненням об'єктів SearchResult (сумісність із тестами Лаб 1-2)."""
    terms = query.lower().split()
    if not terms:
        return []

    matching_docs = None
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


def search_ranked(
    index: InvertedIndex,
    query: str,
    scorer: Optional[Scorer] = None,
    top_k: int = 10,
) -> List[SearchResult]:
    """
    Ранжований пошук із накопиченням балів та відбором top-k за допомогою heapq.nlargest.
    """
    if scorer is None:
        scorer = BM25Scorer()

    terms = query.lower().split()
    if not terms:
        return []

    doc_scores: Dict[str, float] = {}
    num_docs = index.num_docs
    avg_len = index.avg_doc_length

    for term in terms:
        if term not in index:
            continue
        postings = index[term]  # {doc_id: [pos1, pos2, ...]}
        df = index.df(term)

        for doc_id, positions in postings.items():
            tf = len(positions)
            doc_len = index.doc_len(doc_id)

            term_score = scorer.score(
                tf=tf,
                df=df,
                num_docs=num_docs,
                doc_len=doc_len,
                avg_doc_len=avg_len,
            )
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + term_score

    results = [SearchResult(score=score, doc_id=doc_id) for doc_id, score in doc_scores.items()]
    return heapq.nlargest(top_k, results)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="findex Search Engine CLI")
    parser.add_argument("index_path", help="Path to index file")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    args = parser.parse_args()

    idx = load_index(args.index_path)
    results = search_ranked(idx, args.query, top_k=args.top_k)

    print(f"=== Search Results for '{args.query}' ===")
    print(f"Found matches: {len(results)}\n")
    for rank, res in enumerate(results, 1):
        print(f"{rank}. [Score: {res.score:.4f}] Doc ID: {res.doc_id}")


if __name__ == "__main__":
    main()