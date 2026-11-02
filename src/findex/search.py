import argparse

from pathlib import Path
import time
import tracemalloc

from findex.index import InvertedIndex, Posting, DocMeta


def intersect_merge(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Двовказівниковий алгоритм перетину (AND) для відсортованих постінгів - O(N + M)."""
    i, j = 0, 0
    result = []
    while i < len(p1) and j < len(p2):
        if p1[i].doc_id == p2[j].doc_id:
            result.append(Posting(doc_id=p1[i].doc_id, tf=p1[i].tf + p2[j].tf))
            i += 1
            j += 1
        elif p1[i].doc_id < p2[j].doc_id:
            i += 1
        else:
            j += 1
    return result


def intersect_set(p1: list[Posting], p2: list[Posting]) -> list[Posting]:
    """Перетин через множини (set intersection)."""
    map1 = {p.doc_id: p for p in p1}
    map2 = {p.doc_id: p for p in p2}
    common_ids = set(map1.keys()) & set(map2.keys())
    return [
        Posting(doc_id=doc_id, tf=map1[doc_id].tf + map2[doc_id].tf)
        for doc_id in sorted(common_ids)
    ]


def search_boolean(
    index: InvertedIndex, query: str, engine: str = "merge"
) -> list[Posting]:
    from findex.tokenize import tokenize

    terms = list(tokenize(query))
    if not terms:
        return []

    result = index.get_postings(terms[0])
    intersect_fn = intersect_merge if engine == "merge" else intersect_set

    for term in terms[1:]:
        next_postings = index.get_postings(term)
        result = intersect_fn(result, next_postings)

    return result


def main():
    parser = argparse.ArgumentParser(description="Search in inverted index.")
    parser.add_argument("index_path", type=Path, help="Path to index file")
    parser.add_argument("query", type=str, help="Search query (AND space-separated)")
    parser.add_argument(
        "--engine",
        choices=["merge", "set"],
        default="merge",
        help="Intersection engine",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    from findex.storage import load_index

    idx = load_index(args.index_path)

    results = search_boolean(idx, args.query, engine=args.engine)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"=== Search Results (engine: {args.engine}) ===")
    print(f"Query: '{args.query}'")
    print(f"Found matches: {len(results)}")
    for res in results:
        meta = idx.doc_meta.get(res.doc_id)
        path_str = f" ({meta.path})" if meta and meta.path else ""
        print(f"  - doc_id: {res.doc_id}{path_str} | tf_sum: {res.tf}")

    print(f"\nElapsed time: {elapsed:.6f} s")
    print(f"Peak memory:  {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()