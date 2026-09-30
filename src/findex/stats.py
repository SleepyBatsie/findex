import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
import time
import tracemalloc

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def run_streaming_stats(corpus_path: Path, limit: int | None = None):
    doc_stream = iter_documents(corpus_path)
    if limit is not None:
        doc_stream = islice(doc_stream, limit)

    num_docs = 0
    total_tokens = 0
    term_counts = Counter()

    for doc in doc_stream:
        num_docs += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            term_counts[token] += 1

    return {
        "num_docs": num_docs,
        "total_tokens": total_tokens,
        "vocab_size": len(term_counts),
        "top_50": term_counts.most_common(50),
    }


def run_eager_stats(corpus_path: Path, limit: int | None = None):
    doc_stream = iter_documents(corpus_path)
    if limit is not None:
        doc_stream = islice(doc_stream, limit)

    documents = list(doc_stream)
    all_tokens_by_doc = [list(tokenize(doc.text)) for doc in documents]

    num_docs = len(documents)
    all_tokens_flat = [token for doc_tokens in all_tokens_by_doc for token in doc_tokens]
    total_tokens = len(all_tokens_flat)
    term_counts = Counter(all_tokens_flat)

    return {
        "num_docs": num_docs,
        "total_tokens": total_tokens,
        "vocab_size": len(term_counts),
        "top_50": term_counts.most_common(50),
    }


def main():
    parser = argparse.ArgumentParser(description="Corpus stats pipeline")
    parser.add_argument("corpus_path", type=Path, help="Path to corpus directory or .jsonl file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of documents")
    parser.add_argument("--eager", action="store_true", help="Run eager version for comparison")
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    if args.eager:
        stats = run_eager_stats(args.corpus_path, args.limit)
    else:
        stats = run_streaming_stats(args.corpus_path, args.limit)

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    mode = "EAGER" if args.eager else "STREAMING"
    print(f"=== Results ({mode} mode) ===")
    print(f"Documents processed: {stats['num_docs']:,}")
    print(f"Total tokens:        {stats['total_tokens']:,}")
    print(f"Vocabulary size:     {stats['vocab_size']:,}")
    print(f"Elapsed time:        {elapsed_time:.4f} s")
    print(f"Peak memory usage:   {peak_memory / (1024 * 1024):.2f} MB")
    print("\nTop 10 Terms:")
    for term, count in stats["top_50"][:10]:
        print(f"  {term}: {count:,}")


if __name__ == "__main__":
    main()