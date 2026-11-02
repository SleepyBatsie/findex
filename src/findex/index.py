import argparse

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
import tracemalloc
from typing import Iterator

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: str
    tf: int  # Частота терміна в документі


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: str
    length: int  # Загальна кількість токенів у документі
    path: str | None


class InvertedIndex:

    def __init__(self, with_positions: bool = False):
        self.with_positions = with_positions
        # dict[term -> dict[doc_id -> tf або list[positions]]]
        self.index: dict[str, dict[str, int | list[int]]] = defaultdict(dict)
        self.doc_meta: dict[str, DocMeta] = {}

    def build(self, corpus_stream: Iterator) -> None:
        """Будує індекс одним проходом по ледачому пайплайну з Лаби 1."""
        for doc in corpus_stream:
            tokens = list(tokenize(doc.text))
            doc_len = len(tokens)
            self.doc_meta[doc.doc_id] = DocMeta(
                doc_id=doc.doc_id,
                length=doc_len,
                path=str(doc.path) if doc.path else None,
            )

            if not self.with_positions:
                term_counts: dict[str, int] = defaultdict(int)
                for token in tokens:
                    term_counts[token] += 1
                for term, count in term_counts.items():
                    self.index[term][doc.doc_id] = count
            else:
                for pos, token in enumerate(tokens):
                    if doc.doc_id not in self.index[token]:
                        self.index[token][doc.doc_id] = []
                    self.index[token][doc.doc_id].append(pos)  # type: ignore

    def get_postings(self, term: str) -> list[Posting]:
        """Повертає відсортований за doc_id список постінгів."""
        if term not in self.index:
            return []

        raw_postings = self.index[term]
        if self.with_positions:
            postings = [
                Posting(doc_id=doc_id, tf=len(pos_list))  # type: ignore
                for doc_id, pos_list in raw_postings.items()
            ]
        else:
            postings = [
                Posting(doc_id=doc_id, tf=count)  # type: ignore
                for doc_id, count in raw_postings.items()
            ]

        # Важливо: сортуємо за doc_id для подальшого dwo-pointer merge
        return sorted(postings, key=lambda p: p.doc_id)


def main():
    parser = argparse.ArgumentParser(
        description="Build and save inverted index."
    )
    parser.add_argument(
        "data_dir", type=Path, help="Path to corpus directory/file"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("index.bin"),
        help="Output index file path",
    )
    parser.add_argument(
        "--format",
        choices=["pickle", "json"],
        default="pickle",
        help="Serialization format",
    )
    args = parser.parse_args()

    tracemalloc.start()
    start_time = time.perf_counter()

    corpus_stream = iter_documents(args.data_dir)
    idx = InvertedIndex()
    idx.build(corpus_stream)

    from findex.storage import save_index

    save_index(idx, args.out, format=args.format)

    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"=== Indexing Complete ===")
    print(f"Documents indexed: {len(idx.doc_meta)}")
    print(f"Unique terms:     {len(idx.index)}")
    print(f"Output file:      {args.out}")
    print(f"Elapsed time:     {elapsed:.4f} s")
    print(f"Peak memory:      {peak / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()