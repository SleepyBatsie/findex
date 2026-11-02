import json

from pathlib import Path
import pickle

from findex.index import DocMeta, InvertedIndex


def save_index(index: InvertedIndex, path: Path, format: str = "pickle") -> None:
    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(index, f)
    elif format == "json":
        data = {
            "with_positions": index.with_positions,
            "doc_meta": {
                doc_id: {"length": meta.length, "path": meta.path}
                for doc_id, meta in index.doc_meta.items()
            },
            "index": index.index,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def load_index(path: Path) -> InvertedIndex:
    """
    УВАГА (Безпека): Завантаження pickle-файлу з невідомого/чужого джерела є НЕБЕЗПЕЧНИМ,
    оскільки під час розпакування може виконатися довільний Python-код (Arbitrary Code Execution).
    """
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = InvertedIndex(with_positions=data["with_positions"])
        idx.index = data["index"]
        idx.doc_meta = {
            doc_id: DocMeta(
                doc_id=doc_id, length=m["length"], path=m["path"]
            )
            for doc_id, m in data["doc_meta"].items()
        }
        return idx
    else:
        with open(path, "rb") as f:
            return pickle.load(f)