import json
from contextlib import contextmanager
from pathlib import Path
import pickle
from typing import Iterator, Union

from findex.index import DocMeta, InvertedIndex


def save_index(index: InvertedIndex, path: Union[str, Path], format: str = "pickle") -> None:
    """Зберігає індекс у форматі pickle або json."""
    path_obj = Path(path)
    if format == "pickle":
        with open(path_obj, "wb") as f:
            pickle.dump(index, f)
    elif format == "json":
        data = {
            "with_positions": getattr(index, "with_positions", True),
            "doc_meta": {
                doc_id: {"length": meta.length, "path": getattr(meta, "path", None)}
                for doc_id, meta in getattr(index, "doc_meta", {}).items()
            },
            "index": getattr(index, "index", getattr(index, "_index", {})),
        }
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


@contextmanager
def open_index(path: Union[str, Path]) -> Iterator[InvertedIndex]:
    """
    Контекстний менеджер для безпечної роботи з файлом індексу.
    Відкриває файл (JSON або Pickle), завантажує об'єкт та гарантовано закриває
    файловий дескриптор у блоці finally.
    """
    path_obj = Path(path)
    is_json = path_obj.suffix == ".json"
    
    mode = "r" if is_json else "rb"
    encoding = "utf-8" if is_json else None

    f = open(path_obj, mode, encoding=encoding)
    try:
        if is_json:
            data = json.load(f)
            idx = InvertedIndex(with_positions=data.get("with_positions", True))
            idx.index = data["index"]
            idx.doc_meta = {
                doc_id: DocMeta(
                    doc_id=doc_id, length=m["length"], path=m.get("path")
                )
                for doc_id, m in data.get("doc_meta", {}).items()
            }
            yield idx
        else:
            yield pickle.load(f)
    finally:
        f.close()


def load_index(path: Union[str, Path]) -> InvertedIndex:
    """
    Завантажує індекс із файлу через контекстний менеджер open_index.
    
    УВАГА (Безпека): Завантаження pickle-файлу з невідомого/чужого джерела є НЕБЕЗПЕЧНИМ,
    оскільки під час розпакування може виконатися довільний Python-код.
    """
    with open_index(path) as index:
        return index