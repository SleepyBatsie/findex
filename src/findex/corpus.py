import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Document:
    doc_id: str
    path: Path | None
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """
    Ледачий генератор: відкриває по одному файлу/рядку за раз,
    ігнорує помилки кодування та повертає об'єкти Document.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    # Варіант 1: Потокове читання JSON-lines (.jsonl)
    if root.is_file() and root.suffix == ".jsonl":
        with open(root, "r", encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    doc_id = str(data.get("id", f"line_{line_idx}"))
                    text = data.get("text", "")
                    yield Document(doc_id=doc_id, path=root, text=text)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing JSON at line {line_idx}: {e}")
                    continue

    # Варіант 2: Директорія з файлами .txt (rglob)
    elif root.is_dir():
        for path in root.rglob("*.txt"):
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    yield Document(doc_id=path.stem, path=path, text=text)
                except Exception as e:
                    logger.warning(f"Error reading file {path}: {e}")
                    continue
    else:
        raise ValueError(f"Unsupported path structure: {root}")