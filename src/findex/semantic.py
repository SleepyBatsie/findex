"""Dense semantic embeddings, chunking, cosine search, and Reciprocal Rank Fusion (RRF)."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from findex.index import InvertedIndex
from findex.search import SearchResult

logger = logging.getLogger(__name__)

SearchMode = Literal["lexical", "semantic", "hybrid"]
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0).astype(np.float32)
    return (matrix / safe_norms).astype(np.float32)


@dataclass
class SemanticIndex:
    embeddings: NDArray[np.float32]
    chunk_doc_ids: list[str]
    model_name: str = DEFAULT_EMBED_MODEL

    def save(self, npy_path: Path | str) -> Path:
        target = Path(npy_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(target, self.embeddings.astype(np.float32))
        meta_path = target.with_suffix(".meta.json")
        meta_payload = {
            "model_name": self.model_name,
            "num_chunks": len(self.chunk_doc_ids),
            "dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
            "chunk_doc_ids": self.chunk_doc_ids,
        }
        meta_path.write_text(
            json.dumps(meta_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, npy_path: Path | str) -> "SemanticIndex":
        source = Path(npy_path)
        if not source.is_file():
            raise FileNotFoundError(f"Embeddings file not found: {source}")
        meta_path = source.with_suffix(".meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Embeddings metadata file not found: {meta_path}")
        embeddings = np.load(source).astype(np.float32)
        meta = cast(dict[str, object], json.loads(meta_path.read_text(encoding="utf-8")))
        raw_ids = cast(list[object], meta.get("chunk_doc_ids", []))
        return cls(
            embeddings=embeddings,
            chunk_doc_ids=[str(x) for x in raw_ids],
            model_name=str(meta.get("model_name", DEFAULT_EMBED_MODEL)),
        )

    def search_by_vector(
        self,
        query_vec: NDArray[np.float32],
        top_k: int = 10,
    ) -> list[SearchResult]:
        if top_k <= 0 or self.embeddings.size == 0 or not self.chunk_doc_ids:
            return []
        q_norm = l2_normalize(query_vec.astype(np.float32).reshape(1, -1)).reshape(-1)
        chunk_scores = self.embeddings @ q_norm
        best_doc_scores: dict[str, float] = {}
        for doc_id, score_val in zip(self.chunk_doc_ids, chunk_scores.tolist(), strict=True):
            score_f = float(score_val)
            prev = best_doc_scores.get(doc_id)
            if prev is None or score_f > prev:
                best_doc_scores[doc_id] = score_f
        ranked_pairs = sorted(best_doc_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [SearchResult(score=score, doc_id=doc_id) for doc_id, score in ranked_pairs]


_MODEL_CACHE: dict[str, object] = {}


def get_embedding_model(model_name: str = DEFAULT_EMBED_MODEL) -> object:
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def encode_texts(texts: list[str], model_name: str = DEFAULT_EMBED_MODEL) -> NDArray[np.float32]:
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedding_model(model_name)
    raw = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)  # type: ignore[attr-defined]
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return l2_normalize(arr)


def build_semantic_index(
    doc_texts: list[tuple[str, str]],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    chunk_words: int = 120,
    overlap_words: int = 30,
) -> SemanticIndex:
    chunk_doc_ids: list[str] = []
    chunk_strings: list[str] = []
    for doc_id, text in sorted(doc_texts, key=lambda item: item[0]):
        pieces = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words) or [doc_id]
        for piece in pieces:
            chunk_doc_ids.append(doc_id)
            chunk_strings.append(piece)
    embeddings = encode_texts(chunk_strings, model_name=model_name)
    return SemanticIndex(embeddings=embeddings, chunk_doc_ids=chunk_doc_ids, model_name=model_name)


def reciprocal_rank_fusion(
    lexical_results: list[SearchResult],
    semantic_results: list[SearchResult],
    *,
    rrf_k: int = 60,
    top_k: int = 10,
) -> list[SearchResult]:
    if top_k <= 0:
        return []
    fused_scores: dict[str, float] = {}
    for rank, hit in enumerate(lexical_results, start=1):
        fused_scores[hit.doc_id] = fused_scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, hit in enumerate(semantic_results, start=1):
        fused_scores[hit.doc_id] = fused_scores.get(hit.doc_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(fused_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    return [SearchResult(score=score, doc_id=doc_id) for doc_id, score in ordered]


def search_semantic(sem_index: SemanticIndex, query: str, top_k: int = 10) -> list[SearchResult]:
    if not query.strip() or top_k <= 0:
        return []
    q_vec = encode_texts([query], model_name=sem_index.model_name)[0]
    return sem_index.search_by_vector(q_vec, top_k=top_k)


def search_hybrid(
    index: InvertedIndex,
    sem_index: SemanticIndex,
    query: str,
    *,
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    from findex.search import search_ranked

    pool_k = max(top_k * 3, 50)
    lex_hits = search_ranked(index, query, scorer="bm25", top_k=pool_k)
    sem_hits = search_semantic(sem_index, query, top_k=pool_k)
    return reciprocal_rank_fusion(lex_hits, sem_hits, rrf_k=rrf_k, top_k=top_k)
