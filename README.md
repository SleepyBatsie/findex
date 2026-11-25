# findex v1.0.0 ? Full-Text & Semantic Search Engine

`findex` is a search engine in Python 3.13 combining a **NumPy-vectorized BM25 / TF-IDF inverted index** with **dense embeddings (`all-MiniLM-L6-v2`)**, **Reciprocal Rank Fusion (RRF)**, a **FastAPI + Jinja2 web UI**, a **Typer + Rich CLI**, and an **async crawler**.

## Performance Optimization (Lab 8)

### 1. Profiling Hotspots (`cProfile` + Flame Graphs)
- **`search_ranked` + `BM25Scorer.score` (`0.836s` + `0.395s` / 500 runs):** Python interpreter loop overhead and 750k scalar calls (`docs/search_flamegraph.svg`).
- **`InvertedIndex.doc_len` + `heapq.nlargest` (`0.154s` + `0.040s`):** 750k dict lookups and full heap sorting.
- **`InvertedIndex.build` + `iter_tokens` (`0.090s` + `0.074s`):** 375k `dict.setdefault` allocations (`docs/index_flamegraph.svg`).

### 2. NumPy Vectorization (`np.int32` + `np.argpartition`)
| Query Type (2,000 docs) | Python (ms) | NumPy (ms) | Speedup |
| :--- | ---: | ---: | ---: |
| Frequent term (`df=1500`) | 2.1943 | 0.1083 | **20.26x** |
| 3-term query | 3.3518 | 0.1320 | **25.40x** |
| Rare term (`df=5`) | 0.0077 | 0.0311 | 0.25x |

## Semantic & Hybrid Search (`Precision@5`)
| Query Subset | Lexical (BM25) | Semantic (MiniLM) | Hybrid (RRF) |
| :--- | ---: | ---: | ---: |
| 10 Exact-Term Queries (`Q01-Q10`) | 0.200 | 0.200 | 0.200 |
| 5 Paraphrased Queries (`Q11-Q15`) | 0.040 | **0.200** | **0.200** |
| **Mean (All 15 Queries)** | 0.147 | **0.200** | **0.200** |

## CLI & Web Quickstart
```bash
uv sync
findex index ./data -o data/index.json
findex embed data/index.json -o data/embeddings.npy
findex search data/index.json "running background tasks" --mode hybrid
findex serve --host 0.0.0.0 --port 8000
```
