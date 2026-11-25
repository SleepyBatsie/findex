# pyright: reportPrivateUsage=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Benchmark comparing `def` (threadpool) vs `async def` (event-loop blocking)
for 20 concurrent requests on /search with CPU-heavy search/snippet workload.
"""

import asyncio
import math
import time

import httpx
from fastapi import FastAPI

from findex.corpus import Document
from findex.index import InvertedIndex
from findex.web.app import _execute_paginated_search


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, math.ceil((pct / 100.0) * len(sorted_vals)) - 1)
    return sorted_vals[max(0, idx)]


def _build_synthetic_index(num_docs: int = 800) -> InvertedIndex:
    docs = [
        Document(
            f"doc_{i}.txt",
            None,
            (
                f"Document {i} python asyncio event loop fastapi uvicorn threadpool "
                f"gil bm25 search ranking benchmark query token_{i % 50} "
            )
            * 15,
        )
        for i in range(num_docs)
    ]
    idx = InvertedIndex()
    idx.build(docs)
    return idx


def _make_bench_app(idx: InvertedIndex) -> FastAPI:
    bench_app = FastAPI()

    @bench_app.get("/search-def")
    def search_sync_route(q: str = "python asyncio", k: int = 10, page: int = 1) -> dict[str, int]:
        res = _execute_paginated_search(idx=idx, q=q, k=k, scorer="bm25", page=page)
        return {"total_hits": res.total_hits}

    @bench_app.get("/search-async")
    async def search_async_route(
        q: str = "python asyncio", k: int = 10, page: int = 1
    ) -> dict[str, int]:
        res = _execute_paginated_search(idx=idx, q=q, k=k, scorer="bm25", page=page)
        return {"total_hits": res.total_hits}

    @bench_app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    return bench_app


async def _measure_concurrent_batch(
    app: FastAPI,
    endpoint: str,
    concurrent_users: int = 20,
    total_requests: int = 120,
) -> tuple[float, float, float, float, float]:
    transport = httpx.ASGITransport(app=app)
    sem = asyncio.Semaphore(concurrent_users)
    latencies_ms: list[float] = []
    ping_latencies_ms: list[float] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://bench.local") as client:

        async def _one_search() -> None:
            async with sem:
                t0 = time.perf_counter()
                r = await client.get(endpoint, params={"q": "python asyncio", "k": 10, "page": 1})
                r.raise_for_status()
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        async def _background_pings() -> None:
            for _ in range(15):
                t0 = time.perf_counter()
                await client.get("/ping")
                ping_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                await asyncio.sleep(0.005)

        t_start = time.perf_counter()
        await asyncio.gather(
            _background_pings(),
            *[_one_search() for _ in range(total_requests)],
        )
        wall_s = time.perf_counter() - t_start

    latencies_ms.sort()
    rps = total_requests / wall_s
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    p99 = _percentile(latencies_ms, 99)
    max_ping = max(ping_latencies_ms) if ping_latencies_ms else 0.0
    return rps, p50, p95, p99, max_ping


async def main_async() -> None:
    idx = _build_synthetic_index(num_docs=800)
    bench_app = _make_bench_app(idx)

    await _measure_concurrent_batch(bench_app, "/search-def", concurrent_users=5, total_requests=10)

    def_stats = await _measure_concurrent_batch(
        bench_app, "/search-def", concurrent_users=20, total_requests=120
    )
    async_stats = await _measure_concurrent_batch(
        bench_app, "/search-async", concurrent_users=20, total_requests=120
    )

    print("=== 20 Concurrent Requests: `def` (threadpool) vs `async def` (event loop) ===")
    print("| Route Type | Req/sec | p50 (ms) | p95 (ms) | p99 (ms) | Max /ping stall (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    print(
        f"| `def` (threadpool) | {def_stats[0]:.2f} | {def_stats[1]:.2f} | "
        f"{def_stats[2]:.2f} | {def_stats[3]:.2f} | {def_stats[4]:.2f} |"
    )
    print(
        f"| `async def` (blocking loop) | {async_stats[0]:.2f} | {async_stats[1]:.2f} | "
        f"{async_stats[2]:.2f} | {async_stats[3]:.2f} | {async_stats[4]:.2f} |"
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
