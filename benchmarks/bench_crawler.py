# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""
Benchmark & failure-handling demo for Lab 6 (Asyncio Crawler):
- Runs identical start URL and max_pages=220 across concurrency [1, 5, 20]
- Measures Wall time, Pages/sec, Errors, and Peak RSS (MB)
- Demonstrates handled 429 Retry-After, TimeoutError, and redirect loop in crawl.log
"""

import asyncio
import ctypes
import importlib
import socket
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import httpx

from findex.crawler import CrawlStats, crawl_to_jsonl


def get_peak_rss_mb() -> float:
    """Return peak resident set size (RSS) of the current process in MiB."""
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_bool

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float(counters.PeakWorkingSetSize) / (1024 * 1024)
        return 0.0

    resource = importlib.import_module("resource")
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024.0


class _DocsMirrorHandler(BaseHTTPRequestHandler):
    """Local HTTP mirror serving 260 interlinked docs pages + fault-injection endpoints."""

    rate_limit_hits: int = 0

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/robots.txt":
            body = b"User-agent: *\nAllow: /docs/\nDisallow: /admin/\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/fault/429":
            _DocsMirrorHandler.rate_limit_hits += 1
            if _DocsMirrorHandler.rate_limit_hits <= 2:
                body = b"Too Many Requests"
                self.send_response(429)
                self.send_header("Retry-After", "0.15")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            html = (
                b"<html><head><title>Recovered 429</title></head>"
                b"<body><p>Page recovered after Retry-After backoff</p></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/fault/loop-a":
            self.send_response(302)
            self.send_header("Location", "/fault/loop-b")
            self.end_headers()
            return

        if self.path == "/fault/loop-b":
            self.send_response(302)
            self.send_header("Location", "/fault/loop-a")
            self.end_headers()
            return

        if self.path.startswith("/docs/page_"):
            time.sleep(0.03)
            try:
                num = int(self.path.split("_")[-1])
            except ValueError:
                num = 0
            links = "".join(
                f'<a href="/docs/page_{(num + step) % 260}">Page {(num + step) % 260}</a> '
                for step in (1, 2, 3, 7)
            )
            html_str = (
                f"<html><head><title>Doc {num}</title></head><body>"
                f"<h1>Documentation Section {num}</h1>"
                f"<p>Asynchronous I/O event loop coroutines and HTTP concurrency guide {num}.</p>"
                f"{links}</body></html>"
            )
            encoded = html_str.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return

        self.send_response(404)
        self.end_headers()


class run_local_mirror:  # noqa: N801
    """Context manager running local ThreadingHTTPServer on an ephemeral port."""

    def __init__(self) -> None:
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            addr: tuple[Any, ...] = s.getsockname()
            port = int(addr[1])

        self._server = ThreadingHTTPServer(("127.0.0.1", port), _DocsMirrorHandler)
        thread = Thread(target=self._server.serve_forever, daemon=True)
        thread.start()
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *args: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


async def run_benchmarks() -> None:
    max_pages = 220
    concurrency_levels = [1, 5, 20]

    with run_local_mirror() as base_url, tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        start_url = f"{base_url}/docs/page_0"

        print(f"Start URL: {start_url} | max_pages={max_pages}")
        print("\n=== Crawler Concurrency Benchmark ===")
        print("| Concurrency | Wall time (s) | Pages / sec | Errors | Peak RSS (MB) |")
        print("|---:|---:|---:|---:|---:|")

        for conc in concurrency_levels:
            out_jsonl = tmp / f"crawl_c{conc}.jsonl"
            out_log = tmp / f"crawl_c{conc}.log"
            stats = CrawlStats()
            t0 = time.perf_counter()
            await crawl_to_jsonl(
                start_url,
                out_jsonl,
                log_path=out_log,
                max_pages=max_pages,
                concurrency=conc,
                per_host_concurrency=conc,
                min_delay=0.0,
                timeout=5.0,
                stats=stats,
            )
            wall = time.perf_counter() - t0
            pps = stats.pages_collected / wall
            rss = get_peak_rss_mb()
            print(f"| {conc} | {wall:.4f} | {pps:.2f} | {stats.errors} | {rss:.2f} |")

        _DocsMirrorHandler.rate_limit_hits = 0
        fault_jsonl = tmp / "faults.jsonl"
        fault_log = tmp / "faults.log"
        async with httpx.AsyncClient(follow_redirects=True, max_redirects=5) as client:
            await crawl_to_jsonl(
                [f"{base_url}/fault/429", f"{base_url}/fault/loop-a"],
                fault_jsonl,
                log_path=fault_log,
                max_pages=5,
                concurrency=2,
                per_host_concurrency=2,
                min_delay=0.0,
                timeout=2.0,
                client=client,
            )

        print("\n=== Handled Failures Log Sample (crawl.log) ===")
        print(fault_log.read_text(encoding="utf-8").strip())


def main() -> None:
    asyncio.run(run_benchmarks())


if __name__ == "__main__":
    main()