# pyright: reportUnknownMemberType=false
"""Benchmark & GIL experiments for Lab 5."""

import ctypes
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict, cast

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = cast(Any, None)

from findex.index import BuildMetrics, ExecutorName, build_index

RESULTS_JSON = Path("benchmarks/results_data.json")
CHART_PNG = Path("docs/speedup_chart.png")


class CellResult(TypedDict):
    workers: int
    wall_s: float
    cpu_s: float
    peak_rss_mb: float
    merge_s: float
    speedup: float


class BenchmarkReport(TypedDict):
    cpu_model: str
    cpu_count: int
    python_version: str
    worker_counts: list[int]
    series: dict[str, list[CellResult]]


def is_gil_enabled() -> bool:
    checker = getattr(sys, "_is_gil_enabled", None)
    return bool(checker()) if callable(checker) else True


def get_cpu_model() -> str:
    if sys.platform == "win32":
        try:
            cmd = "(Get-CimInstance Win32_Processor).Name"
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", cmd],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return platform.processor() or "Unknown CPU"


def get_peak_rss_mb() -> float:
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
            return counters.PeakWorkingSetSize / (1024 * 1024)
        return 0.0

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024.0


def prepare_benchmark_corpus(target_dir: Path, num_docs: int = 4000, words: int = 800) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    vocab = [
        "python", "concurrency", "parallelism", "threading", "multiprocessing",
        "interpreter", "gil", "lock", "mutex", "atomic", "process", "spawn",
        "pickle", "serialization", "index", "inverted", "postings", "bm25",
        "tfidf", "ranking", "search", "query", "parser", "ast", "token",
        "unicode", "latency", "throughput", "amdahl", "speedup", "memory",
        "cache", "cpu", "kernel", "scheduler", "worker", "queue", "future",
        "executor", "async", "await", "io", "disk", "stream", "індекс",
    ]
    for i in range(num_docs):
        text = " ".join(rng.choice(vocab) for _ in range(words))
        (target_dir / f"doc_{i:04d}.txt").write_text(text, encoding="utf-8")


def measure_cell(
    corpus_dir: Path,
    executor: ExecutorName,
    workers: int,
    repeats: int = 3,
) -> CellResult:
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    rss_samples: list[float] = []
    merge_samples: list[float] = []

    build_index(corpus_dir, workers=workers, executor=executor)

    for _ in range(repeats):
        metrics_box: list[BuildMetrics] = []
        t_wall_0 = time.perf_counter()
        t_cpu_0 = time.process_time()
        build_index(corpus_dir, workers=workers, executor=executor, metrics_out=metrics_box)
        wall_samples.append(time.perf_counter() - t_wall_0)
        cpu_samples.append(time.process_time() - t_cpu_0)
        rss_samples.append(get_peak_rss_mb())
        merge_samples.append(metrics_box[0].merge_seconds if metrics_box else 0.0)

    return CellResult(
        workers=workers,
        wall_s=round(statistics.median(wall_samples), 4),
        cpu_s=round(statistics.median(cpu_samples), 4),
        peak_rss_mb=round(statistics.median(rss_samples), 2),
        merge_s=round(statistics.median(merge_samples), 4),
        speedup=1.0,
    )


def demonstrate_gil_race_condition() -> None:
    print("\n=== Race Condition Demo ===")
    print(f"Python: {sys.version.split()[0]} | sys._is_gil_enabled() = {is_gil_enabled()}")
    expected = 4 * 5_000
    unsafe_counter = 0

    def unsafe_worker() -> None:
        nonlocal unsafe_counter
        for _ in range(5_000):
            tmp = unsafe_counter
            time.sleep(0)
            unsafe_counter = tmp + 1

    threads = [threading.Thread(target=unsafe_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    safe_counter = 0
    lock = threading.Lock()

    def safe_worker() -> None:
        nonlocal safe_counter
        local_count = 0
        for _ in range(5_000):
            local_count += 1
        with lock:
            safe_counter += local_count

    safe_threads = [threading.Thread(target=safe_worker) for _ in range(4)]
    for t in safe_threads:
        t.start()
    for t in safe_threads:
        t.join()

    lost = expected - unsafe_counter
    print(f"Unsafe shared counter: {unsafe_counter} / {expected} (lost={lost})")
    print(f"Fixed (worker-local + Lock): {safe_counter} / {expected}")


def demonstrate_io_bound_scaling(corpus_dir: Path) -> None:
    files = sorted(corpus_dir.glob("*.txt"))[:120]

    def io_task(path: Path) -> int:
        time.sleep(0.005)
        return len(path.read_bytes())

    t0 = time.perf_counter()
    serial_bytes = sum(io_task(p) for p in files)
    t_serial = time.perf_counter() - t0

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        threaded_bytes = sum(pool.map(io_task, files))
    t_threads = time.perf_counter() - t1

    assert serial_bytes == threaded_bytes
    print("\n=== I/O-Bound Counter-Experiment (120 files + 5ms I/O wait) ===")
    print(f"Serial (1 worker):   {t_serial:.3f} s")
    print(f"Threads (8 workers): {t_threads:.3f} s (speedup = {t_serial / t_threads:.2f}x)")


def generate_speedup_chart(all_data: BenchmarkReport, output_png: Path) -> None:
    if plt is None:
        print("\n[Info] matplotlib not installed; skipping PNG chart rendering.")
        return

    output_png.parent.mkdir(parents=True, exist_ok=True)
    worker_counts = all_data["worker_counts"]
    series = all_data["series"]

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(
        worker_counts,
        worker_counts,
        linestyle="--",
        color="#666666",
        linewidth=1.5,
        label="Ideal linear speedup (y = x)",
    )

    styles = {
        "threads_gil": ("o-", "#d95f02", "Threads (GIL enabled)"),
        "processes": ("s-", "#1b9e77", "Processes (spawn)"),
        "threads_nogil": ("^-", "#7570b3", "Threads (Python 3.13t, no-GIL)"),
    }
    for key, (marker_style, color, label) in styles.items():
        if key not in series:
            continue
        rows = series[key]
        xs = [row["workers"] for row in rows]
        ys = [row["speedup"] for row in rows]
        plt.plot(xs, ys, marker_style, color=color, linewidth=2.2, markersize=7, label=label)

    plt.title(
        f"findex Index Build Speedup vs. Workers\n"
        f"({all_data['cpu_model']}, {all_data['cpu_count']} logical cores)",
        fontsize=11,
    )
    plt.xlabel("Number of Workers", fontsize=10)
    plt.ylabel("Speedup (T_serial / T_wall)", fontsize=10)
    plt.xticks(worker_counts)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
    print(f"\nSaved speedup chart to {output_png}")


def print_markdown_table(series: dict[str, list[CellResult]]) -> None:
    print("\n=== Benchmark Summary (Median of 3 runs) ===")
    print(
        "| Executor | Workers | Wall time (s) | CPU time (s) "
        "| Peak RSS (MB) | Merge (s) | Speedup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, key in [
        ("serial", "serial"),
        ("threads (GIL)", "threads_gil"),
        ("processes", "processes"),
        ("threads (3.13t no-GIL)", "threads_nogil"),
    ]:
        for row in series.get(key, []):
            print(
                f"| {label} | {row['workers']} | {row['wall_s']:.4f} | "
                f"{row['cpu_s']:.4f} | {row['peak_rss_mb']:.2f} | "
                f"{row['merge_s']:.4f} | {row['speedup']:.2f}x |"
            )


def main() -> None:
    if "--chart-only" in sys.argv:
        report = cast(BenchmarkReport, json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
        print_markdown_table(report["series"])
        generate_speedup_chart(report, CHART_PNG)
        return

    gil_active = is_gil_enabled()
    cpu_model = get_cpu_model()
    cpu_count = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, 8, cpu_count})

    print(f"CPU: {cpu_model} ({cpu_count} logical cores)")
    print(f"Python: {sys.version}")
    print(f"sys._is_gil_enabled(): {gil_active}")

    series: dict[str, list[CellResult]] = {}
    if RESULTS_JSON.exists():
        loaded = cast(BenchmarkReport, json.loads(RESULTS_JSON.read_text(encoding="utf-8")))
        series = loaded.get("series", {})

    with tempfile.TemporaryDirectory() as tmp_dir:
        corpus_path = Path(tmp_dir) / "bench_corpus"
        prepare_benchmark_corpus(corpus_path, num_docs=4000, words=800)

        if gil_active:
            serial_metrics = measure_cell(corpus_path, "serial", workers=1, repeats=3)
            baseline_wall = serial_metrics["wall_s"]
            series["serial"] = [serial_metrics]

            thread_rows: list[CellResult] = []
            proc_rows: list[CellResult] = []
            for w in worker_counts:
                t_m = measure_cell(corpus_path, "threads", workers=w, repeats=3)
                t_m["speedup"] = round(baseline_wall / t_m["wall_s"], 2)
                thread_rows.append(t_m)

                p_m = measure_cell(corpus_path, "processes", workers=w, repeats=3)
                p_m["speedup"] = round(baseline_wall / p_m["wall_s"], 2)
                proc_rows.append(p_m)

            series["threads_gil"] = thread_rows
            series["processes"] = proc_rows
        else:
            baseline_wall = (
                series["serial"][0]["wall_s"]
                if series.get("serial")
                else measure_cell(corpus_path, "serial", workers=1, repeats=3)["wall_s"]
            )
            nogil_rows: list[CellResult] = []
            for w in worker_counts:
                t_m = measure_cell(corpus_path, "threads", workers=w, repeats=3)
                t_m["speedup"] = round(baseline_wall / t_m["wall_s"], 2)
                nogil_rows.append(t_m)
            series["threads_nogil"] = nogil_rows

        all_data: BenchmarkReport = {
            "cpu_model": cpu_model,
            "cpu_count": cpu_count,
            "python_version": sys.version.split()[0],
            "worker_counts": worker_counts,
            "series": series,
        }
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON.write_text(json.dumps(all_data, indent=2), encoding="utf-8")

        print_markdown_table(series)
        generate_speedup_chart(all_data, CHART_PNG)
        demonstrate_gil_race_condition()
        demonstrate_io_bound_scaling(corpus_path)


if __name__ == "__main__":
    main()