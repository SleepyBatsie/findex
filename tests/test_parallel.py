"""Tests for parallel indexing: build_partial, merge, executors, spawn safety, and worker errors."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from findex.cli import app
from findex.corpus import discover_files, load_corpus
from findex.index import (
    ExecutorName,
    InvertedIndex,
    build_index,
    build_partial,
    chunk_paths,
    merge,
)


def _failing_partial(
    paths: list[Path],
    base_dir: Path | None = None,
    with_positions: bool = True,
) -> InvertedIndex:
    """Top-level helper picklable by ProcessPoolExecutor to simulate worker crash."""
    raise RuntimeError("Simulated worker crash inside build_partial")


def test_merge_single_partial_matches_lab4_serial_build(corpus_dir: Path) -> None:
    """merge([build_partial(all_paths)]) must produce the exact same index as Lab 4."""
    docs = load_corpus(corpus_dir)
    lab4_index = InvertedIndex()
    lab4_index.build(docs)

    all_paths = discover_files(corpus_dir)
    merged = merge([build_partial(all_paths, base_dir=corpus_dir)])

    assert merged == lab4_index
    assert merged.avg_doc_length == pytest.approx(lab4_index.avg_doc_length)


def test_chunk_paths_disjoint_and_exhaustive(tmp_path: Path) -> None:
    paths = [tmp_path / f"doc_{i:03d}.txt" for i in range(11)]
    chunks = chunk_paths(paths, num_chunks=4)
    assert len(chunks) == 4
    flattened = [p for chunk in chunks for p in chunk]
    assert flattened == paths
    assert len(set(flattened)) == len(paths)


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2])
def test_build_index_executors_produce_identical_index(
    tmp_path: Path,
    executor: ExecutorName,
    workers: int,
) -> None:
    """Serial, threads, and spawn-safe processes must build identical indexes."""
    cdir = tmp_path / "multi_corpus"
    cdir.mkdir()
    for i in range(8):
        (cdir / f"file_{i:02d}.txt").write_text(
            f"Document {i} about python concurrency threads processes and indexing",
            encoding="utf-8",
        )

    expected = InvertedIndex()
    expected.build(load_corpus(cdir))

    built = build_index(cdir, workers=workers, executor=executor)
    assert built == expected


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
def test_worker_failure_propagates_and_does_not_hang(
    corpus_dir: Path,
    executor: ExecutorName,
) -> None:
    """When a worker raises an exception, build_index fails immediately with traceback."""
    with (
        patch("findex.index.build_partial", _failing_partial),
        pytest.raises(RuntimeError, match="Simulated worker crash"),
    ):
        build_index(corpus_dir, workers=2, executor=executor)


def test_cli_index_parallel_flags_and_worker_error(tmp_path: Path, corpus_dir: Path) -> None:
    runner = CliRunner()
    out_idx = tmp_path / "parallel_idx.json"

    res = runner.invoke(
        app,
        ["index", str(corpus_dir), "-o", str(out_idx), "--workers", "2", "--executor", "threads"],
    )
    assert res.exit_code == 0
    assert out_idx.exists()

    with patch("findex.index.build_partial", _failing_partial):
        fail_res = runner.invoke(
            app,
            [
                "index",
                str(corpus_dir),
                "-o",
                str(tmp_path / "fail.json"),
                "--workers",
                "2",
                "--executor",
                "threads",
            ],
        )
    assert fail_res.exit_code != 0
    assert "Traceback" in fail_res.stderr
    assert "Simulated worker crash" in fail_res.stderr
