"""CLI integration tests using typer.testing.CliRunner."""

import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "search" in result.stdout
    assert "stats" in result.stdout


def test_cli_index_and_search_table(tmp_path: Path, corpus_dir: Path) -> None:
    out_idx = tmp_path / "cli_index.json"
    res_index = runner.invoke(app, ["-v", "index", str(corpus_dir), "-o", str(out_idx)])
    assert res_index.exit_code == 0
    assert out_idx.exists()

    res_search = runner.invoke(app, ["search", str(out_idx), "python async"])
    assert res_search.exit_code == 0
    assert "alpha.txt" in res_search.stdout


def test_cli_search_json_stdout_clean(saved_index_path: Path) -> None:
    res = runner.invoke(
        app,
        ["-vv", "search", str(saved_index_path), "python", "--scorer", "tfidf", "--json"],
    )
    assert res.exit_code == 0
    raw_parsed: object = json.loads(res.stdout)
    assert isinstance(raw_parsed, list)
    parsed = cast(list[dict[str, Any]], raw_parsed)
    assert len(parsed) >= 1
    assert parsed[0]["doc_id"] in {"alpha.txt", "gamma.txt"}
    assert "DEBUG" in res.stderr


def test_cli_search_boolean_and_stats(saved_index_path: Path) -> None:
    res_bool = runner.invoke(
        app,
        ["search", str(saved_index_path), 'python AND "event loop"', "--boolean", "--json"],
    )
    assert res_bool.exit_code == 0
    raw_data: object = json.loads(res_bool.stdout)
    assert isinstance(raw_data, list)
    data = cast(list[dict[str, Any]], raw_data)
    assert [item["doc_id"] for item in data] == ["alpha.txt"]

    res_stats = runner.invoke(app, ["stats", str(saved_index_path), "--top", "5"])
    assert res_stats.exit_code == 0
    assert "Documents" in res_stats.stdout
    assert "Unique Terms" in res_stats.stdout


def test_cli_expected_errors_no_traceback(tmp_path: Path, saved_index_path: Path) -> None:
    missing_res = runner.invoke(app, ["search", str(tmp_path / "missing.json"), "python"])
    assert missing_res.exit_code != 0
    assert "Error:" in missing_res.stderr
    assert "Traceback" not in missing_res.stderr

    broken_query = runner.invoke(
        app,
        ["search", str(saved_index_path), "python AND (", "--boolean"],
    )
    assert broken_query.exit_code != 0
    assert "Error:" in broken_query.stderr
    assert "Traceback" not in broken_query.stderr
