"""Unit tests for index persistence (save_index, load_index, open_index)."""

from pathlib import Path

import pytest

from findex.index import InvertedIndex
from findex.storage import load_index, open_index, save_index


def test_save_and_load_roundtrip(tmp_path: Path, built_index: InvertedIndex) -> None:
    target = save_index(built_index, tmp_path / "nested" / "idx.json")
    restored = load_index(target)
    assert restored == built_index


def test_open_index_context_manager(tmp_path: Path, built_index: InvertedIndex) -> None:
    target = save_index(built_index, tmp_path / "ctx.json")
    with open_index(target) as loaded:
        assert loaded == built_index


def test_load_missing_or_corrupted_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path / "does_not_exist.json")

    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_index(bad_file)
