"""Parametrized unit tests for Unicode tokenization."""

import pytest

from findex.tokenize import iter_tokens, tokenize


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("", []),
        ("   \t\n  ", []),
        ("Hello, World!", ["hello", "world"]),
        ("Python 3.13 & Async/Await", ["python", "3", "13", "async", "await"]),
        ("Привіт, світе! Їжак і ґанок.", ["привіт", "світе", "їжак", "і", "ґанок"]),
        ("Café naïve résumé über", ["café", "naïve", "résumé", "über"]),
        ("日本語テスト alpha", ["日本語テスト", "alpha"]),
    ],
)
def test_tokenize_parametrized_unicode(raw_text: str, expected: list[str]) -> None:
    assert tokenize(raw_text) == expected


def test_iter_tokens_is_lazy_iterator() -> None:
    it = iter_tokens("one two three")
    assert next(it) == "one"
    assert list(it) == ["two", "three"]
