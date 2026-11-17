"""Unicode-aware tokenization utilities."""

import re
from collections.abc import Iterator

_TOKEN_RE: re.Pattern[str] = re.compile(r"\w+", flags=re.UNICODE)


def iter_tokens(text: str) -> Iterator[str]:
    """Yield normalized lowercase Unicode word tokens from text."""
    for match in _TOKEN_RE.finditer(text.lower()):
        yield match.group(0)


def tokenize(text: str) -> list[str]:
    """Return a list of normalized lowercase Unicode word tokens from text."""
    return list(iter_tokens(text))