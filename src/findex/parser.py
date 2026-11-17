"""Recursive descent parser for boolean and phrase search queries."""

import re

from findex.ast import And, ASTNode, Not, Or, Phrase, Term

_TOKEN_SPECIFICATION: list[tuple[str, str]] = [
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("PHRASE", r'"[^"]*"'),
    ("AND", r"\bAND\b"),
    ("OR", r"\bOR\b"),
    ("NOT", r"\bNOT\b"),
    ("WORD", r'[^\s()"]+'),
]

_TOK_REGEX: re.Pattern[str] = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPECIFICATION),
    flags=re.UNICODE,
)


def tokenize_query(query: str) -> list[tuple[str, str]]:
    """Tokenize a raw boolean query string into (token_kind, token_value) pairs."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    for match in _TOK_REGEX.finditer(query):
        gap = query[pos : match.start()]
        if gap.strip():
            raise ValueError(f"Invalid syntax near: {gap.strip()}")
        pos = match.end()

        kind = match.lastgroup
        if kind is None:
            continue
        value = match.group()
        if kind == "PHRASE":
            value = value[1:-1]
        tokens.append((kind, value))

    trailing = query[pos:]
    if trailing.strip():
        raise ValueError(f"Invalid syntax near: {trailing.strip()}")

    return tokens


class QueryParser:
    """Recursive descent parser producing an ASTNode tree."""

    def __init__(self, query: str) -> None:
        self.tokens: list[tuple[str, str]] = tokenize_query(query)
        self.pos: int = 0

    def current_token(self) -> tuple[str, str] | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def parse(self) -> ASTNode:
        if not self.tokens:
            raise ValueError("Empty query")
        node = self._parse_or()
        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos][1]}")
        return node

    def _parse_or(self) -> ASTNode:
        left = self._parse_and()
        while self.pos < len(self.tokens) and self.tokens[self.pos][0] == "OR":
            self.pos += 1
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self) -> ASTNode:
        left = self._parse_not()
        while self.pos < len(self.tokens):
            kind, _ = self.tokens[self.pos]
            if kind == "AND":
                self.pos += 1
                right = self._parse_not()
                left = And(left, right)
            elif kind in ("WORD", "PHRASE", "NOT", "LPAREN"):
                right = self._parse_not()
                left = And(left, right)
            else:
                break
        return left

    def _parse_not(self) -> ASTNode:
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == "NOT":
            self.pos += 1
            operand = self._parse_not()
            return Not(operand)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        tok = self.current_token()
        if tok is None:
            raise ValueError("Unexpected end of query")

        kind, val = tok
        if kind == "LPAREN":
            self.pos += 1
            node = self._parse_or()
            if self.pos >= len(self.tokens) or self.tokens[self.pos][0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return node
        if kind == "PHRASE":
            self.pos += 1
            words = val.split()
            if not words:
                raise ValueError("Empty phrase in query")
            return Phrase(words)
        if kind == "WORD":
            self.pos += 1
            return Term(val)
        raise ValueError(f"Unexpected token: {val}")


def parse_query(query: str) -> ASTNode:
    """Parse a query string into an ASTNode."""
    return QueryParser(query).parse()