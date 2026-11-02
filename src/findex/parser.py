# src/findex/parser.py
import re
from typing import List, Tuple, Optional
from findex.ast import ASTNode, Term, Phrase, And, Or, Not


def tokenize_query(query: str) -> List[Tuple[str, str]]:
    """
    Розбиває рядок запиту на токени:
    LPAREN, RPAREN, PHRASE, AND, OR, NOT, WORD.
    """
    token_specification = [
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("PHRASE", r'"[^"]*"'),
        ("AND", r"\bAND\b"),
        ("OR", r"\bOR\b"),
        ("NOT", r"\bNOT\b"),
        ("WORD", r'[^\s()"]+'),
    ]
    tok_regex = "|".join(f"(?P<{pair[0]}>{pair[1]})" for pair in token_specification)
    tokens = []
    for mo in re.finditer(tok_regex, query):
        kind = mo.lastgroup
        value = mo.group()
        if kind == "PHRASE":
            value = value[1:-1]  # Прибираємо зовнішні лапки
        tokens.append((kind, value))
    return tokens


class QueryParser:
    """Парсер мови запитів методом рекурсивного спуску."""

    def __init__(self, query: str) -> None:
        self.tokens: List[Tuple[str, str]] = tokenize_query(query)
        self.pos: int = 0

    def current_token(self) -> Optional[Tuple[str, str]]:
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
        if not tok:
            raise ValueError("Unexpected end of query")

        kind, val = tok
        if kind == "LPAREN":
            self.pos += 1
            node = self._parse_or()
            if self.pos >= len(self.tokens) or self.tokens[self.pos][0] != "RPAREN":
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return node
        elif kind == "PHRASE":
            self.pos += 1
            words = val.split()
            return Phrase(words)
        elif kind == "WORD":
            self.pos += 1
            return Term(val)
        else:
            raise ValueError(f"Unexpected token: {val}")


def parse_query(query: str) -> ASTNode:
    """Зручна функція-обгортка для парсингу рядка запиту в AST."""
    return QueryParser(query).parse()