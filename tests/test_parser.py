"""Unit tests for query parser AST tree structure and evaluation."""

import pytest

from findex.ast import And, Not, Or, Phrase, Term
from findex.index import InvertedIndex
from findex.parser import parse_query


def test_parser_builds_expected_ast_tree() -> None:
    parsed = parse_query('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(["event", "loop"]),
    )
    assert parsed == expected


def test_parser_operator_precedence() -> None:
    parsed = parse_query("python OR java AND NOT legacy")
    expected = Or(Term("python"), And(Term("java"), Not(Term("legacy"))))
    assert parsed == expected


@pytest.mark.parametrize(
    "bad_query",
    [
        "",
        "   ",
        "python AND (async OR await",
        "python ) async",
        '""',
    ],
)
def test_parser_rejects_invalid_queries(bad_query: str) -> None:
    with pytest.raises(ValueError):
        parse_query(bad_query)


def test_ast_evaluation_on_fixture(built_index: InvertedIndex) -> None:
    ast = parse_query('python AND "event loop"')
    assert ast.evaluate(built_index) == {"doc1", "doc5"}
