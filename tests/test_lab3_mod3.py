# tests/test_lab3_mod3.py
from findex.ast import And, Not, Or, Phrase, Term
from findex.corpus import Document
from findex.index import Index
from findex.parser import parse_query


def test_ast_operators_overloading():
    """Перевірка перезавантаження побітових операторів (&, |, ~)."""
    t1 = Term("python")
    t2 = Term("async")
    t3 = Term("java")

    ast = t1 & (t2 | ~t3)
    expected = And(t1, Or(t2, Not(t3)))

    assert ast == expected


def test_parser_complex_query():
    """Перевірка парсингу складного запиту в AST (порівняння дерев)."""
    query = 'python AND (async OR await) NOT java "event loop"'
    parsed_ast = parse_query(query)

    expected_ast = And(
        And(And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java"))),
        Phrase(["event", "loop"]),
    )

    assert parsed_ast == expected_ast


def test_ast_evaluation_and_phrase_matching():
    """Перевірка оцінки запитів та фразового пошуку з позиціями."""
    docs = [
        Document("doc1", None, "python event loop is fast"),
        Document("doc2", None, "event python loop is slow"),
        Document("doc3", None, "java event loop is old"),
    ]
    idx = Index()
    idx.build(docs)

    phrase_ast = Phrase(["event", "loop"])
    assert phrase_ast.evaluate(idx) == {"doc1", "doc3"}

    query_ast = parse_query('python AND "event loop" NOT java')
    matching_docs = query_ast.evaluate(idx)

    assert matching_docs == {"doc1"}
