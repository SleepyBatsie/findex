"""Abstract Syntax Tree (AST) nodes for boolean and phrase queries."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from findex.index import InvertedIndex
from findex.tokenize import tokenize


class ASTNode(ABC):
    """Base class for all query AST nodes."""

    @abstractmethod
    def evaluate(self, index: InvertedIndex) -> set[str]:
        """Evaluate node against an InvertedIndex and return matching doc_ids."""

    def __and__(self, other: "ASTNode") -> "And":
        return And(self, other)

    def __or__(self, other: "ASTNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Term(ASTNode):
    """Single term leaf node."""

    term: str

    def evaluate(self, index: InvertedIndex) -> set[str]:
        tokens = tokenize(self.term)
        if not tokens:
            return set()
        clean_term = tokens[0]
        if clean_term in index:
            return set(index[clean_term].keys())
        return set()


@dataclass(frozen=True)
class Phrase(ASTNode):
    """Exact sequential phrase node using positional postings."""

    words: list[str]

    def evaluate(self, index: InvertedIndex) -> set[str]:
        clean_words: list[str] = []
        for word in self.words:
            clean_words.extend(tokenize(word))

        if not clean_words:
            return set()

        first_term = clean_words[0]
        if first_term not in index:
            return set()

        candidate_docs = set(index[first_term].keys())
        for word in clean_words[1:]:
            if word not in index:
                return set()
            candidate_docs &= set(index[word].keys())

        matching_docs: set[str] = set()
        for doc_id in candidate_docs:
            pos_lists = [index[word][doc_id] for word in clean_words]
            for start_pos in pos_lists[0]:
                if all(
                    (start_pos + offset) in pos_lists[offset]
                    for offset in range(1, len(clean_words))
                ):
                    matching_docs.add(doc_id)
                    break

        return matching_docs


@dataclass(frozen=True)
class And(ASTNode):
    """Logical AND node."""

    left: ASTNode
    right: ASTNode

    def evaluate(self, index: InvertedIndex) -> set[str]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True)
class Or(ASTNode):
    """Logical OR node."""

    left: ASTNode
    right: ASTNode

    def evaluate(self, index: InvertedIndex) -> set[str]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True)
class Not(ASTNode):
    """Logical NOT (complement) node."""

    operand: ASTNode

    def evaluate(self, index: InvertedIndex) -> set[str]:
        all_docs = set(index.doc_meta.keys())
        return all_docs - self.operand.evaluate(index)
