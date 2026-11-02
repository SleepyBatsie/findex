# src/findex/ast.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set
from findex.index import InvertedIndex


class ASTNode(ABC):
    """Базовий клас для всіх вузлів синтаксичного дерева (AST)."""

    @abstractmethod
    def evaluate(self, index: InvertedIndex) -> Set[str]:
        """Оцінює вузол та повертає множину doc_id, що відповідають запиту."""
        pass

    def __and__(self, other: "ASTNode") -> "And":
        return And(self, other)

    def __or__(self, other: "ASTNode") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass(frozen=True)
class Term(ASTNode):
    term: str

    def evaluate(self, index: InvertedIndex) -> Set[str]:
        clean_term = self.term.lower()
        if clean_term in index:
            return set(index[clean_term].keys())
        return set()


@dataclass(frozen=True)
class Phrase(ASTNode):
    words: List[str]

    def evaluate(self, index: InvertedIndex) -> Set[str]:
        if not self.words:
            return set()

        clean_words = [w.lower() for w in self.words]

        # 1. Знаходимо документи, які містять УСІ слова з фрази
        first_term = clean_words[0]
        if first_term not in index:
            return set()

        candidate_docs = set(index[first_term].keys())
        for word in clean_words[1:]:
            if word not in index:
                return set()
            candidate_docs &= set(index[word].keys())

        # 2. Перевіряємо послідовність позицій у знайдених документах
        matching_docs = set()
        for doc_id in candidate_docs:
            pos_lists = [index[word][doc_id] for word in clean_words]
            for p0 in pos_lists[0]:
                if all((p0 + i) in pos_lists[i] for i in range(1, len(clean_words))):
                    matching_docs.add(doc_id)
                    break

        return matching_docs


@dataclass(frozen=True)
class And(ASTNode):
    left: ASTNode
    right: ASTNode

    def evaluate(self, index: InvertedIndex) -> Set[str]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True)
class Or(ASTNode):
    left: ASTNode
    right: ASTNode

    def evaluate(self, index: InvertedIndex) -> Set[str]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True)
class Not(ASTNode):
    operand: ASTNode

    def evaluate(self, index: InvertedIndex) -> Set[str]:
        doc_map = getattr(index, "doc_meta", getattr(index, "doc_metas", {}))
        all_docs = set(doc_map.keys())
        return all_docs - self.operand.evaluate(index)