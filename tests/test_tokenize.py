from findex.tokenize import tokenize


def test_tokenize_mixed_case():
    assert list(tokenize("Hello WORLD WoRlD")) == ["hello", "world", "world"]


def test_tokenize_cyrillic():
    assert list(tokenize("Привіт Світ СВІТ")) == ["привіт", "світ", "світ"]


def test_tokenize_combining_accent():
    accented_text = "e\u0301le\u0301phant"  # e + combining mark
    assert list(tokenize(accented_text)) == ["éléphant"]


def test_tokenize_punctuation():
    assert list(tokenize("Hello, world! How's it going?")) == ["hello", "world", "how's", "it", "going"]


def test_tokenize_ukrainian_apostrophe():
    assert list(tokenize("Об'єкт пам'яті м'яч")) == ["об'єкт", "пам'яті", "м'яч"]


def test_tokenize_empty():
    assert list(tokenize("")) == []