import re
import unicodedata
from typing import Iterator

# Регулярний вираз для слів з підтримкою апострофів всередині слів
WORD_RE = re.compile(r"\b[\w]+(?:['’ʼ][\w]+)?\b", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """
    Нормалізує текст (NFC + casefold) та виділяє токени за допомогою regex.
    """
    if not text:
        return

    # NFC нормалізація + casefold (краще ніж lower для Unicode)
    normalized_text = unicodedata.normalize("NFC", text).casefold()

    for match in WORD_RE.finditer(normalized_text):
        token = match.group()
        # Пропускаємо суто цифрові токени
        if not token.isdigit():
            yield token