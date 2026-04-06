from __future__ import annotations

import math
import re
import unicodedata
from typing import List, Sequence, Tuple


def normalize_text(s: str, *, lowercase: bool = True, strip: bool = True) -> str:
    """
    Нормализация строки перед расчётом метрик OCR.

    - NFKC: приводит совместимые символы к каноническому виду (например, разные формы пробелов/кавычек).
    - strip/lowercase: убирает края и (опционально) приводит к нижнему регистру.
    - collapse whitespace: для OCR важно сравнивать содержимое, а не артефакты переносов/множественных пробелов.
    """
    s = unicodedata.normalize("NFKC", s)
    if strip:
        s = s.strip()
    if lowercase:
        s = s.lower()
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    # Levenshtein distance: минимальное число вставок/удалений/замен для превращения a -> b.
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


def cer(reference: str, hypothesis: str, *, lowercase: bool = True) -> float:
    """
    CER (Character Error Rate) = LevenshteinDistance(chars(ref), chars(hyp)) / max(1, len(ref)).

    - 0.0 означает полное совпадение.
    - Значения > 1.0 возможны (например, если гипотеза сильно длиннее эталона).
    """
    ref = normalize_text(reference, lowercase=lowercase)
    hyp = normalize_text(hypothesis, lowercase=lowercase)
    # Character-level distance.
    return _edit_distance(list(ref), list(hyp)) / (len(ref) if len(ref) > 0 else 1)


def wer(reference: str, hypothesis: str, *, lowercase: bool = True) -> float:
    """
    WER (Word Error Rate) = LevenshteinDistance(tokens(ref), tokens(hyp)) / max(1, num_tokens(ref)).

    Токенизация здесь простая: разбиение по пробелам после normalize_text().
    """
    ref = normalize_text(reference, lowercase=lowercase)
    hyp = normalize_text(hypothesis, lowercase=lowercase)
    ref_toks = [t for t in ref.split(" ") if t]
    hyp_toks = [t for t in hyp.split(" ") if t]
    return _edit_distance(ref_toks, hyp_toks) / (len(ref_toks) if len(ref_toks) > 0 else 1)

