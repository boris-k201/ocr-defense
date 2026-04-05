from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


_WORD_SPLIT_RE = re.compile(r"(\s+)")
_LETTER_RE = re.compile(r"[A-Za-zА-Яа-яЁё]")

# Unicode combining diacritics range U+0300–U+036F (canonical combining diacritics).
_DIACRITICS = [chr(cp) for cp in range(0x0300, 0x036F + 1)]


@dataclass(frozen=True)
class DiacriticsAttackConfig:
    budget_per_word: int = 5
    diacritics_probability: float = 0.6  # chance to inject diacritics into a word
    random_seed: Optional[int] = None


def _count_diacritics_in_segment(segment: str) -> int:
    return sum(1 for ch in segment if 0x0300 <= ord(ch) <= 0x036F)


def diacritics_attack(text: str, config: DiacriticsAttackConfig) -> Tuple[str, Dict]:
    """
    Text-level coding attack: insert combining diacritical marks.

    Rule: no more than `budget_per_word` combining marks per word token.
    Human readability is preserved by attaching few diacritics only to letters.
    """
    rng = random.Random(config.random_seed)

    # Split by whitespace, keep spaces.
    parts = _WORD_SPLIT_RE.split(text)
    out_parts = []
    per_word_marks = []

    for part in parts:
        if part.isspace() or not part:
            out_parts.append(part)
            continue

        # Only process segments that contain at least one letter.
        if not _LETTER_RE.search(part):
            out_parts.append(part)
            continue

        if rng.random() > config.diacritics_probability:
            out_parts.append(part)
            per_word_marks.append(0)
            continue

        marks_used = 0
        new_chars = []
        for ch in part:
            new_chars.append(ch)

            if marks_used >= config.budget_per_word:
                continue

            if _LETTER_RE.fullmatch(ch):
                # Insert a diacritic with moderate probability to avoid overloading.
                if rng.random() < 0.55:
                    new_chars.append(rng.choice(_DIACRITICS))
                    marks_used += 1

        new_part = "".join(new_chars)
        # Hard cap (safety).
        if _count_diacritics_in_segment(new_part) > config.budget_per_word:
            # If we overshot due to randomness, truncate to cap.
            # Approach: remove extra diacritics from the end.
            diacritics_positions = [i for i, ch in enumerate(new_part) if 0x0300 <= ord(ch) <= 0x036F]
            excess = len(diacritics_positions) - config.budget_per_word
            if excess > 0:
                # Remove diacritics from the end first.
                to_remove = set(diacritics_positions[-excess:])
                rebuilt = []
                for i, ch in enumerate(new_part):
                    if i in to_remove:
                        continue
                    rebuilt.append(ch)
                new_part = "".join(rebuilt)

        out_parts.append(new_part)
        per_word_marks.append(_count_diacritics_in_segment(new_part))

    marks_total = sum(per_word_marks)
    return "".join(out_parts), {"marks_total": marks_total, "words_affected": sum(1 for x in per_word_marks if x > 0)}

