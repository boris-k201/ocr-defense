from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+|[^A-Za-zА-Яа-яЁё]+", flags=re.UNICODE)


def detect_language(text: str) -> str:
    # Very lightweight heuristic.
    cyr = sum(1 for ch in text if "А" <= ch <= "я" or ch in "Ёё")
    lat = sum(1 for ch in text if "A" <= ch <= "z" or "a" <= ch <= "z")
    return "ru" if cyr >= lat else "en"


def _preserve_case(src: str, repl: str) -> str:
    if not src:
        return repl
    if src.isupper():
        return repl.upper()
    if src[0].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


# Small built-in synonym sets (extensible by user via config in future).
# Keys and values are stored in lowercase.
_RU_SYNONYMS: Dict[str, List[str]] = {
    "хороший": ["отличный", "прекрасный", "замечательный"],
    "плохой": ["ужасный", "нехороший", "ужасающе"],
    "большой": ["крупный", "значительный", "огромный"],
    "маленький": ["небольшой", "крошечный", "малый"],
    "важный": ["значимый", "существенный", "ключевой"],
    "быстрый": ["скорый", "проворный", "оперативный"],
    "медленный": ["неторопливый", "тормозной", "плавный"],
    "разный": ["различный", "неодинаковый"],
    "исследование": ["изучение", "исследование"],  # keep itself as fallback
    "метод": ["способ", "прием"],
    "данные": ["сведения", "информация"],
    "качество": ["уровень", "качество"],  # keep itself as fallback
    "система": ["механизм", "структура"],
    "защита": ["оберег", "охранение"],
    "текст": ["сообщение", "письмо"],
}

_EN_SYNONYMS: Dict[str, List[str]] = {
    "good": ["great", "excellent", "fine"],
    "bad": ["poor", "awful", "terrible"],
    "large": ["big", "huge", "major"],
    "small": ["little", "minor", "tiny"],
    "important": ["significant", "key", "crucial"],
    "fast": ["rapid", "quick", "swift"],
    "slow": ["sluggish", "slower", "gradual"],
    "different": ["distinct", "varied", "diverse"],
    "method": ["approach", "technique", "strategy"],
    "data": ["information", "records", "facts"],
    "system": ["framework", "mechanism", "structure"],
    "text": ["content", "string", "passage"],
    "protection": ["defense", "guard", "shield"],
    "study": ["research", "investigation"],
}


def get_synonyms(word: str, language: str) -> List[str]:
    key = word.lower()
    if language == "ru":
        return _RU_SYNONYMS.get(key, [])
    return _EN_SYNONYMS.get(key, [])


@dataclass(frozen=True)
class SemanticAttackConfig:
    language: str = "auto"  # "auto" | "ru" | "en"
    max_changed_words: int = 3
    population_size: int = 24
    generations: int = 18
    tournament_k: int = 3
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    random_seed: Optional[int] = None


def semantic_synonym_attack(
    text: str,
    config: SemanticAttackConfig,
    *,
    fitness_fn: Optional[Callable[[str], float]] = None,
) -> Tuple[str, Dict]:
    """
    Semantic-level attack by synonym replacement using a GA loop.

    If fitness_fn is provided, GA maximizes fitness_fn(candidate_text).
    Otherwise it maximizes "number of replaced words" under the change budget.
    """
    rng = random.Random(config.random_seed)
    language = config.language if config.language != "auto" else detect_language(text)

    # Tokenize into word/non-word segments, so we keep punctuation/whitespace.
    parts = _WORD_RE.findall(text)
    word_positions: List[int] = []
    candidate_synonyms: List[List[str]] = []

    for idx, part in enumerate(parts):
        if re.fullmatch(r"[A-Za-zА-Яа-яЁё]+", part, flags=re.UNICODE):
            syns = get_synonyms(part, language)
            if syns:
                word_positions.append(idx)
                # Gene 0 means keep original; genes 1..k are synonym options.
                candidate_synonyms.append([part] + syns)

    if len(word_positions < 2):
        return text, {"changed_words": 0, "language": language, "ga_used": True}

    n = len(word_positions)

    def build_candidate(chromosome: Sequence[int]) -> str:
        out = parts[:]
        changed = 0
        for gene_i, part_i in enumerate(word_positions):
            chosen = candidate_synonyms[gene_i][chromosome[gene_i]]
            if chosen != out[part_i]:
                changed += 1
            out[part_i] = _preserve_case(out[part_i], chosen) if chosen != out[part_i] else out[part_i]
        return "".join(out)

    def count_changed(chromosome: Sequence[int]) -> int:
        changed = 0
        for gene_i in range(n):
            if chromosome[gene_i] != 0:
                changed += 1
        return changed

    def fitness(candidate_text: str, chromosome: Sequence[int]) -> float:
        if fitness_fn is not None:
            return float(fitness_fn(candidate_text))
        # Default heuristic fitness.
        changed = count_changed(chromosome)
        # Prefer more changed words, but heavily penalize breaking the budget.
        if changed > config.max_changed_words:
            return -1000.0 - (changed - config.max_changed_words)
        return float(changed)

    def random_chromosome() -> List[int]:
        chrom = [0] * n
        # Ensure we respect change budget by sampling a subset.
        max_to_change = min(config.max_changed_words, n)
        k = rng.randint(0, max_to_change)
        if k == 0:
            return chrom
        positions = rng.sample(range(n), k)
        for gene_i in positions:
            syns = candidate_synonyms[gene_i]
            # Choose a synonym index 1..len-1
            chrom[gene_i] = rng.randint(1, len(syns) - 1)
        return chrom

    # Population initialization: include original plus randoms.
    population: List[List[int]] = []
    population.append([0] * n)
    while len(population) < config.population_size:
        population.append(random_chromosome())

    best_text = text
    best_fit = float("-inf")
    best_chrom = population[0]

    for _gen in range(config.generations):
        scored: List[Tuple[float, List[int]]] = []
        for chrom in population:
            cand_text = build_candidate(chrom)
            fit = fitness(cand_text, chrom)
            scored.append((fit, chrom))
            if fit > best_fit:
                best_fit = fit
                best_chrom = chrom
                best_text = cand_text

        scored.sort(key=lambda x: x[0], reverse=True)
        # Elitism
        new_population: List[List[int]] = [scored[0][1][:]]

        def tournament_select() -> List[int]:
            contenders = rng.sample(scored, k=min(config.tournament_k, len(scored)))
            contenders.sort(key=lambda x: x[0], reverse=True)
            return contenders[0][1][:]

        while len(new_population) < config.population_size:
            parent1 = tournament_select()
            parent2 = tournament_select()
            child = parent1[:]

            if rng.random() < config.crossover_rate:
                # One-point crossover.
                point = rng.randint(1, n - 1)
                child = parent1[:point] + parent2[point:]

            # Mutation
            for gene_i in range(n):
                if rng.random() < config.mutation_rate:
                    syns = candidate_synonyms[gene_i]
                    if len(syns) <= 1:
                        continue
                    # Toggle between keep-original and a random synonym.
                    if child[gene_i] == 0:
                        child[gene_i] = rng.randint(1, len(syns) - 1)
                    else:
                        child[gene_i] = 0

            # Repair to ensure change budget is not violated too much:
            # if too many changes, revert some random genes back to original.
            changed = count_changed(child)
            if changed > config.max_changed_words:
                to_revert = changed - config.max_changed_words
                revert_positions = [i for i in range(n) if child[i] != 0]
                rng.shuffle(revert_positions)
                for i in revert_positions[:to_revert]:
                    child[i] = 0

            new_population.append(child)

        population = new_population

    changed_words = 0
    for gene_i in range(n):
        if best_chrom[gene_i] != 0:
            changed_words += 1

    return best_text, {"changed_words": changed_words, "language": language, "ga_used": True, "best_fitness": best_fit}

