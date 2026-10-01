"""Card validation and a narrow adapter around HenryRLee's PH Evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from phevaluator import evaluate_cards

RANKS = "23456789TJQKA"
SUITS = "CDHS"


def normalize_card(card: str) -> str:
    """Convert ``AH``/``Ah`` into the evaluator's conventional ``Ah`` form."""
    value = card.strip()
    if len(value) != 2 or value[0].upper() not in RANKS or value[1].upper() not in SUITS:
        raise ValueError(f"Invalid card {card!r}; expected a value like 'Ah' or 'AH'.")
    return value[0].upper() + value[1].lower()


def normalize_cards(cards: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(normalize_card(card) for card in cards)
    if len(set(normalized)) != len(normalized):
        raise ValueError("A card may only appear once in a Hold'em state.")
    return normalized


@dataclass(frozen=True)
class PerfectHashEvaluator:
    """HenryRLee PH evaluator wrapper; lower rank values represent stronger hands."""

    def evaluate(self, cards: Iterable[str]) -> int:
        normalized = normalize_cards(cards)
        if not 5 <= len(normalized) <= 7:
            raise ValueError("PH evaluation requires between 5 and 7 cards.")
        return int(evaluate_cards(*normalized))

    def compare(self, hero_cards: Iterable[str], villain_cards: Iterable[str]) -> int:
        """Return 1 for hero win, 0 for tie, and -1 for hero loss."""
        hero_rank = self.evaluate(hero_cards)
        villain_rank = self.evaluate(villain_cards)
        return 1 if hero_rank < villain_rank else -1 if hero_rank > villain_rank else 0
