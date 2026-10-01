"""Bounded PLO5 equity check for a human-requested coaching brief.

This samples one uniformly random five-card opponent. It is not a solved PLO
range or multiway decision tree, so the brief must state that limitation.
"""

from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from phevaluator import evaluate_cards

from backend.engine.hand_evaluator import normalize_cards

DECK = tuple(f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs")
HOLE_PAIRS = tuple(combinations(range(5), 2))
BOARD_TRIPLES = tuple(combinations(range(5), 3))


def best_plo5_rank(hole: tuple[str, ...], board: tuple[str, ...]) -> int:
    """Omaha requires exactly two hole and exactly three board cards."""
    if len(hole) != 5 or len(board) != 5:
        raise ValueError("PLO5 evaluation requires five hole and five board cards.")
    return min(
        evaluate_cards(hole[a], hole[b], board[c], board[d], board[e])
        for a, b in HOLE_PAIRS for c, d, e in BOARD_TRIPLES
    )


def estimate_plo5_equity(hero_cards: list[str], board_cards: list[str], samples: int = 120, seed: int | None = None) -> dict[str, float | int]:
    if len(hero_cards) != 5 or len(board_cards) not in {0, 3, 4, 5}:
        raise ValueError("Enter five PLO5 hole cards and zero, three, four, or five board cards.")
    known = normalize_cards([*hero_cards, *board_cards])
    hero = known[:5]
    board = known[5:]
    deck = tuple(card for card in DECK if card not in known)
    draws = 5 + 5 - len(board)
    if len(deck) < draws:
        raise ValueError("Not enough legal cards remain.")
    trials = max(30, min(int(samples), 200))
    generator = np.random.default_rng(seed)
    wins = ties = 0
    started = perf_counter()
    for _ in range(trials):
        sampled = generator.choice(len(deck), size=draws, replace=False)
        opponent = tuple(deck[int(index)] for index in sampled[:5])
        runout = board + tuple(deck[int(index)] for index in sampled[5:])
        hero_rank = best_plo5_rank(hero, runout)
        opponent_rank = best_plo5_rank(opponent, runout)
        wins += hero_rank < opponent_rank
        ties += hero_rank == opponent_rank
    return {
        "equity": round((wins + ties / 2) / trials, 4),
        "samples": trials,
        "latency_ms": round((perf_counter() - started) * 1000, 2),
    }
