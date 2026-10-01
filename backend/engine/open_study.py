"""Read-only, narrowly scoped open solver data for post-hand study.

The bundled artifact is an *approximate heads-up push/fold game*, not a
solution to unrestricted Hold'em. Keep it out of the live coach's decision
path so an unmatched cash, multiway, or tournament spot cannot inherit it.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from backend.engine.hand_evaluator import RANKS, normalize_cards

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "open_study" / "hu_push_fold.json"
SOURCE_URL = "https://github.com/davidvayn/pokersolver/blob/0036c954f411984e87298e75ed63d7e5554bcf83/data/preflop/solved-scenarios.json"
SOURCE_SHA256 = "6fb89ce7b5ca3919480570f6e14188fb95e400b1dd5fade32d6d01d9d743b59c"
STACKS_BB = (2, 3, 5, 8, 10, 12, 15, 20)
MODEL = "heads-up-push-fold-monte-carlo-v1"
ASSUMPTIONS = (
    "Heads-up NLH with equal effective stacks and 0.5/1 blinds.",
    "No ante, rake, ICM, limps, or non-all-in opens.",
    "SB may fold or shove; BB may fold or call a shove.",
    "Equities are Monte Carlo estimates; exploitability is for this simplified model, not the full game.",
)


def _hand_classes() -> set[str]:
    classes = {rank * 2 for rank in RANKS}
    for high_index in range(1, len(RANKS)):
        for low_index in range(high_index):
            prefix = RANKS[high_index] + RANKS[low_index]
            classes.update((prefix + "s", prefix + "o"))
    return classes


EXPECTED_CLASSES = _hand_classes()


@lru_cache(maxsize=1)
def _catalog() -> dict[int, dict[str, Any]]:
    raw = DATA_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Bundled open study data failed its SHA-256 integrity check.")
    scenarios = json.loads(raw)
    if not isinstance(scenarios, list) or len(scenarios) != len(STACKS_BB):
        raise RuntimeError("Bundled open study data has an unexpected scenario count.")

    catalog: dict[int, dict[str, Any]] = {}
    for scenario in scenarios:
        stack = scenario.get("effective_stack_bb")
        hands = scenario.get("hands")
        action_values = scenario.get("action_values")
        if stack not in STACKS_BB or stack in catalog or scenario.get("model") != MODEL:
            raise RuntimeError("Bundled open study data has an invalid model or stack.")
        if not isinstance(hands, list) or not isinstance(action_values, list):
            raise RuntimeError("Bundled open study data has missing hand tables.")
        frequencies = {row[0]: row[1:] for row in hands if isinstance(row, list) and len(row) == 3}
        values = {row[0]: row[1:] for row in action_values if isinstance(row, list) and len(row) == 5}
        if set(frequencies) != EXPECTED_CLASSES or set(values) != EXPECTED_CLASSES:
            raise RuntimeError("Bundled open study data must contain each of the 169 hand classes once.")
        if any(
            not all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in pair)
            for pair in frequencies.values()
        ) or any(
            not all(isinstance(value, (int, float)) and math.isfinite(value) for value in row)
            for row in values.values()
        ):
            raise RuntimeError("Bundled open study data contains invalid frequencies or action values.")
        catalog[stack] = {**scenario, "frequencies": frequencies, "values": values}
    if set(catalog) != set(STACKS_BB):
        raise RuntimeError("Bundled open study data is missing an expected stack depth.")
    return catalog


def hand_class(hero_hand: list[str]) -> str:
    cards = normalize_cards(hero_hand)
    if len(cards) != 2:
        raise ValueError("Exactly two Hold'em cards are required.")
    first, second = cards
    if first[0] == second[0]:
        return first[0] * 2
    high, low = sorted((first[0], second[0]), key=RANKS.index, reverse=True)
    return high + low + ("s" if first[1] == second[1] else "o")


def list_push_fold_studies() -> dict[str, Any]:
    catalog = _catalog()
    return {
        "game": "nlh",
        "model": MODEL,
        "study_only": True,
        "available_stacks_bb": list(STACKS_BB),
        "scenarios": [
            {"effective_stack_bb": stack, "quality": catalog[stack]["quality"]}
            for stack in STACKS_BB
        ],
        "assumptions": list(ASSUMPTIONS),
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
    }


def lookup_push_fold_study(
    *, effective_stack_bb: int, hero_hand: list[str], decision: str
) -> dict[str, Any]:
    if effective_stack_bb not in STACKS_BB:
        raise ValueError(f"Only exact stack depths {list(STACKS_BB)}bb are available; no interpolation is used.")
    if decision not in {"sb_first", "bb_vs_shove"}:
        raise ValueError("Decision must be sb_first or bb_vs_shove.")
    label = hand_class(hero_hand)
    scenario = _catalog()[effective_stack_bb]
    shove, call = scenario["frequencies"][label]
    sb_fold_ev, sb_shove_ev, bb_fold_ev, bb_call_ev = scenario["values"][label]
    frequencies = {"fold": 1 - shove, "shove": shove} if decision == "sb_first" else {"fold": 1 - call, "call": call}
    action_values = {"fold": sb_fold_ev, "shove": sb_shove_ev} if decision == "sb_first" else {"fold": bb_fold_ev, "call": bb_call_ev}
    return {
        "game": "nlh",
        "study_only": True,
        "model": MODEL,
        "decision": decision,
        "effective_stack_bb": effective_stack_bb,
        "hand_class": label,
        "action_frequencies": frequencies,
        "model_action_values_bb": action_values,
        "quality": scenario["quality"],
        "model_exploitability_bb": scenario["exploitability_bb"],
        "action_value_standard_error_upper_bound_bb": scenario["action_value_standard_error_upper_bound_bb"],
        "assumptions": list(ASSUMPTIONS),
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "artifact_id": scenario["artifact_id"],
    }
