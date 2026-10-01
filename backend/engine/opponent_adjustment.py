"""Conservative player-read overlay for an exact imported solver strategy.

This is a bounded policy mixture, not a new equilibrium or proof of a tell.
Only completed, contextual hand records with confirmed actions supply evidence.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def size_bucket(ratio: float) -> str:
    if ratio <= 0.5:
        return "small"
    if ratio <= 1.0:
        return "medium"
    return "large"


def _trusted(action: dict[str, Any]) -> bool:
    source = action.get("source")
    if source in {"manual", "correction"}:
        return True
    return source in {"speech", "vision"} and (action.get("confidence") or 0) >= 0.8


def summarize_opponent_json(
    archived_hands: list[dict[str, Any]], *, player_id: str,
    game_variant: str, game_format: str, tournament_phase: str,
    table_size_bucket: str,
) -> dict[str, Any]:
    """Measure action responses and size patterns in comparable completed hands.

    Old archives without pot/price metadata are excluded rather than guessed.
    Showdown *hand strength* is not inferred from the size or from the winner.
    """
    facing = Counter()
    by_faced_size: dict[str, Counter] = {key: Counter() for key in ("small", "medium", "large")}
    own_sizes = Counter()
    qualifying_hands: set[int] = set()
    facing_hands: set[int] = set()
    for hand in archived_hands:
        if (hand.get("game_variant") != game_variant or hand.get("game_format") != game_format
                or hand.get("tournament_phase") != tournament_phase
                or hand.get("table_size_bucket") != table_size_bucket):
            continue
        hand_id = hand.get("hand_id")
        if not isinstance(hand_id, int):
            continue
        matched = False
        for action in hand.get("actions", []):
            if not isinstance(action, dict) or str(action.get("player", "")).casefold() != player_id.casefold() or not _trusted(action):
                continue
            pot = action.get("pot_before")
            price = action.get("price_to_call_before")
            if not isinstance(pot, (int, float)) or not math.isfinite(pot) or pot <= 0:
                continue
            kind = action.get("action")
            if isinstance(price, (int, float)) and math.isfinite(price) and price > 0 and kind in {"fold", "call", "raise", "all_in"}:
                facing[kind] += 1
                by_faced_size[size_bucket(price / pot)][kind] += 1
                facing_hands.add(hand_id)
                matched = True
            contributed = action.get("contributed")
            if kind in {"bet", "raise", "all_in"} and isinstance(contributed, (int, float)) and math.isfinite(contributed) and contributed > 0:
                own_sizes[size_bucket(contributed / pot)] += 1
                matched = True
        if matched:
            qualifying_hands.add(hand_id)

    def fold_posterior(counts: Counter, strength: float) -> float:
        total = sum(counts.values())
        return (counts["fold"] + strength * 0.35) / (total + strength)

    total_facing = sum(facing.values())
    own_total = sum(own_sizes.values())
    return {
        "player_id": player_id,
        "source": "hand_archive.json",
        "context": {
            "game_variant": game_variant, "game_format": game_format,
            "tournament_phase": tournament_phase, "table_size_bucket": table_size_bucket,
        },
        "qualifying_hands": len(qualifying_hands),
        "facing_hands": len(facing_hands),
        "facing_actions": dict(facing),
        "posterior_fold_when_facing_bet": round(fold_posterior(facing, 20), 4),
        "by_faced_size": {
            bucket: {
                "samples": sum(counts.values()), "folds": counts["fold"],
                "posterior_fold": round(fold_posterior(counts, 12), 4),
            }
            for bucket, counts in by_faced_size.items()
        },
        "own_aggressive_sizes": {
            "samples": own_total,
            "counts": {bucket: own_sizes[bucket] for bucket in ("small", "medium", "large")},
        },
        "limitations": [
            "Size buckets use recorded contribution divided by the tracked pot before the action; unlogged blinds or actions can distort the ratio.",
            "Size frequency is not evidence that a player is strong or bluffing; no hidden cards are inferred.",
            "Only completed hands with trusted actions and matching format, phase, variant, and table size contribute.",
        ],
    }


def blend_exact_strategy(
    solver_result: dict[str, Any], player_model: dict[str, Any], read: dict[str, Any],
    *, proposed_raise_to: float | None = None,
    observed_opponent_bet_to_pot: float | None = None,
) -> dict[str, Any]:
    """Blend exact-tree frequencies with a separate approximate player policy.

    At least 12 different facing hands are needed. Even with thousands of
    observations the player branch is capped at 35%; an unusual current size
    halves that cap rather than being assumed strong or weak.
    """
    native = solver_result["action_frequencies"]
    labels = list(native)
    categories = []
    for label in labels:
        token = label.split()[0].upper()
        categories.append("call" if token in {"CHECK", "CALL"} else "raise" if token in {"BET", "RAISE", "ALLIN", "ALL-IN"} else "fold" if token == "FOLD" else "unknown")
    reason = None
    if "unknown" in categories or len(set(categories)) != len(categories):
        reason = "The imported tree has multiple sizes or unrecognized actions; no player blend was applied."
    n = int(read["facing_hands"])
    weight = min(0.35, 0.35 * n / (n + 30)) if n >= 12 and reason is None else 0.0
    sized_evidence = None
    if proposed_raise_to is not None and proposed_raise_to > solver_result["spot"]["to_call"]:
        ratio = (proposed_raise_to - solver_result["spot"]["to_call"]) / solver_result["spot"]["pot"]
        bucket = size_bucket(ratio)
        sized_evidence = {"proposed_raise_bucket": bucket, **read["by_faced_size"][bucket]}
    if observed_opponent_bet_to_pot is not None and read["own_aggressive_sizes"]["samples"] >= 20:
        observed_bucket = size_bucket(observed_opponent_bet_to_pot)
        proportion = read["own_aggressive_sizes"]["counts"][observed_bucket] / read["own_aggressive_sizes"]["samples"]
        if proportion < 0.1:
            weight *= 0.5
            reason = "This opponent's observed size is uncommon in their recorded history; the player adjustment was further shrunk."
    exploit = player_model["action_frequencies"]
    if weight and "raise" in categories:
        fold_rate = (sized_evidence["posterior_fold"] if sized_evidence and sized_evidence["samples"] >= 8
                     else read["posterior_fold_when_facing_bet"])
        # Small bounded odds tilt; not a best-response computation.
        raise_factor = math.exp(max(-0.45, min(0.45, 1.5 * (fold_rate - 0.35))))
    else:
        raise_factor = 1.0
    player_weights = [max(0.0, float(exploit[category])) * (raise_factor if category == "raise" else 1.0) for category in categories]
    total = sum(player_weights)
    if total <= 0:
        weight = 0.0
        reason = "Player model has no probability on the imported legal actions."
        player_weights = [float(native[label]) for label in labels]
    else:
        player_weights = [value / total for value in player_weights]
    blended = {
        label: round((1 - weight) * float(native[label]) + weight * player_weights[index], 6)
        for index, label in enumerate(labels)
    }
    return {
        "baseline": native,
        "player_model_on_tree_actions": dict(zip(labels, [round(value, 6) for value in player_weights])),
        "blended_frequencies": blended,
        "player_weight": round(weight, 4),
        "size_evidence": sized_evidence,
        "reason": reason or ("Too few comparable facing hands; baseline unchanged." if n < 12 else "Conservative player-specific study blend."),
        "study_only": True,
        "caveat": "The blend is a heuristic mixture, not a GTO solution or validated best response. Confirm the opponent's identity and all solve assumptions.",
    }
