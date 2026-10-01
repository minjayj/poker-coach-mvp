from __future__ import annotations

from typing import Any

from backend.models import Recommendation

RANK_ORDER = "23456789TJQKA"


def _rank_value(rank: str) -> int:
    return RANK_ORDER.index(rank.upper())


def _normalize_cards(cards: list[str] | None) -> list[str]:
    if not cards:
        return []
    return [card.strip().upper() for card in cards if card and card.strip()]


def classify_preflop_hand(cards: list[str] | None) -> str:
    normalized = _normalize_cards(cards)
    if len(normalized) != 2 or len(normalized[0]) < 2 or len(normalized[1]) < 2:
        return "unknown"

    first_rank, first_suit = normalized[0][0], normalized[0][-1]
    second_rank, second_suit = normalized[1][0], normalized[1][-1]
    suited = first_suit == second_suit
    pair = first_rank == second_rank
    high_rank = max(first_rank, second_rank, key=_rank_value)
    low_rank = min(first_rank, second_rank, key=_rank_value)

    if pair and _rank_value(high_rank) >= _rank_value("T"):
        return "premium"
    if {first_rank, second_rank} == {"A", "K"}:
        return "premium" if suited else "strong"
    if {first_rank, second_rank} == {"A", "Q"} and suited:
        return "strong"
    if {first_rank, second_rank} == {"K", "Q"} and suited:
        return "strong"
    if pair and _rank_value(high_rank) >= _rank_value("7"):
        return "strong"
    if _rank_value(high_rank) >= _rank_value("A") and _rank_value(low_rank) >= _rank_value("T"):
        return "strong"
    if suited and abs(_rank_value(first_rank) - _rank_value(second_rank)) == 1:
        return "playable"
    if pair:
        return "playable"
    if _rank_value(high_rank) >= _rank_value("K") or (
        _rank_value(high_rank) >= _rank_value("Q") and suited
    ):
        return "playable"
    return "weak"


def recommend_move(
    hero_cards: list[str] | None,
    position: str,
    stack_bb: float,
    facing_action: str,
    pot_size: float,
    player_tendencies: dict[str, Any] | None,
) -> Recommendation:
    tendencies = player_tendencies or {}
    aggression = float(tendencies.get("aggression_score", 0.0) or 0.0)
    vpip = float(tendencies.get("vpip", 0.0) or 0.0)
    strength = classify_preflop_hand(hero_cards)
    loose_aggressive = aggression >= 2.0 or vpip >= 0.4
    standard_raise = max(4.0, round(max(pot_size, 2.0) * 0.75, 1))

    if stack_bb <= 10:
        if strength in {"premium", "strong"}:
            return Recommendation(
                move="all_in",
                amount=None,
                reason="Short stack spot: shove/fold mode favors jamming a strong range.",
            )
        if facing_action in {"raise", "bet", "all_in"}:
            return Recommendation(
                move="fold",
                amount=None,
                reason="Short stack with a weak holding facing pressure is a fold in the MVP ruleset.",
            )
        return Recommendation(
            move="check" if facing_action == "none" else "fold",
            amount=None,
            reason="Without a strong hand, preserve the short stack for a better spot.",
        )

    if strength == "premium":
        return Recommendation(
            move="raise" if facing_action != "all_in" else "call",
            amount=round(standard_raise + (2.0 if loose_aggressive else 0.0), 1),
            reason="Premium hands want to build the pot and punish loose action.",
        )

    if strength == "strong":
        if facing_action in {"raise", "bet", "all_in"} and loose_aggressive:
            return Recommendation(
                move="call",
                amount=None,
                reason="Strong range can continue a bit wider against loose-aggressive opponents.",
            )
        return Recommendation(
            move="raise",
            amount=standard_raise,
            reason="Strong preflop strength prefers an aggressive line in this MVP engine.",
        )

    if strength == "playable":
        if facing_action in {"raise", "bet", "all_in"}:
            return Recommendation(
                move="fold" if not loose_aggressive else "call",
                amount=None,
                reason="Playable hands continue only a little wider against opponents who over-pressure pots.",
            )
        return Recommendation(
            move="check",
            amount=None,
            reason=f"No bet is facing and position is {position}, so taking the free option is acceptable.",
        )

    if facing_action == "none":
        return Recommendation(
            move="check",
            amount=None,
            reason="No action is facing, so the MVP defaults to the low-variance option.",
        )

    return Recommendation(
        move="fold",
        amount=None,
        reason="Weak holdings facing aggression fold by default in the MVP ruleset.",
    )
