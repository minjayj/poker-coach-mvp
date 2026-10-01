"""Small, explicit chat-input parser. It never guesses hidden table facts."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Callable

from backend.models import CoachChatContext, GameState

CARD = r"(?:10|[2-9TJQKA])[cdhs]"
CARD_RUN = rf"({CARD}(?:[\s,]+{CARD}){{0,4}})"
MONEY = r"\$?([0-9]+(?:\.[0-9]+)?)"


def _cards(text: str, labels: str) -> list[str] | None:
    match = re.search(rf"\b(?:{labels})\b\s*(?:are|is|:|=)?\s*{CARD_RUN}", text, re.I)
    if not match:
        return None
    return [card.replace("10", "T").upper() for card in re.findall(CARD, match.group(1), re.I)]


def parse_chat_input(message: str, context: CoachChatContext, state: GameState, resolve_alias: Callable[[str], str | None] | None = None) -> tuple[CoachChatContext, dict]:
    """Extract only clearly labelled facts; the caller validates cards before saving."""
    if context.hand_id != state.hand_id or context.game_variant != state.game_variant:
        context = CoachChatContext(hand_id=state.hand_id, game_variant=state.game_variant)
    update: dict = {}
    if context.pending_alias:
        if re.fullmatch(r"\s*(?:yes|yes same|same player|same person|correct)\s*[.!]?\s*", message, re.I):
            update["confirmed_alias"] = (context.pending_alias, context.pending_candidate)
            context.opponent_id = context.pending_candidate
            context.pending_alias = context.pending_candidate = None
        elif re.fullmatch(r"\s*(?:no|not the same|different player)\s*[.!]?\s*", message, re.I):
            context.pending_alias = context.pending_candidate = None
            update["identity_question"] = "Understood. Which seated player is this, or should I treat the player as unknown?"
            return context, update
    hand = _cards(message, r"my (?:hand|cards)|i (?:have|hold)|hero(?: hand| cards)?")
    board = _cards(message, r"board|flop|turn|river")
    if hand is not None:
        update["hero_cards"] = hand
    if board is not None:
        update["board_cards"] = board
        context.opponent_line = None
    pot = re.search(rf"\bpot\s*(?:is|:|=)?\s*{MONEY}\b", message, re.I)
    if pot:
        update["current_pot"] = float(pot.group(1))
    # Exact seated names only. Similar names are not silently merged.
    known_players = list(dict.fromkeys([*state.players, *state.observed_players]))
    mentioned = [name for name in known_players if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", message, re.I)]
    actor_mentioned = bool(mentioned or re.search(r"\b(?:villain|opponent|they|he|she)\b", message, re.I))
    price = re.search(rf"\b(?:to call|call(?:ing)? costs?|facing|price to call)\s*(?:is|:|=)?\s*{MONEY}\b", message, re.I)
    line_change = bool(re.search(r"\b(?:bets?|bet|raised|raises?|reraised|reraises?|3-bets?|shoved|shoves?|all[- ]?in)\b", message, re.I) and actor_mentioned)
    if (board is not None or line_change) and not price:
        context.to_call = None
    if price:
        context.to_call = float(price.group(1))
    elif re.search(r"\b(?:i can check|no bet to call|to call\s*\$?0(?:\.0+)?\b)\b", message, re.I):
        context.to_call = 0.0
    if len(mentioned) == 1:
        context.opponent_id = mentioned[0]
    elif len(mentioned) > 1:
        context.opponent_id = None
        update["identity_question"] = "Which player is the opponent in this decision? I found more than one name."
    elif not update.get("confirmed_alias"):
        actor = re.search(r"\b([A-Za-z][A-Za-z0-9_]{1,24})\s+(?:raised|raises?|bets?|checked|checks?|called|calls?|folded|folds?|shoved|shoves?)\b", message, re.I)
        if actor and actor.group(1).lower() not in {"i", "he", "she", "they", "villain", "opponent", "hero"} and known_players:
            detected = actor.group(1)
            canonical = resolve_alias(detected) if resolve_alias else None
            if canonical in known_players:
                context.opponent_id = canonical
                context.opponent_line = message.strip()
                return context, update
            closest = max(known_players, key=lambda name: SequenceMatcher(None, detected.lower(), name.lower()).ratio())
            score = SequenceMatcher(None, detected.lower(), closest.lower()).ratio()
            context.opponent_id = None
            context.opponent_line = message.strip()
            if score >= .55:
                context.pending_alias, context.pending_candidate = detected, closest
                update["identity_question"] = f"Is {detected} the same player as {closest}? Please say yes or no before I use that profile."
            else:
                update["identity_question"] = f"I don't recognize {detected} among the seated players. Which player is this?"
    if re.search(r"\b(?:bets?|raised|raises?|reraised|reraises?|3-bets?|checked|checks?|called|calls?|folded|folds?|shoved|shoves?|all[- ]?in)\b", message, re.I) and actor_mentioned:
        context.opponent_line = message.strip()
    return context, update


def clarification(state: GameState, context: CoachChatContext) -> str | None:
    needed = 5 if state.game_variant == "plo5" else 2
    if len(state.hero_cards) != needed:
        return f"What are your {needed} hole cards? For example: 'I have Ah Kh' (or all five cards for PLO5)."
    if len(state.board_cards) not in {0, 3, 4, 5}:
        return "Please confirm the full board so far, for example: 'board Qs Jd 3c'."
    if state.observed_pot is None and state.pot <= 0:
        return "What is the pot right now? For example: 'pot 24'."
    if context.to_call is None:
        return "How much do you need to call? If checking is free, say 'I can check'."
    return None


def preflop_line_hint(state: GameState, context: CoachChatContext) -> str | None:
    """Use only an explicit preflop verb as a broad range prior."""
    if state.board_cards or state.current_street != "preflop" or not context.opponent_line:
        return None
    line = context.opponent_line.lower()
    if re.search(r"\b(?:4[- ]?bet|four[- ]?bet)\b", line):
        return "four_bet"
    if re.search(r"\b(?:3[- ]?bet|three[- ]?bet|rerais|re-rais)\b", line):
        return "three_bet"
    if re.search(r"\b(?:rais|open|bet)\w*", line):
        return "open"
    if re.search(r"\blimp\w*", line):
        return "limp"
    return None


def general_coaching_answer(message: str) -> str | None:
    """Answer a few teaching questions without pretending a spot was observed."""
    text = message.lower()
    if re.search(r"\b(?:what is|explain|is this|is it)\s+gto\b", text):
        return "GTO is a strategy that cannot be exploited in the full modeled game. This app does not solve full poker GTO: Hold'em uses a small regret-matching decision model, while PLO5 uses a limited equity study. I can explain the model's line for a specific hand once you confirm the cards, pot and price."
    if re.search(r"\bpot odds\b", text):
        return "Pot odds compare the call price with the pot after your call: required equity is call / (pot + call), using the pot before your call. That is a starting point; future betting, rake, multiway play and tournament payouts can change the decision."
    if re.search(r"\b(?:what is|explain)\s+(?:vpip|pfr|af|aggression factor)\b", text):
        return "VPIP measures how often a player voluntarily puts chips in preflop. PFR measures how often they raise preflop. AF is postflop bets and raises divided by calls. Small samples are noisy, so the range model shrinks player rates toward a population prior."
    if re.search(r"\b(?:what is|explain)\s+icm\b", text):
        return "ICM estimates tournament prize equity from every remaining stack and the payout ladder. Near bubbles and final tables, losing chips can hurt prize equity more than winning the same number helps. Without complete stacks and payouts, I won't claim an ICM-adjusted answer."
    return None


def is_visibility_question(message: str) -> bool:
    """Recognize a direct capture-status question, not a hand description."""
    return bool(re.fullmatch(
        r"\s*(?:can you see (?:it|my screen|the (?:screen|table|game))|"
        r"do you see (?:it|my screen|the (?:screen|table|game))|"
        r"are you seeing (?:my screen|the (?:screen|table|game)))\s*[?.!]*\s*",
        message, re.I,
    ))


def wants_specific_decision(message: str) -> bool:
    """Keep action requests on the verified-facts/engine path."""
    return bool(re.search(
        r"\b(?:what should i do|what (?:move|action) should i|should i (?:fold|call|raise|bet|check)|"
        r"(?:tell me|recommend) (?:what|whether) to (?:fold|call|raise|bet|check)|"
        r"analy[sz]e (?:this|my) (?:hand|spot))\b",
        message, re.I,
    ))
