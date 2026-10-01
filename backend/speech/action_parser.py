from __future__ import annotations

import re
from typing import Protocol

from backend.models import ParsedSpeechAction

NAME_AND_ACTION_RE = re.compile(
    r"^(?P<player>[A-Za-z][A-Za-z0-9_-]*)\s+(?P<verb>raises?|calls?|folds?|checks?|bets?)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
SELF_ACTION_RE = re.compile(
    r"^i\s+(?P<verb>raise|call|fold|check|bet)\b(?P<rest>.*)$",
    re.IGNORECASE,
)
ALL_IN_RE = re.compile(
    r"^(?:(?P<player>[A-Za-z][A-Za-z0-9_-]*)\s+)?(?:goes\s+)?all[\s-]?in\b(?P<rest>.*)$",
    re.IGNORECASE,
)
MAKE_IT_RE = re.compile(r"^(?:make it|raise to)\s+(?P<amount>\d+(?:\.\d+)?)\b", re.IGNORECASE)
RAISE_MORE_RE = re.compile(r"^raise\s+(?P<amount>\d+(?:\.\d+)?)\s+more\b", re.IGNORECASE)
TO_AMOUNT_RE = re.compile(r"\bto\s+(?P<amount>\d+(?:\.\d+)?)\b", re.IGNORECASE)
MORE_AMOUNT_RE = re.compile(r"\b(?P<amount>\d+(?:\.\d+)?)\s+more\b", re.IGNORECASE)
PLAIN_AMOUNT_RE = re.compile(r"\b(?P<amount>\d+(?:\.\d+)?)\b")

ACTION_MAP = {
    "raise": "raise",
    "raises": "raise",
    "call": "call",
    "calls": "call",
    "fold": "fold",
    "folds": "fold",
    "check": "check",
    "checks": "check",
    "bet": "bet",
    "bets": "bet",
}


class ActionParser(Protocol):
    def parse(self, raw_text: str) -> ParsedSpeechAction:
        ...


def _clean_player(player: str | None) -> str | None:
    if not player:
        return None
    return player.strip().strip(",")


def _extract_amount(rest: str, default_type: str = "unknown") -> tuple[float | None, str]:
    if not rest:
        return None, default_type

    match = TO_AMOUNT_RE.search(rest)
    if match:
        return float(match.group("amount")), "total"

    match = MORE_AMOUNT_RE.search(rest)
    if match:
        return float(match.group("amount")), "additional"

    match = PLAIN_AMOUNT_RE.search(rest)
    if match:
        return float(match.group("amount")), default_type

    return None, default_type


class RuleBasedActionParser:
    def parse(self, raw_text: str) -> ParsedSpeechAction:
        text = raw_text.strip()
        lowered = text.lower()
        if not text:
            return ParsedSpeechAction(raw_text=raw_text, confidence=0.0)

        all_in_match = ALL_IN_RE.match(text)
        if all_in_match:
            player = _clean_player(all_in_match.group("player"))
            amount, amount_type = _extract_amount(all_in_match.group("rest"), "unknown")
            return ParsedSpeechAction(
                player=player,
                action="all_in",
                amount=amount,
                amount_type=amount_type,
                confidence=0.78 if not player else 0.9,
                raw_text=raw_text,
            )

        match = NAME_AND_ACTION_RE.match(text)
        if match:
            player = _clean_player(match.group("player"))
            action = ACTION_MAP.get(match.group("verb").lower(), "unknown")
            amount, amount_type = _extract_amount(match.group("rest"), "unknown")
            confidence = 0.95 if action in {"raise", "bet"} and amount is not None else 0.9
            return ParsedSpeechAction(
                player=player,
                action=action,
                amount=amount,
                amount_type=amount_type,
                confidence=confidence,
                raw_text=raw_text,
            )

        match = SELF_ACTION_RE.match(text)
        if match:
            action = ACTION_MAP.get(match.group("verb").lower(), "unknown")
            amount, amount_type = _extract_amount(match.group("rest"), "unknown")
            return ParsedSpeechAction(
                player=None,
                action=action,
                amount=amount,
                amount_type=amount_type,
                confidence=0.86,
                raw_text=raw_text,
            )

        match = MAKE_IT_RE.match(text)
        if match:
            return ParsedSpeechAction(
                player=None,
                action="raise",
                amount=float(match.group("amount")),
                amount_type="total",
                confidence=0.88,
                raw_text=raw_text,
            )

        match = RAISE_MORE_RE.match(text)
        if match:
            return ParsedSpeechAction(
                player=None,
                action="raise",
                amount=float(match.group("amount")),
                amount_type="additional",
                confidence=0.85,
                raw_text=raw_text,
            )

        for keyword, action in (
            ("calls", "call"),
            ("folds", "fold"),
            ("checks", "check"),
            ("raises", "raise"),
            ("bets", "bet"),
            ("call", "call"),
            ("fold", "fold"),
            ("check", "check"),
            ("raise", "raise"),
            ("bet", "bet"),
        ):
            if lowered == keyword:
                return ParsedSpeechAction(
                    player=None,
                    action=action,
                    amount=None,
                    amount_type="unknown",
                    confidence=0.72,
                    raw_text=raw_text,
                )

        return ParsedSpeechAction(
            player=None,
            action="unknown",
            amount=None,
            amount_type="unknown",
            confidence=0.2,
            raw_text=raw_text,
        )


class LLMActionParser:
    def parse(self, raw_text: str) -> ParsedSpeechAction:
        raise NotImplementedError("Plug in an LLM-backed parser when needed.")


class ParserPipeline:
    def __init__(
        self,
        deterministic_parser: ActionParser | None = None,
        llm_parser: ActionParser | None = None,
    ) -> None:
        self.deterministic_parser = deterministic_parser or RuleBasedActionParser()
        self.llm_parser = llm_parser

    def parse(self, raw_text: str) -> ParsedSpeechAction:
        deterministic = self.deterministic_parser.parse(raw_text)
        if deterministic.action != "unknown" or self.llm_parser is None:
            return deterministic
        return self.llm_parser.parse(raw_text)


def build_action_parser(llm_parser: ActionParser | None = None) -> ActionParser:
    return ParserPipeline(llm_parser=llm_parser)
