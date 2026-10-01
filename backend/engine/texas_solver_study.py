"""Exact-match, post-hand reader for TexasSolver's exported strategy trees.

TexasSolver is not invoked in a live request. Its output is useful only with
the same board, ranges, betting tree, pot, stack, rake and action path used to
build the original solve. This module deliberately refuses fuzzy matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.engine.hand_evaluator import normalize_cards


class TexasSolverSpot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board: list[str] = Field(min_length=3, max_length=5)
    pot: float = Field(gt=0)
    to_call: float = Field(ge=0)
    effective_stack: float = Field(gt=0)
    rake_percent: float = Field(ge=0, le=100)
    hero_position: Literal["IP", "OOP"]
    action_path: list[str] = Field(default_factory=list)
    game_format: Literal["cash"] = "cash"
    players: Literal[2] = 2

    @model_validator(mode="after")
    def valid_cards_and_amounts(self) -> "TexasSolverSpot":
        cards = normalize_cards(self.board)
        if len(cards) != len(set(cards)):
            raise ValueError("Board contains duplicate cards.")
        if any(not math.isfinite(amount) for amount in (self.pot, self.to_call, self.effective_stack, self.rake_percent)):
            raise ValueError("Amounts must be finite.")
        if self.to_call > self.effective_stack:
            raise ValueError("Call price exceeds the effective stack.")
        if any(not item or len(item) > 80 for item in self.action_path):
            raise ValueError("Action path has an empty or overlong label.")
        return self


class TexasSolverLookup(TexasSolverSpot):
    hero_hand: list[str] = Field(min_length=2, max_length=2)


class AdjustedStudyRequest(TexasSolverLookup):
    opponent_id: str = Field(min_length=1, max_length=150)
    proposed_raise_to: float | None = Field(default=None, ge=0)
    observed_opponent_bet_to_pot: float | None = Field(default=None, ge=0, le=10)

    @model_validator(mode="after")
    def valid_raise(self) -> "AdjustedStudyRequest":
        if self.proposed_raise_to is not None and not self.to_call < self.proposed_raise_to <= self.effective_stack:
            raise ValueError("proposed_raise_to must exceed the call price and fit the effective stack.")
        return self


def _canonical_spot(spot: TexasSolverSpot) -> dict[str, Any]:
    data = {key: getattr(spot, key) for key in TexasSolverSpot.model_fields}
    data["board"] = sorted(normalize_cards(spot.board))
    # Decimal chip amounts are matched exactly to cents; no nearest-neighbor
    # substitution of a different SPR, rake, or price.
    for key in ("pot", "to_call", "effective_stack", "rake_percent"):
        data[key] = round(data[key], 2)
    return data


def _target_node(export: dict[str, Any], path: list[str]) -> dict[str, Any]:
    node = export
    for action in path:
        children = node.get("childrens")
        if not isinstance(children, dict) or action not in children:
            raise ValueError(f"TexasSolver export has no node for action {action!r}.")
        node = children[action]
    if not isinstance(node, dict):
        raise ValueError("TexasSolver action node is invalid.")
    return node


def _validated_strategy(node: dict[str, Any]) -> tuple[list[str], dict[str, list[float]]]:
    strategy = node.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("Selected node has no solved strategy.")
    actions = strategy.get("actions")
    hands = strategy.get("strategy")
    if not isinstance(actions, list) or not actions or len(actions) > 20 or not all(isinstance(a, str) and a for a in actions):
        raise ValueError("TexasSolver action labels are invalid.")
    if not isinstance(hands, dict) or not hands:
        raise ValueError("TexasSolver hand strategy is missing.")
    clean: dict[str, list[float]] = {}
    for combo, frequencies in hands.items():
        if not isinstance(combo, str) or len(combo) != 4 or not isinstance(frequencies, list) or len(frequencies) != len(actions):
            raise ValueError("TexasSolver combo or frequency vector is malformed.")
        cards = normalize_cards([combo[:2], combo[2:]])
        if cards[0] == cards[1]:
            raise ValueError("TexasSolver combo repeats a card.")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or value > 1 for value in frequencies):
            raise ValueError("TexasSolver frequency is not a probability.")
        if abs(sum(frequencies) - 1.0) > 0.005:
            raise ValueError("TexasSolver action frequencies do not sum to one.")
        key = "".join(sorted(cards))
        if key in clean:
            raise ValueError("TexasSolver export repeats a hand combo.")
        clean[key] = [round(float(value), 6) for value in frequencies]
    return actions, clean


def import_texas_solver_export(
    export_path: Path, manifest_path: Path, library_dir: Path
) -> Path:
    """Convert one native JSON node to a bounded, provenance-tagged library item.

    The manifest is supplied by the person who configured the solve; the raw
    export alone does not prove pot, rake, stack, ranges, or position. A wrong
    manifest produces a wrong comparison, so study results remain advisory.
    """
    if export_path.stat().st_size > 100_000_000:
        raise ValueError("TexasSolver export exceeds the 100 MB import limit.")
    raw = export_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("provider") != "TexasSolver":
        raise ValueError("Manifest must identify provider TexasSolver.")
    spot = TexasSolverSpot.model_validate(manifest.get("spot"))
    range_notes = manifest.get("range_notes")
    if not isinstance(range_notes, str) or not range_notes.strip():
        raise ValueError("Manifest must describe both assumed player ranges.")
    export = json.loads(raw)
    actions, hands = _validated_strategy(_target_node(export, spot.action_path))
    board = set(normalize_cards(spot.board))
    hands = {combo: vector for combo, vector in hands.items() if not board.intersection((combo[:2], combo[2:]))}
    if not hands:
        raise ValueError("No unblocked hand combos remain in the selected strategy.")
    record = {
        "schema_version": 1,
        "provider": "TexasSolver",
        "study_only": True,
        "spot": _canonical_spot(spot),
        "actions": actions,
        "hands": hands,
        "range_notes": range_notes.strip(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_name": export_path.name,
    }
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()
    library_dir.mkdir(parents=True, exist_ok=True)
    target = library_dir / f"{digest}.json"
    target.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    return target


def lookup_texas_solver(library_dir: Path, query: TexasSolverLookup) -> dict[str, Any] | None:
    cards = normalize_cards(query.hero_hand)
    if cards[0] == cards[1] or set(cards).intersection(normalize_cards(query.board)):
        raise ValueError("Hero cards must be distinct and cannot overlap the board.")
    wanted = _canonical_spot(query)
    combo = "".join(sorted(cards))
    for path in library_dir.glob("*.json") if library_dir.exists() else ():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != 1 or record.get("provider") != "TexasSolver" or record.get("spot") != wanted:
            continue
        frequencies = record["hands"].get(combo)
        if frequencies is None:
            return None
        return {
            "matched": True,
            "study_only": True,
            "provider": "TexasSolver",
            "actions": record["actions"],
            "action_frequencies": dict(zip(record["actions"], frequencies)),
            "hero_combo": combo,
            "spot": wanted,
            "range_notes": record["range_notes"],
            "source_sha256": record["source_sha256"],
            "source_name": record["source_name"],
            "caveat": "This is one assumed heads-up cash-game tree, not a solution to the observed hand or a profitability guarantee.",
        }
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import one TexasSolver JSON node for exact post-hand study.")
    parser.add_argument("export", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("library", type=Path)
    arguments = parser.parse_args()
    print(import_texas_solver_export(arguments.export, arguments.manifest, arguments.library))
