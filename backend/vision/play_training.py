"""Offline suit calibration from reviewed local play frames, never self-labels."""

from __future__ import annotations

import base64
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend.vision.clubgg_holdem import (
    HEIGHT, WIDTH, HERO_TEMPLATES, TEMPLATES, _classify_mask,
    _normalized_mask, _white_card,
)


def _extract(frame: np.ndarray, cards: list[str], hero: bool) -> list[tuple[str, np.ndarray]]:
    output: list[tuple[str, np.ndarray]] = []
    for index, card in enumerate(cards):
        suit = card[-1].upper()
        if hero:
            x = (615, 690)[index]
            if not _white_card(frame, x, 755):
                continue
            crop = frame[795:845, x + 5:x + 52]
        else:
            x, y = 436 + 107 * index, 422
            if not _white_card(frame, x, y):
                continue
            crop = frame[y + 56:y + 98, x + 4:x + 44]
        mask = _normalized_mask(crop, suit in "DH")
        if mask is not None:
            output.append((suit, mask))
    return output


def _predict(mask: np.ndarray, suit: str, hero: bool, extras: dict[str, list[np.ndarray]] | None) -> str | None:
    red = suit in "DH"
    base = _classify_mask(mask, red, HERO_TEMPLATES if hero else TEMPLATES, .78, .05)
    if base is not None or not extras:
        return base
    possibilities = "DH" if red else "SC"
    scores = sorted((
        (max((float((mask == example).mean()) for example in extras.get(candidate, [])), default=0.0), candidate)
        for candidate in possibilities
    ), reverse=True)
    return scores[0][1] if scores[0][0] >= .84 and scores[0][0] - scores[1][0] >= .08 else None


def train_reviewed_suits(play_root: Path) -> dict[str, object]:
    """Promote only when held-out reviewed frames improve coverage without errors."""
    labeled: list[tuple[str, list[tuple[bool, str, np.ndarray]]]] = []
    variants: dict[Path, str] = {}
    card_patterns: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for metadata_path in sorted(play_root.glob("*/frames/*.json")):
        try:
            session_dir = metadata_path.parent.parent
            if session_dir not in variants:
                variants[session_dir] = json.loads((session_dir / "session.json").read_text(encoding="utf-8")).get("game_variant", "")
            if variants[session_dir] != "nlh":
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            label = metadata.get("label") or {}
            if not label.get("verified"):
                continue
            image = cv2.imread(str(metadata_path.with_suffix(".jpg")))
            if image is None:
                continue
            height, width = image.shape[:2]
            if not (1.31 <= width / height <= 1.43 and width >= 1000 and height >= 700):
                continue
            image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            samples = [(True, suit, mask) for suit, mask in _extract(image, label.get("hero_cards", []), True)]
            samples += [(False, suit, mask) for suit, mask in _extract(image, label.get("board_cards", []), False)]
            if samples:
                labeled.append((str(metadata_path), samples))
                card_patterns.add((tuple(label.get("hero_cards", [])), tuple(label.get("board_cards", []))))
        except (OSError, ValueError, KeyError, IndexError):
            continue
    report: dict[str, object] = {"reviewed_frames": len(labeled), "unique_card_patterns": len(card_patterns), "activated": False}
    if len(labeled) < 12:
        report["reason"] = "Review at least 12 distinct captured frames with visible labeled cards."
        return report
    if len(card_patterns) < 4:
        report["reason"] = "Review at least four different hands or boards before training."
        return report

    # Hold out whole frames, not individual cards within a frame. Nearby video
    # frames remain correlated, so this is a smoke test, not general accuracy.
    held_out = [samples for index, (_, samples) in enumerate(labeled) if index % 5 == 0]
    training = [samples for index, (_, samples) in enumerate(labeled) if index % 5 != 0]
    if len(held_out) < 3:
        report["reason"] = "Not enough independent review frames for validation."
        return report
    banks: dict[bool, dict[str, list[np.ndarray]]] = {True: defaultdict(list), False: defaultdict(list)}
    seen: dict[bool, dict[str, set[bytes]]] = {True: defaultdict(set), False: defaultdict(set)}
    for samples in training:
        for hero, suit, mask in samples:
            packed = np.packbits(mask.ravel()).tobytes()
            if packed not in seen[hero][suit] and len(banks[hero][suit]) < 32:
                seen[hero][suit].add(packed)
                banks[hero][suit].append(mask)
    baseline_correct = candidate_correct = baseline_errors = candidate_errors = total = 0
    for samples in held_out:
        for hero, suit, mask in samples:
            baseline = _predict(mask, suit, hero, None)
            candidate = _predict(mask, suit, hero, banks[hero])
            total += 1
            baseline_correct += baseline == suit
            candidate_correct += candidate == suit
            baseline_errors += baseline is not None and baseline != suit
            candidate_errors += candidate is not None and candidate != suit
    report.update({
        "validation_cards": total,
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "baseline_errors": baseline_errors,
        "candidate_errors": candidate_errors,
        "template_counts": {
            "hero": {suit: len(banks[True][suit]) for suit in "SHDC"},
            "board": {suit: len(banks[False][suit]) for suit in "SHDC"},
        },
    })
    if total < 8 or candidate_errors or candidate_correct <= baseline_correct:
        report["reason"] = "Candidate did not show a safe held-out improvement; current reader was kept."
        return report
    data = {
        "version": 1, "trained_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_frames": len(labeled), "validation_cards": total,
        "baseline_correct": baseline_correct, "candidate_correct": candidate_correct,
        "hero": {}, "board": {},
    }
    for hero, key in ((True, "hero"), (False, "board")):
        data[key] = {
            suit: [base64.b64encode(np.packbits(mask.ravel()).tobytes()).decode() for mask in banks[hero][suit]]
            for suit in "SHDC"
        }
    path = play_root / "suit_model.json"
    temporary = play_root / "suit_model.json.tmp"
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)
    report["activated"] = True
    report["model_path"] = str(path)
    return report
