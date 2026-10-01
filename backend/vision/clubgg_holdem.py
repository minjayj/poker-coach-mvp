"""Conservative reader for the single-window ClubGG NLH layout in GamePlay 1.

This is intentionally a calibrated adapter, not a general computer-vision claim.
It abstains on other layouts, unreadable cards, overlays, and unknown actions.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from backend.vision.variant import detect_table_variant_frame


WIDTH, HEIGHT = 1404, 1025
RANKS = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}
# 24x24 binary masks sampled only from the suit glyphs in the supplied NLH
# recording. No player image or identifying information is embedded here.
SUIT_MASKS = {
    "S": "AP8AAf+AA//AA//AB//gD//wH//4H//4P//+f//+f///////////////////////f///f//+P+P8H5n4D5z4ADwAAH4AAH8A",
    "D": "Af8AAf+AA//AA//AD//gD//gP//4f//+///+///////////+P//4H//4D//gD//gA//AA//AAf8AAP8AAH4AAHwAADwAABAA",
    "C": "Af/AAf/AAf/AAf/AAf/AAf/AAf/AAP+ADxx4H5z8P//+f///////////////////////////f///P5z+Dhw4ADwAAD4AAP+A",
    "H": "f/P/f/f/f///f///////////f///f///f///f///H//+H//+D//4D//4B//4A//wAf/gAf/AAf/AAH+AAH+AAB4AAB4AAAwA",
}
TEMPLATES = {
    suit: np.unpackbits(np.frombuffer(base64.b64decode(data), dtype=np.uint8)).reshape(24, 24)
    for suit, data in SUIT_MASKS.items()
}
HERO_SUIT_MASKS = {
    "D": "ADAAADAAAHwAAHwAAP8AAf+AAf+AAf/AB//4B//4H//+H//+f//////////+f//+H//4D//4B//gA//gAP+AAP+AAD+AAAMA",
    "S": "AAgAAAwAABwAAH8AAP+AAf+AA//AB//gD//gf//4///8///+/////////////////////////////5//Pzf+APhgA/wAA/4A",
    "H": "AADwAAH4P8f+P8f+///+///////////////////////+///+///+P//4P//4H//wB//wB//wAf/AAf/AAP8AAP8AAD8AAD4A",
    "C": "AD4AAP+AAf/AA//AA//AA//gA//gA//AA//AAf/AAP+A/xhw/9n8///+/////////////////////7f/fjP+APgwA/wAA/4A",
}
HERO_TEMPLATES = {
    suit: np.unpackbits(np.frombuffer(base64.b64decode(data), dtype=np.uint8)).reshape(24, 24)
    for suit, data in HERO_SUIT_MASKS.items()
}


@dataclass
class VisionRead:
    status: str = "unsupported_layout"
    hero_cards: list[str] | None = None
    board_cards: list[str] | None = None
    current_pot: float | None = None
    visible_names: list[str] = field(default_factory=list)
    active_names: list[str] = field(default_factory=list)
    hero_player: str | None = None
    hero_turn: bool = False
    hero_to_call: float | None = None
    raise_to: float | None = None
    observed_profile: dict[str, object] | None = None
    bottom_name: str | None = None
    hole_cards_absent: bool = False
    missing: list[str] = field(default_factory=list)

    def public(self) -> dict[str, object]:
        return {
            "status": self.status,
            "hero_cards": self.hero_cards,
            "board_cards": self.board_cards,
            "current_pot": self.current_pot,
            "visible_names": self.visible_names,
            "active_names": self.active_names,
            "hero_player": self.hero_player,
            "hero_turn": self.hero_turn,
            "hero_to_call": self.hero_to_call,
            "raise_to": self.raise_to,
            "observed_profile": self.observed_profile,
            "bottom_name": self.bottom_name,
            "missing": self.missing,
        }


def _decode(image_data: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(image_data.split(",", 1)[1], validate=True)
        return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (IndexError, ValueError):
        return None


def _normalized_mask(crop: np.ndarray, red: bool) -> np.ndarray | None:
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if red:
        binary = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45)).astype(np.uint8)
    else:
        binary = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 95).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[component]
    if area < 100 or w < 10 or h < 10:
        return None
    glyph = (labels[y:y + h, x:x + w] == component).astype(np.uint8)
    return cv2.resize(glyph, (24, 24), interpolation=cv2.INTER_NEAREST)


def _classify_mask(mask: np.ndarray, red: bool, templates: dict[str, np.ndarray], threshold: float, margin: float) -> str | None:
    possibilities = "DH" if red else "SC"
    scores = sorted(((float((mask == templates[s]).mean()), s) for s in possibilities), reverse=True)
    return scores[0][1] if scores[0][0] >= threshold and scores[0][0] - scores[1][0] >= margin else None


def _suit(crop: np.ndarray, *, hero: bool = False, extras: dict[str, list[np.ndarray]] | None = None) -> str | None:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red_pixels = int(((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45)).sum())
    red = red_pixels > 80
    mask = _normalized_mask(crop, red)
    if mask is None:
        return None
    templates = HERO_TEMPLATES if hero else TEMPLATES
    base = _classify_mask(mask, red, templates, .78, .05)
    if base is not None:
        return base
    if extras:
        possibilities = "DH" if red else "SC"
        scores = sorted((
            (max((float((mask == example).mean()) for example in extras.get(suit, [])), default=0.0), suit)
            for suit in possibilities
        ), reverse=True)
        if scores[0][0] >= .84 and scores[0][0] - scores[1][0] >= .08:
            return scores[0][1]
    # In this ClubGG layout, the printed black board suits differ mainly
    # at the top: a spade has a narrow point, a club has a broad lobe.
    # Older templates score both highly, so abstention alone lost clear
    # board cards. Apply this only to plausible black board glyphs.
    if not hero and not red and max(float((mask == templates[s]).mean()) for s in "SC") >= .78:
        top_width = float(mask[:6].sum(axis=1).mean())
        if top_width <= 5.5:
            return "S"
        if top_width >= 7.5 and float((mask == templates["C"]).mean()) > float((mask == templates["S"]).mean()) + .025:
            return "C"
    return None


def _ocr_text(engine: object, crop: np.ndarray) -> tuple[str, float]:
    text, confidence = _ocr_raw(engine, crop)
    return text.upper(), confidence


def _ocr_raw(engine: object, crop: np.ndarray) -> tuple[str, float]:
    enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    result = engine(enlarged, use_det=False, use_cls=False, use_rec=True)
    if not result.txts:
        return "", 0.0
    return str(result.txts[0]).strip(), float(result.scores[0])


def _rank(engine: object, crop: np.ndarray) -> str | None:
    text, confidence = _ocr_text(engine, crop)
    if text == "T":
        text = "10"
    return ("T" if text == "10" else text) if text in RANKS and confidence >= .92 else None


def _white_card(frame: np.ndarray, x: int, y: int) -> bool:
    patch = frame[y + 8:y + 18, x + 48:x + 70]
    return bool(patch.size and (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) > 190).mean() > .75)


def _hero_card_present(frame: np.ndarray, x: int) -> bool:
    """Detect either exposed white area of an overlapping hole card.

    The right card can cover the first card's right-edge patch, and its own
    edge can be shaded by the table UI. Requiring that one particular patch
    be 75% white rejected clearly legible cards in the supplied capture.
    Rank/suit recognition below is still required before accepting a card.
    """
    gray = cv2.cvtColor(frame[763:773, x + 4:x + 70], cv2.COLOR_BGR2GRAY)
    if gray.shape != (10, 66):
        return False
    return max(
        float((gray[:, :40] > 190).mean()),
        float((gray[:, 44:] > 190).mean()),
    ) > .70


# Name boxes and card-back patches share this fixed seven-seat layout. A
# visible card back means "cards are displayed", not proof of a past action.
NAME_BOXES = (
    (439, 210, 601, 243), (803, 209, 972, 243),
    (69, 397, 235, 435), (1170, 402, 1355, 440),
    (1036, 798, 1210, 836), (610, 882, 803, 922),
    (190, 798, 385, 836),
)
CARD_BACK_BOXES = (
    (460, 145, 520, 185), (820, 145, 880, 185),
    (85, 335, 145, 375), (1190, 335, 1250, 375),
    (1055, 730, 1115, 770), None,
    (210, 730, 270, 770),
)


def _card_back_visible(frame: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    # The ClubGG crosshatch is both bright and edge-dense. Requiring both
    # prevents most circular player avatars from looking like active cards.
    return bool(gray.size and (gray > 160).mean() > .35
                and (cv2.Canny(gray, 80, 160) > 0).mean() > .25)


def _hero_action_visible(frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame[940:1015, 1030:1200], cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 100) & (hsv[:, :, 2] > 70)
    return bool(red.mean() > .80)


def _button_amount(engine: object, crop: np.ndarray) -> float | None:
    text, confidence = _ocr_text(engine, crop)
    if confidence < .85 or not re.fullmatch(r"\d+(?:\.\d{1,2})?", text):
        return None
    return float(text)


def _single_table_frame(frame: np.ndarray) -> np.ndarray | None:
    """Normalize a window share or find exactly one NLH table on a desktop.

    The ratios below come from the supplied ClubGG recording. A green felt
    candidate must occupy a substantial, filled ellipse-shaped region; two
    candidates are ambiguous and must not be silently combined.
    """
    height, width = frame.shape[:2]
    ratio = width / height
    if 1.31 <= ratio <= 1.43 and width >= 700 and height >= 500:
        return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA if width >= WIDTH else cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 65, 30), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    count, _, components, _ = cv2.connectedComponentsWithStats(green)
    candidates: list[np.ndarray] = []
    for x, y, felt_width, felt_height, area in components[1:count]:
        if (felt_width < 400 or felt_height < 180 or area < 20_000
                or not 1.7 <= felt_width / felt_height <= 2.65
                or area / (felt_width * felt_height) < .42):
            continue
        table_width = round(felt_width / .84)
        table_height = round(table_width * HEIGHT / WIDTH)
        left = round(x - .078 * table_width)
        top = round(y - .241 * table_height)
        if table_width > width or table_height > height:
            continue
        # Felt edges vary a few pixels with overlays and JPEG compression.
        # A maximized table can start exactly at the screen's top edge.
        clamped_left = min(max(left, 0), width - table_width)
        clamped_top = min(max(top, 0), height - table_height)
        if (abs(clamped_left - left) > .02 * table_width
                or abs(clamped_top - top) > .02 * table_height):
            continue
        crop = cv2.resize(
            frame[clamped_top:clamped_top + table_height, clamped_left:clamped_left + table_width],
            (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA if table_width >= WIDTH else cv2.INTER_CUBIC,
        )
        if detect_table_variant_frame(crop) is None:
            continue
        candidates.append(crop)
    return candidates[0] if len(candidates) == 1 else None


class ClubGGHoldemReader:
    def __init__(self, model_path: Path | None = None) -> None:
        self._engine = None
        self.model_path = model_path
        self._extra_hero: dict[str, list[np.ndarray]] = {}
        self._extra_board: dict[str, list[np.ndarray]] = {}
        self.reload_model()
        self._names_source = ""
        self._names_at = 0.0
        self._names_cache: list[str] = []
        self._seat_names_cache: list[str | None] = []
        self._bottom_cache: str | None = None

    def reload_model(self) -> None:
        self._extra_hero = {}
        self._extra_board = {}
        if not self.model_path or not self.model_path.is_file():
            return
        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                return
            for key, destination in (("hero", self._extra_hero), ("board", self._extra_board)):
                for suit, values in data.get(key, {}).items():
                    if suit in "SHDC":
                        destination[suit] = [
                            np.unpackbits(np.frombuffer(base64.b64decode(value, validate=True), dtype=np.uint8)).reshape(24, 24)
                            for value in values[:32]
                        ]
        except (OSError, ValueError, TypeError):
            self._extra_hero = {}
            self._extra_board = {}

    def _ocr(self) -> object:
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    def warmup(self) -> None:
        """Load local OCR models once before the first capture frame."""
        self._ocr()

    def read(self, image_data: str, source_id: str = "") -> VisionRead:
        frame = _decode(image_data)
        if frame is None:
            return VisionRead(status="invalid_frame")
        frame = _single_table_frame(frame)
        if frame is None:
            return VisionRead(status="unsupported_layout", missing=["I can't isolate one ClubGG table. Share one game window, or keep just one table visible on your screen."])
        result = VisionRead(status="partial", missing=["Prior betting actions and price to call need confirmation."])
        engine = self._ocr()

        # ClubGG's explicit profile modal is the only trustworthy source for
        # platform VPIP/PFR. Do not infer them from a single observed hand.
        header, header_score = _ocr_text(engine, frame[400:446, 640:760])
        if header == "NLH" and header_score >= .95:
            name, name_score = _ocr_raw(engine, frame[211:253, 550:715])
            hands, hands_score = _ocr_text(engine, frame[484:526, 475:550])
            vpip, vpip_score = _ocr_text(engine, frame[485:525, 596:690])
            pfr, pfr_score = _ocr_text(engine, frame[485:525, 709:811])
            if (name_score >= .95 and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_ -]{1,23}", name)
                    and hands_score >= .95 and hands.isdigit()
                    and vpip_score >= .95 and pfr_score >= .95
                    and re.fullmatch(r"\d+(?:\.\d+)?%", vpip)
                    and re.fullmatch(r"\d+(?:\.\d+)?%", pfr)):
                result.observed_profile = {
                    "player": name,
                    "hands_seen": int(hands),
                    "vpip": float(vpip[:-1]),
                    "pfr": float(pfr[:-1]),
                    "source": "clubgg_profile",
                    "confidence": .95,
                }
                result.status = "profile_read"
                result.missing = ["Profile captured; betting actions still need confirmation."]
                return result

        cue = detect_table_variant_frame(frame)
        if cue is None or cue[0] != "nlh":
            return VisionRead(status="not_holdem", missing=["The shared window is not a recognized Hold'em table."])

        if source_id and source_id == self._names_source and time.monotonic() - self._names_at < 5:
            result.visible_names = self._names_cache.copy()
            seat_names = self._seat_names_cache.copy()
            result.bottom_name = self._bottom_cache
        else:
            seat_names = [None] * len(NAME_BOXES)
            for index, (x1, y1, x2, y2) in enumerate(NAME_BOXES):
                name, score = _ocr_raw(engine, frame[y1:y2, x1:x2])
                if score >= .95 and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_ -]{1,23}", name) and name not in result.visible_names:
                    result.visible_names.append(name)
                    seat_names[index] = name
                    if index == 5:
                        result.bottom_name = name
            if source_id:
                self._names_source = source_id
                self._names_at = time.monotonic()
                self._names_cache = result.visible_names.copy()
                self._seat_names_cache = seat_names.copy()
                self._bottom_cache = result.bottom_name

        result.hero_player = result.bottom_name
        result.active_names = [name for name, box in zip(seat_names, CARD_BACK_BOXES)
                               if name and box is not None and _card_back_visible(frame, box)]

        if _hero_action_visible(frame) and result.bottom_name:
            label, label_score = _ocr_text(engine, frame[938:978, 1030:1200])
            if label_score >= .90 and label in {"CALL", "CHECK"}:
                result.hero_turn = True
                result.hero_to_call = 0.0 if label == "CHECK" else _button_amount(engine, frame[968:1024, 1030:1200])
                if result.hero_to_call is not None:
                    result.raise_to = _button_amount(engine, frame[968:1024, 1220:1390])
                    result.missing = ["The bettor and prior action are not verified; confirm them before using a player-specific read."]
                else:
                    result.missing = ["The amount to call is unclear; please confirm it and the prior action."]

        # Pot label: preserve only a complete OCR match, never a nearby stack.
        pot_crop = cv2.cvtColor(frame[355:416, 607:801], cv2.COLOR_BGR2GRAY)
        text, score = _ocr_text(engine, pot_crop)
        match = re.fullmatch(r"TOTAL\s*POT\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", text)
        if match and score >= .90:
            result.current_pot = float(match.group(1).replace(",", ""))
        else:
            result.missing.append("Pot amount is unreadable; please confirm it.")

        board: list[str] = []
        board_uncertain = False
        for i in range(5):
            x, y = 436 + 107 * i, 422
            if not _white_card(frame, x, y):
                if board:
                    break
                continue
            rank = _rank(engine, frame[y + 5:y + 55, x + 4:x + 57])
            suit = _suit(frame[y + 56:y + 98, x + 4:x + 44], extras=self._extra_board)
            if rank is None or suit is None:
                board_uncertain = True
                break
            board.append(rank + suit)
        if not board_uncertain and len(board) in {0, 3, 4, 5}:
            result.board_cards = board
        else:
            result.missing.append("Board cards are unclear; please confirm them.")

        # Hole cards overlap; read rank and the visible left suit glyph of each.
        hole: list[str] = []
        result.hole_cards_absent = not any(_hero_card_present(frame, x) for x in (615, 690))
        if not result.hole_cards_absent and result.bottom_name:
            result.active_names.append(result.bottom_name)
        for x in (615, 690):
            if not _hero_card_present(frame, x):
                break
            rank = _rank(engine, frame[755:804, x:x + 45])
            suit = _suit(frame[795:845, x + 5:x + 52], hero=True, extras=self._extra_hero)
            if rank is None or suit is None:
                break
            hole.append(rank + suit)
        if len(hole) == 2 and len(set(hole + (result.board_cards or []))) == 2 + len(result.board_cards or []):
            result.hero_cards = hole
        else:
            result.missing.append("Your two hole cards are unclear; please confirm them.")

        result.status = "read" if not result.missing else "partial"
        return result
