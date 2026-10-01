"""Provider-neutral latest-frame and event de-duplication boundary."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time

from backend.models import FrameInputRequest
from backend.vision.variant import detect_table_variant
from backend.vision.clubgg_holdem import ClubGGHoldemReader, VisionRead
from backend.models import InputObservation


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InputStreamHub:
    """Keeps only the most recent preview; a CNN/camera can be swapped freely."""

    MAX_DATA_URL_CHARS = 4_500_000
    STATIC_FRAME_SECONDS = 12

    def __init__(self, model_path: Path | None = None) -> None:
        self.latest_image_data: str | None = None
        self.source_id: str | None = None
        self.last_received_at: str | None = None
        self.last_observation_at: str | None = None
        self.last_error: str | None = None
        self._frame_digest: bytes | None = None
        self._last_changed_monotonic: float | None = None
        self._identical_frame_count = 0
        self._recent_event_ids: deque[str] = deque(maxlen=2_000)
        self._stopped_sources: deque[str] = deque(maxlen=100)
        self._variant_candidate: str | None = None
        self._variant_votes = 0
        self._variant_source: str | None = None
        self.holdem_reader = ClubGGHoldemReader(model_path)
        self._read_source: str | None = None
        self._read_candidates: dict[str, tuple[object, int]] = {}
        self._read_emitted: dict[str, object] = {}
        self.latest_vision: VisionRead | None = None

    def stable_holdem_observation(self, source_id: str, read: VisionRead) -> InputObservation | None:
        """Emit changed, identical readings only after two successive frames."""
        if source_id != self._read_source:
            self._read_source = source_id
            self._read_candidates = {}
            self._read_emitted = {}
        values: dict[str, object] = {
            "hero_cards": read.hero_cards,
            "board_cards": read.board_cards,
            "current_pot": read.current_pot,
            "visible_players": read.visible_names if read.visible_names else None,
            "active_players": read.active_names if read.visible_names else None,
            "hero_player": read.hero_player,
            "hero_turn": read.hero_turn,
            "hero_to_call": read.hero_to_call if read.hero_turn else None,
            "raise_to": read.raise_to if read.hero_turn else None,
            "observed_profile": read.observed_profile,
        }
        changed: dict[str, object] = {}
        for key, value in values.items():
            if value is None:
                self._read_candidates.pop(key, None)
                continue
            old, votes = self._read_candidates.get(key, (None, 0))
            votes = votes + 1 if old == value else 1
            self._read_candidates[key] = (value, votes)
            if votes >= 2 and self._read_emitted.get(key) != value:
                changed[key] = value
                self._read_emitted[key] = value
        if not changed:
            return None
        profile = changed.pop("observed_profile", None)
        return InputObservation.model_validate({
            **changed,
            "player_stats": [profile] if profile else [],
            "confidence": 0.90,
            "field_confidences": {key: 0.90 for key in changed},
        })

    def observed_variant(self, source_id: str, image_data: str | None) -> tuple[str, float] | None:
        """Require three matching window frames before changing strategy type."""
        if not image_data or not source_id.startswith("clubgg-window-share-"):
            return None
        result = detect_table_variant(image_data)
        if source_id != self._variant_source or result is None or result[0] != self._variant_candidate:
            self._variant_source = source_id
            self._variant_candidate = result[0] if result else None
            self._variant_votes = 1 if result else 0
        else:
            self._variant_votes += 1
        return result if self._variant_votes >= 3 and self._variant_votes % 3 == 0 else None

    def ingest(self, request: FrameInputRequest) -> tuple[list[str], bool]:
        """Store preview metadata and return new action-event ids only once."""
        if request.source_id in self._stopped_sources:
            raise ValueError("This capture source has stopped; start a new capture session.")
        if request.source_id != self.source_id:
            self.latest_vision = None
        if request.image_data is not None:
            if not request.image_data.startswith("data:image/"):
                raise ValueError("image_data must be an image data URL.")
            if len(request.image_data) > self.MAX_DATA_URL_CHARS:
                raise ValueError("Frame is too large; send a compressed JPEG/PNG under 4.5 MB.")
            digest = hashlib.sha256(request.image_data.encode("utf-8")).digest()
            if digest != self._frame_digest or request.source_id != self.source_id:
                self._frame_digest = digest
                self._last_changed_monotonic = time.monotonic()
                self._identical_frame_count = 1
            else:
                self._identical_frame_count += 1
            self.latest_image_data = request.image_data

        self.source_id = request.source_id.strip() or "external"
        self.last_received_at = utc_now()
        new_event_ids: list[str] = []
        if request.observation is not None:
            self.last_observation_at = self.last_received_at
            for event in request.observation.actions:
                if event.event_id not in self._recent_event_ids:
                    self._recent_event_ids.append(event.event_id)
                    new_event_ids.append(event.event_id)
        return new_event_ids, request.observation is not None

    def status(self) -> dict[str, object]:
        unchanged_seconds = (time.monotonic() - self._last_changed_monotonic) if self._last_changed_monotonic is not None else None
        return {
            "source_id": self.source_id,
            "last_received_at": self.last_received_at,
            "last_observation_at": self.last_observation_at,
            "has_preview": self.latest_image_data is not None,
            "last_error": self.last_error,
            "unchanged_seconds": round(unchanged_seconds, 1) if unchanged_seconds is not None else None,
            "frame_static": bool(unchanged_seconds is not None and unchanged_seconds >= self.STATIC_FRAME_SECONDS
                                 and self._identical_frame_count >= 4),
        }

    def clear_if_current(self, source_id: str) -> bool:
        """Clear a stopped browser source without erasing a newer adapter feed."""
        if source_id not in self._stopped_sources:
            self._stopped_sources.append(source_id)
        if self.source_id != source_id:
            return False
        self._variant_source = None
        self._variant_candidate = None
        self._variant_votes = 0
        self._read_source = None
        self._read_candidates = {}
        self._read_emitted = {}
        self.latest_image_data = None
        self.latest_vision = None
        self._frame_digest = None
        self._last_changed_monotonic = None
        self._identical_frame_count = 0
        self.source_id = None
        self.last_received_at = None
        self.last_observation_at = None
        return True

    def latest_frame(self) -> dict[str, object]:
        return {"image_data": self.latest_image_data, **self.status()}
