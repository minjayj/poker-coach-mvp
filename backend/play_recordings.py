"""Opt-in local window recordings and post-game, human-verified frame labels."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from backend.engine.hand_evaluator import normalize_cards


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlayRecordingStore:
    MAX_CHUNK_BYTES = 8_000_000
    MAX_VIDEO_BYTES = 4_000_000_000
    FRAME_INTERVAL_SECONDS = 3.0
    MAX_FRAMES_PER_SESSION = 5_000

    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "play"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._active: dict[str, str] = {}
        self._last_frame: dict[str, float] = {}

    def _directory(self, session_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise ValueError("Invalid play session ID.")
        path = self.root / session_id
        if not path.is_dir():
            raise ValueError("Play session not found.")
        return path

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def _manifest(self, session_id: str) -> dict:
        return json.loads((self._directory(session_id) / "session.json").read_text(encoding="utf-8"))

    def start(self, source_id: str, mime_type: str, consent: bool, game_variant: str) -> dict:
        if not consent:
            raise ValueError("Recording requires explicit consent.")
        if not source_id.startswith("clubgg-window-share-"):
            raise ValueError("Record a single shared game window, not a desktop or tab.")
        if not mime_type.startswith("video/webm"):
            raise ValueError("The browser must provide a WebM recording.")
        with self._lock:
            if source_id in self._active:
                raise ValueError("This window is already being recorded.")
            session_id = uuid.uuid4().hex
            directory = self.root / session_id
            (directory / "frames").mkdir(parents=True)
            manifest = {
                "id": session_id, "source_id": source_id, "game_variant": game_variant,
                "mime_type": mime_type, "status": "recording", "started_at": utc_now(),
                "stopped_at": None, "next_chunk": 0, "video_bytes": 0, "frame_count": 0,
            }
            self._write_json(directory / "session.json", manifest)
            self._active[source_id] = session_id
            return manifest

    def append_chunk(self, session_id: str, index: int, data: bytes) -> dict:
        if not data or len(data) > self.MAX_CHUNK_BYTES:
            raise ValueError("Recording chunk must be 1–8 MB or smaller.")
        with self._lock:
            manifest = self._manifest(session_id)
            if manifest["status"] != "recording" or self._active.get(manifest["source_id"]) != session_id:
                raise ValueError("Recording session is not active.")
            if index != manifest["next_chunk"]:
                raise ValueError(f"Expected chunk {manifest['next_chunk']}.")
            if manifest["video_bytes"] + len(data) > self.MAX_VIDEO_BYTES:
                raise ValueError("Recording reached its 4 GB local size limit.")
            with (self._directory(session_id) / "capture.webm").open("ab") as output:
                output.write(data)
            manifest["next_chunk"] += 1
            manifest["video_bytes"] += len(data)
            self._write_json(self._directory(session_id) / "session.json", manifest)
            return {"next_chunk": manifest["next_chunk"], "video_bytes": manifest["video_bytes"]}

    def stop(self, session_id: str, incomplete: bool = False) -> dict:
        with self._lock:
            manifest = self._manifest(session_id)
            if manifest["status"] != "recording":
                return manifest
            manifest["status"] = "incomplete" if incomplete or not manifest["next_chunk"] else "saved"
            manifest["stopped_at"] = utc_now()
            self._write_json(self._directory(session_id) / "session.json", manifest)
            self._active.pop(manifest["source_id"], None)
            self._last_frame.pop(session_id, None)
            return manifest

    def stop_source(self, source_id: str) -> None:
        with self._lock:
            session_id = self._active.get(source_id)
            if session_id:
                self.stop(session_id, incomplete=True)

    def list_sessions(self) -> list[dict]:
        sessions = []
        for path in self.root.glob("*/session.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["status"] == "recording" and self._active.get(item["source_id"]) != item["id"]:
                    item["status"] = "interrupted"
                sessions.append(item)
            except (OSError, ValueError, KeyError):
                continue
        return sorted(sessions, key=lambda item: item["started_at"], reverse=True)

    def get_session(self, session_id: str) -> dict:
        item = self._manifest(session_id)
        if item["status"] == "recording" and self._active.get(item["source_id"]) != session_id:
            item["status"] = "interrupted"
        return item

    def record_frame(self, source_id: str, image_data: str, vision: dict | None, hand_id: int) -> None:
        with self._lock:
            session_id = self._active.get(source_id)
            if not session_id:
                return
            now = time.monotonic()
            if now - self._last_frame.get(session_id, 0) < self.FRAME_INTERVAL_SECONDS:
                return
            manifest = self._manifest(session_id)
            if manifest["frame_count"] >= self.MAX_FRAMES_PER_SESSION:
                return
            try:
                image = base64.b64decode(image_data.split(",", 1)[1], validate=True)
            except (IndexError, ValueError):
                return
            if not image.startswith(b"\xff\xd8") or len(image) > self.MAX_CHUNK_BYTES:
                return
            frame_id = f"{manifest['frame_count']:06d}"
            directory = self._directory(session_id) / "frames"
            (directory / f"{frame_id}.jpg").write_bytes(image)
            self._write_json(directory / f"{frame_id}.json", {
                "id": frame_id, "captured_at": utc_now(), "hand_id": hand_id,
                "prediction": vision, "label": None,
            })
            manifest["frame_count"] += 1
            self._write_json(self._directory(session_id) / "session.json", manifest)
            self._last_frame[session_id] = now

    def record_event(self, kind: str, payload: dict) -> None:
        """Save confirmed/manual actions alongside every currently recorded window."""
        with self._lock:
            for session_id in set(self._active.values()):
                with (self._directory(session_id) / "events.jsonl").open("a", encoding="utf-8") as output:
                    output.write(json.dumps({"at": utc_now(), "kind": kind, "payload": payload}, ensure_ascii=False) + "\n")

    def list_frames(self, session_id: str) -> list[dict]:
        directory = self._directory(session_id) / "frames"
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]

    def list_events(self, session_id: str) -> list[dict]:
        path = self._directory(session_id) / "events.jsonl"
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8") as source:
            recent = deque(source, maxlen=1000)
        return [json.loads(line) for line in recent if line.strip()]

    def frame_path(self, session_id: str, frame_id: str) -> Path:
        if not re.fullmatch(r"\d{6}", frame_id):
            raise ValueError("Invalid frame ID.")
        path = self._directory(session_id) / "frames" / f"{frame_id}.jpg"
        if not path.is_file():
            raise ValueError("Frame not found.")
        return path

    def video_path(self, session_id: str) -> Path:
        path = self._directory(session_id) / "capture.webm"
        if not path.is_file():
            raise ValueError("No video was saved for this session.")
        return path

    def label_frame(self, session_id: str, frame_id: str, hero_cards: list[str], board_cards: list[str], action_summary: str) -> dict:
        self.frame_path(session_id, frame_id)
        if len(hero_cards) not in (0, 2) or len(board_cards) not in (0, 3, 4, 5):
            raise ValueError("Use zero or two Hero cards and zero, three, four, or five board cards.")
        cards = normalize_cards([*hero_cards, *board_cards])
        with self._lock:
            path = self._directory(session_id) / "frames" / f"{frame_id}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            data["label"] = {
                "verified": True, "hero_cards": [card.upper() for card in cards[:len(hero_cards)]],
                "board_cards": [card.upper() for card in cards[len(hero_cards):]],
                "action_summary": action_summary.strip()[:300], "reviewed_at": utc_now(),
            }
            self._write_json(path, data)
            return data
