from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - import failure depends on local environment
    cv2 = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_cards(frame: Any) -> list[dict[str, Any]]:
    """Placeholder card detection interface for future model integration."""
    _ = frame
    return []


class CameraManager:
    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.last_capture_at: str | None = None
        self.last_detected: list[dict[str, Any]] = []
        self.last_error: str | None = None

    def capture_frame(self) -> Any | None:
        if cv2 is None:
            self.last_error = "OpenCV is not installed or failed to import."
            return None

        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            self.last_error = "Could not open the default webcam."
            capture.release()
            return None

        ok, frame = capture.read()
        capture.release()
        if not ok:
            self.last_error = "Camera opened, but no frame was returned."
            return None

        self.last_capture_at = _utc_now()
        self.last_error = None
        self.last_detected = detect_cards(frame)
        return frame

    def status(self, refresh: bool = False) -> dict[str, Any]:
        if refresh:
            self.capture_frame()

        return {
            "available": cv2 is not None,
            "device_index": self.device_index,
            "last_capture_at": self.last_capture_at,
            "last_detected": self.last_detected,
            "last_error": self.last_error,
        }
