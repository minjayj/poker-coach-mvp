"""Opt-in recording is local, bounded, reviewable, and not self-training."""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import create_app


def tiny_window_frame() -> str:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    encoded = cv2.imencode(".jpg", frame)[1].tobytes()
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode()


def test_record_upload_review_and_no_training_without_labels(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    source = "clubgg-window-share-test"
    frame = {"source_id": source, "image_data": tiny_window_frame()}
    assert client.post("/api/input/frame", json=frame).status_code == 200
    assert client.post("/api/play/start", json={"source_id": source, "mime_type": "video/webm", "consent": False}).status_code == 422
    started = client.post("/api/play/start", json={"source_id": source, "mime_type": "video/webm", "consent": True})
    assert started.status_code == 200
    session_id = started.json()["id"]
    assert client.post(f"/api/play/{session_id}/chunks/1", content=b"bad order").status_code == 422
    assert client.post(f"/api/play/{session_id}/chunks/0", content=b"webm chunk").status_code == 200
    assert client.post("/api/input/frame", json=frame).status_code == 200
    frames = client.get(f"/api/play/{session_id}/frames").json()
    assert len(frames) == 1
    frame_id = frames[0]["id"]
    assert client.get(f"/api/play/{session_id}/frames/{frame_id}/image").status_code == 200
    assert client.post(f"/api/play/{session_id}/frames/{frame_id}/label", json={
        "verified": False, "hero_cards": ["9D", "7S"], "board_cards": [],
    }).status_code == 422
    saved = client.post(f"/api/play/{session_id}/frames/{frame_id}/label", json={
        "verified": True, "hero_cards": ["9D", "7S"], "board_cards": [],
        "action_summary": "Opponent raised to 10",
    })
    assert saved.status_code == 200
    assert saved.json()["label"]["hero_cards"] == ["9D", "7S"]
    assert client.post("/api/play/train", json={}).json()["activated"] is False
    assert client.post(f"/api/play/{session_id}/stop", json={}).json()["status"] == "saved"
    assert client.get(f"/api/play/{session_id}/video").content == b"webm chunk"
    assert client.get("/api/play/sessions").json()[0]["frame_count"] == 1


def test_confirmed_actions_are_logged_with_recording(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    client.post("/api/config/seats", json={
        "seats": [
            {"seat": 1, "player": "FantaMJ", "stack": 100, "is_hero": True},
            {"seat": 2, "player": "Alex", "stack": 100},
        ], "hero_seat": 1, "blinds": {"small": .5, "big": 1},
    })
    source = "clubgg-window-share-events"
    client.post("/api/input/frame", json={"source_id": source, "image_data": tiny_window_frame()})
    session_id = client.post("/api/play/start", json={"source_id": source, "mime_type": "video/webm", "consent": True}).json()["id"]
    client.post("/api/action", json={"player": "Alex", "action": "raise", "amount": 3, "amount_type": "total"})
    events = client.get(f"/api/play/{session_id}/events").json()
    assert events[0]["kind"] == "confirmed_action"
    assert events[0]["payload"]["player"] == "Alex"
    client.post("/api/input/stop", json={"source_id": source})
    assert client.get("/api/play/sessions").json()[0]["status"] == "incomplete"
