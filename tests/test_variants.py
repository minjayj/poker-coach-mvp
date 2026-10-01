import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.engine.plo5 import best_plo5_rank, estimate_plo5_equity
from backend.main import create_app
from backend.vision.variant import detect_table_variant


def felt_data_url(hue: int) -> str:
    pixel = cv2.cvtColor(np.uint8([[[hue, 180, 130]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frame = np.full((480, 640, 3), pixel, dtype=np.uint8)
    encoded = cv2.imencode(".jpg", frame)[1].tobytes()
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def test_variant_detector_abstains_and_requires_three_window_frames(tmp_path) -> None:
    nlh = felt_data_url(60)
    plo5 = felt_data_url(105)
    assert detect_table_variant(nlh)[0] == "nlh"
    assert detect_table_variant(plo5)[0] == "plo5"
    assert detect_table_variant("data:image/jpeg;base64,AA==") is None
    client = TestClient(create_app(tmp_path / "data"))
    for index in range(3):
        response = client.post("/api/input/frame", json={"source_id": "clubgg-window-share-test", "image_data": plo5})
        assert response.status_code == 200
        assert response.json()["state"]["game_variant"] == ("plo5" if index == 2 else "nlh")


def test_variant_profiles_and_plo5_coach_are_separate(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    seats = {"seats": [
        {"seat": 1, "player": "FantaMJ", "stack": 100, "is_hero": True},
        {"seat": 2, "player": "Alex", "stack": 100},
    ], "hero_seat": 1}
    assert client.post("/api/config/seats", json=seats).status_code == 200
    client.post("/api/action", json={"player": "Alex", "action": "raise", "amount": 4, "amount_type": "total"})
    client.post("/api/hand/end", json={"winner": "Alex"})
    assert client.get("/api/players/Alex").json()["profile"]["local_hands_seen"] == 1

    switched = client.post("/api/config/game-format", json={"game_format": "cash", "game_variant": "plo5"})
    assert switched.status_code == 200
    assert client.get("/api/players/Alex").json()["profile"]["local_hands_seen"] == 0
    client.post("/api/players/Alex/observed-stats", json={"hands_seen": 40, "vpip": 55, "pfr": 12})
    assert client.get("/api/players/Alex").json()["profile"]["vpip"] == 0.55
    cards = ["Ah", "Kh", "Qh", "Jd", "Tc"]
    assert client.post("/api/vision/card_manual", json={"hero_cards": cards, "board_cards": ["2h", "3h", "4s"]}).status_code == 200
    client.post("/api/input/frame", json={"source_id": "manual-context", "observation": {"current_pot": 20}})
    brief = client.post("/api/coach/brief", json={"opponent_id": "Alex", "to_call": 4, "simulations": 100}).json()
    assert brief["ready"] is True
    assert brief["solution"]["decision_model"] == "plo5_random_opponent"
    assert client.post("/api/solver/recommend", json={"hero_hand": ["Ah", "Kh"], "current_pot": 20}).status_code == 422
    client.post("/api/action", json={"player": "Alex", "action": "call", "amount": 2})
    client.post("/api/hand/end", json={"winner": "FantaMJ"})
    assert client.get("/api/players/Alex").json()["profile"]["local_hands_seen"] == 1
    history = client.get("/api/players/Alex/history").json()
    assert {hand["game_variant"] for hand in history["hands"]} == {"nlh", "plo5"}

    client.post("/api/config/game-format", json={"game_format": "cash", "game_variant": "nlh"})
    nlh_profile = client.get("/api/players/Alex").json()["profile"]
    assert nlh_profile["local_hands_seen"] == 1
    assert nlh_profile["observed_hands_seen"] is None


def test_plo5_exact_two_hole_and_three_board() -> None:
    # Three aces in the hole do not make three of a kind on an unpaired board.
    rank = best_plo5_rank(("Ah", "Ac", "Ad", "Kc", "Qc"), ("2s", "3h", "4d", "5s", "9c"))
    comparison = best_plo5_rank(("Ah", "Ac", "Kd", "Kc", "Qc"), ("2s", "3h", "4d", "5s", "9c"))
    assert rank == comparison
    result = estimate_plo5_equity(["Ah", "Kh", "Qh", "Jd", "Tc"], ["2h", "3h", "4s"], 50, seed=1)
    assert 0 <= result["equity"] <= 1
    assert result["samples"] == 50
