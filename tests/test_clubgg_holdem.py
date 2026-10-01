"""The window adapter abstains on ambiguity and requires temporal agreement."""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.input_stream import InputStreamHub
from backend.main import create_app
from backend.vision.clubgg_holdem import ClubGGHoldemReader, VisionRead, _hero_card_present, _single_table_frame, _suit


def test_suit_masks_and_unsupported_layout() -> None:
    # Reconstructed glyphs exercise the classifier without shipping a player's
    # personal recording or screenshot as a test fixture.
    from backend.vision.clubgg_holdem import HERO_TEMPLATES

    for suit, mask in HERO_TEMPLATES.items():
        crop = np.full((45, 45, 3), 245, dtype=np.uint8)
        color = (0, 0, 180) if suit in "DH" else (20, 20, 20)
        crop[7:31, 8:32][mask.astype(bool)] = color
        assert _suit(crop, hero=True) == suit
    frame = np.zeros((600, 900, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", frame)
    data = "data:image/jpeg;base64," + base64.b64encode(encoded).decode()
    assert ClubGGHoldemReader().read(data).status == "unsupported_layout"


def test_overlapping_hero_cards_can_use_either_visible_white_patch() -> None:
    frame = np.zeros((1025, 1404, 3), dtype=np.uint8)
    # The left card exposes its right edge; the overlapping right card has a
    # shaded right edge but leaves a larger white area near the rank.
    frame[763:773, 663:685] = 245
    frame[763:773, 694:734] = 245
    assert _hero_card_present(frame, 615)
    assert _hero_card_present(frame, 690)
    assert not _hero_card_present(frame, 500)


def test_one_table_can_be_isolated_from_screen_but_two_cannot() -> None:
    window = np.full((1025, 1404, 3), (18, 18, 18), dtype=np.uint8)
    cv2.ellipse(window, (700, 505), (590, 260), 0, 0, 360, (35, 120, 45), -1)
    desktop = np.full((1080, 1728, 3), (80, 60, 50), dtype=np.uint8)
    desktop[:1025, 162:1566] = window
    isolated = _single_table_frame(desktop)
    assert isolated is not None
    assert isolated.shape == (1025, 1404, 3)

    smaller = cv2.resize(window, (700, 511))
    two_tables = np.full((700, 1600, 3), (80, 60, 50), dtype=np.uint8)
    two_tables[80:591, 50:750] = smaller
    two_tables[80:591, 850:1550] = smaller
    assert _single_table_frame(two_tables) is None


def test_two_identical_frames_required_for_state_update(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path / "data")
    read = VisionRead(status="partial", hero_cards=["9D", "7S"], board_cards=[], current_pot=2.5)
    monkeypatch.setattr(app.state.input_stream.holdem_reader, "read", lambda *_: read)
    client = TestClient(app)
    payload = {"source_id": "clubgg-window-share-test", "image_data": "data:image/jpeg;base64,AA=="}
    first = client.post("/api/input/frame", json=payload)
    assert first.status_code == 200
    assert first.json()["vision"]["hero_cards"] == ["9D", "7S"]
    assert first.json()["state"]["hero_cards"] == []
    second = client.post("/api/input/frame", json=payload)
    assert second.status_code == 200
    state = second.json()["state"]
    assert state["hero_cards"] == ["9D", "7S"]
    assert state["board_cards"] == []
    assert state["observed_pot"] == 2.5
    assert state["last_input_verified_spot"] is False
    assert second.json()["new_action_events"] == 0


def test_screen_share_routes_single_table_to_reader(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path / "data")
    read = VisionRead(status="partial", hero_cards=["9D", "7S"], board_cards=[], current_pot=2.5)
    monkeypatch.setattr(app.state.input_stream.holdem_reader, "read", lambda *_: read)
    client = TestClient(app)
    payload = {"source_id": "clubgg-monitor-share-test", "image_data": "data:image/jpeg;base64,AA=="}
    first = client.post("/api/input/frame", json=payload)
    assert first.status_code == 200
    assert first.json()["vision"]["hero_cards"] == ["9D", "7S"]
    second = client.post("/api/input/frame", json=payload)
    assert second.json()["state"]["hero_cards"] == ["9D", "7S"]


def test_unreadable_frame_cannot_clear_last_confirmed_cards() -> None:
    hub = InputStreamHub()
    source = "clubgg-window-share-a"
    good = VisionRead(hero_cards=["AH", "7C"])
    assert hub.stable_holdem_observation(source, good) is None
    assert hub.stable_holdem_observation(source, good).hero_cards == ["AH", "7C"]
    assert hub.stable_holdem_observation(source, VisionRead(hero_cards=None)) is None
    assert hub.stable_holdem_observation(source, good) is None


def test_static_capture_is_flagged_and_recovers(monkeypatch) -> None:
    from backend.models import FrameInputRequest

    clock = [100.0]
    monkeypatch.setattr("backend.input_stream.time.monotonic", lambda: clock[0])
    hub = InputStreamHub()
    source = "clubgg-window-share-static"
    original = FrameInputRequest(source_id=source, image_data="data:image/jpeg;base64,AA==")
    hub.ingest(original)
    clock[0] += 13
    hub.ingest(original)
    assert hub.status()["frame_static"] is False
    hub.ingest(original)
    hub.ingest(original)
    assert hub.status()["frame_static"] is True
    hub.ingest(FrameInputRequest(source_id=source, image_data="data:image/jpeg;base64,AQ=="))
    assert hub.status()["frame_static"] is False
    hub.clear_if_current(source)
    assert hub.status()["unchanged_seconds"] is None


def test_displayed_profile_requires_two_reads_and_stays_separate(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path / "data")
    read = VisionRead(status="profile_read", observed_profile={
        "player": "FantaMJ", "hands_seen": 204, "vpip": 46.39,
        "pfr": 15.46, "source": "clubgg_profile", "confidence": .95,
    })
    monkeypatch.setattr(app.state.input_stream.holdem_reader, "read", lambda *_: read)
    client = TestClient(app)
    client.post("/api/config/seats", json={
        "seats": [{"seat": 1, "player": "FantaMJ", "stack": 100, "is_hero": True}],
        "hero_seat": 1, "blinds": {"small": .5, "big": 1},
    })
    payload = {"source_id": "clubgg-window-share-profile", "image_data": "data:image/jpeg;base64,AA=="}
    client.post("/api/input/frame", json=payload)
    assert client.get("/api/players/FantaMJ").json()["profile"]["observed_hands_seen"] is None
    client.post("/api/input/frame", json=payload)
    profile = client.get("/api/players/FantaMJ").json()["profile"]
    assert profile["observed_hands_seen"] == 204
    assert round(profile["observed_vpip"], 4) == .4639
    assert round(profile["observed_pfr"], 4) == .1546
    assert profile["local_hands_seen"] == 0


def test_hero_cards_clear_only_when_confirmed_hero_seat_is_empty(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path / "data")
    current = {"read": VisionRead(hero_cards=["9D", "7S"], board_cards=[], current_pot=2.5)}
    monkeypatch.setattr(app.state.input_stream.holdem_reader, "read", lambda *_: current["read"])
    client = TestClient(app)
    client.post("/api/config/seats", json={
        "seats": [{"seat": 1, "player": "FantaMJ", "stack": 100, "is_hero": True}],
        "hero_seat": 1,
    })
    payload = {"source_id": "clubgg-window-share-clear", "image_data": "data:image/jpeg;base64,AA=="}
    client.post("/api/input/frame", json=payload)
    client.post("/api/input/frame", json=payload)
    assert client.get("/api/state").json()["hero_cards"] == ["9D", "7S"]
    current["read"] = VisionRead(hole_cards_absent=True, bottom_name="FantaMJ")
    client.post("/api/input/frame", json=payload)
    assert client.get("/api/state").json()["hero_cards"] == ["9D", "7S"]
    client.post("/api/input/frame", json=payload)
    assert client.get("/api/state").json()["hero_cards"] == []


def test_live_names_active_seats_and_call_price_reach_coach(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path / "data")
    current = {"read": VisionRead(
        status="partial", hero_cards=["KC", "3S"], board_cards=["8H", "QC", "TC"],
        current_pot=5.20, visible_names=["FantaMJ", "McFelon", "oksb"],
        active_names=["FantaMJ", "McFelon", "oksb"], hero_player="FantaMJ",
        hero_turn=True, hero_to_call=2.60, raise_to=5.20,
    )}
    monkeypatch.setattr(app.state.input_stream.holdem_reader, "read", lambda *_: current["read"])
    client = TestClient(app)
    payload = {"source_id": "clubgg-window-share-live", "image_data": "data:image/jpeg;base64,AA=="}
    client.post("/api/input/frame", json=payload)
    state = client.post("/api/input/frame", json=payload).json()["state"]
    assert state["observed_players"] == ["FantaMJ", "McFelon", "oksb"]
    assert state["observed_active_players"] == ["FantaMJ", "McFelon", "oksb"]
    assert state["observed_hero"] == "FantaMJ"
    assert state["observed_hero_to_call"] == 2.60
    assert state["observed_raise_to"] == 5.20
    assert app.state.service.player_database.get_player("McFelon") is not None
    reply = client.post("/api/coach/chat", json={"message": "What should I do?"}).json()
    assert "3 players with cards" in reply["reply"]
    assert "who made the bet" in reply["reply"]
    assert reply["context"]["to_call"] == 2.60

    current["read"] = VisionRead(
        status="partial", hero_cards=["KC", "3S"], board_cards=["8H", "QC", "TC"],
        current_pot=5.20, visible_names=["FantaMJ", "McFelon", "oksb"],
        active_names=["FantaMJ", "McFelon", "oksb"], hero_player="FantaMJ",
        hero_turn=False,
    )
    client.post("/api/input/frame", json=payload)
    state = client.post("/api/input/frame", json=payload).json()["state"]
    assert state["observed_hero_to_call"] is None
    client.post("/api/input/stop", json={"source_id": payload["source_id"]})
    stopped = client.get("/api/state").json()
    assert stopped["observed_players"] == []
    assert stopped["observed_active_players"] == []
    assert app.state.service.player_database.get_player("McFelon") is not None


def test_similar_seen_name_waits_for_identity_confirmation(tmp_path, monkeypatch) -> None:
    app = create_app(tmp_path / "data")
    app.state.service.player_database.ensure_player("FantaMJ")
    read = VisionRead(status="partial", visible_names=["Fanta"], active_names=["Fanta"])
    monkeypatch.setattr(app.state.input_stream.holdem_reader, "read", lambda *_: read)
    client = TestClient(app)
    payload = {"source_id": "clubgg-window-share-alias", "image_data": "data:image/jpeg;base64,AA=="}
    client.post("/api/input/frame", json=payload)
    state = client.post("/api/input/frame", json=payload).json()["state"]
    review = next(item for item in state["review_queue"] if item["field"] == "identity_seen_player")
    assert review["payload"]["candidate_player"] == "FantaMJ"
    assert app.state.service.player_database.get_player("Fanta") is None
    client.post(f"/api/reviews/{review['id']}", json={"decision": "approve"})
    assert app.state.service.player_database.resolve_alias("Fanta") == "FantaMJ"
