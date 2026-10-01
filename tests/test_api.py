import json

from fastapi.testclient import TestClient

from backend.main import create_app


def test_full_api_flow(tmp_path) -> None:
    app = create_app(tmp_path / "data")
    client = TestClient(app)

    response = client.post(
        "/api/config/seats",
        json={
            "seats": [
                {"seat": 1, "player": "Jason", "stack": 100, "is_hero": True},
                {"seat": 2, "player": "Alex", "stack": 100, "is_hero": False},
            ],
            "hero_seat": 1,
            "blinds": {"small": 1, "big": 2},
        },
    )
    assert response.status_code == 200
    configured = response.json()
    assert configured["hero_seat"] == 1
    assert configured["players"] == ["Jason", "Alex"]

    response = client.post(
        "/api/action",
        json={
            "player": "Jason",
            "action": "raise",
            "amount": 6,
            "amount_type": "total",
            "street": "preflop",
        },
    )
    assert response.status_code == 200
    assert response.json()["pot"] == 6

    response = client.post("/api/speech/transcript", json={"text": "Alex calls"})
    assert response.status_code == 200
    parsed = response.json()
    assert parsed["applied"] is True
    assert parsed["parsed"]["player"] == "Alex"
    assert parsed["state"]["pot"] == 12

    response = client.post(
        "/api/vision/card_manual",
        json={"hero_cards": ["Ah", "Kh"], "board_cards": ["Qs", "Jd", "3c"]},
    )
    assert response.status_code == 200
    cards_state = response.json()
    assert cards_state["hero_cards"] == ["AH", "KH"]
    assert cards_state["board_cards"] == ["QS", "JD", "3C"]

    response = client.get("/api/players")
    assert response.status_code == 200
    players = {entry["name"]: entry for entry in response.json()}
    # Aggregates are committed when a hand ends, avoiding double-counting if a
    # captured action is later corrected or undone.
    assert players["Jason"]["hands_seen"] == 0

    response = client.post("/api/hand/end", json={"winner": "Jason", "notes": "Test hand"})
    assert response.status_code == 200
    archived = response.json()
    assert archived["message"] == "Hand archived and stats updated."
    assert archived["state"]["pot"] == 0
    assert archived["state"]["actions"] == []
    saved_hand = json.loads((tmp_path / "data" / "hand_archive.json").read_text(encoding="utf-8"))[0]
    assert saved_hand["game_format"] == "cash"
    assert saved_hand["table_size_bucket"] == "1-2"
    assert saved_hand["actions"][1]["pot_before"] == 6
    assert saved_hand["actions"][1]["price_to_call_before"] == 6

    response = client.get("/api/players")
    assert response.status_code == 200
    players = {entry["name"]: entry for entry in response.json()}
    assert players["Jason"]["hands_played"] == 1
    assert players["Jason"]["vpip"] == 1.0
    assert players["Jason"]["pfr"] == 1.0
    assert players["Jason"]["raise_count"] == 1
    assert players["Alex"]["call_count"] == 1
    assert players["Alex"]["aggression_factor"] == 0.0

    breakdown = client.get("/api/players/Alex/table-breakdown")
    assert breakdown.status_code == 200
    assert breakdown.json()["breakdown"][0]["table_size_bucket"] == "1-2"
    assert breakdown.json()["breakdown"][0]["call_count"] == 1

    history = client.get("/api/players/Alex/history")
    assert history.status_code == 200
    assert history.json()["hands"][0]["table_size_bucket"] == "1-2"
    assert history.json()["actions"][0]["action"] == "call"

    response = client.post(
        "/api/solver/recommend",
        json={
            "board": ["Qs", "Jd", "3c"],
            "hero_hand": ["Ah", "Kh"],
            "current_pot": 12,
            "to_call": 4,
            "raise_to": 14,
            "opponent_id": "Alex",
            "simulations": 300,
            "cfr_iterations": 100,
        },
    )
    assert response.status_code == 200
    solution = response.json()
    assert solution["stats_source"] == "local_database"
    assert set(solution["action_frequencies"]) == {"fold", "call", "raise"}
    assert solution["equity_samples"] == 300
    assert solution["bayesian_adjustment"]["table_size"] == 2


def test_generic_input_feed_is_idempotent_and_supports_coaching(tmp_path) -> None:
    app = create_app(tmp_path / "data")
    client = TestClient(app)
    client.post(
        "/api/config/seats",
        json={
            "seats": [
                {"seat": 1, "player": "Hero", "stack": 100, "is_hero": True},
                {"seat": 2, "player": "Alex", "stack": 100, "is_hero": False},
            ],
            "hero_seat": 1,
        },
    )
    payload = {
        "source_id": "test-cnn",
        "image_data": "data:image/jpeg;base64,AA==",
        "observation": {
            "hero_cards": ["Ah", "Kh"],
            "board_cards": ["Qs", "Jd", "3c"],
            "current_pot": 12,
            "active_players": ["Hero", "Alex"],
            "confidence": 0.91,
            "actions": [
                {
                    "event_id": "unique-raise-1",
                    "player": "Alex",
                    "action": "raise",
                    "amount": 6,
                    "amount_type": "total",
                    "street": "preflop",
                }
            ],
        },
    }
    first = client.post("/api/input/frame", json=payload)
    assert first.status_code == 200
    assert first.json()["new_action_events"] == 1
    assert first.json()["state"]["observed_pot"] == 12
    assert first.json()["state"]["hero_cards"] == ["AH", "KH"]

    repeated = client.post("/api/input/frame", json=payload)
    assert repeated.status_code == 200
    assert repeated.json()["new_action_events"] == 0
    assert len(repeated.json()["state"]["actions"]) == 1

    latest = client.get("/api/input/latest")
    assert latest.status_code == 200
    assert latest.json()["source_id"] == "test-cnn"
    assert latest.json()["has_preview"] is True

    brief = client.post(
        "/api/coach/brief",
        json={"opponent_id": "Alex", "to_call": 6, "simulations": 300},
    )
    assert brief.status_code == 200
    assert brief.json()["ready"] is True
    assert brief.json()["recommended_action"] in {"fold", "call", "raise"}


def test_hand_truth_layer_and_low_confidence_review_queue(tmp_path) -> None:
    app = create_app(tmp_path / "data")
    client = TestClient(app)
    client.post(
        "/api/config/seats",
        json={
            "seats": [
                {"seat": 1, "player": "Hero", "stack": 100, "is_hero": True},
                {"seat": 2, "player": "Alex", "stack": 100},
                {"seat": 3, "player": "Blair", "stack": 100},
            ],
            "hero_seat": 1,
            "dealer_seat": 1,
        },
    )
    for action in [
        {"player": "Hero", "action": "raise", "amount": 6, "amount_type": "total"},
        {"player": "Alex", "action": "call"},
        {"player": "Blair", "action": "fold"},
    ]:
        response = client.post("/api/action", json=action)
        assert response.status_code == 200

    state = client.get("/api/state").json()
    assert state["active_players"] == ["Hero", "Alex"]
    assert state["folded_players"] == ["Blair"]
    assert state["positions"]["Hero"] == "BTN"
    assert state["total_contributions"]["Hero"] == 6
    assert state["side_pots"][0]["amount"] == 12

    low_confidence = client.post(
        "/api/input/frame",
        json={
            "source_id": "test-vision",
            "observation": {
                "current_pot": 30,
                "field_confidences": {"current_pot": 0.4},
            },
        },
    )
    assert low_confidence.status_code == 200
    assert low_confidence.json()["state"]["observed_pot"] is None
    review = low_confidence.json()["state"]["review_queue"][0]
    assert review["field"] == "current_pot"

    approved = client.post(f"/api/reviews/{review['id']}", json={"decision": "approve"})
    assert approved.status_code == 200
    assert approved.json()["observed_pot"] == 30

    uncertain_action = client.post(
        "/api/input/frame",
        json={
            "source_id": "test-vision",
            "observation": {
                "actions": [{"event_id": "uncertain-fold", "player": "Alex", "action": "fold", "confidence": 0.2}],
            },
        },
    )
    assert uncertain_action.status_code == 200
    assert all(
        not (action["player"] == "Alex" and action["action"] == "fold")
        for action in uncertain_action.json()["state"]["actions"]
    )
    assert uncertain_action.json()["state"]["review_queue"][0]["field"] == "action"


def test_fuzzy_player_identity_requires_confirmation_then_remembers_alias(tmp_path) -> None:
    app = create_app(tmp_path / "data")
    client = TestClient(app)
    client.post(
        "/api/config/seats",
        json={
            "seats": [
                {"seat": 1, "player": "Hero", "stack": 100, "is_hero": True},
                {"seat": 2, "player": "Fanta", "stack": 100},
            ],
            "hero_seat": 1,
        },
    )
    detected = client.post(
        "/api/input/frame",
        json={
            "source_id": "name-detector",
            "observation": {
                "actions": [{
                    "event_id": "fuzzy-raise", "player": "Fantamj", "action": "raise",
                    "amount": 30, "amount_type": "total", "confidence": 0.95,
                }],
            },
        },
    )
    assert detected.status_code == 200
    state = detected.json()["state"]
    assert state["actions"] == []
    review = state["review_queue"][0]
    assert review["field"] == "identity_action"
    assert review["payload"]["candidate_player"] == "Fanta"

    confirmed = client.post(f"/api/reviews/{review['id']}", json={"decision": "approve"})
    assert confirmed.status_code == 200
    assert confirmed.json()["actions"][0]["player"] == "Fanta"

    alias_reused = client.post(
        "/api/input/frame",
        json={
            "source_id": "name-detector",
            "observation": {
                "actions": [{
                    "event_id": "known-alias-call", "player": "Fantamj", "action": "call",
                    "amount": 30, "amount_type": "total", "confidence": 0.95,
                }],
            },
        },
    )
    assert alias_reused.status_code == 200
    assert alias_reused.json()["state"]["actions"][-1]["player"] == "Fanta"


def test_displayed_profile_stats_are_saved_separately_and_feed_the_model(tmp_path) -> None:
    app = create_app(tmp_path / "data")
    client = TestClient(app)
    client.post(
        "/api/config/seats",
        json={
            "seats": [
                {"seat": 1, "player": "Hero", "stack": 100, "is_hero": True},
                {"seat": 2, "player": "Alex", "stack": 100},
            ],
            "hero_seat": 1,
        },
    )
    saved = client.post(
        "/api/players/Alex/observed-stats",
        json={"hands_seen": 240, "vpip": 42, "pfr": 18, "aggression_factor": 2.1},
    )
    assert saved.status_code == 200
    profile = saved.json()["profile"]
    assert profile["local_hands_seen"] == 0
    assert profile["observed_hands_seen"] == 240
    assert profile["vpip"] == 0.42
    assert profile["stats_origin"] == "observed"

    vision_saved = client.post(
        "/api/input/frame",
        json={
            "source_id": "profile-ocr",
            "observation": {
                "player_stats": [{
                    "player": "Alex", "hands_seen": 250, "vpip": 40,
                    "pfr": 17, "aggression_factor": 1.8, "confidence": 0.96,
                }]
            },
        },
    )
    assert vision_saved.status_code == 200
    player = client.get("/api/players/Alex").json()["profile"]
    assert player["observed_hands_seen"] == 250
    assert player["observed_vpip"] == 0.4
    assert player["observed_source"] == "clubgg_profile"
