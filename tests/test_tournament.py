from fastapi.testclient import TestClient

from backend.engine.tournament import ICMDecisionContext, icm_equity, tournament_utilities
from backend.engine.cfr_solver import HoldemCFRSolver, OpponentStats, RangeContext
from backend.main import create_app


def test_icm_preserves_winner_take_all_and_values_survival() -> None:
    assert icm_equity((60, 40), (100, 0), 0) == 60.0
    context = ICMDecisionContext((100, 100, 10), 0, 1, (100, 100, 0))
    utilities, details = tournament_utilities(
        context, pot=20, call_cost=30, raise_cost=60,
        equity=0.5, fold_equity=0.0,
    )
    assert details["risk_premium_ratio"] > 1.0
    assert utilities[0] == 0.0
    assert utilities[1] < 0.0


def test_tournament_format_phase_memory_and_input_round_trip(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    response = client.post("/api/config/seats", json={
        "seats": [
            {"seat": 1, "player": "Hero", "stack": 100, "is_hero": True},
            {"seat": 2, "player": "Alex", "stack": 100},
            {"seat": 3, "player": "Blair", "stack": 10},
        ],
        "hero_seat": 1,
    })
    assert response.status_code == 200
    configured = client.post("/api/config/game-format", json={
        "game_format": "tournament",
        "tournament": {
            "players_remaining": 3, "paid_places": 2,
            "payouts": [100, 100], "final_table_size": 9, "ante": 1,
        },
    })
    assert configured.status_code == 200
    assert configured.json()["tournament_phase"] == "bubble"

    # A vision source can update only the number it saw without clearing prizes.
    observed = client.post("/api/input/frame", json={
        "source_id": "test-layout-adapter",
        "image_data": "data:image/jpeg;base64,AA==",
        "observation": {"tournament": {"players_remaining": 3}, "confidence": 0.97},
    })
    assert observed.status_code == 200
    assert observed.json()["state"]["tournament"]["payouts"] == [100, 100]
    assert client.get("/api/input/latest").json()["has_preview"] is True
    assert client.post("/api/input/stop", json={"source_id": "other"}).json()["cleared"] is False
    assert client.post("/api/input/stop", json={"source_id": "test-layout-adapter"}).json()["cleared"] is True
    assert client.get("/api/input/latest").json()["has_preview"] is False
    assert client.post("/api/input/frame", json={"source_id": "test-layout-adapter", "image_data": "data:image/jpeg;base64,AA=="}).status_code == 422

    client.post("/api/vision/card_manual", json={"hero_cards": ["Ah", "Kh"]})
    coached = client.post("/api/coach/brief", json={"opponent_id": "Alex", "to_call": 30, "simulations": 300})
    assert coached.status_code == 200
    assert coached.json()["solution"]["decision_model"] == "tournament_icm"
    assert coached.json()["solution"]["icm"] is not None

    action = client.post("/api/action", json={"player": "Alex", "action": "raise", "amount": 12, "amount_type": "total"})
    assert action.status_code == 200
    ended = client.post("/api/hand/end", json={"winner": "Alex"})
    assert ended.status_code == 200
    assert ended.json()["state"]["game_format"] == "tournament"
    history = client.get("/api/players/Alex/history").json()
    assert history["hands"][0]["tournament_phase"] == "bubble"
    assert history["actions"][0]["game_format"] == "tournament"
    phases = client.get("/api/players/Alex/phase-breakdown").json()["phases"]
    assert phases[0]["hands_seen"] == 1
    assert phases[0]["raise_count"] == 1


def test_cash_mode_does_not_use_tournament_payouts(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    client.post("/api/config/seats", json={"seats": [
        {"seat": 1, "player": "Hero", "stack": 100, "is_hero": True},
        {"seat": 2, "player": "Alex", "stack": 100},
    ], "hero_seat": 1})
    client.post("/api/vision/card_manual", json={"hero_cards": ["Ah", "Kh"]})
    result = client.post("/api/coach/brief", json={"opponent_id": "Alex", "to_call": 10, "simulations": 300})
    assert result.status_code == 200
    assert result.json()["solution"]["decision_model"] == "chip_ev"
    assert result.json()["solution"]["icm"] is None


def test_cash_rake_reduces_the_value_of_continuing() -> None:
    arguments = dict(
        board=["Qs", "Jd", "3c", "2s", "9h"], hero_hand=["Ah", "Kh"],
        opponent_stats=OpponentStats(), current_pot=20, to_call=4,
        simulations=100, cfr_iterations=30,
        range_context=RangeContext(street="flop"),
    )
    no_rake = HoldemCFRSolver(seed=17).solve(**arguments)
    with_rake = HoldemCFRSolver(seed=17).solve(**arguments, rake_percent=10, rake_cap=5)
    assert with_rake["utilities"]["call"] < no_rake["utilities"]["call"]
    assert with_rake["rake"]["estimated_call_rake"] > 0
