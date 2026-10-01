import json

from fastapi.testclient import TestClient

from backend.engine.opponent_adjustment import blend_exact_strategy, summarize_opponent_json
from backend.engine.texas_solver_study import import_texas_solver_export
from backend.engine.coach import build_brief
from backend.main import create_app
from backend.models import GameState


def _archive(count=20, *, phase="cash", confidence=1.0):
    return [
        {
            "hand_id": number, "game_variant": "nlh", "game_format": "cash",
            "tournament_phase": phase, "table_size_bucket": "1-2",
            "actions": [
                {"player": "Alex", "action": "fold" if number < count * 3 // 4 else "call",
                 "pot_before": 10, "price_to_call_before": 6,
                 "contributed": 0, "source": "vision", "confidence": confidence},
                {"player": "Alex", "action": "bet", "pot_before": 20,
                 "price_to_call_before": 0, "contributed": 12,
                 "source": "manual", "confidence": None},
            ],
        }
        for number in range(count)
    ]


def _baseline():
    return {
        "action_frequencies": {"FOLD": 0.2, "CALL": 0.4, "RAISE 12": 0.4},
        "spot": {"pot": 10, "to_call": 6},
    }


def _player_model():
    return {"action_frequencies": {"fold": 0.1, "call": 0.2, "raise": 0.7}}


def _read(archive):
    return summarize_opponent_json(
        archive, player_id="Alex", game_variant="nlh", game_format="cash",
        tournament_phase="cash", table_size_bucket="1-2",
    )


def test_json_read_shrinks_and_filters_context_and_vision_confidence():
    read = _read(_archive())
    assert read["facing_hands"] == 20
    assert read["facing_actions"]["fold"] == 15
    assert read["by_faced_size"]["medium"]["samples"] == 20
    assert read["own_aggressive_sizes"]["counts"]["medium"] == 20
    assert 0.35 < read["posterior_fold_when_facing_bet"] < 0.75
    assert _read(_archive(20, phase="bubble"))["facing_hands"] == 0
    assert _read(_archive(20, confidence=0.3))["facing_hands"] == 0


def test_blend_is_capped_and_abstains_when_data_or_tree_is_weak():
    base = _baseline()
    weak = blend_exact_strategy(base, _player_model(), _read(_archive(8)))
    assert weak["player_weight"] == 0
    assert weak["blended_frequencies"] == base["action_frequencies"]
    strong = blend_exact_strategy(base, _player_model(), _read(_archive(100)), proposed_raise_to=12)
    assert 0 < strong["player_weight"] <= 0.35
    assert strong["blended_frequencies"]["RAISE 12"] > 0.4
    assert strong["size_evidence"]["proposed_raise_bucket"] == "medium"
    assert abs(sum(strong["blended_frequencies"].values()) - 1) < 1e-5
    multiple_sizes = {**base, "action_frequencies": {"CALL": 0.2, "RAISE 12": 0.4, "RAISE 20": 0.4}}
    refused = blend_exact_strategy(multiple_sizes, _player_model(), _read(_archive(100)))
    assert refused["player_weight"] == 0


def test_coach_brief_exposes_contextual_read_without_claiming_hidden_cards():
    brief = build_brief(GameState(), None, None, 0, _read(_archive(20)))
    assert any("comparable recorded hands" in line for line in brief["what_i_see"])
    assert any("does not reveal" in line for line in brief["watch_out_for"])
    assert brief["opponent_read"]["facing_hands"] == 20


def test_exact_study_adjustment_api_reads_completed_json(tmp_path):
    data = tmp_path / "data"
    app = create_app(data)
    client = TestClient(app)
    (data / "hand_archive.json").write_text(json.dumps(_archive(20)), encoding="utf-8")
    export = tmp_path / "export.json"
    export.write_text(json.dumps({"strategy": {
        "actions": ["FOLD", "CALL", "RAISE 12"],
        "strategy": {"Kc3s": [0.2, 0.4, 0.4]},
    }}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "provider": "TexasSolver", "range_notes": "Explicit hero and villain ranges",
        "spot": {"board": ["8h", "Qc", "Tc"], "pot": 10, "to_call": 6,
                 "effective_stack": 100, "rake_percent": 0, "hero_position": "IP"},
    }), encoding="utf-8")
    import_texas_solver_export(export, manifest, data / "solver_study")
    payload = {
        "board": ["8h", "Qc", "Tc"], "hero_hand": ["Kc", "3s"],
        "pot": 10, "to_call": 6, "effective_stack": 100,
        "rake_percent": 0, "hero_position": "IP", "opponent_id": "Alex",
        "proposed_raise_to": 12,
    }
    response = client.post("/api/study/texas-solver/adjusted", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["opponent_read"]["facing_hands"] == 20
    assert result["adjustment"]["player_weight"] > 0
    assert result["solver_spot"]["study_only"] is True
    assert client.post("/api/study/texas-solver/adjusted", json={**payload, "pot": 11}).status_code == 404
