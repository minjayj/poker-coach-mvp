"""The open study catalog must not silently expand beyond its solved model."""

from fastapi.testclient import TestClient

from backend.engine.open_study import hand_class, lookup_push_fold_study
from backend.main import create_app


def test_hand_classes_are_canonical_and_validate_cards() -> None:
    assert hand_class(["Ah", "Kh"]) == "AKs"
    assert hand_class(["Kd", "As"]) == "AKo"
    assert hand_class(["Ad", "Ah"]) == "AA"


def test_open_study_lookup_keeps_both_decisions_separate() -> None:
    sb = lookup_push_fold_study(effective_stack_bb=2, hero_hand=["Ah", "Ad"], decision="sb_first")
    bb = lookup_push_fold_study(effective_stack_bb=2, hero_hand=["Ah", "Ad"], decision="bb_vs_shove")
    assert sb["study_only"] is True
    assert sb["action_frequencies"]["shove"] > 0.99
    assert "call" not in sb["action_frequencies"]
    assert bb["action_frequencies"]["call"] > 0.99
    assert "shove" not in bb["action_frequencies"]
    assert sb["quality"] == "high-precision"


def test_open_study_api_is_exact_stack_and_nlh_only(tmp_path) -> None:
    client = TestClient(create_app(tmp_path / "data"))
    catalog = client.get("/api/study/push-fold")
    assert catalog.status_code == 200
    assert catalog.json()["available_stacks_bb"] == [2, 3, 5, 8, 10, 12, 15, 20]
    assert catalog.json()["study_only"] is True

    spot = client.post("/api/study/push-fold", json={
        "effective_stack_bb": 10,
        "hero_hand": ["7h", "2c"],
        "decision": "sb_first",
    })
    assert spot.status_code == 200
    result = spot.json()
    assert result["hand_class"] == "72o"
    assert result["quality"] == "advisory"
    assert abs(sum(result["action_frequencies"].values()) - 1) < 1e-12
    assert "No ante" in " ".join(result["assumptions"])

    for payload in (
        {"effective_stack_bb": 9, "hero_hand": ["Ah", "Kh"], "decision": "sb_first"},
        {"effective_stack_bb": 10, "hero_hand": ["Ah", "Ah"], "decision": "sb_first"},
        {"effective_stack_bb": 10, "hero_hand": ["Ah", "Kh", "Qh", "Jh", "Th"], "decision": "sb_first"},
    ):
        assert client.post("/api/study/push-fold", json=payload).status_code == 422
