import json

import pytest
from fastapi.testclient import TestClient

from backend.engine.texas_solver_study import (
    TexasSolverLookup, import_texas_solver_export, lookup_texas_solver,
)
from backend.main import create_app


def _fixture(tmp_path, frequencies=None):
    export = tmp_path / "raw.json"
    manifest = tmp_path / "manifest.json"
    export.write_text(json.dumps({
        "actions": ["CHECK", "BET 50"],
        "childrens": {"CHECK": {
            "strategy": {
                "actions": ["CALL", "FOLD"],
                "strategy": {"Kc3s": frequencies or [0.65, 0.35]},
            }
        }},
    }), encoding="utf-8")
    manifest.write_text(json.dumps({
        "provider": "TexasSolver",
        "range_notes": "Hero 30% opening range, villain 35% defending range.",
        "spot": {
            "board": ["8h", "Qc", "Tc"], "pot": 5.20, "to_call": 2.60,
            "effective_stack": 100, "rake_percent": 0, "hero_position": "IP",
            "action_path": ["CHECK"],
        },
    }), encoding="utf-8")
    return export, manifest


def _query(**changes):
    payload = {
        "board": ["8H", "QC", "TC"], "pot": 5.2, "to_call": 2.6,
        "effective_stack": 100, "rake_percent": 0, "hero_position": "IP",
        "action_path": ["CHECK"], "hero_hand": ["3S", "KC"],
    }
    payload.update(changes)
    return TexasSolverLookup.model_validate(payload)


def test_native_export_import_and_exact_lookup(tmp_path):
    export, manifest = _fixture(tmp_path)
    library = tmp_path / "data" / "solver_study"
    imported = import_texas_solver_export(export, manifest, library)
    assert imported.is_file()
    result = lookup_texas_solver(library, _query())
    assert result["study_only"] is True
    assert result["action_frequencies"] == {"CALL": 0.65, "FOLD": 0.35}
    assert len(result["source_sha256"]) == 64
    assert lookup_texas_solver(library, _query(to_call=2.61)) is None
    assert lookup_texas_solver(library, _query(rake_percent=5)) is None
    assert lookup_texas_solver(library, _query(hero_hand=["Ah", "Kh"])) is None
    with pytest.raises(ValueError):
        lookup_texas_solver(library, _query(hero_hand=["8H", "KC"]))


def test_malformed_or_unproven_export_is_rejected(tmp_path):
    export, manifest = _fixture(tmp_path, [0.8, 0.8])
    with pytest.raises(ValueError, match="sum to one"):
        import_texas_solver_export(export, manifest, tmp_path / "library")
    assert not (tmp_path / "library").exists()
    export, manifest = _fixture(tmp_path)
    metadata = json.loads(manifest.read_text())
    metadata.pop("range_notes")
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="ranges"):
        import_texas_solver_export(export, manifest, tmp_path / "library")


def test_study_api_does_not_fallback_to_a_similar_spot(tmp_path):
    export, manifest = _fixture(tmp_path)
    data = tmp_path / "appdata"
    import_texas_solver_export(export, manifest, data / "solver_study")
    client = TestClient(create_app(data))
    exact = client.post("/api/study/texas-solver", json=_query().model_dump())
    assert exact.status_code == 200
    assert exact.json()["provider"] == "TexasSolver"
    wrong = client.post("/api/study/texas-solver", json=_query(pot=5.3).model_dump())
    assert wrong.status_code == 404
    wrong_game = client.post("/api/study/texas-solver", json={**_query().model_dump(), "game_format": "tournament"})
    assert wrong_game.status_code == 422
