from fastapi.testclient import TestClient

from backend.main import create_app
from backend.models import FrameInputRequest


def test_focus_page_and_advanced_page(tmp_path):
    client = TestClient(create_app(tmp_path))
    focus = client.get("/").text
    assert "Share table or display" in focus
    assert 'id="setupToggle"' in focus
    assert 'id="setupOverlay"' in focus
    assert 'focus-layout.css' in focus
    advanced = client.get("/advanced").text
    assert 'id="seatSetup"' in advanced
    assert 'advanced-theme.css' in advanced
    assert client.get("/static/advanced-theme.css").status_code == 200


def test_general_coach_question_does_not_require_cards(tmp_path):
    client = TestClient(create_app(tmp_path))
    result = client.post("/api/coach/chat", json={"message": "What is GTO?"}).json()
    assert "does not solve full poker GTO" in result["reply"]
    assert result["brief"] is None


def test_can_you_see_it_reports_actual_capture_status(tmp_path):
    app = create_app(tmp_path)
    client = TestClient(app)
    absent = client.post("/api/coach/chat", json={"message": "Can you see it?"}).json()
    assert "no fresh shared image" in absent["reply"]
    assert absent["brief"] is None

    app.state.input_stream.ingest(FrameInputRequest(
        source_id="clubgg-window-share-test",
        image_data="data:image/jpeg;base64,AA==",
    ))
    present = client.post("/api/coach/chat", json={"message": "Can you see my screen?"}).json()
    assert "receiving frames" in present["reply"]
    assert "every card or action" in present["reply"]


def test_general_conversation_uses_local_model_but_action_request_stays_grounded(tmp_path):
    app = create_app(tmp_path)

    class StubModel:
        calls = []

        async def converse(self, message, history, state):
            self.calls.append((message, history))
            return "Let's review that concept together."

        async def status(self):
            return {"available": True, "model": "test:1b", "reason": "ready"}

    stub = StubModel()
    app.state.local_coach = stub
    client = TestClient(app)
    response = client.post("/api/coach/chat", json={
        "message": "Can we talk through position?",
        "history": [{"role": "user", "text": "Hi coach"}, {"role": "coach", "text": "Hi!"}],
    }).json()
    assert response["reply"] == "Let's review that concept together."
    assert response["reply_source"] == "local_model"
    assert len(stub.calls[0][1]) == 2
    assert client.get("/api/coach/model-status").json()["available"] is True

    specific = client.post("/api/coach/chat", json={"message": "What should I do?"}).json()
    assert "hole cards" in specific["reply"]
    assert len(stub.calls) == 1


def test_unavailable_local_model_is_disclosed(tmp_path):
    app = create_app(tmp_path)

    class UnavailableModel:
        async def converse(self, message, history, state):
            return None

        async def status(self):
            return {"available": False, "model": "gemma3:1b", "reason": "Ollama is not running on this computer."}

    app.state.local_coach = UnavailableModel()
    reply = TestClient(app).post("/api/coach/chat", json={"message": "Hi coach"}).json()["reply"]
    assert "conversational model isn't ready" in reply
    assert "Ollama is not running" in reply


def test_chat_asks_for_missing_facts_then_explains(tmp_path):
    client = TestClient(create_app(tmp_path))
    first = client.post("/api/coach/chat", json={"message": "What should I do?"}).json()
    assert "hole cards" in first["reply"]
    second = client.post("/api/coach/chat", json={"message": "I have Ah Kh", "context": first["context"]}).json()
    assert "pot" in second["reply"]
    third = client.post("/api/coach/chat", json={"message": "pot 24", "context": second["context"]}).json()
    assert "call" in third["reply"]
    fourth = client.post("/api/coach/chat", json={"message": "to call 6", "context": third["context"]}).json()
    assert fourth["brief"]["ready"] is True
    assert "not a full GTO solve" in fourth["reply"]
    assert fourth["state"]["hero_cards"] == ["AH", "KH"]


def test_opponent_raise_requires_fresh_price_and_bad_cards_are_rejected(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.post("/api/config/seats", json={"seats": [{"seat": 1, "player": "Hero", "is_hero": True}, {"seat": 2, "player": "Alex"}], "hero_seat": 1})
    bad = client.post("/api/coach/chat", json={"message": "I have Ah Ah"}).json()
    assert "couldn't use those cards" in bad["reply"]
    good = client.post("/api/coach/chat", json={"message": "I have Ah Kh pot 20 to call 4 Alex raised"}).json()
    assert good["context"]["to_call"] == 4
    raised = client.post("/api/coach/chat", json={"message": "Alex raises again to 30", "context": good["context"]}).json()
    assert raised["context"]["to_call"] is None
    assert "How much" in raised["reply"]


def test_similar_username_requires_confirmation_before_profile_merge(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.post("/api/config/seats", json={"seats": [{"seat": 1, "player": "Hero", "is_hero": True}, {"seat": 2, "player": "FantaMJ"}], "hero_seat": 1})
    uncertain = client.post("/api/coach/chat", json={"message": "Fanta raised to 30"}).json()
    assert "same player" in uncertain["reply"]
    assert uncertain["context"]["opponent_id"] is None
    confirmed = client.post("/api/coach/chat", json={"message": "yes", "context": uncertain["context"]}).json()
    assert confirmed["context"]["opponent_id"] == "FantaMJ"
    later = client.post("/api/coach/chat", json={"message": "Fanta raised again", "context": confirmed["context"]}).json()
    assert "same player" not in later["reply"]


def test_chat_preflop_raise_changes_range_prior(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.post("/api/config/seats", json={"seats": [{"seat": 1, "player": "Hero", "is_hero": True}, {"seat": 2, "player": "Alex"}], "hero_seat": 1})
    result = client.post("/api/coach/chat", json={"message": "I have Ah Kh, pot 20, to call 4. Alex raised to 8."}).json()
    assert result["brief"]["ready"] is True
    assert "open preflop range prior" in result["reply"]
    assert result["brief"]["solution"]["bayesian_adjustment"]["preflop_context"] == "open"


def test_automatic_coaching_gate_requires_complete_verified_observation(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.post("/api/config/seats", json={"seats": [{"seat": 1, "player": "Hero", "is_hero": True}, {"seat": 2, "player": "Alex"}], "hero_seat": 1})
    partial = client.post("/api/input/frame", json={"source_id": "test", "observation": {"game_variant": "nlh", "confidence": .95}}).json()
    assert partial["state"]["last_input_verified_spot"] is False
    full = client.post("/api/input/frame", json={"source_id": "test", "observation": {
        "hero_cards": ["Ah", "Kh"], "board_cards": [], "current_pot": 20,
        "current_actor": "Hero", "current_street": "preflop", "confidence": .95,
        "actions": [{"event_id": "alex-raise", "player": "Alex", "action": "raise", "amount": 4, "amount_type": "total"}],
    }}).json()
    assert full["state"]["last_input_verified_spot"] is True
