import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.azure_coach_llm import AzureCoachLLM
from backend.main import create_app
from backend.models import CoachChatTurn


def test_azure_status_requires_server_side_key(monkeypatch, tmp_path):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    app = create_app(tmp_path)
    status = TestClient(app).get("/api/coach/model-status").json()
    assert status["provider"] == "azure"
    assert status["model"] == "gpt-6-luna"
    assert status["available"] is False
    assert "AZURE_OPENAI_API_KEY" in status["reason"]


def test_azure_chat_uses_selected_deployment_without_exposing_key(monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-test-key")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(output_text="Let's discuss position and ranges.")

    monkeypatch.setattr("backend.azure_coach_llm.AsyncOpenAI", FakeClient)
    model = AzureCoachLLM()
    state = create_app(tmp_path).state.service.get_state()
    history = [CoachChatTurn(role="user", text="Hi"), CoachChatTurn(role="coach", text="Hello")]
    reply = asyncio.run(model.converse("Can we discuss position?", history, state))

    assert reply == "Let's discuss position and ranges."
    assert captured["client"]["api_key"] == "fake-test-key"
    assert captured["client"]["base_url"] == "https://pokeragentmodel.services.ai.azure.com/openai/v1/"
    assert captured["request"]["model"] == "gpt-6-luna"
    assert captured["request"]["reasoning"] == {"effort": "none"}
    assert captured["request"]["max_output_tokens"] == 500
    assert captured["request"]["store"] is False
    assert captured["request"]["input"][-1] == {"role": "user", "content": "Can we discuss position?"}
    assert "fake-test-key" not in str(asyncio.run(model.status()))


def test_live_page_can_connect_azure_key_in_memory(monkeypatch, tmp_path):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    app = create_app(tmp_path)
    client = TestClient(app, base_url="http://127.0.0.1:8000")
    credential = "fake-test-key-that-is-long-enough"

    async def fake_converse(message, history, state):
        assert message == "Reply with the single word READY."
        assert history == []
        assert app.state.local_coach.api_key == credential
        app.state.local_coach.verified = True
        return "READY"

    monkeypatch.setattr(app.state.local_coach, "converse", fake_converse)
    denied = client.post("/api/coach/connect-azure", json={"api_key": credential})
    assert denied.status_code == 403
    assert app.state.local_coach.api_key == ""

    connected = client.post("/api/coach/connect-azure", json={"api_key": credential}, headers={"Origin": "http://127.0.0.1:8000"})
    assert connected.status_code == 200
    assert connected.json()["verified"] is True
    assert credential not in str(connected.json())
    assert client.get("/api/coach/model-status").json()["available"] is True
