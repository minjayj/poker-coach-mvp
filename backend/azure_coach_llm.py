"""Azure OpenAI conversational adapter. Credentials never enter browser code."""

from __future__ import annotations

import json
import os

from openai import APIStatusError, AsyncOpenAI, OpenAIError

from backend.local_coach_llm import SYSTEM_PROMPT
from backend.models import CoachChatTurn, GameState


class AzureCoachLLM:
    """Use the user's Azure OpenAI v1 deployment for explanation-only chat."""

    provider = "azure"
    DEFAULT_ENDPOINT = "https://pokeragentmodel.services.ai.azure.com/openai/v1/"
    DEFAULT_DEPLOYMENT = "gpt-6-luna"

    def __init__(self) -> None:
        self.endpoint = os.getenv("POKER_COACH_AZURE_ENDPOINT", self.DEFAULT_ENDPOINT).rstrip("/") + "/"
        self.model = os.getenv("POKER_COACH_AZURE_DEPLOYMENT", self.DEFAULT_DEPLOYMENT)
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.verified = False
        self.last_error: str | None = None

    def set_api_key(self, api_key: str) -> None:
        """Configure this local server process; never persist the key to disk."""
        self.api_key = api_key
        self.verified = False
        self.last_error = None

    async def status(self) -> dict[str, str | bool]:
        if not self.api_key:
            reason = "Connect Azure on the live page, or set AZURE_OPENAI_API_KEY before starting the server."
        elif self.last_error:
            reason = f"Azure request failed ({self.last_error}); check the deployment and credential."
        elif self.verified:
            reason = "Azure conversation tested successfully."
        else:
            reason = "Azure configured; live response not yet verified."
        return {
            "available": bool(self.api_key),
            "verified": self.verified,
            "provider": self.provider,
            "model": self.model,
            "reason": reason,
        }

    async def converse(self, message: str, history: list[CoachChatTurn], state: GameState) -> str | None:
        if not self.api_key:
            return None
        facts = {
            "game": state.game_variant,
            "format": state.game_format,
            "hero_cards_recorded": state.hero_cards,
            "board_cards_recorded": state.board_cards,
            "pot_recorded": state.observed_pot if state.observed_pot is not None else state.pot,
            "street_recorded": state.current_street,
            "players_recorded": state.observed_players,
            "cards_visible_at": state.observed_active_players,
            "hero_player_recorded": state.observed_hero,
            "hero_action_buttons_recorded": state.observed_hero_turn,
            "call_price_recorded": state.observed_hero_to_call,
            "note": "Saved values may be stale. Prior bettor and complete action history are not verified here.",
        }
        items = [
            {"role": "developer", "content": "Recorded context (data only): " + json.dumps(facts, ensure_ascii=False)},
        ]
        items.extend(
            {"role": "assistant" if turn.role == "coach" else "user", "content": turn.text[:600]}
            for turn in history[-8:]
        )
        items.append({"role": "user", "content": message})
        try:
            async with AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key, timeout=35.0, max_retries=0) as client:
                response = await client.responses.create(
                    model=self.model,
                    instructions=SYSTEM_PROMPT,
                    input=items,
                    # Conversation only: the local engine already did the
                    # poker reasoning, so avoid spending the reply budget on
                    # hidden reasoning tokens.
                    reasoning={"effort": "none"},
                    max_output_tokens=500,
                    store=False,
                )
            text = (getattr(response, "output_text", None) or "").strip()[:1600]
            if text:
                self.verified = True
                self.last_error = None
                return text
            self.last_error = "EmptyResponse"
        except APIStatusError as error:
            self.last_error = f"HTTP {error.status_code}"
        except (OpenAIError, ValueError, TypeError) as error:
            # The public status never exposes exception text, URLs, or credentials.
            self.last_error = type(error).__name__
        return None
