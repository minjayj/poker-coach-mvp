from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from threading import RLock
from typing import Any

from backend.engine.recommend import recommend_move
from backend.engine.hand_evaluator import normalize_cards
from backend.player_database import PlayerStatsDatabase, table_size_bucket
from backend.models import (
    ActionRecord,
    ActionRequest,
    CorrectionRequest,
    GameState,
    GameFormatRequest,
    HandEndRequest,
    InputObservation,
    ObservedStatsInput,
    PlayerProfile,
    TournamentContext,
    Recommendation,
    ReviewItem,
    SeatConfigRequest,
    SidePot,
    StreetName,
    VisionManualCardRequest,
    utc_now,
)


class JSONStore:
    def read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


class GameService:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.players_dir = self.data_dir / "players"
        self.state_path = self.data_dir / "game_state.json"
        self.log_path = self.data_dir / "session_log.json"
        self.archive_path = self.data_dir / "hand_archive.json"
        self.player_database = PlayerStatsDatabase(self.data_dir / "player_stats.sqlite3")
        self.store = JSONStore()
        self.lock = RLock()
        self._bootstrap()

    def _bootstrap(self) -> None:
        self.players_dir.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self.store.write_json(self.state_path, GameState().model_dump(mode="json"))
        if not self.log_path.exists():
            self.store.write_json(self.log_path, [])
        if not self.archive_path.exists():
            self.store.write_json(self.archive_path, [])

    def get_state(self) -> GameState:
        with self.lock:
            payload = self.store.read_json(
                self.state_path, GameState().model_dump(mode="json")
            )
            state = GameState.model_validate(payload)
            return self._recompute_state(state)

    def save_state(self, state: GameState) -> GameState:
        with self.lock:
            state = self._recompute_state(state)
            self.store.write_json(self.state_path, state.model_dump(mode="json"))
            return state

    def clear_live_observations(self) -> GameState:
        """Drop transient screen-only facts when the capture source changes."""
        state = self.get_state()
        state.observed_players = []
        state.observed_active_players = []
        state.observed_hero = None
        state.observed_actor = None
        state.observed_hero_turn = False
        state.observed_hero_to_call = None
        state.observed_raise_to = None
        return self.save_state(state)

    def list_player_profiles(self, variant: str | None = None) -> list[PlayerProfile]:
        variant = variant or self.get_state().game_variant
        return [self.get_player_profile_for_variant(str(row["player_id"]), variant) for row in self.player_database.list_players()]

    def get_player_profile_for_variant(self, player_id: str, variant: str) -> PlayerProfile | None:
        if self.player_database.get_player(player_id) is None:
            return None
        row = self.player_database.get_player_variant_stats(player_id, variant)
        if row is None:
            row = {"player_id": player_id, **{column: 0 for column in self.player_database.SUMMARY_COLUMNS}}
        observed = self.player_database.get_observed_stats(player_id, variant)
        profile = self._profile_from_database(row, observed)
        profile.game_variant = variant  # type: ignore[assignment]
        return profile

    def get_player_profile(self, player_id: str) -> PlayerProfile | None:
        return self.get_player_profile_for_variant(player_id, self.get_state().game_variant)

    def get_player_profile_for_table_size(self, player_id: str, player_count: int) -> PlayerProfile | None:
        """Use the matching table-size sample when it exists, then global history."""
        row = self.player_database.get_player_table_stats(player_id, player_count)
        return self._profile_from_database(row, self.player_database.get_observed_stats(player_id)) if row else self.get_player_profile(player_id)

    def get_player_profile_for_context(self, player_id: str, player_count: int, phase: str) -> PlayerProfile | None:
        variant = self.get_state().game_variant
        phase_row = self.player_database.get_player_variant_stats(player_id, variant, phase)
        if phase_row and int(phase_row["hands_seen"]) >= 10:
            profile = self._profile_from_database(phase_row)
            profile.game_variant = variant  # type: ignore[assignment]
            profile.stats_origin = f"{phase}_sample"
            return profile
        return self.get_player_profile_for_variant(player_id, variant)

    def get_player_phase_breakdown(self, player_id: str) -> list[dict[str, Any]]:
        phases = ("cash", "tournament", "bubble", "final_table")
        variant = self.get_state().game_variant
        return [{**row, "tournament_phase": phase} for phase in phases if (row := self.player_database.get_player_variant_stats(player_id, variant, phase))]

    def get_player_table_breakdown(self, player_id: str) -> list[PlayerProfile]:
        variant = self.get_state().game_variant
        observed = self.player_database.get_observed_stats(player_id, variant)
        profiles = [self._profile_from_database(row, observed) for row in self.player_database.get_player_table_breakdown(player_id, variant)]
        for profile in profiles:
            profile.game_variant = variant  # type: ignore[assignment]
        return profiles

    def get_player_history(self, player_id: str, limit: int = 100) -> dict[str, list[dict[str, Any]]]:
        return self.player_database.get_player_history(player_id, limit)

    def update_player_stats(self, player_id: str, action_taken: str | dict[str, Any]) -> None:
        """Public updater for integrations that submit a completed-hand event."""
        self.player_database.update_player_stats(player_id, action_taken)

    def configure_game_format(self, request: GameFormatRequest) -> GameState:
        state = self.get_state()
        variant = request.game_variant or state.game_variant
        if variant != state.game_variant and state.actions:
            raise ValueError("Finish or correct the current hand before switching game variant.")
        if variant != state.game_variant:
            state.hero_cards = []
            state.board_cards = []
        state.game_format = request.game_format
        state.game_variant = variant
        state.cash = request.cash
        state.tournament = request.tournament
        self._append_log("game_format", request.model_dump(mode="json"))
        return self.save_state(state)

    def save_observed_stats(self, player_id: str, stats: ObservedStatsInput, variant: str | None = None) -> PlayerProfile:
        """Save platform-visible stats as a baseline, distinct from our own hand log."""
        player = self._match_player_name(player_id, self.get_state()) or player_id.strip()
        self.player_database.upsert_observed_stats(
            player,
            hands_seen=stats.hands_seen,
            vpip=stats.vpip,
            pfr=stats.pfr,
            aggression_factor=stats.aggression_factor,
            source=f"{stats.source}:{variant or self.get_state().game_variant}",
            confidence=stats.confidence,
        )
        self._append_log("observed_player_stats", {"player": player, **stats.model_dump(mode="json")})
        return self.get_player_profile_for_variant(player, variant or self.get_state().game_variant) or PlayerProfile(name=player)

    def configure_seats(self, request: SeatConfigRequest) -> GameState:
        state = self.get_state()
        hero_seat = request.hero_seat
        if hero_seat is None:
            hero = next((seat.seat for seat in request.seats if seat.is_hero), None)
            hero_seat = hero

        normalized_seats = [
            seat.model_copy(update={"is_hero": seat.seat == hero_seat})
            for seat in sorted(request.seats, key=lambda seat: seat.seat)
        ]
        state.seats = normalized_seats
        state.players = [seat.player for seat in normalized_seats]
        state.hero_seat = hero_seat
        state.dealer_seat = request.dealer_seat or state.dealer_seat or (normalized_seats[0].seat if normalized_seats else None)
        state.blinds = request.blinds
        state.current_actor = state.players[0] if state.players else None
        for player in state.players:
            self._ensure_profile(player)
        self._append_log(
            "config_seats",
            {
                "seats": [seat.model_dump(mode="json") for seat in normalized_seats],
                "hero_seat": hero_seat,
                "dealer_seat": state.dealer_seat,
                "blinds": request.blinds.model_dump(mode="json"),
            },
        )
        state = self.save_state(state)
        return state

    def apply_action(self, request: ActionRequest) -> GameState:
        state = self.get_state()
        player = self._match_player_name(request.player, state)
        if request.street:
            state.current_street = request.street
        record = ActionRecord(
            street=request.street or state.current_street,
            player=player,
            action=request.action,
            amount=request.amount,
            amount_type=request.amount_type,
            source=request.source,
            raw_text=request.raw_text,
            confidence=request.confidence,
        )
        state.actions.append(record)
        self._append_log("action", record.model_dump(mode="json"))
        state = self.save_state(state)
        return state

    def apply_correction(self, request: CorrectionRequest) -> GameState:
        state = self.get_state()
        if not state.actions:
            return state

        if request.type == "undo":
            removed = state.actions.pop()
            self._append_log("correction", {"type": "undo", "removed_action": removed.model_dump(mode="json")})
        else:
            last_action = state.actions[-1]
            if request.player is not None:
                last_action.player = self._match_player_name(request.player, state) or request.player
            if request.action is not None:
                last_action.action = request.action
            if request.amount is not None:
                last_action.amount = request.amount
                if last_action.amount_type == "unknown":
                    last_action.amount_type = "total"
            last_action.source = "correction"
            self._append_log("correction", {"type": request.type, "updated_action": last_action.model_dump(mode="json")})

        state = self.save_state(state)
        return state

    def finalize_hand(self, request: HandEndRequest) -> GameState:
        state = self.get_state()
        archived_hands = self.store.read_json(self.archive_path, [])
        winner = self._match_player_name(request.winner, state)
        hand_summary = {
            "hand_id": state.hand_id,
            "game_format": state.game_format,
            "game_variant": state.game_variant,
            "tournament_phase": state.tournament_phase,
            "table_size_bucket": table_size_bucket(len(state.players)),
            "blinds": state.blinds.model_dump(mode="json"),
            "ended_at": utc_now(),
            "winner": winner,
            "notes": request.notes,
            "players": state.players,
            "pot": state.pot,
            "current_street": state.current_street,
            "hero_cards": state.hero_cards,
            "board_cards": state.board_cards,
            "hand_flags": state.hand_flags,
            "actions": [action.model_dump(mode="json") for action in state.actions],
        }
        archived_hands.append(hand_summary)
        self.store.write_json(self.archive_path, archived_hands)
        self._append_log("hand_end", hand_summary)
        self._update_database_for_completed_hand(state, winner)

        next_state = GameState(
            game_format=state.game_format,
            game_variant=state.game_variant,
            cash=state.cash,
            tournament=state.tournament,
            seats=state.seats,
            players=state.players,
            blinds=state.blinds,
            hero_seat=state.hero_seat,
            current_street="preflop",
            current_actor=state.players[0] if state.players else None,
            hand_id=state.hand_id + 1,
        )
        next_state = self.save_state(next_state)
        return next_state

    def set_manual_cards(self, request: VisionManualCardRequest) -> GameState:
        state = self.get_state()
        required = 5 if state.game_variant == "plo5" else 2
        if request.hero_cards and len(request.hero_cards) != required:
            raise ValueError(f"{state.game_variant.upper()} requires exactly {required} hero cards.")
        normalize_cards([*request.hero_cards, *request.board_cards])
        state.hero_cards = [card.strip().upper() for card in request.hero_cards if card.strip()]
        state.board_cards = [card.strip().upper() for card in request.board_cards if card.strip()]
        self._append_log(
            "card_manual",
            {"hero_cards": state.hero_cards, "board_cards": state.board_cards},
        )
        return self.save_state(state)

    def apply_input_observation(self, observation: InputObservation, source_id: str = "external") -> GameState:
        """Apply high-confidence fields and queue uncertain recognition for review."""
        state = self.get_state()
        threshold = 0.85

        def confident(field: str) -> bool:
            confidence = observation.field_confidences.get(field, observation.confidence if observation.confidence is not None else 1.0)
            if confidence >= (0.80 if field == "game_variant" else threshold):
                return True
            value = getattr(observation, field)
            state.review_queue.append(
                ReviewItem(field=field, payload={"value": value}, confidence=confidence, source_id=source_id)
            )
            return False

        if observation.game_variant is not None and confident("game_variant"):
            if observation.game_variant != state.game_variant and state.actions:
                if not any(item.field == "game_variant" and item.source_id == source_id and item.payload.get("value") == observation.game_variant for item in state.review_queue):
                    state.review_queue.append(ReviewItem(
                        field="game_variant", payload={"value": observation.game_variant, "reason": "hand_in_progress"},
                        confidence=observation.field_confidences.get("game_variant", observation.confidence or 1.0), source_id=source_id,
                    ))
            elif observation.game_variant != state.game_variant:
                state.game_variant = observation.game_variant
                state.hero_cards = []
                state.board_cards = []
        required = 5 if state.game_variant == "plo5" else 2
        if observation.hero_cards is not None and len(observation.hero_cards) in {0, required} and confident("hero_cards"):
            state.hero_cards = [card.strip().upper() for card in observation.hero_cards if card.strip()]
        elif observation.hero_cards is not None and len(observation.hero_cards) not in {0, required}:
            state.review_queue.append(ReviewItem(field="hero_cards", payload={"value": observation.hero_cards, "reason": "wrong_variant_card_count"}, confidence=0.0, source_id=source_id))
        if observation.board_cards is not None and confident("board_cards"):
            state.board_cards = [card.strip().upper() for card in observation.board_cards if card.strip()]
        if observation.current_street is not None and confident("current_street"):
            state.current_street = observation.current_street
        if observation.current_pot is not None and confident("current_pot"):
            state.observed_pot = observation.current_pot
        if observation.visible_players is not None and confident("visible_players"):
            known = {str(row["player_id"]).casefold(): str(row["player_id"]) for row in self.player_database.list_players()}
            observed: list[str] = []
            for name in observation.visible_players:
                name = name.strip()
                if not name or name in observed:
                    continue
                canonical = self.player_database.resolve_alias(name) or known.get(name.casefold())
                if canonical is None:
                    closest = max(known.values(), key=lambda player: SequenceMatcher(None, name.casefold(), player.casefold()).ratio(), default=None)
                    similarity = SequenceMatcher(None, name.casefold(), closest.casefold()).ratio() if closest else 0.0
                    if closest and similarity >= .65:
                        if not any(item.field == "identity_seen_player" and item.payload.get("detected_player") == name for item in state.review_queue):
                            state.review_queue.append(ReviewItem(
                                field="identity_seen_player",
                                payload={"detected_player": name, "candidate_player": closest},
                                confidence=round(similarity, 3), source_id=source_id,
                            ))
                    else:
                        self.player_database.ensure_player(name)
                        known[name.casefold()] = name
                observed.append(canonical or name)
            state.observed_players = observed
        if observation.hero_player is not None and confident("hero_player"):
            state.observed_hero = self.player_database.resolve_alias(observation.hero_player) or observation.hero_player
        if observation.hero_turn is not None and confident("hero_turn"):
            state.observed_hero_turn = observation.hero_turn
            if observation.hero_turn:
                state.observed_actor = state.observed_hero
                state.observed_hero_to_call = None
                state.observed_raise_to = None
            else:
                if state.observed_actor == state.observed_hero:
                    state.observed_actor = None
                state.observed_hero_to_call = None
                state.observed_raise_to = None
        if observation.hero_to_call is not None and state.observed_hero_turn and confident("hero_to_call"):
            state.observed_hero_to_call = observation.hero_to_call
        if observation.raise_to is not None and state.observed_hero_turn and confident("raise_to"):
            state.observed_raise_to = observation.raise_to
        if observation.current_actor is not None and confident("current_actor"):
            resolved_player, identity_review = self.resolve_observed_player(observation.current_actor, state)
            if resolved_player:
                state.observed_actor = resolved_player
            elif identity_review:
                state.review_queue.append(
                    ReviewItem(
                        field="identity_actor", payload=identity_review,
                        confidence=observation.field_confidences.get("current_actor", observation.confidence or 1.0),
                        source_id=source_id,
                    )
                )
        if observation.active_players is not None and confident("active_players"):
            state.observed_active_players = [
                self._match_player_name(player, state) or player.strip()
                for player in observation.active_players
                if player.strip()
            ]
        if observation.game_format is not None and confident("game_format"):
            state.game_format = observation.game_format
        if observation.tournament is not None and confident("tournament"):
            state.tournament = TournamentContext.model_validate({
                **state.tournament.model_dump(),
                **observation.tournament.model_dump(exclude_unset=True),
            })
        for observed in observation.player_stats:
            confidence = observed.confidence if observed.confidence is not None else observation.field_confidences.get(
                "player_stats", observation.confidence if observation.confidence is not None else 1.0
            )
            resolved_player, identity_review = self.resolve_observed_player(observed.player, state)
            payload = observed.model_dump(mode="json")
            if identity_review is not None:
                state.review_queue.append(ReviewItem(
                    field="identity_profile_stats", payload={**payload, **identity_review},
                    confidence=confidence, source_id=source_id,
                ))
            elif confidence < threshold:
                state.review_queue.append(ReviewItem(
                    field="player_stats", payload={**payload, "player": resolved_player},
                    confidence=confidence, source_id=source_id,
                ))
            elif resolved_player:
                self.save_observed_stats(resolved_player, observed, state.game_variant)
        if observation.confidence is not None:
            state.input_confidence = observation.confidence
        state.last_input_at = utc_now()
        state.last_input_source = source_id
        try:
            explicit_cards_valid = (
                observation.hero_cards is not None
                and len(observation.hero_cards) == required
                and observation.board_cards is not None
                and len(observation.board_cards) in {0, 3, 4, 5}
                and bool(normalize_cards([*observation.hero_cards, *observation.board_cards]))
            )
        except ValueError:
            explicit_cards_valid = False
        required_fields = ("hero_cards", "board_cards", "current_pot", "current_actor", "current_street", "action")
        state.last_input_verified_spot = bool(
            observation.confidence is not None and observation.confidence >= 0.85
            and all(observation.field_confidences.get(field, observation.confidence) >= 0.85 for field in required_fields)
            and explicit_cards_valid and observation.current_pot is not None
            and observation.current_actor is not None and observation.current_street is not None
            and observation.actions
        )
        state.review_queue = state.review_queue[-100:]
        self._append_log("input_observation", observation.model_dump(mode="json"))
        return self.save_state(state)

    def queue_review(self, field: str, payload: dict[str, object], confidence: float, source_id: str) -> GameState:
        """Hold uncertain vision output for a visible human review before use."""
        state = self.get_state()
        state.review_queue.append(
            ReviewItem(field=field, payload=payload, confidence=confidence, source_id=source_id)
        )
        state.review_queue = state.review_queue[-100:]
        self._append_log("vision_review_queued", state.review_queue[-1].model_dump(mode="json"))
        return self.save_state(state)

    def resolve_observed_player(self, detected_name: str | None, state: GameState | None = None) -> tuple[str | None, dict[str, object] | None]:
        """Return an exact/confirmed identity or a review payload for fuzzy names."""
        if not detected_name or not detected_name.strip():
            return None, {"reason": "missing_player"}
        state = state or self.get_state()
        detected = detected_name.strip()
        exact = next((player for player in [*state.players, *state.observed_players] if player.lower() == detected.lower()), None)
        if exact:
            return exact, None
        aliased = self.player_database.resolve_alias(detected)
        if aliased and aliased in state.players:
            return aliased, None

        candidates = []
        for player in state.players:
            score = SequenceMatcher(None, detected.lower(), player.lower()).ratio()
            if score >= 0.55:
                candidates.append((score, player))
        if candidates:
            score, candidate = max(candidates)
            return None, {
                "reason": "possible_alias",
                "detected_player": detected,
                "candidate_player": candidate,
                "identity_score": round(score, 3),
            }
        return None, {"reason": "unknown_player", "detected_player": detected}

    def resolve_review(self, review_id: str, approved: bool) -> GameState:
        state = self.get_state()
        item = next((review for review in state.review_queue if review.id == review_id), None)
        if item is None:
            return state
        if approved and item.field == "game_variant" and state.actions:
            raise ValueError("Finish the current hand before applying a different game variant.")
        state.review_queue = [review for review in state.review_queue if review.id != review_id]
        if approved:
            if item.field == "hero_cards":
                state.hero_cards = [str(card).strip().upper() for card in item.payload.get("value", [])]
            elif item.field == "board_cards":
                state.board_cards = [str(card).strip().upper() for card in item.payload.get("value", [])]
            elif item.field == "current_pot":
                state.observed_pot = float(item.payload.get("value", 0.0))
            elif item.field == "current_street":
                state.current_street = str(item.payload.get("value", state.current_street))  # type: ignore[assignment]
            elif item.field == "current_actor":
                state.observed_actor = self._match_player_name(str(item.payload.get("value", "")), state)
            elif item.field == "active_players":
                state.observed_active_players = [
                    self._match_player_name(str(player), state) or str(player)
                    for player in item.payload.get("value", [])
                ]
            elif item.field == "game_format":
                state.game_format = str(item.payload.get("value", "cash"))  # type: ignore[assignment]
            elif item.field == "game_variant":
                if not state.actions:
                    state.game_variant = str(item.payload.get("value", "nlh"))  # type: ignore[assignment]
                    state.hero_cards = []
                    state.board_cards = []
            elif item.field == "tournament":
                state.tournament = TournamentContext.model_validate({
                    **state.tournament.model_dump(),
                    **dict(item.payload.get("value") or {}),
                })
            elif item.field == "action":
                payload = item.payload
                request = ActionRequest(
                    player=payload.get("player"),
                    action=payload["action"],  # type: ignore[arg-type]
                    amount=payload.get("amount"),  # type: ignore[arg-type]
                    amount_type=payload.get("amount_type", "unknown"),  # type: ignore[arg-type]
                    street=payload.get("street"),  # type: ignore[arg-type]
                    source="vision",
                    confidence=item.confidence,
                )
                self._append_log("vision_review_approved", item.model_dump(mode="json"))
                self.store.write_json(self.state_path, state.model_dump(mode="json"))
                return self.apply_action(request)
            elif item.field == "identity_action":
                payload = item.payload
                candidate = str(payload.get("candidate_player", ""))
                detected = str(payload.get("detected_player", ""))
                if candidate and detected:
                    self.player_database.remember_alias(detected, candidate)
                request = ActionRequest(
                    player=candidate or None,
                    action=payload["action"],  # type: ignore[arg-type]
                    amount=payload.get("amount"),  # type: ignore[arg-type]
                    amount_type=payload.get("amount_type", "unknown"),  # type: ignore[arg-type]
                    street=payload.get("street"),  # type: ignore[arg-type]
                    source="vision",
                    confidence=item.confidence,
                )
                self._append_log("vision_identity_confirmed", item.model_dump(mode="json"))
                self.store.write_json(self.state_path, state.model_dump(mode="json"))
                return self.apply_action(request)
            elif item.field == "identity_actor":
                candidate = str(item.payload.get("candidate_player", ""))
                detected = str(item.payload.get("detected_player", ""))
                if candidate and detected:
                    self.player_database.remember_alias(detected, candidate)
                    state.observed_actor = candidate
            elif item.field == "identity_seen_player":
                candidate = str(item.payload.get("candidate_player", ""))
                detected = str(item.payload.get("detected_player", ""))
                if candidate and detected:
                    self.player_database.remember_alias(detected, candidate)
                    state.observed_players = [candidate if name == detected else name for name in state.observed_players]
                    state.observed_active_players = [candidate if name == detected else name for name in state.observed_active_players]
                    if state.observed_hero == detected:
                        state.observed_hero = candidate
            elif item.field in {"player_stats", "identity_profile_stats"}:
                payload = item.payload
                candidate = str(payload.get("candidate_player", ""))
                player = candidate or str(payload.get("player", ""))
                detected = str(payload.get("detected_player", ""))
                if candidate and detected:
                    self.player_database.remember_alias(detected, candidate)
                if player:
                    self.save_observed_stats(player, ObservedStatsInput(
                        hands_seen=payload.get("hands_seen"),
                        vpip=payload.get("vpip"),
                        pfr=payload.get("pfr"),
                        aggression_factor=payload.get("aggression_factor"),
                        source=str(payload.get("source") or "vision_profile"),
                        confidence=item.confidence,
                    ))
        elif item.field == "identity_seen_player":
            detected = str(item.payload.get("detected_player", ""))
            if detected:
                self.player_database.ensure_player(detected)
        self._append_log(
            "vision_review_approved" if approved else "vision_review_dismissed",
            item.model_dump(mode="json"),
        )
        return self.save_state(state)

    def rebuild_profiles(self, state: GameState | None = None) -> list[PlayerProfile]:
        """Compatibility shim: profiles now come from the SQLite source of truth."""
        return self.list_player_profiles()

    def _apply_profile_action(
        self, profiles: dict[str, PlayerProfile], action: ActionRecord
    ) -> None:
        if not action.player:
            return
        profile = profiles.setdefault(action.player, PlayerProfile(name=action.player))
        if action.action == "raise":
            profile.raises += 1
            if action.amount is not None:
                profile.total_raise_amount += float(action.amount)
                profile.raise_samples += 1
        elif action.action == "call":
            profile.calls += 1
        elif action.action == "fold":
            profile.folds += 1
        elif action.action == "check":
            profile.checks += 1
        elif action.action == "all_in":
            profile.all_ins += 1
            if action.amount is not None:
                profile.total_raise_amount += float(action.amount)
                profile.raise_samples += 1

    def _append_log(self, event_type: str, payload: dict[str, Any]) -> None:
        with self.lock:
            events = self.store.read_json(self.log_path, [])
            events.append(
                {"timestamp": utc_now(), "type": event_type, "payload": payload}
            )
            self.store.write_json(self.log_path, events)

    def _ensure_profile(self, name: str) -> None:
        self.player_database.ensure_player(name)

    @staticmethod
    def _profile_from_database(row: dict[str, Any], observed: dict[str, Any] | None = None) -> PlayerProfile:
        local_hands_seen = int(row["hands_seen"])
        hands_seen = local_hands_seen
        calls = int(row["call_count"])
        raises = int(row["raise_count"])
        aggressive = int(row["aggressive_actions"])
        local_vpip = round(int(row["vpip_count"]) / hands_seen, 3) if hands_seen else 0.0
        local_pfr = round(int(row["pfr_count"]) / hands_seen, 3) if hands_seen else 0.0
        local_af = round(aggressive / calls, 2) if calls else float(aggressive)
        observed_hands = int(observed["hands_seen"] or 0) if observed else 0
        observed_vpip = float(observed["vpip"]) if observed and observed["vpip"] is not None else None
        observed_pfr = float(observed["pfr"]) if observed and observed["pfr"] is not None else None
        observed_af = float(observed["aggression_factor"]) if observed and observed["aggression_factor"] is not None else None
        observed_confidence = float(observed["confidence"]) if observed else None
        prior_weight = observed_hands * (observed_confidence if observed_confidence is not None else 1.0)
        total_weight = hands_seen + prior_weight
        vpip = round(((local_vpip * hands_seen) + ((observed_vpip or 0.0) * prior_weight)) / total_weight, 3) if observed_vpip is not None and total_weight else local_vpip
        pfr = round(((local_pfr * hands_seen) + ((observed_pfr or 0.0) * prior_weight)) / total_weight, 3) if observed_pfr is not None and total_weight else local_pfr
        af = round(((local_af * hands_seen) + ((observed_af or 0.0) * prior_weight)) / total_weight, 2) if observed_af is not None and total_weight else local_af
        effective_hands = int(round(total_weight)) if observed else hands_seen
        return PlayerProfile(
            name=str(row["player_id"]),
            hands_played=effective_hands,
            hands_seen=effective_hands,
            vpip=vpip,
            pfr=pfr,
            vpip_hands=round(vpip * effective_hands),
            pfr_hands=round(pfr * effective_hands),
            total_pots=float(row["total_pots"]),
            calls=calls,
            raises=raises,
            call_count=calls,
            raise_count=raises,
            aggression_score=af,
            aggression_factor=af,
            folds=int(row.get("fold_count", 0)),
            checks=int(row.get("check_count", 0)),
            bets=int(row.get("bet_count", 0)),
            all_ins=int(row.get("all_in_count", 0)),
            all_in_count=int(row.get("all_in_count", 0)),
            showdown_count=int(row.get("showdown_count", 0)),
            win_count=int(row.get("win_count", 0)),
            total_contributed=float(row.get("total_contributed", 0.0)),
            action_count=int(row.get("action_count", 0)),
            table_size_bucket=row.get("table_size_bucket"),
            local_hands_seen=local_hands_seen,
            observed_hands_seen=observed_hands if observed else None,
            observed_vpip=observed_vpip,
            observed_pfr=observed_pfr,
            observed_aggression_factor=observed_af,
            observed_source=str(observed["source"]).rsplit(":", 1)[0] if observed else None,
            observed_confidence=observed_confidence,
            observed_at=str(observed["observed_at"]) if observed else None,
            stats_origin="combined" if observed and local_hands_seen else "observed" if observed else "local",
        )

    def _update_database_for_completed_hand(self, state: GameState, winner: str | None) -> None:
        """Persist raw action history and summaries after—not during—a hand."""
        vpip_players = set(state.hand_flags.get("vpip", []))
        pfr_players = set(state.hand_flags.get("pfr", []))
        hero_name = self._hero_name(state)
        player_count = len(state.observed_active_players) or len(state.players)
        actions = [action.model_dump(mode="json") for action in state.actions]
        ended_at = utc_now()
        for player in state.players:
            starting_stack = next((seat.stack for seat in state.seats if seat.player == player), 0.0)
            self.player_database.record_completed_hand(
                hand_id=state.hand_id,
                player_id=player,
                table_size=player_count,
                is_hero=player == hero_name,
                starting_stack=starting_stack,
                ending_stack=state.stacks.get(player, starting_stack),
                total_pot=state.pot,
                final_street=state.current_street,
                won=player == winner,
                vpip=player in vpip_players,
                pfr=player in pfr_players,
                position=state.positions.get(player, "unknown"),
                preflop_context=self._preflop_context_for(state, player),
                effective_stack_bb=self._effective_stack_bb(state, player),
                actions=actions,
                ended_at=ended_at,
                game_format=state.game_format,
                tournament_phase=state.tournament_phase,
                players_remaining=state.tournament.players_remaining,
                game_variant=state.game_variant,
            )

    def _preflop_context_for(self, state: GameState, player: str) -> str:
        preflop_actions = [action for action in state.actions if action.street == "preflop"]
        first_index = next((index for index, action in enumerate(preflop_actions) if action.player == player), None)
        if first_index is None:
            return "unknown"
        action = preflop_actions[first_index]
        earlier_raises = sum(previous.action in {"raise", "bet", "all_in"} for previous in preflop_actions[:first_index])
        if action.action == "call":
            return "call_raise" if earlier_raises else "limp"
        if action.action in {"raise", "bet", "all_in"}:
            return "open" if earlier_raises == 0 else "three_bet" if earlier_raises == 1 else "four_bet"
        return "unknown"

    def _effective_stack_bb(self, state: GameState, player: str) -> float | None:
        hero_name = self._hero_name(state)
        big_blind = state.blinds.big
        player_stack = next((seat.stack for seat in state.seats if seat.player == player), 0.0)
        hero_stack = next((seat.stack for seat in state.seats if seat.player == hero_name), 0.0)
        if not big_blind or not player_stack or not hero_stack:
            return None
        return round(min(player_stack, hero_stack) / big_blind, 2)

    def _match_player_name(self, candidate: str | None, state: GameState) -> str | None:
        if not candidate:
            return None
        lowered = candidate.strip().lower()
        for player in [*state.players, *state.observed_players]:
            if player.lower() == lowered:
                return player
        return self.player_database.resolve_alias(candidate) or candidate.strip()

    def _seat_map(self, state: GameState) -> dict[str, float]:
        return {seat.player: float(seat.stack) for seat in state.seats}

    def _hero_name(self, state: GameState) -> str | None:
        for seat in state.seats:
            if seat.seat == state.hero_seat:
                return seat.player
        return None

    def _hero_position_label(self, state: GameState) -> str:
        hero_name = self._hero_name(state)
        ordered_players = [seat.player for seat in sorted(state.seats, key=lambda seat: seat.seat)]
        if not hero_name or hero_name not in ordered_players:
            return "unknown"
        index = ordered_players.index(hero_name)
        last_index = len(ordered_players) - 1
        if index <= 1:
            return "early"
        if index == last_index:
            return "late"
        return "middle"

    def _recommendation_for(self, state: GameState) -> Recommendation:
        if state.game_variant == "plo5":
            return Recommendation(reason="PLO5 uses a separate coach; Hold'em recommendations are disabled.")
        hero_name = self._hero_name(state)
        if not hero_name:
            return Recommendation(reason="Set the hero seat to unlock recommendations.")

        opponent_profiles = {profile.name: profile for profile in self.list_player_profiles(state.game_variant)}
        facing_action = "none"
        tendencies: dict[str, Any] = {}
        for action in reversed(state.actions):
            if action.player and action.player != hero_name:
                if action.action in {"raise", "bet", "all_in"}:
                    facing_action = action.action
                    profile = opponent_profiles.get(action.player)
                    tendencies = profile.model_dump(mode="json") if profile else {}
                    break

        stack_bb = 0.0
        if state.blinds.big > 0:
            stack_bb = round(state.stacks.get(hero_name, 0.0) / state.blinds.big, 2)

        return recommend_move(
            hero_cards=state.hero_cards,
            position=self._hero_position_label(state),
            stack_bb=stack_bb,
            facing_action=facing_action,
            pot_size=state.pot,
            player_tendencies=tendencies,
        )

    def _next_player(self, state: GameState, current_player: str | None) -> str | None:
        ordered_players = [
            seat.player for seat in sorted(state.seats, key=lambda seat: seat.seat)
            if seat.player in state.active_players
        ]
        if not ordered_players:
            return None
        if not current_player or current_player not in ordered_players:
            return ordered_players[0]
        index = ordered_players.index(current_player)
        return ordered_players[(index + 1) % len(ordered_players)]

    def _positions(self, state: GameState) -> dict[str, str]:
        seats = sorted(state.seats, key=lambda seat: seat.seat)
        if not seats:
            return {}
        dealer_index = next((index for index, seat in enumerate(seats) if seat.seat == state.dealer_seat), 0)
        ordered = seats[dealer_index:] + seats[:dealer_index]
        count = len(ordered)
        if count == 2:
            labels = ["BTN/SB", "BB"]
        elif count == 3:
            labels = ["BTN", "SB", "BB"]
        else:
            middle = {
                4: ["UTG"], 5: ["UTG", "CO"], 6: ["UTG", "HJ", "CO"],
                7: ["UTG", "LJ", "HJ", "CO"], 8: ["UTG", "UTG+1", "LJ", "HJ", "CO"],
            }.get(count, ["UTG", "UTG+1", "MP", "LJ", "HJ", "CO"])
            labels = ["BTN", "SB", "BB", *middle]
            labels = labels[:count]
        return {seat.player: labels[index] for index, seat in enumerate(ordered)}

    @staticmethod
    def _side_pots(total_contributions: dict[str, float], folded_players: set[str]) -> list[dict[str, object]]:
        levels = sorted({amount for amount in total_contributions.values() if amount > 0})
        previous = 0.0
        pots: list[dict[str, object]] = []
        for level in levels:
            contributors = [player for player, amount in total_contributions.items() if amount >= level]
            amount = round((level - previous) * len(contributors), 2)
            if amount > 0:
                pots.append({
                    "amount": amount,
                    "contributors": contributors,
                    "eligible_players": [player for player in contributors if player not in folded_players],
                })
            previous = level
        return pots

    def _format_current_action(self, action: ActionRecord | None) -> str:
        if action is None:
            return "Waiting for action"
        parts = [action.player or "Unknown player", action.action.replace("_", " ")]
        if action.amount is not None:
            parts.append(f"{action.amount:g}")
        return " ".join(parts)

    def _round_dict(self, values: dict[str, float]) -> dict[str, float]:
        return {key: round(value, 2) for key, value in values.items()}

    def _contribution_for_action(
        self,
        action: ActionRecord,
        stacks: dict[str, float],
        street_contributions: dict[str, float],
        current_bet: float,
    ) -> tuple[float, float]:
        if not action.player:
            return 0.0, current_bet

        previous = street_contributions.get(action.player, 0.0)
        remaining = stacks.get(action.player, 0.0)
        amount = float(action.amount) if action.amount is not None else None
        contributed = 0.0

        if action.action in {"raise", "bet"}:
            if amount is None:
                return 0.0, current_bet
            target = previous + amount if action.amount_type == "additional" else amount
            target = max(target, previous)
            contributed = min(max(0.0, target - previous), remaining)
            street_contributions[action.player] = previous + contributed
            current_bet = max(current_bet, street_contributions[action.player])
        elif action.action == "call":
            if current_bet > 0:
                target = current_bet
                if amount is not None:
                    if action.amount_type == "additional":
                        target = previous + amount
                    elif action.amount_type == "total":
                        target = amount
                contributed = min(max(0.0, target - previous), remaining)
                street_contributions[action.player] = previous + contributed
            elif amount is not None:
                target = previous + amount if action.amount_type == "additional" else amount
                contributed = min(max(0.0, target - previous), remaining)
                street_contributions[action.player] = previous + contributed
                current_bet = max(current_bet, street_contributions[action.player])
        elif action.action == "all_in":
            if amount is None:
                contributed = remaining
                street_contributions[action.player] = previous + contributed
            else:
                target = previous + amount if action.amount_type == "additional" else amount
                contributed = min(max(0.0, target - previous), remaining)
                street_contributions[action.player] = previous + contributed
            current_bet = max(current_bet, street_contributions[action.player])

        return round(contributed, 2), round(current_bet, 2)

    def _recompute_state(self, state: GameState) -> GameState:
        state.tournament_phase = self._tournament_phase(state)
        starting_stacks = self._seat_map(state)
        current_street: StreetName = "preflop"
        street_contributions = {player: 0.0 for player in starting_stacks}
        total_contributions = {player: 0.0 for player in starting_stacks}
        stacks = starting_stacks.copy()
        pot = 0.0
        current_bet = 0.0
        vpip: set[str] = set()
        pfr: set[str] = set()
        folded: set[str] = set()

        for action in state.actions:
            if action.street != current_street:
                current_street = action.street
                street_contributions = {player: 0.0 for player in starting_stacks}
                current_bet = 0.0

            action.pot_before = round(pot, 2)
            action.price_to_call_before = round(max(0.0, current_bet - street_contributions.get(action.player or "", 0.0)), 2)
            contributed, current_bet = self._contribution_for_action(
                action,
                stacks=stacks,
                street_contributions=street_contributions,
                current_bet=current_bet,
            )
            action.contributed = contributed
            pot += contributed

            if action.player:
                stacks[action.player] = max(0.0, round(stacks.get(action.player, 0.0) - contributed, 2))
                total_contributions[action.player] = round(total_contributions.get(action.player, 0.0) + contributed, 2)
                if action.action == "fold":
                    folded.add(action.player)
                if current_street == "preflop" and action.action in {"call", "bet", "raise", "all_in"} and contributed > 0:
                    vpip.add(action.player)
                if current_street == "preflop" and action.action in {"bet", "raise", "all_in"}:
                    pfr.add(action.player)

        state.players = [seat.player for seat in sorted(state.seats, key=lambda seat: seat.seat)]
        state.stacks = self._round_dict(stacks)
        state.pot = round(pot, 2)
        state.current_street = current_street if state.actions else "preflop"
        state.current_bet = round(current_bet, 2)
        state.street_contributions = self._round_dict(street_contributions)
        state.total_contributions = self._round_dict(total_contributions)
        state.folded_players = sorted(folded)
        state.active_players = [player for player in state.players if player not in folded]
        state.positions = self._positions(state)
        state.side_pots = [
            SidePot.model_validate(side_pot)
            for side_pot in self._side_pots(total_contributions, folded)
        ]
        state.current_action = self._format_current_action(state.actions[-1] if state.actions else None)
        state.current_actor = self._next_player(
            state, state.actions[-1].player if state.actions else None
        )
        state.hand_flags = {
            "vpip": sorted(vpip),
            "pfr": sorted(pfr),
        }
        hero_name = self._hero_name(state)
        hero_contribution = street_contributions.get(hero_name or "", 0.0)
        state.hero_to_call = round(max(0.0, current_bet - hero_contribution), 2) if hero_name in state.active_players else 0.0
        if hero_name in state.active_players and stacks.get(hero_name, 0.0) > 0:
            state.hero_legal_actions = ["check", "raise", "all_in"] if state.hero_to_call == 0 else ["fold", "call", "raise", "all_in"]
        else:
            state.hero_legal_actions = []
        state.recommendation = self._recommendation_for(state)
        return state

    @staticmethod
    def _tournament_phase(state: GameState) -> str:
        if state.game_format == "cash":
            return "cash"
        remaining = state.tournament.players_remaining
        paid = state.tournament.paid_places or (len(state.tournament.payouts) or None)
        if remaining is not None and paid is not None and remaining == paid + 1:
            return "bubble"
        if remaining is not None and remaining <= state.tournament.final_table_size:
            return "final_table"
        return "tournament"
