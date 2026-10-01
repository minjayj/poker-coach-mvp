from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from datetime import datetime, timezone
import asyncio
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.models import (
    ActionRequest,
    AzureCredentialRequest,
    CorrectionRequest,
    CoachRequest,
    CoachChatRequest,
    FrameInputRequest,
    GameFormatRequest,
    InputObservation,
    InputStopRequest,
    PlayFrameLabelRequest,
    PlayStartRequest,
    PlayStopRequest,
    ReviewResolutionRequest,
    HandEndRequest,
    ObservedStatsInput,
    PushFoldStudyRequest,
    SeatConfigRequest,
    SpeechTranscriptRequest,
    SolverRequest,
    VisionManualCardRequest,
)
from backend.engine.coach import build_brief, build_plo5_brief, icm_context_from_state, pick_opponent, range_context_from_state, stats_from_profile
from backend.coach_chat import clarification, general_coaching_answer, is_visibility_question, parse_chat_input, preflop_line_hint, wants_specific_decision
from backend.engine.cfr_solver import HoldemCFRSolver, OpponentStats, RangeContext
from backend.engine.open_study import list_push_fold_studies, lookup_push_fold_study
from backend.engine.texas_solver_study import AdjustedStudyRequest, TexasSolverLookup, lookup_texas_solver
from backend.engine.opponent_adjustment import blend_exact_strategy, summarize_opponent_json
from backend.player_database import table_size_bucket
from backend.input_stream import InputStreamHub
from backend.azure_coach_llm import AzureCoachLLM
from backend.local_coach_llm import LocalCoachLLM
from backend.play_recordings import PlayRecordingStore
from backend.speech.action_parser import build_action_parser
from backend.storage import GameService
from backend.vision.camera import CameraManager
from backend.vision.clubgg_holdem import VisionRead
from backend.vision.play_training import train_reviewed_suits

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DEFAULT_DATA_DIR = ROOT_DIR / "data"


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    try:
        await run_in_threadpool(app.state.input_stream.holdem_reader.warmup)
    except Exception as error:
        app.state.input_stream.last_error = f"Hold'em reader unavailable: {type(error).__name__}"
    yield


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Poker Coach MVP", lifespan=app_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=r"^https?://(?:127\.0\.0\.1|localhost)(?::\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    resolved_data_dir = data_dir or DEFAULT_DATA_DIR
    app.state.service = GameService(resolved_data_dir)
    app.state.play = PlayRecordingStore(resolved_data_dir)
    app.state.parser = build_action_parser()
    app.state.camera = CameraManager()
    app.state.solver = HoldemCFRSolver()
    app.state.local_coach = LocalCoachLLM() if os.getenv("POKER_COACH_LLM_PROVIDER", "azure").lower() == "ollama" else AzureCoachLLM()
    app.state.input_stream = InputStreamHub(app.state.play.root / "suit_model.json")
    app.state.vision_lock = asyncio.Lock()
    app.state.connections: set[WebSocket] = set()
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    async def broadcast_state() -> None:
        state = app.state.service.get_state().model_dump(mode="json")
        dead_connections: list[WebSocket] = []
        for connection in list(app.state.connections):
            try:
                await connection.send_json(state)
            except Exception:
                dead_connections.append(connection)
        for connection in dead_connections:
            app.state.connections.discard(connection)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/advanced")
    async def advanced() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "advanced.html")

    @app.get("/play")
    async def play_review() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "play.html")

    @app.get("/api/play/sessions")
    async def list_play_sessions() -> list[dict]:
        return app.state.play.list_sessions()

    @app.post("/api/play/start")
    async def start_play_recording(request: PlayStartRequest) -> dict:
        if app.state.input_stream.source_id != request.source_id:
            raise HTTPException(status_code=422, detail="Start a single-window share before recording.")
        try:
            return app.state.play.start(request.source_id, request.mime_type, request.consent, app.state.service.get_state().game_variant)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/play/{session_id}/chunks/{index}")
    async def upload_play_chunk(session_id: str, index: int, request: Request) -> dict:
        if index < 0 or int(request.headers.get("content-length", "0")) > app.state.play.MAX_CHUNK_BYTES:
            raise HTTPException(status_code=413, detail="Recording chunk is too large.")
        data = await request.body()
        try:
            return await run_in_threadpool(app.state.play.append_chunk, session_id, index, data)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/play/{session_id}/stop")
    async def stop_play_recording(session_id: str, request: PlayStopRequest) -> dict:
        try:
            return app.state.play.stop(session_id, request.incomplete)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/play/{session_id}/frames")
    async def list_play_frames(session_id: str) -> list[dict]:
        try:
            return app.state.play.list_frames(session_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/play/{session_id}/events")
    async def list_play_events(session_id: str) -> list[dict]:
        try:
            return app.state.play.list_events(session_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.get("/api/play/{session_id}/frames/{frame_id}/image")
    async def get_play_frame(session_id: str, frame_id: str) -> FileResponse:
        try:
            return FileResponse(app.state.play.frame_path(session_id, frame_id), media_type="image/jpeg")
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/play/{session_id}/frames/{frame_id}/label")
    async def label_play_frame(session_id: str, frame_id: str, request: PlayFrameLabelRequest) -> dict:
        if not request.verified:
            raise HTTPException(status_code=422, detail="Review the image and explicitly verify this label.")
        try:
            return app.state.play.label_frame(session_id, frame_id, request.hero_cards, request.board_cards, request.action_summary)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/api/play/{session_id}/video")
    async def get_play_video(session_id: str) -> FileResponse:
        try:
            return FileResponse(app.state.play.video_path(session_id), media_type="video/webm")
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    @app.post("/api/play/train")
    async def train_play_reader() -> dict:
        report = await run_in_threadpool(train_reviewed_suits, app.state.play.root)
        if report["activated"]:
            async with app.state.vision_lock:
                app.state.input_stream.holdem_reader.reload_model()
        return report

    @app.get("/api/state")
    async def get_state() -> dict:
        return app.state.service.get_state().model_dump(mode="json")

    @app.post("/api/config/seats")
    async def configure_seats(request: SeatConfigRequest) -> dict:
        state = app.state.service.configure_seats(request)
        await broadcast_state()
        return state.model_dump(mode="json")

    @app.post("/api/config/game-format")
    async def configure_game_format(request: GameFormatRequest) -> dict:
        try:
            state = app.state.service.configure_game_format(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        await broadcast_state()
        return state.model_dump(mode="json")

    @app.post("/api/action")
    async def add_action(request: ActionRequest) -> dict:
        state = app.state.service.apply_action(request)
        app.state.play.record_event("confirmed_action", request.model_dump(mode="json"))
        await broadcast_state()
        return state.model_dump(mode="json")

    @app.post("/api/correction")
    async def correct_action(request: CorrectionRequest) -> dict:
        state = app.state.service.apply_correction(request)
        app.state.play.record_event("action_correction", request.model_dump(mode="json"))
        await broadcast_state()
        return state.model_dump(mode="json")

    @app.get("/api/players")
    async def get_players() -> list[dict]:
        return [
            profile.model_dump(mode="json")
            for profile in app.state.service.list_player_profiles()
        ]

    @app.get("/api/players/{player_id}")
    async def get_player(player_id: str) -> dict:
        profile = app.state.service.get_player_profile(player_id)
        if profile is None:
            return {"player_id": player_id, "found": False}
        return {"found": True, "profile": profile.model_dump(mode="json")}

    @app.get("/api/players/{player_id}/table-breakdown")
    async def get_player_table_breakdown(player_id: str) -> dict:
        breakdown = app.state.service.get_player_table_breakdown(player_id)
        return {
            "player_id": player_id,
            "breakdown": [profile.model_dump(mode="json") for profile in breakdown],
        }

    @app.get("/api/players/{player_id}/history")
    async def get_player_history(player_id: str, limit: int = 100) -> dict:
        return app.state.service.get_player_history(player_id, limit)

    @app.get("/api/players/{player_id}/phase-breakdown")
    async def get_player_phase_breakdown(player_id: str) -> dict:
        return {"player_id": player_id, "phases": app.state.service.get_player_phase_breakdown(player_id)}

    @app.post("/api/players/{player_id}/observed-stats")
    async def save_observed_player_stats(player_id: str, request: ObservedStatsInput) -> dict:
        """Save numbers visibly shown on a consented table/profile panel."""
        profile = app.state.service.save_observed_stats(player_id, request)
        return {"saved": True, "profile": profile.model_dump(mode="json")}

    @app.post("/api/solver/recommend")
    async def solver_recommendation(request: SolverRequest) -> dict:
        """On-demand endpoint; callers decide when to show its result in the UI."""
        live_state = app.state.service.get_state()
        if live_state.game_variant != "nlh" or request.game_variant == "plo5":
            raise HTTPException(status_code=422, detail="Hold'em CFR cannot analyze PLO5. Use the PLO5 coaching brief instead.")
        table_size = len(live_state.observed_active_players) or len(live_state.players)
        stored_profile = app.state.service.get_player_profile_for_context(request.opponent_id, table_size, live_state.tournament_phase) if request.opponent_id else None
        input_stats = request.opponent_stats
        if input_stats is not None:
            stats = OpponentStats(
                vpip=input_stats.vpip, pfr=input_stats.pfr, af=input_stats.af,
                hands_seen=input_stats.hands_seen, vpip_count=input_stats.vpip_count,
                pfr_count=input_stats.pfr_count,
            )
        elif stored_profile is not None:
            stats = OpponentStats(
                vpip=stored_profile.vpip, pfr=stored_profile.pfr,
                af=stored_profile.aggression_factor, hands_seen=stored_profile.hands_seen,
                vpip_count=stored_profile.vpip_count, pfr_count=stored_profile.pfr_count,
            )
        else:
            stats = OpponentStats()

        icm_context, icm_reason = icm_context_from_state(live_state, request.opponent_id)
        cash_mode = (request.game_format or live_state.game_format) == "cash"
        if cash_mode:
            icm_context, icm_reason = None, None
        result = app.state.solver.solve(
            board=request.board, hero_hand=request.hero_hand,
            opponent_stats=stats, current_pot=request.current_pot,
            to_call=request.to_call, raise_to=request.raise_to,
            simulations=request.simulations, cfr_iterations=request.cfr_iterations,
            range_context=RangeContext(
                opponent_position=request.opponent_position or live_state.positions.get(request.opponent_id or "", "unknown"),
                preflop_context=request.preflop_context or range_context_from_state(live_state, request.opponent_id).preflop_context,
                effective_stack_bb=request.effective_stack_bb,
                table_size=request.table_size or table_size,
                street=live_state.current_street,
            ),
            icm_context=icm_context,
            hero_stack=live_state.stacks.get(next((seat.player for seat in live_state.seats if seat.is_hero), "")),
            opponent_stack=live_state.stacks.get(request.opponent_id or ""),
            rake_percent=live_state.cash.rake_percent if cash_mode else 0.0,
            rake_cap=live_state.cash.rake_cap if cash_mode else None,
            no_flop_no_drop=live_state.cash.no_flop_no_drop,
        )
        result["opponent_id"] = request.opponent_id
        result["stats_source"] = "request" if input_stats else "local_database" if stored_profile else "population_prior"
        result["icm_unavailable_reason"] = icm_reason
        return result

    @app.get("/api/study/push-fold")
    async def push_fold_study_catalog() -> dict:
        """List exact supported spots; this data is not used by the live coach."""
        return list_push_fold_studies()

    @app.post("/api/study/push-fold")
    async def push_fold_study_lookup(request: PushFoldStudyRequest) -> dict:
        try:
            return lookup_push_fold_study(
                effective_stack_bb=request.effective_stack_bb,
                hero_hand=request.hero_hand,
                decision=request.decision,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/api/study/texas-solver")
    async def texas_solver_study_lookup(request: TexasSolverLookup) -> dict:
        """Exact post-hand lookup; never substituted into the live coach."""
        try:
            result = lookup_texas_solver(resolved_data_dir / "solver_study", request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if result is None:
            raise HTTPException(status_code=404, detail="No exact TexasSolver spot and hand combo is imported.")
        return result

    @app.post("/api/study/texas-solver/adjusted")
    async def texas_solver_player_adjustment(request: AdjustedStudyRequest) -> dict:
        """Separate exact-tree reference from bounded opponent exploitation."""
        try:
            baseline = lookup_texas_solver(resolved_data_dir / "solver_study", request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if baseline is None:
            raise HTTPException(status_code=404, detail="No exact TexasSolver spot and hand combo is imported.")
        archive = app.state.service.store.read_json(app.state.service.archive_path, [])
        read = summarize_opponent_json(
            archive, player_id=request.opponent_id, game_variant="nlh", game_format="cash",
            tournament_phase="cash", table_size_bucket="1-2",
        )
        profile = app.state.service.get_player_profile_for_variant(request.opponent_id, "nlh")
        player_model = app.state.solver.solve(
            board=request.board, hero_hand=request.hero_hand,
            opponent_stats=stats_from_profile(profile), current_pot=request.pot,
            to_call=request.to_call, raise_to=request.proposed_raise_to,
            simulations=800, cfr_iterations=200,
            range_context=RangeContext(table_size=2, street={3: "flop", 4: "turn", 5: "river"}[len(request.board)]),
            hero_stack=request.effective_stack, opponent_stack=request.effective_stack,
            rake_percent=request.rake_percent,
        )
        return {
            "solver_spot": baseline,
            "opponent_read": read,
            "adjustment": blend_exact_strategy(
                baseline, player_model, read, proposed_raise_to=request.proposed_raise_to,
                observed_opponent_bet_to_pot=request.observed_opponent_bet_to_pot,
            ),
            "player_model": {
                "decision_model": player_model["decision_model"],
                "action_frequencies": player_model["action_frequencies"],
                "equity_samples": player_model["equity_samples"],
                "stats_source": profile.stats_origin if profile and profile.hands_seen else "population_prior",
            },
        }

    @app.get("/api/players/{player_id}/read")
    async def player_json_read(player_id: str) -> dict:
        state = app.state.service.get_state()
        archive = app.state.service.store.read_json(app.state.service.archive_path, [])
        return summarize_opponent_json(
            archive, player_id=player_id, game_variant=state.game_variant,
            game_format=state.game_format, tournament_phase=state.tournament_phase,
            table_size_bucket=table_size_bucket(len(state.observed_active_players) or len(state.players)),
        )

    @app.post("/api/coach/brief")
    async def coach_brief(request: CoachRequest) -> dict:
        """Create an explanation-first coaching brief for an explicitly requested spot."""
        state = app.state.service.get_state()
        opponent_id = pick_opponent(state, request.opponent_id)
        table_size = len(state.observed_active_players) or len(state.players)
        opponent = app.state.service.get_player_profile_for_context(opponent_id, table_size, state.tournament_phase) if opponent_id else None
        hero_name = next((seat.player for seat in state.seats if seat.is_hero), None) or state.observed_hero
        default_to_call = (state.observed_hero_to_call if state.observed_hero_turn and state.observed_hero_to_call is not None
                           else max(0.0, state.current_bet - state.street_contributions.get(hero_name or "", 0.0)))
        to_call = request.to_call if request.to_call is not None else default_to_call
        if state.game_variant == "plo5":
            brief = build_plo5_brief(state, opponent, to_call, request.simulations)
            brief["opponent_id"] = opponent_id
            return brief
        solution = None
        icm_context, icm_reason = icm_context_from_state(state, opponent_id)
        if len(state.hero_cards) == 2:
            pot = state.observed_pot if state.observed_pot is not None else state.pot
            solution = app.state.solver.solve(
                board=state.board_cards,
                hero_hand=state.hero_cards,
                opponent_stats=stats_from_profile(opponent),
                current_pot=pot,
                to_call=to_call,
                raise_to=request.raise_to,
                simulations=request.simulations,
                range_context=replace(range_context_from_state(state, opponent_id), preflop_context=request.preflop_context) if request.preflop_context else range_context_from_state(state, opponent_id),
                icm_context=icm_context,
                hero_stack=state.stacks.get(hero_name or ""),
                opponent_stack=state.stacks.get(opponent_id or ""),
                rake_percent=state.cash.rake_percent if state.game_format == "cash" else 0.0,
                rake_cap=state.cash.rake_cap if state.game_format == "cash" else None,
                no_flop_no_drop=state.cash.no_flop_no_drop,
            )
        opponent_read = None
        if opponent_id and table_size >= 2:
            opponent_read = summarize_opponent_json(
                app.state.service.store.read_json(app.state.service.archive_path, []),
                player_id=opponent_id, game_variant=state.game_variant,
                game_format=state.game_format, tournament_phase=state.tournament_phase,
                table_size_bucket=table_size_bucket(table_size),
            )
        brief = build_brief(state, opponent, solution, to_call, opponent_read)
        brief["opponent_id"] = opponent_id
        brief["icm_unavailable_reason"] = icm_reason
        if icm_reason and state.game_format == "tournament":
            brief["watch_out_for"].append(icm_reason)
        return brief

    @app.post("/api/coach/chat")
    async def coach_chat(request: CoachChatRequest) -> dict:
        """Conversational entry point; facts must be stated or supplied by an adapter."""
        state = app.state.service.get_state()
        if is_visibility_question(request.message):
            stream = app.state.input_stream
            capture_status = stream.status()
            last_frame_at = stream.last_received_at
            fresh_frame = bool(
                stream.source_id and stream.latest_image_data and last_frame_at
                and (datetime.now(timezone.utc) - datetime.fromisoformat(last_frame_at)).total_seconds() < 6
            )
            if fresh_frame and capture_status["frame_static"]:
                reply = ("I'm receiving frames, but the picture has not changed for over 12 seconds. "
                         "If ClubGG is moving, the shared image has frozen; keep the game visible and unminimized, "
                         "or stop and re-share your display. I won't treat this picture as a current hand.")
            elif fresh_frame:
                reply = ("Yes, I'm receiving frames from your shared window. That doesn't mean every card or action is readable; "
                         "I'll ask you to confirm anything I can't verify.")
            else:
                reply = ("Not yet—no fresh shared image is reaching the coach. Click Share table or display and select ClubGG. "
                         "If you're already sharing, check the capture status under the preview.")
            return {"reply": reply, "context": request.context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None}
        previous_context = request.context.model_copy(deep=True)
        context, extracted = parse_chat_input(request.message, request.context, state, app.state.service.player_database.resolve_alias)
        live_read = app.state.input_stream.latest_vision
        last_frame_at = app.state.input_stream.last_received_at
        live_read_fresh = bool(
            live_read and last_frame_at and not app.state.input_stream.status()["frame_static"]
            and (datetime.now(timezone.utc) - datetime.fromisoformat(last_frame_at)).total_seconds() < 6
        )
        explicit_price = bool(re.search(r"\b(?:to call|call costs?|facing|price to call)\s*(?:is|:|=)?\s*\$?\d|\bi can check\b", request.message, re.I))
        if live_read_fresh and live_read and live_read.hero_turn and live_read.hero_to_call is not None:
            # Explicitly supplied prices take precedence over the OCR button.
            if not explicit_price:
                context.to_call = live_read.hero_to_call
            active_opponents = [name for name in state.observed_active_players if name != state.observed_hero]
            if context.opponent_id and context.opponent_id not in active_opponents:
                context.opponent_id = None
            if context.opponent_id is None and len(active_opponents) == 1:
                context.opponent_id = active_opponents[0]
        elif live_read_fresh and live_read and not live_read.hero_turn and not explicit_price:
            context.to_call = None
        if "confirmed_alias" in extracted:
            alias, canonical = extracted["confirmed_alias"]
            if alias and canonical and canonical in [*state.players, *state.observed_players]:
                app.state.service.player_database.remember_alias(alias, canonical)
        if "identity_question" in extracted:
            return {"reply": extracted["identity_question"], "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None}
        if "hero_cards" in extracted or "board_cards" in extracted:
            try:
                state = app.state.service.set_manual_cards(VisionManualCardRequest(
                    hero_cards=extracted.get("hero_cards", state.hero_cards),
                    board_cards=extracted.get("board_cards", state.board_cards),
                ))
            except ValueError as error:
                return {"reply": f"I couldn't use those cards: {error} Please confirm the exact cards.", "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None}
        if "current_pot" in extracted:
            state = app.state.service.apply_input_observation(InputObservation(current_pot=extracted["current_pot"]), "coach-chat")
        if extracted:
            await broadcast_state()
        teaching_reply = general_coaching_answer(request.message) if not extracted else None
        if teaching_reply:
            return {"reply": teaching_reply, "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None}
        context_changed = any(
            getattr(context, field) != getattr(previous_context, field)
            for field in ("to_call", "opponent_id", "opponent_line", "pending_alias", "pending_candidate")
        )
        if not extracted and not context_changed and not wants_specific_decision(request.message):
            local_reply = await app.state.local_coach.converse(request.message, request.history, state)
            if local_reply:
                source = getattr(app.state.local_coach, "provider", "local") + "_model"
                return {"reply": local_reply, "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None, "reply_source": source}
            model_status = await app.state.local_coach.status()
            reply = f"The conversational model isn't ready: {model_status['reason']} I can still analyze a specific hand if you give me your cards, pot, price to call, and the action before you."
            return {"reply": reply, "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None, "reply_source": "rules_fallback"}
        question = clarification(state, context)
        if question:
            reply = question
            if extracted.get("hero_cards") or extracted.get("board_cards") or "current_pot" in extracted:
                reply = "Got it. " + question
            return {"reply": reply, "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None}
        if (live_read_fresh and live_read and live_read.hero_turn and context.to_call
                and len(state.observed_active_players) > 2 and not context.opponent_id):
            reply = (f"I can see {len(state.observed_active_players)} players with cards and a {context.to_call:.2f} call price, "
                     "but I can't verify who made the bet. Which player bet or raised? I won't pretend this is heads-up.")
            return {"reply": reply, "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": None}
        brief = await coach_brief(CoachRequest(
            opponent_id=context.opponent_id,
            to_call=context.to_call,
            simulations=800,
            preflop_context=preflop_line_hint(state, context),
        ))
        if not brief["ready"]:
            reply = brief["headline"] + " " + " ".join(brief["why"][:1])
        else:
            line_hint = preflop_line_hint(state, context)
            line = (f"You said: {context.opponent_line}. " + (f"I treated that as a {line_hint.replace('_', '-')} preflop range prior. " if line_hint else "That postflop line is not separately solved here. ")) if context.opponent_line else "I don't have a confirmed opponent action yet. "
            profile = app.state.service.get_player_profile(context.opponent_id) if context.opponent_id else None
            read = (f"{profile.name}'s {profile.hands_seen} recorded {state.game_variant.upper()} hands give a tentative range read; this is not proof of their holdings. " if profile and profile.hands_seen else "No reliable player-specific sample is available, so the range is a population prior. ")
            contextual_read = brief.get("opponent_read") or {}
            if contextual_read.get("facing_hands", 0) >= 12:
                actions = contextual_read["facing_actions"]
                read += (f"In {contextual_read['facing_hands']} comparable hands they folded "
                         f"{actions.get('fold', 0)} of {sum(actions.values())} tracked decisions facing a bet. "
                         "A bet size alone does not tell me their hidden cards. ")
            move = brief.get("recommended_action")
            lead = f"My model leans {move.upper()}. " if move else "I can't justify a specific move from this model yet. "
            why = " ".join(brief["why"][:2])
            caveat = "This is a bounded regret-matching estimate, not a full GTO solve. " if state.game_variant == "nlh" else "This is a random-opponent PLO5 study, not GTO. "
            reply = lead + why + " " + line + read + caveat + "Please verify the displayed cards, pot, price, and active players before acting."
        return {"reply": reply, "context": context.model_dump(mode="json"), "state": state.model_dump(mode="json"), "brief": brief}

    @app.get("/api/coach/model-status")
    async def coach_model_status() -> dict:
        return await app.state.local_coach.status()

    @app.post("/api/coach/connect-azure")
    async def connect_azure(credential: AzureCredentialRequest, request: Request) -> dict:
        """One-time local key entry. No browser or server-side persistence."""
        if not isinstance(app.state.local_coach, AzureCoachLLM):
            raise HTTPException(status_code=409, detail="Azure is not the selected conversation provider.")
        origin = request.headers.get("origin", "")
        expected_origin = f"{request.url.scheme}://{request.url.netloc}"
        if request.url.hostname not in {"127.0.0.1", "localhost"} or origin != expected_origin:
            raise HTTPException(status_code=403, detail="Open the local live page to connect Azure.")
        app.state.local_coach.set_api_key(credential.api_key.get_secret_value())
        # Verify the credential without sending table frames or chat history.
        await app.state.local_coach.converse("Reply with the single word READY.", [], app.state.service.get_state())
        return await app.state.local_coach.status()

    @app.get("/api/reviews")
    async def get_reviews() -> list[dict]:
        return [review.model_dump(mode="json") for review in app.state.service.get_state().review_queue]

    @app.post("/api/reviews/{review_id}")
    async def resolve_review(review_id: str, request: ReviewResolutionRequest) -> dict:
        try:
            state = app.state.service.resolve_review(review_id, request.decision == "approve")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        app.state.play.record_event("review_decision", {"review_id": review_id, "decision": request.decision})
        await broadcast_state()
        return state.model_dump(mode="json")

    @app.post("/api/speech/transcript")
    async def parse_transcript(request: SpeechTranscriptRequest) -> dict:
        parsed = app.state.parser.parse(request.text)
        state = app.state.service.get_state()

        if parsed.player is None and request.text.strip().lower().startswith("i "):
            for seat in state.seats:
                if seat.seat == state.hero_seat:
                    parsed.player = seat.player
                    break

        if parsed.player is None and request.speaker:
            parsed.player = app.state.service._match_player_name(request.speaker, state)

        if parsed.player is None:
            parsed.player = state.current_actor

        applied = False
        updated_state = state
        if parsed.confidence >= request.auto_apply_threshold and parsed.action != "unknown" and parsed.player:
            updated_state = app.state.service.apply_action(
                ActionRequest(
                    player=parsed.player,
                    action=parsed.action,
                    amount=parsed.amount,
                    amount_type=parsed.amount_type,
                    raw_text=parsed.raw_text,
                    source="speech",
                    confidence=parsed.confidence,
                )
            )
            applied = True
            app.state.play.record_event("confirmed_action", parsed.model_dump(mode="json"))
            await broadcast_state()

        return {
            "parsed": parsed.model_dump(mode="json"),
            "applied": applied,
            "state": updated_state.model_dump(mode="json"),
        }

    @app.post("/api/hand/end")
    async def end_hand(request: HandEndRequest) -> dict:
        state = app.state.service.finalize_hand(request)
        app.state.play.record_event("hand_end", request.model_dump(mode="json"))
        await broadcast_state()
        return {
            "message": "Hand archived and stats updated.",
            "state": state.model_dump(mode="json"),
            "players": [
                profile.model_dump(mode="json")
                for profile in app.state.service.list_player_profiles()
            ],
        }

    @app.get("/api/vision/status")
    async def vision_status() -> dict:
        return app.state.camera.status(refresh=True)

    @app.get("/api/input/latest")
    async def latest_input() -> dict:
        """Latest generic preview frame and its transport status."""
        return app.state.input_stream.latest_frame()

    @app.post("/api/input/stop")
    async def stop_input(request: InputStopRequest) -> dict:
        app.state.play.stop_source(request.source_id)
        cleared = app.state.input_stream.clear_if_current(request.source_id)
        if cleared:
            app.state.service.clear_live_observations()
            await broadcast_state()
        return {"cleared": cleared}

    @app.post("/api/input/frame")
    async def ingest_input(request: FrameInputRequest) -> dict:
        """Accept a browser capture, screen capture, camera adapter, or CNN result."""
        previous_source = app.state.input_stream.source_id
        try:
            new_event_ids, has_observation = app.state.input_stream.ingest(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        state = (app.state.service.clear_live_observations()
                 if previous_source and previous_source != request.source_id
                 else app.state.service.get_state())
        observation = request.observation
        vision_read: VisionRead | None = None
        detected = app.state.input_stream.observed_variant(request.source_id, request.image_data)
        browser_share = request.source_id.startswith("clubgg-") and "-share-" in request.source_id
        if request.image_data and browser_share and observation is None:
            try:
                async with app.state.vision_lock:
                    vision_read = await run_in_threadpool(
                        app.state.input_stream.holdem_reader.read, request.image_data, request.source_id
                    )
                    app.state.input_stream.latest_vision = vision_read
                if state.game_variant == "nlh" or (detected and detected[0] == "nlh"):
                    hero_name = next((seat.player for seat in state.seats if seat.is_hero), None)
                    if (vision_read.hole_cards_absent and hero_name and vision_read.bottom_name
                            and vision_read.bottom_name.casefold() == hero_name.casefold()):
                        vision_read.hero_cards = []
                    observation = app.state.input_stream.stable_holdem_observation(request.source_id, vision_read)
                app.state.input_stream.last_error = None
            except Exception as error:
                # A missing OCR runtime or changed game layout must never become
                # a fabricated observation or a live recommendation.
                app.state.input_stream.last_error = f"Hold'em reader unavailable: {type(error).__name__}"
                vision_read = VisionRead(status="reader_unavailable", missing=["Please confirm cards, pot and action in chat."])
                app.state.input_stream.latest_vision = vision_read
        if detected and detected[0] != state.game_variant and (observation is None or observation.game_variant is None):
            observation = (observation or InputObservation()).model_copy(update={
                "game_variant": detected[0],
                "field_confidences": {**(observation.field_confidences if observation else {}), "game_variant": detected[1]},
            })
        if observation is not None:
            state = app.state.service.apply_input_observation(observation, request.source_id)
            for event in observation.actions:
                if event.event_id not in new_event_ids:
                    continue
                confidence = event.confidence if event.confidence is not None else observation.field_confidences.get("action", observation.confidence if observation.confidence is not None else 1.0)
                resolved_player, identity_review = app.state.service.resolve_observed_player(event.player, state)
                action_needs_amount = event.action in {"raise", "bet"} and event.amount is None
                if identity_review is not None:
                    payload = event.model_dump(mode="json")
                    payload.update(identity_review)
                    state = app.state.service.queue_review(
                        "identity_action" if payload.get("candidate_player") else "ambiguous_action",
                        payload, confidence, request.source_id
                    )
                    continue
                if action_needs_amount:
                    payload = event.model_dump(mode="json")
                    payload["reason"] = "missing_amount"
                    state = app.state.service.queue_review(
                        "action", payload, confidence, request.source_id
                    )
                    continue
                if confidence < 0.85:
                    state = app.state.service.queue_review(
                        "action", {**event.model_dump(mode="json"), "player": resolved_player}, confidence, request.source_id
                    )
                    continue
                state = app.state.service.apply_action(
                    ActionRequest(
                        player=resolved_player,
                        action=event.action,
                        amount=event.amount,
                        amount_type=event.amount_type,
                        street=event.street,
                        source="vision",
                        confidence=confidence,
                    )
                )
                app.state.play.record_event("confirmed_action", event.model_dump(mode="json"))
            await broadcast_state()
        if request.image_data and request.source_id.startswith("clubgg-window-share-"):
            await run_in_threadpool(
                app.state.play.record_frame, request.source_id, request.image_data,
                vision_read.public() if vision_read else None, state.hand_id,
            )
        return {
            "accepted": True,
            "new_action_events": len(new_event_ids),
            "observation_received": has_observation or detected is not None,
            "input": app.state.input_stream.status(),
            "vision": vision_read.public() if vision_read else None,
            "state": state.model_dump(mode="json"),
        }

    @app.post("/api/vision/card_manual")
    async def set_manual_cards(request: VisionManualCardRequest) -> dict:
        try:
            state = app.state.service.set_manual_cards(request)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        await broadcast_state()
        return state.model_dump(mode="json")

    @app.websocket("/ws")
    async def websocket_state(websocket: WebSocket) -> None:
        await websocket.accept()
        app.state.connections.add(websocket)
        await websocket.send_json(app.state.service.get_state().model_dump(mode="json"))
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            app.state.connections.discard(websocket)
        except Exception:
            app.state.connections.discard(websocket)

    return app


app = create_app()
