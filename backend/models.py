from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, model_validator

ActionName = Literal["raise", "call", "fold", "check", "bet", "all_in", "unknown"]
AmountType = Literal["total", "additional", "unknown"]
StreetName = Literal["preflop", "flop", "turn", "river", "showdown"]
GameFormat = Literal["cash", "tournament"]
GameVariant = Literal["nlh", "plo5"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BlindConfig(BaseModel):
    small: float = 1.0
    big: float = 2.0


class TournamentContext(BaseModel):
    players_remaining: int | None = Field(default=None, ge=2)
    paid_places: int | None = Field(default=None, ge=1)
    payouts: list[float] = Field(default_factory=list, max_length=9)
    final_table_size: int = Field(default=9, ge=2, le=9)
    ante: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_payouts(self) -> "TournamentContext":
        if any(amount < 0 for amount in self.payouts):
            raise ValueError("Payouts must be nonnegative.")
        if any(first < second for first, second in zip(self.payouts, self.payouts[1:])):
            raise ValueError("Payouts must be ordered from first place downward.")
        if self.paid_places is not None and self.payouts and len(self.payouts) > self.paid_places:
            raise ValueError("Payouts cannot include more places than paid_places.")
        return self


class TournamentContextUpdate(BaseModel):
    """Fields a vision adapter actually saw; omitted fields keep saved values."""

    players_remaining: int | None = Field(default=None, ge=2)
    paid_places: int | None = Field(default=None, ge=1)
    payouts: list[float] | None = Field(default=None, max_length=9)
    final_table_size: int | None = Field(default=None, ge=2, le=9)
    ante: float | None = Field(default=None, ge=0.0)


class CashContext(BaseModel):
    rake_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    rake_cap: float | None = Field(default=None, ge=0.0)
    no_flop_no_drop: bool = True


class SeatState(BaseModel):
    seat: int
    player: str
    stack: float = 0.0
    is_hero: bool = False


class SidePot(BaseModel):
    amount: float
    contributors: list[str] = Field(default_factory=list)
    eligible_players: list[str] = Field(default_factory=list)


class ReviewItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    field: str
    payload: dict[str, object] = Field(default_factory=dict)
    confidence: float
    source_id: str
    created_at: str = Field(default_factory=utc_now)


class Recommendation(BaseModel):
    move: Literal["fold", "check", "call", "raise", "all_in"] = "check"
    amount: Optional[float] = None
    reason: str = "Waiting for enough context."


class ActionRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: str = Field(default_factory=utc_now)
    street: StreetName = "preflop"
    player: Optional[str] = None
    action: ActionName = "unknown"
    amount: Optional[float] = None
    amount_type: AmountType = "unknown"
    source: Literal["manual", "speech", "vision", "correction", "system"] = "manual"
    raw_text: Optional[str] = None
    confidence: Optional[float] = None
    contributed: float = 0.0
    pot_before: float | None = None
    price_to_call_before: float | None = None


class ParsedSpeechAction(BaseModel):
    player: Optional[str] = None
    action: ActionName = "unknown"
    amount: Optional[float] = None
    amount_type: AmountType = "unknown"
    confidence: float = 0.0
    raw_text: str


class GameState(BaseModel):
    game_format: GameFormat = "cash"
    game_variant: GameVariant = "nlh"
    cash: CashContext = Field(default_factory=CashContext)
    tournament: TournamentContext = Field(default_factory=TournamentContext)
    tournament_phase: str = "cash"
    seats: list[SeatState] = Field(default_factory=list)
    players: list[str] = Field(default_factory=list)
    blinds: BlindConfig = Field(default_factory=BlindConfig)
    hero_seat: Optional[int] = None
    current_street: StreetName = "preflop"
    pot: float = 0.0
    stacks: dict[str, float] = Field(default_factory=dict)
    actions: list[ActionRecord] = Field(default_factory=list)
    recommendation: Recommendation = Field(default_factory=Recommendation)
    current_action: str = "Waiting for action"
    current_actor: Optional[str] = None
    hero_cards: list[str] = Field(default_factory=list)
    board_cards: list[str] = Field(default_factory=list)
    current_bet: float = 0.0
    street_contributions: dict[str, float] = Field(default_factory=dict)
    hand_id: int = 1
    hand_flags: dict[str, list[str]] = Field(
        default_factory=lambda: {"vpip": [], "pfr": []}
    )
    observed_pot: float | None = None
    observed_actor: Optional[str] = None
    observed_players: list[str] = Field(default_factory=list)
    observed_hero: str | None = None
    observed_active_players: list[str] = Field(default_factory=list)
    observed_hero_turn: bool = False
    observed_hero_to_call: float | None = None
    observed_raise_to: float | None = None
    input_confidence: float | None = None
    last_input_at: str | None = None
    last_input_source: str | None = None
    last_input_verified_spot: bool = False
    dealer_seat: int | None = None
    positions: dict[str, str] = Field(default_factory=dict)
    active_players: list[str] = Field(default_factory=list)
    folded_players: list[str] = Field(default_factory=list)
    total_contributions: dict[str, float] = Field(default_factory=dict)
    side_pots: list[SidePot] = Field(default_factory=list)
    hero_to_call: float = 0.0
    hero_legal_actions: list[str] = Field(default_factory=list)
    review_queue: list[ReviewItem] = Field(default_factory=list)


class SeatConfigRequest(BaseModel):
    seats: list[SeatState]
    blinds: BlindConfig = Field(default_factory=BlindConfig)
    hero_seat: Optional[int] = None
    dealer_seat: Optional[int] = None


class GameFormatRequest(BaseModel):
    game_format: GameFormat
    game_variant: GameVariant | None = None
    cash: CashContext = Field(default_factory=CashContext)
    tournament: TournamentContext = Field(default_factory=TournamentContext)


class ActionRequest(BaseModel):
    player: Optional[str] = None
    action: ActionName
    amount: Optional[float] = None
    amount_type: AmountType = "unknown"
    street: Optional[StreetName] = None
    raw_text: Optional[str] = None
    source: Literal["manual", "speech", "vision"] = "manual"
    confidence: Optional[float] = None


class CorrectionRequest(BaseModel):
    type: Literal["wrong_player", "wrong_amount", "wrong_action", "undo"]
    player: Optional[str] = None
    action: Optional[ActionName] = None
    amount: Optional[float] = None


class SpeechTranscriptRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    auto_apply_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class HandEndRequest(BaseModel):
    winner: Optional[str] = None
    notes: Optional[str] = None


class VisionManualCardRequest(BaseModel):
    hero_cards: list[str] = Field(default_factory=list, max_length=5)
    board_cards: list[str] = Field(default_factory=list)


class PlayerProfile(BaseModel):
    name: str
    game_variant: GameVariant = "nlh"
    hands_played: int = 0
    vpip: float = 0.0
    pfr: float = 0.0
    raises: int = 0
    calls: int = 0
    folds: int = 0
    checks: int = 0
    all_ins: int = 0
    avg_raise_amount: float = 0.0
    aggression_score: float = 0.0
    vpip_hands: int = 0
    pfr_hands: int = 0
    total_raise_amount: float = 0.0
    raise_samples: int = 0
    last_updated: str = Field(default_factory=utc_now)
    hands_seen: int = 0
    vpip_count: int = 0
    pfr_count: int = 0
    total_pots: float = 0.0
    call_count: int = 0
    raise_count: int = 0
    aggression_factor: float = 0.0
    bets: int = 0
    all_in_count: int = 0
    showdown_count: int = 0
    win_count: int = 0
    total_contributed: float = 0.0
    action_count: int = 0
    table_size_bucket: str | None = None
    local_hands_seen: int = 0
    observed_hands_seen: int | None = None
    observed_vpip: float | None = None
    observed_pfr: float | None = None
    observed_aggression_factor: float | None = None
    observed_source: str | None = None
    observed_confidence: float | None = None
    observed_at: str | None = None
    stats_origin: str = "local"


class OpponentStatsInput(BaseModel):
    """Rates accept either decimal form (0.40) or percent form (40)."""

    vpip: float = 0.25
    pfr: float = 0.18
    af: float = 1.5
    hands_seen: int = Field(default=0, ge=0)
    vpip_count: int | None = Field(default=None, ge=0)
    pfr_count: int | None = Field(default=None, ge=0)


class SolverRequest(BaseModel):
    board: list[str] = Field(default_factory=list, max_length=5)
    hero_hand: list[str] = Field(min_length=2, max_length=2)
    current_pot: float = Field(ge=0.0)
    opponent_id: str | None = None
    opponent_stats: OpponentStatsInput | None = None
    to_call: float = Field(default=0.0, ge=0.0)
    raise_to: float | None = Field(default=None, ge=0.0)
    simulations: int = Field(default=3000, ge=100, le=20000)
    cfr_iterations: int = Field(default=400, ge=20, le=2000)
    opponent_position: str | None = None
    preflop_context: str | None = None
    effective_stack_bb: float | None = Field(default=None, ge=0.0)
    table_size: int | None = Field(default=None, ge=1, le=9)
    game_format: GameFormat | None = None
    game_variant: GameVariant | None = None


class InputActionEvent(BaseModel):
    """An idempotent action emitted by a vision/CNN adapter."""

    event_id: str
    player: str | None = None
    action: ActionName
    amount: float | None = Field(default=None, ge=0.0)
    amount_type: AmountType = "unknown"
    street: StreetName | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ObservedStatsInput(BaseModel):
    """A displayed player-profile snapshot, kept separate from local hand history."""

    hands_seen: int | None = Field(default=None, ge=0)
    vpip: float | None = Field(default=None, ge=0.0, le=100.0)
    pfr: float | None = Field(default=None, ge=0.0, le=100.0)
    aggression_factor: float | None = Field(default=None, ge=0.0, le=100.0)
    source: str = "clubgg_profile"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ObservedPlayerStatsInput(ObservedStatsInput):
    player: str


class InputObservation(BaseModel):
    """Provider-neutral structured output from a camera, screen, or CNN."""

    hero_cards: list[str] | None = Field(default=None, max_length=5)
    board_cards: list[str] | None = Field(default=None, max_length=5)
    current_pot: float | None = Field(default=None, ge=0.0)
    current_street: StreetName | None = None
    current_actor: str | None = None
    visible_players: list[str] | None = None
    hero_player: str | None = None
    hero_turn: bool | None = None
    hero_to_call: float | None = Field(default=None, ge=0.0)
    raise_to: float | None = Field(default=None, ge=0.0)
    active_players: list[str] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    field_confidences: dict[str, float] = Field(default_factory=dict)
    actions: list[InputActionEvent] = Field(default_factory=list)
    player_stats: list[ObservedPlayerStatsInput] = Field(default_factory=list)
    game_format: GameFormat | None = None
    game_variant: GameVariant | None = None
    tournament: TournamentContextUpdate | None = None


class FrameInputRequest(BaseModel):
    """A generic video frame plus optional recognition result.

    ``image_data`` is a JPEG/PNG data URL.  Providers may omit it and submit
    only structured observations when the video stream lives elsewhere.
    """

    source_id: str = Field(default="external")
    image_data: str | None = None
    observation: InputObservation | None = None


class InputStopRequest(BaseModel):
    source_id: str


class PlayStartRequest(BaseModel):
    source_id: str = Field(min_length=1, max_length=150)
    mime_type: str = Field(min_length=1, max_length=100)
    consent: bool


class PlayStopRequest(BaseModel):
    incomplete: bool = False


class PlayFrameLabelRequest(BaseModel):
    hero_cards: list[str] = Field(default_factory=list, max_length=2)
    board_cards: list[str] = Field(default_factory=list, max_length=5)
    action_summary: str = Field(default="", max_length=300)
    verified: bool


class CoachRequest(BaseModel):
    opponent_id: str | None = None
    to_call: float | None = Field(default=None, ge=0.0)
    raise_to: float | None = Field(default=None, ge=0.0)
    simulations: int = Field(default=3000, ge=100, le=20000)
    preflop_context: Literal["open", "three_bet", "four_bet", "limp", "call_raise"] | None = None


class PushFoldStudyRequest(BaseModel):
    """An offline lookup in the bundled, narrowly scoped NLH study artifact."""

    effective_stack_bb: int = Field(ge=2)
    hero_hand: list[str] = Field(min_length=2, max_length=2)
    decision: Literal["sb_first", "bb_vs_shove"]


class CoachChatContext(BaseModel):
    hand_id: int | None = None
    game_variant: GameVariant | None = None
    to_call: float | None = Field(default=None, ge=0.0)
    opponent_id: str | None = None
    opponent_line: str | None = None
    pending_alias: str | None = None
    pending_candidate: str | None = None


class CoachChatTurn(BaseModel):
    role: Literal["user", "coach"]
    text: str = Field(min_length=1, max_length=600)


class CoachChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    context: CoachChatContext = Field(default_factory=CoachChatContext)
    history: list[CoachChatTurn] = Field(default_factory=list, max_length=8)


class AzureCredentialRequest(BaseModel):
    # SecretStr keeps the credential masked in model representations and errors.
    api_key: SecretStr = Field(min_length=20, max_length=512)


class ReviewResolutionRequest(BaseModel):
    decision: Literal["approve", "dismiss"]
