"""Explanation-first coaching layer built from state, stats, and solver output."""

from __future__ import annotations

from typing import Any

from backend.engine.cfr_solver import OpponentStats, RangeContext
from backend.engine.tournament import ICMDecisionContext
from backend.engine.plo5 import estimate_plo5_equity
from backend.models import GameState, PlayerProfile


def pick_opponent(state: GameState, requested: str | None) -> str | None:
    known = state.players or state.observed_players
    if requested and requested in known:
        return requested
    hero_name = next((seat.player for seat in state.seats if seat.is_hero), None) or state.observed_hero
    for action in reversed(state.actions):
        if action.player and action.player != hero_name and action.action in {"raise", "bet", "all_in"}:
            return action.player
    opponents = [player for player in (state.observed_active_players or known) if player != hero_name]
    return opponents[0] if len(opponents) == 1 else None


def range_context_from_state(state: GameState, opponent_id: str | None) -> RangeContext:
    """Derive a cautious position/action prior from the deterministic hand ledger."""
    hero_name = next((seat.player for seat in state.seats if seat.is_hero), None)
    opponent_actions = [
        action for action in state.actions
        if action.player == opponent_id and action.street == "preflop"
    ]
    preflop_context = "unknown"
    if opponent_actions:
        first = opponent_actions[0]
        earlier_raises = sum(
            action.action in {"raise", "bet", "all_in"}
            for action in state.actions[:state.actions.index(first)]
            if action.street == "preflop"
        )
        if first.action == "call":
            preflop_context = "call_raise" if earlier_raises else "limp"
        elif first.action in {"raise", "bet", "all_in"}:
            preflop_context = "open" if earlier_raises == 0 else "three_bet" if earlier_raises == 1 else "four_bet"
    big_blind = state.blinds.big or 1.0
    hero_stack = state.stacks.get(hero_name or "", 0.0)
    opponent_stack = state.stacks.get(opponent_id or "", 0.0)
    effective_stack_bb = min(hero_stack, opponent_stack) / big_blind if hero_stack and opponent_stack else None
    return RangeContext(
        opponent_position=state.positions.get(opponent_id or "", "unknown"),
        preflop_context=preflop_context,
        effective_stack_bb=round(effective_stack_bb, 2) if effective_stack_bb is not None else None,
        table_size=len(state.observed_active_players) or len(state.active_players) or len(state.observed_players) or len(state.players),
        street={0: "preflop", 3: "flop", 4: "turn", 5: "river"}.get(len(state.board_cards), state.current_street),
    )


def icm_context_from_state(state: GameState, opponent_id: str | None) -> tuple[ICMDecisionContext | None, str | None]:
    """Use payout equity only when the entire remaining final table is known."""
    if state.game_format != "tournament":
        return None, None
    tournament = state.tournament
    count = tournament.players_remaining
    paid_places = tournament.paid_places or len(tournament.payouts)
    if count is None or count > tournament.final_table_size or count != len(state.seats):
        return None, "ICM needs the stacks of every remaining player at the final table."
    if not tournament.payouts or len(tournament.payouts) < paid_places:
        return None, "Enter the payouts for every paid place to enable prize-equity calculations."
    hero = next((seat.player for seat in state.seats if seat.is_hero), None)
    names = [seat.player for seat in state.seats]
    if hero not in names or opponent_id not in names or hero == opponent_id:
        return None, "Choose a seated opponent and Hero to calculate tournament risk."
    stacks = tuple(float(state.stacks.get(name, 0.0)) for name in names)
    if sum(stacks) <= 0 or any(stack < 0 for stack in stacks):
        return None, "Enter current stacks before calculating payout pressure."
    return ICMDecisionContext(
        stacks=stacks,
        hero_index=names.index(hero),
        opponent_index=names.index(opponent_id),
        payouts=tuple(tournament.payouts),
    ), None


def build_brief(
    state: GameState,
    opponent: PlayerProfile | None,
    solution: dict[str, Any] | None,
    to_call: float,
    opponent_read: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate quantitative analysis into compact, teachable coaching notes."""
    hero_name = next((seat.player for seat in state.seats if seat.is_hero), None) or state.observed_hero or "Hero"
    board = " ".join(state.board_cards) if state.board_cards else "preflop"
    hero_cards = " ".join(state.hero_cards) if state.hero_cards else "not detected"
    pot = state.observed_pot if state.observed_pot is not None else state.pot
    what_i_see = [
        f"{hero_name} holds {hero_cards}; board: {board}.",
        f"Pot used for analysis: ${pot:.2f}. You need ${to_call:.2f} to continue.",
        f"{len(state.observed_active_players) or len(state.active_players) or len(state.observed_players) or len(state.players)} players are currently represented in the hand context.",
    ]
    if state.observed_active_players:
        what_i_see.append("Visible cards at: " + ", ".join(state.observed_active_players) + ".")
    watch_out_for = [
        "Confirm cards, pot, and the amount to call before relying on any analysis.",
        "Treat a low-sample opponent profile as a starting point; the Bayesian range model intentionally shrinks it toward population tendencies.",
    ]
    if state.game_format == "tournament":
        what_i_see.append(
            f"Tournament: {state.tournament_phase.replace('_', ' ')}; "
            f"{state.tournament.players_remaining or 'unknown'} remain, "
            f"{state.tournament.paid_places or len(state.tournament.payouts) or 'unknown'} paid."
        )
        if state.tournament_phase in {"bubble", "final_table"}:
            watch_out_for.append(
                "Elimination and payout jumps can change risk tolerance. Check this player's measured behavior in this stage before assuming they are waiting for someone else to bust."
            )
    else:
        what_i_see.append("Cash game: analysis uses chip value; tournament payouts do not apply.")
    if opponent is not None:
        what_i_see.append(
            f"Opponent memory for {opponent.name}: {opponent.hands_seen} hands, "
            f"VPIP {opponent.vpip:.0%}, PFR {opponent.pfr:.0%}, AF {opponent.aggression_factor:.2f}."
        )
        if opponent.stats_origin.endswith("_sample"):
            what_i_see.append(f"These tendencies come from {opponent.local_hands_seen} recorded hands in the current phase.")
        if opponent.vpip >= 0.40:
            watch_out_for.append("This is a loose profile, so the range model includes substantially more weak and suited trash combinations.")
        if opponent.aggression_factor >= 2.5:
            watch_out_for.append("High aggression can make a raise less likely to fold this opponent; avoid assuming automatic fold equity.")
    if opponent_read and opponent_read["facing_hands"] >= 12:
        faced = opponent_read["facing_actions"]
        samples = sum(faced.values())
        what_i_see.append(
            f"In {opponent_read['facing_hands']} comparable recorded hands, this player folded "
            f"{faced.get('fold', 0)} of {samples} tracked decisions facing a bet."
        )
        sizes = opponent_read["own_aggressive_sizes"]
        if sizes["samples"] >= 12:
            counts = sizes["counts"]
            what_i_see.append(
                f"Recorded aggressive size mix: small {counts['small']}, medium {counts['medium']}, large {counts['large']}."
            )
        watch_out_for.append("Bet-size frequency alone does not reveal whether this player holds a strong hand or is bluffing.")
    if state.review_queue:
        watch_out_for.insert(0, f"{len(state.review_queue)} low-confidence vision item(s) still need review before treating the state as final.")

    if solution is None:
        return {
            "ready": False,
            "headline": "Coach is waiting for two hero cards before analyzing the spot.",
            "recommended_action": None,
            "what_i_see": what_i_see,
            "why": ["Add hero cards and request coaching to calculate equity and action frequencies."],
            "watch_out_for": watch_out_for,
            "next_rep": "Practice saying the pot, price to call, board texture, and opponent tendency out loud before choosing an action.",
            "opponent_read": opponent_read,
        }

    frequencies = solution["action_frequencies"]
    move = str(solution["recommended_action"])
    why = [
        f"Estimated equity is {solution['equity']:.0%} across {solution['equity_samples']:,} weighted runouts.",
        f"Regret-matching mix: fold {frequencies['fold']:.0%}, {'check' if to_call == 0 else 'call'} {frequencies['call']:.0%}, raise {frequencies['raise']:.0%}.",
        f"The raise model estimates {solution['fold_equity']:.0%} fold equity against the selected range.",
    ]
    if state.game_format == "cash" and solution.get("rake") and solution["rake"]["percent"] > 0:
        rake_cap = solution["rake"]["cap"]
        cap_text = f" capped at {rake_cap:.2f}" if rake_cap is not None else ""
        why.append(
            f"Cash-game estimate includes {solution['rake']['percent']:g}% rake{cap_text}."
        )
    if state.game_format == "tournament":
        if solution.get("decision_model") == "tournament_icm":
            icm = solution["icm"]
            why.append(
                f"ICM estimates prize equity after folding at {icm['fold_prize_equity']:.2f}; "
                f"modeled call at {icm['call_prize_equity']:.2f} and raise at {icm['raise_prize_equity']:.2f}."
            )
            why.append(f"Prize-equity downside/upside ratio for a call: {icm['risk_premium_ratio']:.2f}.")
        else:
            watch_out_for.append("Payout equity is unavailable with the current tournament inputs; this result uses chip EV.")
    if move == "raise":
        why.append(f"The selected raise size is ${solution['raise_to']:.2f}; use it as a sizing reference, not an instruction.")
    elif move == "call":
        why.append("Calling preserves your equity while avoiding the extra risk assigned to the raise branch.")
    elif move == "check":
        why.append("Checking continues without investing additional chips.")
    else:
        why.append("Folding protects the stack when the price and modeled range do not justify continuing.")
    return {
        "ready": True,
        "headline": f"Coaching view: {move.upper()} is the highest-frequency line in this model.",
        "recommended_action": move,
        "what_i_see": what_i_see,
        "why": why,
        "watch_out_for": watch_out_for,
        "next_rep": "After the hand, compare the actual showdown or action to the predicted range and update your read rather than memorizing one result.",
        "solution": solution,
        "opponent_read": opponent_read,
    }


def build_plo5_brief(state: GameState, opponent: PlayerProfile | None, to_call: float, simulations: int) -> dict[str, Any]:
    """Separate, cautious PLO5 coaching path; never reuse Hold'em CFR."""
    pot = state.observed_pot if state.observed_pot is not None else state.pot
    see = [
        f"PLO5: Hero holds {' '.join(state.hero_cards) or 'unknown'}; board {' '.join(state.board_cards) or 'preflop'}.",
        f"Observed pot {pot:.2f}; price to continue {to_call:.2f}.",
    ]
    watch = [
        "This is a heads-up comparison against one random five-card opponent, not a calibrated PLO5 range or solved strategy.",
        "Check the action history, pot, stack sizes, and number of live opponents before acting.",
    ]
    if state.game_format == "cash" and state.cash.rake_percent > 0:
        watch.append("This PLO5 comparison does not yet subtract cash-game rake from the call value.")
    if opponent:
        see.append(f"{opponent.name}: {opponent.hands_seen} PLO5-specific hands, VPIP {opponent.vpip:.0%}, PFR {opponent.pfr:.0%}.")
    if state.game_format == "tournament":
        watch.append("Tournament payout pressure is not modeled by this PLO5 check.")
    if len(state.hero_cards) != 5 or len(state.board_cards) not in {0, 3, 4, 5}:
        return {
            "ready": False, "headline": "Enter five PLO5 hole cards and a valid board before coaching.",
            "recommended_action": None, "what_i_see": see,
            "why": ["PLO5 must use exactly two of your five hole cards and exactly three board cards."],
            "watch_out_for": watch, "next_rep": "Verify all five hole cards first.",
        }
    try:
        result = estimate_plo5_equity(state.hero_cards, state.board_cards, simulations)
    except ValueError as error:
        return {
            "ready": False, "headline": "Card input needs correction before PLO5 coaching.",
            "recommended_action": None, "what_i_see": see, "why": [str(error)],
            "watch_out_for": watch, "next_rep": "Correct duplicate or invalid cards and retry.",
        }
    equity = float(result["equity"])
    pot_odds = to_call / (pot + to_call) if pot + to_call > 0 else 0.0
    multiway = (len(state.active_players) or len(state.players)) > 2
    action = None
    if len(state.board_cards) < 5:
        watch.append("With cards still to come, this raw equity does not model future betting or equity realization; no move is recommended.")
    if state.game_format == "cash" and not multiway and pot > 0 and len(state.board_cards) == 5:
        if to_call <= 0:
            action = "check"
        elif equity > pot_odds + .15:
            action = "call"
        elif equity < pot_odds - .15:
            action = "fold"
    if multiway:
        watch.append("Several players may contest the pot; heads-up equity can overstate your chance to win.")
    if state.review_queue:
        watch.insert(0, f"{len(state.review_queue)} uncertain input item(s) need review.")
    headline = f"PLO5 study view: {action.upper()} is plausible under this narrow model." if action else "PLO5 study view: no reliable action call from this sample."
    return {
        "ready": True, "headline": headline, "recommended_action": action,
        "what_i_see": see,
        "why": [
            f"Estimated equity versus one random PLO5 hand: {equity:.0%} from {result['samples']} runouts.",
            f"Immediate pot-odds threshold for a call: {pot_odds:.0%}.",
            "The comparison enforces exactly two hole cards and three board cards; it does not estimate opponent fold equity or justify a raise.",
        ],
        "watch_out_for": watch,
        "next_rep": "After the hand, compare the shown hand and actual action to the random-opponent baseline.",
        "solution": {"decision_model": "plo5_random_opponent", **result, "pot_odds": round(pot_odds, 4)},
    }


def stats_from_profile(profile: PlayerProfile | None) -> OpponentStats:
    if profile is None:
        return OpponentStats()
    return OpponentStats(
        vpip=profile.vpip,
        pfr=profile.pfr,
        af=profile.aggression_factor,
        hands_seen=profile.hands_seen,
        vpip_count=profile.vpip_count,
        pfr_count=profile.pfr_count,
    )
