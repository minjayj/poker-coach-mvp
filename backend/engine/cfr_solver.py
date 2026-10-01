"""Fast, bounded Hold'em decision abstraction using regret matching (CFR)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter
from typing import Iterable

import numpy as np

from backend.engine.hand_evaluator import PerfectHashEvaluator, RANKS, SUITS, normalize_cards
from backend.engine.tournament import ICMDecisionContext, tournament_utilities

ACTIONS = ("fold", "call", "raise")
FULL_DECK = tuple(f"{rank}{suit.lower()}" for rank in RANKS for suit in SUITS)


@dataclass(frozen=True)
class OpponentStats:
    vpip: float = 0.25
    pfr: float = 0.18
    af: float = 1.5
    hands_seen: int = 0
    vpip_count: int | None = None
    pfr_count: int | None = None

    @staticmethod
    def _rate(value: float) -> float:
        """Accept either 0.40 or 40 as a percentage input."""
        return float(np.clip(value / 100.0 if value > 1.0 else value, 0.0, 1.0))

    def posterior_rates(self, prior_hands: float = 20.0) -> tuple[float, float]:
        """Beta-binomial shrinkage keeps small samples near population priors."""
        hands = max(0, int(self.hands_seen))
        observed_vpip = self.vpip_count if self.vpip_count is not None else self._rate(self.vpip) * hands
        observed_pfr = self.pfr_count if self.pfr_count is not None else self._rate(self.pfr) * hands
        vpip = (observed_vpip + prior_hands * 0.25) / (hands + prior_hands)
        pfr = (observed_pfr + prior_hands * 0.18) / (hands + prior_hands)
        return float(vpip), float(pfr)


@dataclass(frozen=True)
class RangeContext:
    """Situation variables that change a player's plausible hole-card range."""

    opponent_position: str = "unknown"
    preflop_context: str = "unknown"
    effective_stack_bb: float | None = None
    table_size: int = 2
    street: str = "preflop"


@dataclass(frozen=True)
class EquityResult:
    equity: float
    wins: int
    ties: int
    losses: int
    samples: int


def combo_label(card_a: str, card_b: str) -> str:
    """Return conventional range labels such as AA, AKs, 72o, and J4s."""
    rank_a, rank_b = card_a[0].upper(), card_b[0].upper()
    suit_a, suit_b = card_a[1].lower(), card_b[1].lower()
    if rank_a == rank_b:
        return rank_a + rank_b
    high, low = sorted((rank_a, rank_b), key=RANKS.index, reverse=True)
    return f"{high}{low}{'s' if suit_a == suit_b else 'o'}"


class BayesianRangeModel:
    """Weights legal hole-card combinations from VPIP/PFR/AF observations."""

    def weights(self, legal_cards: Iterable[str], stats: OpponentStats, context: RangeContext | None = None) -> tuple[list[tuple[str, str]], np.ndarray, dict[str, float]]:
        cards = tuple(legal_cards)
        combos = list(combinations(cards, 2))
        if not combos:
            raise ValueError("No legal opponent combinations remain.")

        posterior_vpip, posterior_pfr = stats.posterior_rates()
        # Positive values mean a wider-than-population range.  This is the
        # Bayesian adjustment consumed by both equity sampling and CFR utility.
        loose_adjustment = float(np.clip((posterior_vpip - 0.25) / 0.25, -0.75, 1.5))
        aggression = float(np.clip(stats.af, 0.0, 8.0))
        context = context or RangeContext()
        position = context.opponent_position.upper()
        position_width = 0.0
        if position in {"UTG", "UTG+1", "MP"}:
            position_width = -0.18
        elif position in {"CO", "BTN", "BTN/SB"}:
            position_width = 0.16
        action_context = context.preflop_context.lower()
        stack_bb = context.effective_stack_bb
        multiway_pressure = 0.10 if context.table_size >= 3 else 0.0
        weights = np.empty(len(combos), dtype=np.float64)

        for index, (first, second) in enumerate(combos):
            high, low = sorted((RANKS.index(first[0].upper()), RANKS.index(second[0].upper())), reverse=True)
            suited = first[1] == second[1]
            pair = high == low
            strength = (high + low) / 24.0 + (0.24 if pair else 0.0) + (0.08 if suited else 0.0)
            trashness = 1.0 - min(1.0, strength)
            # Loose players receive materially more weak combos.  Explicitly
            # boost the requested examples without excluding any legal combo.
            weight = 0.15 + strength + loose_adjustment * (0.9 * trashness)
            if combo_label(first, second) in {"72o", "J4s"}:
                weight += max(0.0, loose_adjustment) * 1.5
            # Higher PFR/AF makes the continuing range modestly stronger.
            weight += max(0.0, posterior_pfr - 0.18) * strength
            weight += max(0.0, aggression - 1.5) * 0.015 * strength
            # Position and the preflop sequence act as priors before player
            # history is applied. Early opens/3-bets are stronger; late opens
            # and limps contain more weak combinations.
            weight += position_width * (trashness if position_width > 0 else strength)
            if action_context == "limp":
                weight += 0.20 * trashness
            elif action_context in {"call_raise", "three_bet", "four_bet"}:
                weight += 0.24 * strength - 0.12 * trashness
            elif action_context == "open":
                weight += 0.06 * strength
            if stack_bb is not None and stack_bb <= 25:
                weight += 0.10 * strength + (0.08 if pair else 0.0)
            if multiway_pressure:
                weight += multiway_pressure * strength - 0.05 * trashness
            weights[index] = max(0.001, weight)

        weights /= weights.sum()
        return combos, weights, {
            "posterior_vpip": round(posterior_vpip, 4),
            "posterior_pfr": round(posterior_pfr, 4),
            "loose_adjustment": round(loose_adjustment, 4),
            "position_width": round(position_width, 4),
            "table_size": context.table_size,
            "preflop_context": action_context,
            "effective_stack_bb": stack_bb,
        }


class HoldemCFRSolver:
    """Latency-bounded CFR decision solver for one live Hold'em decision point.

    This is deliberately a one-node betting abstraction, not a claim of a full
    no-limit game-tree Nash solution.  It combines sampled board equity, a
    Bayesian opponent range, and regret matching across fold/call/raise in a
    predictable request budget suitable for an on-demand local UI.
    """

    def __init__(self, evaluator: PerfectHashEvaluator | None = None, seed: int | None = None) -> None:
        self.evaluator = evaluator or PerfectHashEvaluator()
        self.range_model = BayesianRangeModel()
        self.rng = np.random.default_rng(seed)

    def estimate_equity(
        self,
        board: list[str],
        hero_hand: list[str],
        stats: OpponentStats,
        range_context: RangeContext | None = None,
        simulations: int = 3_000,
    ) -> tuple[EquityResult, dict[str, float]]:
        board_cards = normalize_cards(board)
        hero_cards = normalize_cards(hero_hand)
        if len(hero_cards) != 2 or not 0 <= len(board_cards) <= 5:
            raise ValueError("Use exactly two hero cards and zero to five board cards.")
        known = board_cards + hero_cards
        if len(set(known)) != len(known):
            raise ValueError("Board and hero hand cannot share a card.")
        remaining = tuple(card for card in FULL_DECK if card not in known)
        combos, weights, adjustment = self.range_model.weights(remaining, stats, range_context)
        draw_count = 5 - len(board_cards)
        samples = max(100, min(int(simulations), 20_000))
        choice_indexes = self.rng.choice(len(combos), size=samples, p=weights)
        wins = ties = losses = 0

        for choice in choice_indexes:
            villain = combos[int(choice)]
            available = [card for card in remaining if card not in villain]
            runout = tuple(self.rng.choice(available, size=draw_count, replace=False)) if draw_count else ()
            outcome = self.evaluator.compare(hero_cards + board_cards + runout, villain + board_cards + runout)
            if outcome > 0:
                wins += 1
            elif outcome == 0:
                ties += 1
            else:
                losses += 1
        equity = (wins + ties * 0.5) / samples
        return EquityResult(equity=equity, wins=wins, ties=ties, losses=losses, samples=samples), adjustment

    @staticmethod
    def _regret_matching(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
        positive = np.maximum(regrets, 0.0) * legal
        return positive / positive.sum() if positive.sum() > 0 else legal / legal.sum()

    def solve(
        self,
        board: list[str],
        hero_hand: list[str],
        opponent_stats: OpponentStats,
        current_pot: float,
        to_call: float = 0.0,
        raise_to: float | None = None,
        simulations: int = 3_000,
        cfr_iterations: int = 400,
        range_context: RangeContext | None = None,
        icm_context: ICMDecisionContext | None = None,
        hero_stack: float | None = None,
        opponent_stack: float | None = None,
        rake_percent: float = 0.0,
        rake_cap: float | None = None,
        no_flop_no_drop: bool = True,
    ) -> dict[str, object]:
        started = perf_counter()
        pot = max(0.0, float(current_pot))
        call_cost = max(0.0, float(to_call))
        if hero_stack is not None:
            call_cost = min(call_cost, max(0.0, hero_stack))
        raise_cost = max(call_cost, float(raise_to) if raise_to is not None else max(call_cost * 2.5, pot * 0.75, 1.0))
        if hero_stack is not None:
            raise_cost = min(raise_cost, max(0.0, hero_stack))
        if opponent_stack is not None:
            raise_cost = min(raise_cost, call_cost + max(0.0, opponent_stack))
        equity_result, adjustment = self.estimate_equity(board, hero_hand, opponent_stats, range_context, simulations)
        equity = equity_result.equity

        # Opponent fold equity falls against loose/high-AF players and increases
        # as the chosen raise becomes larger relative to the pot.
        fold_equity = float(np.clip(
            0.16 + 0.42 * (raise_cost / max(1.0, pot + call_cost))
            - 0.32 * adjustment["posterior_vpip"] - 0.025 * min(opponent_stats.af, 8.0),
            0.03, 0.82,
        ))
        rake_rate = float(np.clip(rake_percent / 100.0, 0.0, 1.0))
        def rake(total_pot: float) -> float:
            value = rake_rate * total_pot
            return min(value, rake_cap) if rake_cap is not None else value

        called_rake = rake(pot + call_cost * 2)
        raised_rake = rake(pot + raise_cost * 2)
        fold_rake = 0.0 if no_flop_no_drop and (range_context is None or range_context.street == "preflop") else rake(pot)
        utilities = np.array([
            0.0,
            equity * (pot - called_rake) - (1.0 - equity) * call_cost,
            fold_equity * (pot - fold_rake) + (1.0 - fold_equity) * (equity * (pot + call_cost - raised_rake) - (1.0 - equity) * raise_cost),
        ], dtype=np.float64)

        icm_details: dict[str, float] | None = None
        icm_error: str | None = None
        if icm_context is not None:
            try:
                adjusted, icm_details = tournament_utilities(
                    icm_context, pot=pot, call_cost=call_cost, raise_cost=raise_cost,
                    equity=equity, fold_equity=fold_equity,
                )
                utilities = np.asarray(adjusted, dtype=np.float64)
            except ValueError as error:
                icm_error = str(error)

        legal = np.array([
            call_cost > 0,
            True,
            raise_cost > call_cost and (hero_stack is None or hero_stack > call_cost)
            and (opponent_stack is None or opponent_stack > 0),
        ], dtype=np.float64)

        regrets = np.zeros(3, dtype=np.float64)
        strategy_sum = np.zeros(3, dtype=np.float64)
        for _ in range(max(20, min(int(cfr_iterations), 2_000))):
            strategy = self._regret_matching(regrets, legal)
            node_value = float(strategy @ utilities)
            regrets += utilities - node_value
            strategy_sum += strategy
        average_strategy = strategy_sum / strategy_sum.sum()
        best_index = int(np.argmax(average_strategy))
        recommended = "check" if best_index == 1 and call_cost == 0 else ACTIONS[best_index]
        return {
            "recommended_action": recommended,
            "action_frequencies": {action: round(float(average_strategy[index]), 4) for index, action in enumerate(ACTIONS)},
            "raise_to": round(raise_cost, 2) if ACTIONS[best_index] == "raise" else None,
            "equity": round(equity, 4),
            "equity_samples": equity_result.samples,
            "fold_equity": round(fold_equity, 4),
            "utilities": {action: round(float(utilities[index]), 4) for index, action in enumerate(ACTIONS)},
            "bayesian_adjustment": adjustment,
            "decision_model": "tournament_icm" if icm_details is not None else "chip_ev",
            "icm": icm_details,
            "icm_error": icm_error,
            "rake": {"percent": rake_percent, "cap": rake_cap, "estimated_call_rake": round(called_rake, 4), "estimated_raise_rake": round(raised_rake, 4)} if icm_details is None else None,
            "latency_ms": round((perf_counter() - started) * 1_000, 2),
        }
