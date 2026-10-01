"""Small-field ICM payout equity for tournament coaching.

This models finishing order from chip shares. It is a payout model, not a
prediction that a specific opponent will fold or wait for an elimination.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ICMDecisionContext:
    stacks: tuple[float, ...]
    hero_index: int
    opponent_index: int
    payouts: tuple[float, ...]


def icm_equity(stacks: tuple[float, ...], payouts: tuple[float, ...], hero_index: int) -> float:
    """Exact independent chip model equity via subset dynamic programming."""
    size = len(stacks)
    if not 2 <= size <= 9 or not 0 <= hero_index < size:
        raise ValueError("ICM requires two to nine remaining players and a valid hero index.")
    if any(stack < 0 for stack in stacks) or sum(stacks) <= 0:
        raise ValueError("ICM requires nonnegative stacks with chips in play.")
    if any(payout < 0 for payout in payouts):
        raise ValueError("Payouts must be nonnegative.")
    prizes = tuple(payouts[:size]) + (0.0,) * max(0, size - len(payouts))
    masses = [0.0] * (1 << size)
    masses[0] = 1.0
    equity = 0.0
    for mask in range(1 << size):
        mass = masses[mask]
        if mass == 0.0 or mask == (1 << size) - 1:
            continue
        rank = mask.bit_count()
        remaining = [index for index in range(size) if not mask & (1 << index)]
        remaining_chips = sum(stacks[index] for index in remaining)
        for index in remaining:
            # Zero-stack players split the remaining places only after every
            # player with chips has been selected.
            probability = (stacks[index] / remaining_chips) if remaining_chips else (1.0 / len(remaining))
            next_mass = mass * probability
            if index == hero_index:
                equity += next_mass * prizes[rank]
            else:
                masses[mask | (1 << index)] += next_mass
    return equity


def tournament_utilities(
    context: ICMDecisionContext,
    *,
    pot: float,
    call_cost: float,
    raise_cost: float,
    equity: float,
    fold_equity: float,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Compare simplified fold, call and raise outcomes in prize equity.

    Pot chips are awarded in every terminal branch, preserving total chips.
    Opponent matching on a raise is capped by their available stack.
    """
    stacks = list(context.stacks)
    hero, opponent = context.hero_index, context.opponent_index
    if hero == opponent or pot < 0 or call_cost > stacks[hero] or raise_cost > stacks[hero]:
        raise ValueError("Incomplete or inconsistent tournament decision context.")

    def value(changes: dict[int, float]) -> float:
        scenario = stacks.copy()
        for index, delta in changes.items():
            scenario[index] += delta
        if any(amount < -1e-8 for amount in scenario):
            raise ValueError("Decision branch would create a negative stack.")
        return icm_equity(tuple(max(0.0, amount) for amount in scenario), context.payouts, hero)

    folded = value({opponent: pot})
    called_win = value({hero: pot})
    called_loss = value({hero: -call_cost, opponent: pot + call_cost})
    match = min(max(raise_cost - call_cost, 0.0), stacks[opponent])
    raised_win = value({hero: pot + match, opponent: -match})
    raised_loss = value({hero: -raise_cost, opponent: pot + raise_cost})
    raised_fold = value({hero: pot})
    call_value = equity * called_win + (1.0 - equity) * called_loss
    raise_value = fold_equity * raised_fold + (1.0 - fold_equity) * (
        equity * raised_win + (1.0 - equity) * raised_loss
    )
    gain = max(0.0, called_win - folded)
    loss = max(0.0, folded - called_loss)
    details = {
        "fold_prize_equity": round(folded, 4),
        "call_prize_equity": round(call_value, 4),
        "raise_prize_equity": round(raise_value, 4),
        "risk_premium_ratio": round(loss / gain, 3) if gain > 0 else 0.0,
    }
    return (0.0, call_value - folded, raise_value - folded), details
