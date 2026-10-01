"""Optional offline CFR benchmarks; never queried for a live Hold'em action.

Kuhn and Leduc are small imperfect-information games. They help check a CFR
research setup, but their policies cannot be transplanted to full NL Hold'em.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def openspiel_cfr_plus(iterations: int = 1_000) -> dict[str, object]:
    """Train OpenSpiel CFR+ on Kuhn poker and report actual exploitability."""
    if not 1 <= iterations <= 100_000:
        raise ValueError("iterations must be between 1 and 100000")
    try:
        import pyspiel
    except ImportError as error:
        raise RuntimeError("Install optional open_spiel in a supported research environment.") from error
    game = pyspiel.load_game("kuhn_poker")
    solver = pyspiel.CFRPlusSolver(game)
    for _ in range(iterations):
        solver.evaluate_and_update_policy()
    return {
        "framework": "OpenSpiel", "game": "kuhn_poker", "algorithm": "CFR+",
        "iterations": iterations,
        "exploitability": float(pyspiel.exploitability(game, solver.average_policy())),
        "study_only": True,
    }


def rlcard_leduc_cfr(iterations: int = 100, model_dir: Path | None = None) -> dict[str, object]:
    """Train RLCard's chance-sampling CFR on Leduc in a separate lab directory."""
    if not 1 <= iterations <= 10_000:
        raise ValueError("iterations must be between 1 and 10000")
    try:
        import rlcard
        from rlcard.agents import CFRAgent
    except ImportError as error:
        raise RuntimeError("Install optional rlcard in a research environment.") from error
    destination = model_dir or Path.cwd() / "data" / "research" / "rlcard_leduc"
    destination.mkdir(parents=True, exist_ok=True)
    env = rlcard.make("leduc-holdem", config={"seed": 7, "allow_step_back": True})
    agent = CFRAgent(env, str(destination))
    for _ in range(iterations):
        agent.train()
    agent.save()
    return {
        "framework": "RLCard", "game": "leduc-holdem", "algorithm": "chance-sampling CFR",
        "iterations": iterations, "model_dir": str(destination), "study_only": True,
        "note": "No full Hold'em exploitability or live action policy is implied.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an optional small-game CFR research check.")
    parser.add_argument("framework", choices=("openspiel", "rlcard"))
    parser.add_argument("--iterations", type=int, default=100)
    arguments = parser.parse_args()
    result = (openspiel_cfr_plus if arguments.framework == "openspiel" else rlcard_leduc_cfr)(arguments.iterations)
    print(json.dumps(result, indent=2))
