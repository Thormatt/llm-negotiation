"""
Smoke tests for the Phase B protocol scaffolding.

Run as a script: python3 test_protocols.py

These tests don't use pytest because we're keeping the dependency surface
minimal; they're lightweight asserts that catch wiring errors immediately
and serve as executable documentation of the protocols' interfaces.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from protocols.nash_bargaining import CooperativeBlend2P, NashBargaining2P
from protocols.vickrey import Vickrey


def test_vickrey_basic():
    """Highest bidder wins, pays second-highest, payoff = v - p."""
    auction = Vickrey(seed=0)
    # 3 bidders: bids 0.6, 0.7, 0.4 ; valuations 0.8, 0.7, 0.5
    inputs = [(0.6, 0.8), (0.7, 0.7), (0.4, 0.5)]
    result = auction.run(inputs)
    assert result.allocation["winner"] == 1, f"winner should be index 1, got {result.allocation}"
    assert abs(result.allocation["price"] - 0.6) < 1e-9
    assert abs(result.payoffs[1] - (0.7 - 0.6)) < 1e-9
    assert result.payoffs[0] == 0.0 and result.payoffs[2] == 0.0
    print(f"test_vickrey_basic: PASS  (winner={result.allocation['winner']}, "
          f"price={result.allocation['price']:.3f}, payoff_winner={result.payoffs[1]:.3f})")


def test_vickrey_truthful_dominant():
    """Truthful bidding (b = v) gives non-negative payoff in expectation."""
    auction = Vickrey(seed=1)
    rng = np.random.default_rng(2)
    payoffs = []
    for _ in range(2000):
        v = rng.uniform(0.0, 1.0, 3)  # 3 bidders
        inputs = [(float(v[i]), float(v[i])) for i in range(3)]  # all truthful
        out = auction.run(inputs)
        payoffs.extend(out.payoffs)
    arr = np.array(payoffs)
    assert (arr >= -1e-9).all(), f"truthful bidding produced negative payoff: min={arr.min()}"
    print(f"test_vickrey_truthful_dominant: PASS  "
          f"(min payoff={arr.min():.3f}, mean payoff={arr.mean():.3f})")


def test_nash_bargaining_asymmetric():
    """Party who weights an issue more should win that issue (corner solution)."""
    proto = NashBargaining2P()
    direction = np.array([1, 1, 1, 1, 1])
    weights_a = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
    weights_b = np.array([0.1, 0.225, 0.225, 0.225, 0.225])
    inputs = [
        {"weights": weights_a, "direction": direction},
        {"weights": weights_b, "direction": -direction},
    ]
    out = proto.run(inputs)
    proposal = np.array(out.allocation["proposal"])
    # On issue 0, A heavily prefers high → corner at 0.95
    assert proposal[0] >= 0.94, f"issue 0 should be at the high corner, got {proposal[0]}"
    # On issues 1-4, B prefers high → A's perspective wants low corner
    assert all(proposal[i] <= 0.06 for i in range(1, 5)), \
        f"issues 1-4 should be at the low corner from A's view, got {proposal[1:]}"
    print(f"test_nash_bargaining_asymmetric: PASS  "
          f"(proposal={[f'{x:.2f}' for x in proposal]}, payoffs={tuple(f'{p:.3f}' for p in out.payoffs)})")


def test_nash_bargaining_pareto_efficient():
    """Nash corner-solution should be on the Pareto frontier."""
    proto = NashBargaining2P()
    rng = np.random.default_rng(7)
    direction = np.array([1, -1, 1, -1, 1])
    for trial in range(100):
        wa = rng.dirichlet(np.ones(5))
        wb = rng.dirichlet(np.ones(5))
        inputs = [
            {"weights": wa, "direction": direction},
            {"weights": wb, "direction": -direction},
        ]
        out = proto.run(inputs)
        ua, ub = out.payoffs
        for _ in range(200):
            alt = rng.uniform(0.05, 0.95, 5)
            tilde_a = np.where(direction > 0, alt, 1 - alt)
            tilde_b = np.where(direction > 0, 1 - alt, alt)
            ua_alt = float(np.dot(wa, tilde_a))
            ub_alt = float(np.dot(wb, tilde_b))
            assert not (ua_alt > ua + 1e-6 and ub_alt > ub + 1e-6), (
                f"trial {trial}: alternative {alt} dominates Nash solution: "
                f"({ua_alt:.3f}, {ub_alt:.3f}) > ({ua:.3f}, {ub:.3f})"
            )
    print("test_nash_bargaining_pareto_efficient: PASS  "
          "(100 trials, 200 random alternatives each, no dominator found)")


def test_cooperative_blend_smooth():
    """CooperativeBlend2P matches the paper's compute_target_proposal at aggression=1.

    Symmetric weights → 0.5 per issue; asymmetric → fractional split."""
    proto = CooperativeBlend2P()
    weights = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
    direction = np.array([1, -1, 1, -1, 1])
    inputs = [
        {"weights": weights, "direction": direction},
        {"weights": weights, "direction": -direction},
    ]
    out = proto.run(inputs)
    proposal = np.array(out.allocation["proposal"])
    assert np.allclose(proposal, 0.5, atol=1e-6), \
        f"symmetric weights should give 0.5, got {proposal}"
    print(f"test_cooperative_blend_smooth: PASS  (symmetric → 0.5, payoffs equal)")


if __name__ == "__main__":
    test_vickrey_basic()
    test_vickrey_truthful_dominant()
    test_nash_bargaining_asymmetric()
    test_nash_bargaining_pareto_efficient()
    test_cooperative_blend_smooth()
    print("\nAll Phase B protocol smoke tests passed.")
