"""
Offline pre-flight: validate the auction infrastructure without spending API
credits. Runs analytical bidders against each other, confirms revenue and
winner-surplus statistics match the closed-form predictions.

Usage:
    python3 run_offline.py
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np

from auction_env import (
    FirstPriceAuction,
    expected_equilibrium_revenue,
    expected_equilibrium_winner_surplus,
)
from equilibrium_bidder import EquilibriumBidder, NaiveBidder, ShadingBidder


def run_match(bidders, n_trials: int, seed: int = 0) -> dict:
    n = len(bidders)
    env = FirstPriceAuction(n=n, seed=seed)

    revenues = []
    payoffs_by_bidder = defaultdict(list)
    bids_by_bidder = defaultdict(list)
    valuations_by_bidder = defaultdict(list)
    eff = []

    for _ in range(n_trials):
        v = env.sample_valuations()
        b = np.array([bidders[i].bid(float(v[i])) for i in range(n)])
        out = env.resolve(v, b)
        revenues.append(out.revenue)
        eff.append(out.allocation_efficient)
        for i in range(n):
            payoffs_by_bidder[i].append(out.payoffs[i])
            bids_by_bidder[i].append(b[i])
            valuations_by_bidder[i].append(v[i])

    rev = np.array(revenues)
    return {
        "n_bidders": n,
        "trials": n_trials,
        "rev_mean": float(rev.mean()),
        "rev_std": float(rev.std()),
        "alloc_efficient_rate": float(np.mean(eff)),
        "bidders": [
            {
                "name": repr(bidders[i]),
                "payoff_mean": float(np.mean(payoffs_by_bidder[i])),
                "payoff_std": float(np.std(payoffs_by_bidder[i])),
                "bid_mean_over_valuation": float(
                    np.mean(np.divide(
                        bids_by_bidder[i],
                        np.maximum(valuations_by_bidder[i], 1e-9),
                    ))
                ),
            }
            for i in range(n)
        ],
    }


def main() -> None:
    n = 2
    trials = 5_000

    expected_rev = expected_equilibrium_revenue(n)
    expected_winner = expected_equilibrium_winner_surplus(n)

    print(f"FPSB pre-flight (N={n} bidders, {trials} trials per match)")
    print(f"  Theoretical equilibrium: bid(v) = v * (n-1)/n = v * {(n-1)/n}")
    print(f"  Theoretical E[revenue] = {expected_rev:.4f}")
    print(f"  Theoretical E[winner_surplus] = {expected_winner:.4f}")
    print()

    matches = [
        ("Equilibrium vs Equilibrium",
            [EquilibriumBidder(n), EquilibriumBidder(n)]),
        ("Naive vs Naive",
            [NaiveBidder(), NaiveBidder()]),
        ("Equilibrium vs Naive",
            [EquilibriumBidder(n), NaiveBidder()]),
        ("Shade=0.7 vs Equilibrium  (over-shaded)",
            [ShadingBidder(0.7), EquilibriumBidder(n)]),
        ("Shade=0.3 vs Equilibrium  (under-shaded)",
            [ShadingBidder(0.3), EquilibriumBidder(n)]),
    ]

    print(f"{'Match':<48} {'Rev mean':>10} {'Eff rate':>10}")
    print("-" * 75)
    for name, bidders in matches:
        r = run_match(bidders, trials)
        print(f"{name:<48} {r['rev_mean']:>10.4f} {r['alloc_efficient_rate']:>10.1%}")
        for b in r["bidders"]:
            print(f"    {b['name']:<40}  payoff={b['payoff_mean']:.4f}  "
                  f"b/v={b['bid_mean_over_valuation']:.3f}")
        print()


if __name__ == "__main__":
    main()
