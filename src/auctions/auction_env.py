"""
Sealed-bid first-price auction (FPSB) with independent private values.

Standard textbook setup:
  - N risk-neutral bidders.
  - Each draws valuation v_i ~ Uniform[0, 1] independently.
  - Each submits a sealed bid b_i >= 0.
  - Highest bidder wins and pays own bid (first-price rule).
  - Winner's payoff = v_i - b_i.  Losers' payoff = 0.
  - Ties broken uniformly at random.

Theoretical Bayesian-Nash equilibrium for risk-neutral bidders:
  b*(v) = v * (N - 1) / N.

For N = 2: b*(v) = v / 2.  Expected revenue = 1/3.  Expected winner surplus = 1/3.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


N_DEFAULT = 2


@dataclass
class AuctionOutcome:
    valuations: np.ndarray   # (N,)  private valuations
    bids: np.ndarray         # (N,)  bid each bidder submitted
    winner: int              # index of winning bidder
    revenue: float           # price paid (= winning bid)
    payoffs: np.ndarray      # (N,)  v_i - b_i for winner, 0 for losers
    allocation_efficient: bool  # did the highest-valuation bidder win?


class FirstPriceAuction:
    """Single-shot sealed-bid first-price auction.

    Use:
        env = FirstPriceAuction(n=2, seed=42)
        valuations = env.sample_valuations()
        bids = np.array([bidder.bid(v) for bidder, v in zip(bidders, valuations)])
        outcome = env.resolve(valuations, bids)
    """

    def __init__(self, n: int = N_DEFAULT, seed: int | None = None):
        if n < 2:
            raise ValueError("Need at least 2 bidders.")
        self.n = n
        self.rng = np.random.default_rng(seed)

    def sample_valuations(self) -> np.ndarray:
        return self.rng.uniform(0.0, 1.0, size=self.n).astype(np.float64)

    def resolve(self, valuations: np.ndarray, bids: np.ndarray) -> AuctionOutcome:
        if valuations.shape != (self.n,) or bids.shape != (self.n,):
            raise ValueError("valuations and bids must have shape (n,).")
        if np.any(bids < 0):
            raise ValueError("Negative bids are not allowed.")

        # Determine winner: highest bid; uniform random tie-break.
        max_bid = bids.max()
        top = np.flatnonzero(bids == max_bid)
        winner = int(self.rng.choice(top))

        revenue = float(bids[winner])
        payoffs = np.zeros(self.n, dtype=np.float64)
        payoffs[winner] = valuations[winner] - bids[winner]

        max_v = valuations.max()
        top_value = np.flatnonzero(valuations == max_v)
        allocation_efficient = winner in set(top_value.tolist())

        return AuctionOutcome(
            valuations=valuations.copy(),
            bids=bids.copy(),
            winner=winner,
            revenue=revenue,
            payoffs=payoffs,
            allocation_efficient=allocation_efficient,
        )


def equilibrium_bid(valuation: float, n: int = N_DEFAULT) -> float:
    """Risk-neutral Bayesian-Nash equilibrium bid for FPSB with U[0,1] IPV."""
    return valuation * (n - 1) / n


def expected_equilibrium_revenue(n: int = N_DEFAULT) -> float:
    """E[max_i v_i] * (n-1)/n where max of n U[0,1] draws has mean n/(n+1)."""
    return (n - 1) / (n + 1)


def expected_equilibrium_winner_surplus(n: int = N_DEFAULT) -> float:
    """E[max_i v_i - b*(max_i v_i)] = (n/(n+1)) * (1/n) = 1/(n+1)."""
    return 1.0 / (n + 1)
