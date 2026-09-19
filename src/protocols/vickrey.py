"""
Vickrey (sealed-bid second-price) auction protocol.

N bidders submit sealed bids. The highest bidder wins and pays the
second-highest bid. Truthful bidding (b_i = v_i) is a dominant strategy.

This is the auction-theory canonical example of a mechanism that aligns
participant incentives with truth-telling: an LLMIO that simply reports
the human's true valuation as the bid is optimal here. No sub-agent
strategy is needed — the protocol does the work.
"""

from __future__ import annotations

import numpy as np

from framework.protocol import Outcome, Protocol


class Vickrey(Protocol):
    """Sealed-bid second-price auction."""

    name = "Vickrey"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def run(self, inputs: list[tuple[float, float]]) -> Outcome:
        """
        Args:
            inputs: list of (bid, valuation) tuples, one per bidder.
                    Valuation is needed only for payoff calculation,
                    not for outcome determination.

        Returns:
            Outcome with allocation = winning bidder index, payoffs per bidder.
        """
        bids = np.array([b for b, _ in inputs], dtype=np.float64)
        valuations = np.array([v for _, v in inputs], dtype=np.float64)

        if np.any(bids < 0):
            raise ValueError("Negative bids not allowed.")

        max_bid = bids.max()
        winners = np.flatnonzero(bids == max_bid)
        winner = int(self.rng.choice(winners))

        # Second-price rule: winner pays the highest non-self bid
        others = np.delete(bids, winner)
        price = float(others.max()) if others.size > 0 else float(max_bid)

        payoffs = [0.0] * len(bids)
        payoffs[winner] = float(valuations[winner] - price)

        return Outcome(
            allocation={"winner": winner, "price": price},
            payoffs=tuple(payoffs),
            metadata={
                "bids": bids.tolist(),
                "valuations": valuations.tolist(),
                "highest_bid": float(max_bid),
                "second_price": price,
            },
        )
