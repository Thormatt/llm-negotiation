"""
Analytical bidders for FPSB. No learned parameters.

EquilibriumBidder    — exact Bayesian-Nash bid: b(v) = v * (n-1) / n
NaiveBidder          — bids own valuation b(v) = v  (truthful, leaves no surplus)
ShadingBidder        — bids b(v) = v * shade  (parametric ablation knob)

These are the analogs of the negotiation paper's HeuristicOM. The
EquilibriumBidder is what the paper would call the "oracle" bidder for FPSB.
"""

from __future__ import annotations

from dataclasses import dataclass

from auction_env import equilibrium_bid


@dataclass
class EquilibriumBidder:
    """Risk-neutral Bayesian-Nash bidder. The 'oracle' for FPSB-IPV-U[0,1]."""
    n: int = 2

    def bid(self, valuation: float) -> float:
        return equilibrium_bid(valuation, self.n)

    def __repr__(self) -> str:
        return f"EquilibriumBidder(n={self.n})"


@dataclass
class NaiveBidder:
    """Truth-teller: bids own valuation. Wins more often, makes zero profit."""

    def bid(self, valuation: float) -> float:
        return valuation

    def __repr__(self) -> str:
        return "NaiveBidder()"


@dataclass
class ShadingBidder:
    """Parametric bidder: b(v) = v * shade.

    shade = 1.0  ↔  NaiveBidder (truthful)
    shade = (n-1)/n  ↔  EquilibriumBidder
    shade = 0.0  ↔  always bids zero (loses always, but if it ever wins, 100% profit)

    This is the analogue of the negotiation paper's bias scalar.
    """
    shade: float

    def bid(self, valuation: float) -> float:
        return max(0.0, valuation * self.shade)

    def __repr__(self) -> str:
        return f"ShadingBidder(shade={self.shade:.3f})"
