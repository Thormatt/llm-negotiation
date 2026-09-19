"""
SubAgent abstraction.

A SubAgent takes a participant's private state (their utility weights,
budget, valuation, whatever the domain calls for) and any public observation
(what the protocol has revealed so far) and emits a structured numeric
directive — a bid, a target proposal, a preference ordering.

Concrete sub-agents live in domain directories:
    belief_gru/heuristic_om.py — bias-scalar opponent model for negotiation
    auctions/equilibrium_bidder.py — Bayesian-Nash bidder for FPSB

Sub-agents do not call LLMs and do not produce natural language. They are
the deterministic, inspectable, tunable strategy module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SubAgent(ABC):
    """ABC for strategy sub-agents."""

    name: str = "SubAgent"

    @abstractmethod
    def directive(self, private_state: Any, observation: Any = None) -> Any:
        """Compute a numeric directive given private state and any observation.

        Args:
            private_state: the agent's private utility weights / valuation /
                budget / whatever the domain calls for.
            observation: any public information from the protocol or history
                (default None — single-shot settings have no observation).

        Returns:
            A numeric directive. Type is domain-specific (a single bid,
            a target proposal vector, a ranking).
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
