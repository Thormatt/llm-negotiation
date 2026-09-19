"""
Protocol abstraction. The "middleman" in the architectural pattern.

A Protocol takes structured numeric inputs from each participant and produces
a deterministic outcome. By construction the LLMs never communicate directly —
they each independently translate human intent to structured input, and the
protocol does the rest.

Concrete protocols live in `protocols/` (Vickrey, Nash bargaining, matching).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Outcome:
    """Deterministic result of a protocol run.

    All protocols return an Outcome. The `payoffs` array gives each
    participant's utility under the chosen allocation; the `allocation`
    field carries protocol-specific outcome data (winning bidder index,
    final proposal vector, matched pairs, etc.).
    """
    allocation: Any
    payoffs: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


class Protocol(ABC):
    """ABC for middleman protocols. Subclass and implement `run`."""

    name: str = "Protocol"

    @abstractmethod
    def run(self, inputs: list) -> Outcome:
        """Compute the outcome from structured inputs.

        Args:
            inputs: list of participant-specific structured inputs. Shape
                    and meaning are protocol-specific (bids, weight vectors,
                    preference rankings, etc.).

        Returns:
            Outcome with the chosen allocation and per-participant payoffs.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
