"""
LLMIO — the natural-language interface layer.

LLMIO has exactly two jobs:

    1. Translate a human's natural-language intent into structured private
       state (weight vectors, valuations, budgets, preference rankings).
    2. Translate a protocol's outcome back into natural-language explanation
       for the human.

LLMIO does not negotiate. It does not converse with another LLM. It sits at
the boundary between human and the protocol, and is the only place where
natural language exists in the system.

This is intentionally small and focused. The LLM is good at fluent language
translation; that is the job we let it do. Strategic reasoning happens in
the sub-agent; outcome computation happens in the protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Translation:
    """Result of translating intent into structured preferences."""
    structured: Any           # parsed structured preference / valuation
    raw_response: str = ""
    confidence: float = 1.0


class LLMIO(ABC):
    """ABC for LLM-mediated I/O between human and protocol."""

    @abstractmethod
    def intent_to_structured(self, intent: str) -> Translation:
        """Parse a human intent string into structured private state."""
        raise NotImplementedError

    @abstractmethod
    def outcome_to_natural_language(self, outcome: Any) -> str:
        """Render a protocol outcome as a human-readable explanation."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
