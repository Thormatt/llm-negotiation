"""
Framework abstractions for the LLM-as-mouth + sub-agent + middleman pattern.

The pattern, applied to any adversarial agentic system:

    Human-A intent  →  LLM-IO_A  →  structured-preferences_A
                                                            \\
                                                             →  Sub-agent_A  →  numeric-directive_A
                                                                                                  \\
                                                                                                   →  Protocol  →  outcome
                                                                                                  /
                                                             →  Sub-agent_B  →  numeric-directive_B
                                                            /
    Human-B intent  →  LLM-IO_B  →  structured-preferences_B
                                                                                                                       \\
                                                                              outcome  →  LLM-IO  →  natural-language explanation back to humans

The LLMs never directly converse. They translate intent in and explanations out.
The sub-agents compute strategy from structured private information.
The protocol takes structured inputs and produces a deterministic outcome.

This module exports three abstract interfaces:
    LLMIO      — natural-language ↔ structured-preferences translator
    SubAgent   — structured-preferences → numeric-directive
    Protocol   — list[numeric-directive] → outcome

These are the contracts. Concrete implementations live in `protocols/` and
under domain-specific directories like `auctions/` and `belief_gru/`.
"""

from framework.protocol import Protocol, Outcome
from framework.sub_agent import SubAgent
from framework.llm_io import LLMIO

__all__ = ["LLMIO", "Outcome", "Protocol", "SubAgent"]
