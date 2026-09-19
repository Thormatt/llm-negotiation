"""
Concrete middleman protocols.

Each protocol takes structured numeric inputs from each participant and
returns a deterministic Outcome. No language, no LLMs, no negotiation —
just a designed mechanism that maps inputs to allocations.

Protocols here:
    Vickrey                   — sealed-bid second-price auction
    NashBargaining2P          — Nash-bargaining solution for 2-party multi-issue
    GaleShapley (planned)     — two-sided stable matching
"""

from protocols.vickrey import Vickrey
from protocols.nash_bargaining import NashBargaining2P, CooperativeBlend2P

__all__ = ["CooperativeBlend2P", "NashBargaining2P", "Vickrey"]
