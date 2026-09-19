"""
LLM bidder with four conditions matching the negotiation paper's pattern:

  A0  LLM internal strategy            (no advice; "you have valuation v, choose a bid")
  A1  LLM with English advice          (told the equilibrium strategy in prose)
  A2  LLM with numeric directive       (given the exact optimal bid as a number)
  A3  Sub-agent only, no LLM           (just the EquilibriumBidder)

The hypothesis from the negotiation paper, applied here:
  - A0  ≈ noisy, high-variance bidding around equilibrium
  - A1  ≈ A0 (English advice doesn't override the LLM's internal pricing instinct)
  - A2  ≪ A0/A1 in variance (variance collapse from numeric directive)
  - A3  is the analytical optimum

If the pattern holds, A2 should approach A3's expected payoff with a much
tighter bid distribution than A0 or A1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Optional

from auction_env import equilibrium_bid


@dataclass
class BidDecision:
    bid: float
    reasoning: str = ""
    raw_response: str = ""
    error: Optional[str] = None
    condition: str = ""


@dataclass
class LLMBidder:
    """LLM bidder. `condition` ∈ {"A0", "A1", "A2"}. For A3 use EquilibriumBidder."""

    client: object             # OpenAI-compatible client (OpenRouter)
    model_name: str
    condition: str
    n: int = 2                 # bidder count (used to compute equilibrium bid for A2)
    temperature: float = 0.3
    max_tokens: int = 800

    def __post_init__(self):
        if self.condition not in ("A0", "A1", "A2"):
            raise ValueError(f"condition must be A0, A1, or A2 (got {self.condition})")

    def bid(self, valuation: float) -> BidDecision:
        prompt = self._build_prompt(valuation)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            # On API error fall back to equilibrium so a single failure doesn't poison the run.
            return BidDecision(
                bid=equilibrium_bid(valuation, self.n),
                reasoning=f"[API error fallback to equilibrium: {e}]",
                raw_response="",
                error=str(e),
                condition=self.condition,
            )

        bid_amount, reasoning = _parse_bid(raw)
        if bid_amount is None:
            # Malformed output — fallback so the experiment continues.
            return BidDecision(
                bid=equilibrium_bid(valuation, self.n),
                reasoning=f"[Parse failure fallback: {raw[:200]}]",
                raw_response=raw,
                error="parse_failure",
                condition=self.condition,
            )

        # Defensive clamp: nonsense bids (negative, > valuation + 1) get pulled in.
        bid_amount = max(0.0, min(float(bid_amount), 1.0))

        return BidDecision(
            bid=bid_amount,
            reasoning=reasoning,
            raw_response=raw,
            condition=self.condition,
        )

    def _system_prompt(self) -> str:
        return (
            "You are bidding in a sealed-bid first-price auction. "
            "There are exactly two bidders. The other bidder's valuation is "
            "private and drawn from Uniform[0, 1] independently of yours. "
            "Highest bid wins; the winner pays their own bid and receives the "
            "item; the loser pays nothing. Your goal is to maximize your "
            "expected profit, which is your_valuation - your_bid if you win "
            "and 0 if you lose.\n\n"
            "Respond with JSON only, in this exact form:\n"
            '  {"bid": 0.34, "reasoning": "..."}\n'
            "where bid is a number between 0 and 1."
        )

    def _build_prompt(self, valuation: float) -> str:
        base = (
            f"Your private valuation for the item is {valuation:.4f}.\n"
            f"Submit your sealed bid."
        )

        if self.condition == "A0":
            return base + "\n\nRespond with JSON only."

        if self.condition == "A1":
            advice = (
                "\n\nADVICE: In a sealed-bid first-price auction with two "
                "bidders and uniform private valuations, the Bayesian-Nash "
                "equilibrium strategy is to bid exactly half of your "
                "valuation. Bidding more than that gives away surplus; "
                "bidding less reduces your win probability. The equilibrium "
                "is what a strategically optimal bidder does."
            )
            return base + advice + "\n\nRespond with JSON only."

        # A2: numeric directive
        eq = equilibrium_bid(valuation, self.n)
        directive = (
            f"\n\n=== COMPUTED OPTIMAL BID ===\n"
            f"Submit this exact bid: {eq:.4f}\n"
            f"This bid is calculated to maximize your expected profit "
            f"given the equilibrium strategy.\n"
            f"INSTRUCTIONS:\n"
            f"- Submit this bid exactly. Do not deviate.\n"
            f"- This is the strategically optimal value for your valuation.\n"
            f"=== END DIRECTIVE ==="
        )
        return base + directive + "\n\nRespond with JSON only."


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_bid(raw: str) -> tuple[Optional[float], str]:
    """Extract bid and reasoning from an LLM JSON response. Returns (bid, reasoning)."""
    text = raw
    if "```" in text:
        chunks = text.split("```")
        if len(chunks) >= 2:
            text = chunks[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    if not text.startswith("{"):
        m = _JSON_RE.search(raw)
        if m:
            text = m.group(0)

    try:
        decision = json.loads(text, strict=False)
        bid_value = decision.get("bid")
        reasoning = decision.get("reasoning", "")
        if isinstance(bid_value, (int, float)):
            return float(bid_value), str(reasoning)
        if isinstance(bid_value, str):
            try:
                return float(bid_value.strip()), str(reasoning)
            except ValueError:
                pass
    except Exception:
        pass
    return None, raw[:300]
