# Phase A: Auctions — testing the architectural pattern in a second domain

This directory replicates the negotiation-paper's experimental pattern in a
sealed-bid first-price auction (FPSB) setting. The goal: test whether
"LLM-as-mouth + sub-agent" beats "LLM-as-strategist" outside negotiation.

## Setting

- **Two-bidder sealed-bid first-price auction**
- Independent private valuations, $v_i \sim \text{Uniform}[0, 1]$
- Risk-neutral bidders
- Theoretical Bayesian-Nash equilibrium: $b^*(v) = v \cdot (n-1)/n = v/2$ for $n=2$
- Theoretical revenue: $\mathbb{E}[\text{rev}] = (n-1)/(n+1) = 1/3$
- Theoretical winner surplus: $\mathbb{E}[\text{surplus}] = 1/(n+1) = 1/3$

## Conditions

| Code | Name | Bidder logic |
|---|---|---|
| **A0** | LLM internal strategy | LLM is given valuation, chooses bid. No advice. |
| **A1** | LLM with English advice | A0 + system prompt explains the equilibrium strategy in prose. |
| **A2** | LLM with numeric directive | A0 + system prompt provides the exact equilibrium bid as a number. |
| **A3** | Sub-agent only | `EquilibriumBidder` directly. No LLM. |

The opponent (bidder 1) always plays equilibrium. This isolates the focal
bidder's strategy effect.

## Predicted result (from the negotiation-paper hypothesis)

| Condition | Expected payoff | Bid-to-equilibrium σ |
|---|---|---|
| A0 | ≈ 0 to 0.10 (variable) | high |
| A1 | ≈ A0 (English advice doesn't override LLM) | high |
| A2 | ≈ A3 = 0.083 | low (variance collapse) |
| A3 | 0.083 (theoretical) | 0 |

If A2 ≈ A3 with a much tighter bid distribution than A0/A1, the architectural
pattern (LLM-as-mouth + numeric-directive) generalizes from negotiation to
auctions.

## Files

- `auction_env.py` — `FirstPriceAuction` environment and equilibrium constants
- `equilibrium_bidder.py` — `EquilibriumBidder`, `NaiveBidder`, `ShadingBidder`
- `llm_bidder.py` — `LLMBidder` with conditions A0, A1, A2
- `run_offline.py` — pre-flight that runs analytical bidders only (no API spend)
- `modal_auction.py` — Modal entrypoint for live LLM A0–A3 comparison
- `analyze_auction.py` — post-hoc analysis on saved results

## Running

Pre-flight (free, validates infrastructure against analytical theory):
```bash
cd auctions
python3 run_offline.py
```

Smoke test on Modal (just A3, no API spend):
```bash
modal run auctions/modal_auction.py --experiment a3 --quick
```

Full live A/B (needs OpenRouter credits, ~$2 for N=200):
```bash
modal run auctions/modal_auction.py --experiment all --n-auctions 200
```

## Status

- [x] FPSB environment and equilibrium constants
- [x] Analytical bidders (Equilibrium / Naive / Shading)
- [x] LLMBidder with A0–A2
- [x] Modal entrypoint
- [x] Offline pre-flight passes (revenue 0.335 ≈ theoretical 1/3, allocation efficiency 100%)
- [x] Modal A3 baseline validated on 200 auctions (revenue 0.328 ± 0.116, σ(b−eq)=0, 100% efficient)
- [x] Post-hoc analysis script (`analyze_auction.py`)
- [ ] Live A/B for A0/A1/A2 (blocked on OpenRouter top-up; ~$2 for N=200 each)
- [ ] Cross-domain comparison vs negotiation result (waiting on live A/B)

## Next steps

1. Top up OpenRouter credits.
2. Quick LLM smoke (~$0.05): `modal run auctions/modal_auction.py --experiment a0,a1,a2 --quick` — 5 auctions each, verifies the LLM bidder pipeline without committing to a full run.
3. Full live A/B (~$2): `modal run auctions/modal_auction.py --experiment a0,a1,a2 --n-auctions 200`.
4. Post-hoc: `python3 auctions/analyze_auction.py results/auction_fpsb_*.json`.
5. Compare to A3 baseline (already in `results/auction_fpsb_A3_20260428_135430.json`).
