# Phase B scaffolding: middleman protocols

Concrete protocol implementations for the Phase B "middleman hypothesis" experiments:

> If LLMs cannot be steered through language and the strategic work is done by deterministic sub-agents, the LLM-to-LLM language channel is vestigial. Replace it with a designed protocol (the middleman) that takes structured inputs from each side and computes the outcome.

Each protocol implements `framework.Protocol`: a single `run(inputs) -> Outcome` call that takes structured numeric inputs from each participant and returns a deterministic allocation.

## Protocols

| Protocol | Domain | Mechanism | Strategy-proof? |
|---|---|---|---|
| `Vickrey` | sealed-bid auction (n bidders) | second-price, highest bid wins, pays second-highest | yes (truthful is dominant) |
| `NashBargaining2P` | 2-party multi-issue contract | maximises log(U_A) + log(U_B); corner-solution on a box | yes for the additive-utility model |
| `CooperativeBlend2P` | 2-party multi-issue contract | proportional split per issue, weights ratio | no (smoother but Pareto-suboptimal); included for direct comparison to the paper's `compute_target_proposal` formula |

## Running

```bash
python3 protocols/test_protocols.py
```

Validates each protocol against its theoretical properties (Vickrey truthfulness, Nash Pareto-efficiency, cooperative-blend symmetry).

## Why two negotiation protocols?

`NashBargaining2P` is the textbook bargaining solution. `CooperativeBlend2P` is what the negotiation paper's `compute_target_proposal` formula computes when aggression=1. Including both means the Phase B middleman experiment can compare:

- LLM-LLM language channel (paper's Condition A)
- LLM with sub-agent providing `compute_target_proposal` directives (paper's Condition B, equivalent to using `CooperativeBlend2P` as the strategy module)
- Sub-agents communicating only through `NashBargaining2P` (the user's middleman hypothesis — no LLM-LLM channel, exact Pareto-optimal outcome)

The cooperative-blend protocol is included so the comparison covers the paper's specific blend choice in addition to the textbook optimum.

## Status

- [x] Protocol ABC + Outcome dataclass
- [x] Vickrey (sealed-bid second-price)
- [x] NashBargaining2P (corner-solution Nash)
- [x] CooperativeBlend2P (paper's compute_target equivalent)
- [x] Smoke tests for all three
- [ ] GaleShapley (matching markets, planned for Phase C)
- [ ] Modal experiment driver: B1/B2/B3 comparison (after Phase A lands)
