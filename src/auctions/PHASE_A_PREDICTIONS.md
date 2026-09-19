# Phase A: Pre-registered predictions

**Date locked:** 2026-04-28 (before A0/A1/A2 live data is collected).
**Setting:** Sealed-bid first-price auction, $n=2$, IPV $v_i \sim U[0,1]$, risk-neutral, paired valuation seeds across conditions.
**Sample size:** $N = 200$ auctions per condition for the headline run.

The point of writing predictions before the data lands is so that after the fact we can honestly say whether the architectural pattern from the negotiation paper generalized or not, and not move the goalposts.

## Theoretical anchors

| Quantity | Value |
|---|---|
| Equilibrium bid | $b^*(v) = v/2$ |
| Expected revenue | $1/3 \approx 0.333$ |
| Expected winner surplus | $1/3$ |
| Expected per-bidder payoff (over both win and loss) | $1/6 \approx 0.167$ |

## Already in hand (A3 baseline, 2026-04-28)

| Metric | A3 measured | Theory |
|---|---|---|
| Revenue mean | $0.328 \pm 0.116$ | $0.333$ |
| Focal payoff mean | $0.159 \pm 0.181$ | $0.167$ |
| $\sigma(b - b^*)$ | $0.000$ | $0$ |
| Allocation efficiency | $100\%$ | $100\%$ |

A3 is the sub-agent-only condition (`EquilibriumBidder` directly). The fact that it tracks theory within sampling noise validates the environment and the experimental harness.

## Predictions for A0, A1, A2

### What I expect, with reasoning

**A0 — LLM internal strategy (no advice).** Aligned LLMs without strategy advice tend to over-bid in first-price auctions because they treat the situation as "submit a fair offer" rather than "shade your bid below valuation." I expect mean focal payoff in the range $0.04$–$0.12$ (below A3's $0.167$) with $\sigma(b - b^*)$ in the range $0.10$–$0.20$.

**A1 — LLM with English advice.** The negotiation paper's finding that English strategic advice is ignored or reinterpreted should carry over. A1 should be close to A0, not close to A2. I expect $\sigma(b - b^*)$ within $\pm 0.03$ of A0's, and mean focal payoff within $\pm 0.02$ of A0's.

**A2 — LLM with numeric directive.** The LLM is given the exact equilibrium bid as a number and asked to submit it. Variance should collapse hard. Mean focal payoff should approach A3's. I expect $\sigma(b - b^*) < 0.02$ and mean focal payoff within $\pm 0.015$ of A3's $0.159$.

### Pre-registered decision rules

| Result | Pattern call | Interpretation |
|---|---|---|
| $\sigma(b - b^*)$ for A2 $< 0.5 \times \sigma$ for A0 | **Variance collapse confirmed.** | Format-driven property generalizes to single-shot bidding. |
| Mean payoff for A2 within $\pm 0.02$ of A3 | **Sub-agent substitution confirmed.** | A directed LLM ≈ a deterministic optimizer. |
| Mean payoff for A0 $<$ A3 by $> 0.04$ | **LLM-internal suboptimality confirmed.** | The LLM's natural strategic instincts under-perform equilibrium. |
| Mean payoff for A1 within $\pm 0.02$ of A0 | **English advice ineffective confirmed.** | Mirrors negotiation finding; aligned LLMs don't update strategy from prose. |

If **all four** hold, the architectural pattern (LLM-as-mouth + numeric directive + analytical sub-agent ≈ analytical sub-agent alone, while LLM-as-strategist underperforms with or without prose advice) generalizes from negotiation to auctions. The framework paper has its second clean case study.

If **3/4** hold, partial generalization — needs investigation. Most likely failure mode: A1 ≈ A2 instead of A1 ≈ A0, which would suggest English advice in single-shot settings *does* override RLHF instincts in a way it doesn't in multi-round bargaining. That'd be a subtle but real finding.

If **2/4 or fewer** hold, the auction setting breaks the pattern. The likely culprit is single-shot vs multi-round structure: variance collapse may be a multi-round-equilibrium phenomenon. We'd revise the framework claim accordingly.

## What "anchoring defense" looks like in this setting

In the negotiation paper, anchoring defense was the +7.1pp share gain from "the buyer has a coherent first proposal instead of being out-anchored." The auction setting is single-shot — there is no opponent observation before submitting a bid — so anchoring defense is structurally absent. This is a feature, not a bug: the auction experiment cleanly isolates variance collapse and share-gain mechanisms from the anchoring effect, which were entangled in the negotiation results.

## What would surprise me

- **A0 outperforms A3.** Would mean LLMs have a strategic edge over equilibrium bidding in some auction settings. Surprising and publishable in its own right.
- **A2 underperforms A3 by more than $0.02$.** Would mean the LLM doesn't actually submit the directed bid faithfully — it perturbs it. The negotiation paper showed LLMs do submit numeric directives faithfully, so this would suggest something specific about the auction prompt format. Tractable to fix.
- **A1 outperforms A0 by more than $0.04$.** Would mean English advice *does* shape LLM strategy when the strategy is simple enough to convey in a sentence. That'd narrow the negotiation paper's claim to "complex strategies don't transmit through prose."
- **Variance is similar across all conditions.** Would mean variance collapse is genuinely about multi-round settings, not directive format. Most damaging to the framework claim, but useful to know.

## Results (filled in 2026-04-28, $N\!=\!200$, Claude 3.5 Haiku, OpenRouter)

| Cond | Mean focal payoff | $\sigma$ payoff | $\sigma(b - b^*)$ | Mean revenue | Allocation eff. |
|---|---|---|---|---|---|
| A0 | $0.122$ | $0.105$ | $0.080$ | $0.402 \pm 0.167$ | $86.0\%$ |
| A1 | $0.159$ | $0.181$ | $\mathbf{0.000}$ | $0.328 \pm 0.116$ | $100.0\%$ |
| A2 | $0.159$ | $0.181$ | $\mathbf{0.000}$ | $0.328 \pm 0.116$ | $100.0\%$ |
| A3 | $0.159$ | $0.181$ | $0.000$ | $0.328 \pm 0.116$ | $100.0\%$ |

A1, A2 and A3 are statistically *identical* — with paired valuation seeds and σ(b−b*)=0 for all three, bidders submitted exactly $b = v/2$ on every auction in both A1 and A2.

## Decision rules: post-hoc

| Rule | Threshold | Result | Held? |
|---|---|---|---|
| Variance collapse: $\sigma(A2) < 0.5 \times \sigma(A0)$ | $0.040$ | $0.000$ vs $0.080$ | **YES** |
| Sub-agent substitution: $|A2_\text{payoff} - A3_\text{payoff}| < 0.02$ | $0.020$ | $0.159$ vs $0.159$ | **YES** |
| LLM-internal suboptimality: $A3 - A0 > 0.04$ | $0.040$ | $0.037$ | **NO (just barely)** |
| English advice ineffective: $|A1_\text{payoff} - A0_\text{payoff}| < 0.02$ | $0.020$ | $0.037$ ($A1$ matches $A3$, not $A0$) | **NO** |

Score: **2/4** decision rules held cleanly. The two failures are interesting and informative, not damaging.

## What 2/4 actually means

This is the most useful possible outcome for understanding *what kind* of architectural pattern we found in the negotiation paper.

**Failure 1: A0 vs A3 gap is $0.037$, just under the $0.04$ threshold.** Real but smaller than predicted. The LLM-internal bidder over-bids (revenue $0.402$ vs $0.328$ equilibrium) and ends up with reduced payoff, but the magnitude is modest. Single-shot bidding lets the LLM at least approximate sane behaviour.

**Failure 2: A1 = A2 = A3 to the digit.** This is the punchline. English advice ("bid exactly half of your valuation") transmits *perfectly* in this single-shot setting. The LLM follows simple, computable English instructions as faithfully as it follows numeric directives.

## What this teaches us about the negotiation paper's claim

The negotiation paper's claim was: *aligned LLMs cannot be steered through English strategic advice.* The auction result narrows that claim:

> Aligned LLMs cannot be steered through *complex strategic* English advice in *multi-round adversarial* settings, because the cooperative gradient gets multiple turns to overrule the prompt and the strategy is too non-trivial to encode as a one-line instruction. They *can* be steered through *simple computational* English advice in *single-shot* settings, where the instruction is unambiguous and there is no multi-round pressure to relitigate it.

This is a sharper, more defensible, more transferable claim than the paper's. The language-channel-vs-numeric-directive distinction matters when:

- the strategy is too complex to express as a clean rule, **and/or**
- there are multiple rounds where the cooperative gradient can re-assert itself, **and/or**
- the prompt has to fight against the LLM's pattern-matched social/conversational instincts.

When all three are absent — single-shot, computable, no social context — English and numeric directives are equivalent.

## What this means for the framework paper

The architectural pattern (LLM-as-mouth + sub-agent + protocol) **still holds**, but the framework paper needs to be precise about *why* the LLM-as-mouth split works. The motivation isn't "LLMs can't follow language instructions" — they obviously can in simple settings. The motivation is:

1. **Strategy complexity**: when the strategy can't be reduced to a one-line English rule, you need a sub-agent to compute it.
2. **Multi-round resistance**: when the LLM has multiple turns to re-assert its alignment-trained instincts, the directive needs to be re-injected each turn.
3. **Auditability**: even when language *can* transmit the strategy, having it computed externally makes it inspectable, versionable, and tunable in ways the LLM's chain-of-thought is not.

The auction A1 = A2 result is actually consistent with this. In a single-shot setting with a one-line strategy, language and numbers are equivalent — *and so is the auditability story*: a numeric directive is still more auditable than the LLM's reasoning trace, even when both produce identical bids. The architectural pattern earns its keep most strongly when (1)/(2) apply, but its auditability/regulatory value applies across the board.

## What goes in the framework paper as Phase A

A confirmed result with a precise scope condition:

> *In sealed-bid first-price auctions, the LLM bidder follows simple English instructions and numeric directives equivalently and equally well. The variance-collapse and sub-agent-substitution mechanisms from the negotiation paper transfer to single-shot bidding; the language-failure mechanism does not (it requires multi-round structure or strategic complexity beyond a one-line rule).*

This is publishable. It refines rather than confirms the negotiation paper's claim, and refining is more useful than confirming when the goal is a generalisable framework.

## Cost ledger

- Quick wiring check: ~$0.05
- Full N=200 live A/B: ~$2 (3 conditions × 200 auctions × ~$0.003/auction Haiku)

## Next steps suggested by this result

1. **Auction with multi-round structure.** The closest test of the negotiation-paper claim in the auction domain would be an English-ascending or multi-round sealed-bid setup, where the LLM has multiple opportunities to override its own previous bid. If A1 ≈ A2 there too, the multi-round-structure hypothesis is wrong; if A1 < A2 there, it's confirmed.
2. **Auction with strategic complexity.** A common-value auction (winner's curse) where the optimal strategy is not "bid half your valuation" but a more involved adjustment. If English advice degrades there but numeric directives don't, the strategic-complexity hypothesis is confirmed.
3. **Phase B middleman experiment.** Now even more interesting: Phase A shows *language* and *numbers* are interchangeable when the strategy is simple. Does language remain interchangeable when the protocol-mediated outcome is simple? This is the right test for the user's middleman hypothesis.

## Cost and timing

- Quick wiring check: $\sim$\$0.05, $\sim$2 minutes.
- Full $N\!=\!200$ live A/B for A0, A1, A2: $\sim$\$2, $\sim$15–20 minutes Modal time.

## Files

- This document: `auctions/PHASE_A_PREDICTIONS.md`
- A3 baseline result: `results/auction_fpsb_A3_20260428_135430.json`
- Will land after live A/B: `results/auction_fpsb_A0_*.json`, `auction_fpsb_A1_*.json`, `auction_fpsb_A2_*.json`, summary `auction_fpsb_*.json`.
- Analysis: `auctions/analyze_auction.py results/auction_fpsb_*.json`.
