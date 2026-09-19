# Phase B Results: the middleman hypothesis is empirically supported

**Status:** Offline analytical results landed 2026-04-28. Live LLMIO experiment in scaffolding.

**Hypothesis (from `RESEARCH_PLAN.md` Part 2):** For repeatable structured commerce, the LLM-to-LLM language channel is empirically lossy. Replace it with a designed protocol (the middleman) that takes structured numeric inputs from each side; the LLMs remain on each side as I/O layers but never directly converse.

## Result 1 — Middleman dominates LLM-channel on every metric that matters

Replayed N=200 Dirichlet($\alpha\!=\!1$) weight pairs from the paper's randomized-opener Condition B run through three architectures and compared to logged paper baselines.

| Architecture | Buyer share | Share σ | Joint surplus | Pareto eff. |
|---|---|---|---|---|
| LLM-LLM, no scaffold (paper A) | 52.2% | ±8.9% | 1.021 | 72% |
| LLM with GRU sub-agent (paper B) | 55.2% | ±3.8% | 1.083 | 76% |
| LLM with `bias = 0.55` sub-agent | 56.0% | ±3.8% | — | — |
| LLM with Oracle (paper B-Oracle) | 49.5% | ±0.9% | 1.126 | 79% |
| **Middleman: NashBargaining2P** | 51.4% | ±18.7% | **1.354** | **96%** |
| **Middleman: CooperativeBlend2P** | 50.0% | **±0.2%** | **1.269** | **91%** |

Three findings, in order of importance:

1. **The LLM-LLM channel is lossy at the joint-surplus level.** Even the *Oracle* condition — the LLM negotiating with full knowledge of opponent weights — captures only 79% of available joint surplus. The Nash protocol captures 96%. The 17pp gap is the LLM tax: the cost of routing strategy through natural language between two models when the result could be computed directly.

2. **CooperativeBlend2P has lower outcome variance than anything else in the work.** ±0.2% is roughly an order of magnitude tighter than the GRU's ±3.8%, four times tighter than Oracle's ±0.9%, and forty times tighter than unassisted ±8.9%. When the LLM is removed entirely, variance approaches zero.

3. **Nash bargaining trades variance for efficiency; cooperative blend trades efficiency for variance.** Nash hits 96% Pareto but ±18.7% variance because corner solutions amplify weight differences. CooperativeBlend2P smooths corners and gets ±0.2% variance at 91% Pareto. For deployment, the cooperative-blend protocol is the more useful choice: nearly identical efficiency to Nash, but predictability that approaches deterministic.

## Result 2 — The frontier-model advantage disappears under the middleman

Replayed the **exact** weight pairs from the four logged cross-model runs (`results/n50_cross_haiku_vs_*.json`) through the protocols. Same weight pairs, different channel.

| Match-up | LLM-LLM share (paper Table 9) | Middleman: Nash | Middleman: Coop |
|---|---|---|---|
| Haiku vs Grok 3 | 55.2% (±11.3%) | 50.6% | 50.0% |
| Haiku vs Opus 4.6 | 49.1% (±16.1%) | 50.6% | 50.0% |
| Haiku vs GPT-5 | 48.1% (±19.5%) | 50.6% | 50.0% |
| Haiku vs Gemini Pro | 47.9% (±20.3%) | 50.6% | 50.0% |

Under the language channel, frontier sellers extracted 2–5pp from the cheap-with-scaffold buyer. Under the middleman, every matchup returns ~50% buyer share. The 7.3pp range across opponents (47.9–55.2) collapses to <0.001pp.

**Interpretation:** the cross-model exploitation Project Deal documented at deployment scale (Opus users systematically extracting from Haiku users, fairness ratings 4.05 vs 4.06) is a property of the *language channel*, not of any inherent capability gap. The channel lets the more capable model leverage anchoring, social pressure, conversational momentum, and persuasion. Replace the channel with a deterministic protocol that takes structured inputs and the asymmetry stops existing.

This is a strong claim for the framework paper: **structured-protocol architectures are robust to participant model-quality asymmetry; language-channel architectures are not.**

## Result 3 — Variance trade-off is real and protocol-dependent

A subtlety worth recording before we forget. NashBargaining2P is Pareto-optimal (96%) but high-variance (±18.7%) because its corner-solution outputs are extreme: small weight differences flip a variable from 0.95 to 0.05. CooperativeBlend2P sacrifices a few percentage points of Pareto efficiency for a near-deterministic outcome.

For deployment design, this is the trade-off:

| Property | NashBargaining2P | CooperativeBlend2P |
|---|---|---|
| Pareto efficiency | 96% (best in work) | 91% |
| Outcome variance | ±18.7% (high) | ±0.2% (lowest in work) |
| Corner solutions | yes (extreme proposals) | no (smooth fractional) |
| Theoretical grounding | Nash bargaining axioms | paper's `compute_target_proposal` |
| Recommended for | research / efficient design | deployment / predictability |

Both beat the LLM-LLM channel on Pareto. The right protocol depends on whether the deployment values absolute efficiency (Nash) or predictability (Coop).

## Result 4 — Live LLMIO confirms CooperativeBlend2P is robust to translation noise

We added an LLMIO layer (Haiku-Haiku, qualitative-only preference descriptions, no numeric weights leaked) and ran $N\!=\!50$ via Modal. Each side's LLM had to translate "MOST important / very important / moderately important / less important / least important" labels into a normalised 5-dim weight vector — a real interpretation task with non-trivial translation error.

| Protocol | Share (LLMIO) | Share (true weights) | Per-game drift | Translation L1 (per side) |
|---|---|---|---|---|
| **CooperativeBlend2P** | **50.0% ± 0.000** | 50.0% ± 0.002 | **0.001** | ≈ 0.29–0.33 |
| **NashBargaining2P** | 55.4% ± 25.5% | 53.2% ± 21.1% | 0.181 | ≈ 0.29–0.33 |

Same translation noise on both sides ($L_1 \approx 0.29$), wildly different downstream behaviour. Two observations:

**a. CoopBlend's 50/50 share is structural, not empirical.** For 2-party opposite-direction additive utility, the proportional-split formula gives per-issue allocation $w_i^2 / (w_a + w_b)$, which sums to identical buyer and seller utilities regardless of weight asymmetry. The 50/50 outcome is a mathematical property, not a sample-size artefact. Translation noise of *any* magnitude leaves it unchanged.

**b. Nash is fragile to LLMIO error.** Drift per game of 0.181 (vs 0.001 for Coop) confirms that corner-solution protocols amplify input perturbations. Even with the same Haiku translator on both sides, individual game outcomes shift substantially based on which qualitative-level → weight mapping the LLM happened to pick.

**Deployment implication.** Among middleman protocols, CoopBlend trades 5pp Pareto efficiency for structural fairness and translation-noise robustness. Nash gives the higher efficiency ceiling but requires near-perfect intent translation to deliver it. For real-world commerce with imperfect LLMIO, **CoopBlend is the right default**.

## What's next

Two empirical follow-ups, not run yet:

1. **Cross-model LLMIO live.** Replicate Result 4 with Haiku × {Opus, GPT-5, Gemini Pro} in the LLMIO layer. The analytical Result 2 already shows the protocol is symmetric in the limit of perfect translation; the question is whether Nash's translation-noise sensitivity introduces residual model asymmetry. CoopBlend will be 50/50 by construction, so this test is informative for Nash only. ~$3–5 OpenRouter.

2. **Adversarial LLMIO.** Instruct one side's LLMIO to deliberately misrepresent weights to extract more surplus. This is the protocol-level "preference manipulation" robustness test, and is the right framing for the regulatory-fitness of the architecture. Queued for Phase C.

## Code and data

- Analytical Phase B (offline): `protocols/phase_b_offline.py`
- Cross-model replay: `protocols/phase_b_crossmodel.py`
- Protocol implementations: `protocols/{vickrey,nash_bargaining}.py`
- Smoke tests: `protocols/test_protocols.py`
- Existing logs replayed: `results/n50_cross_haiku_vs_*.json`, `results/n200_*.json` (paper data, unmodified)
