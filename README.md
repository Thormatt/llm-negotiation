# Computed Targets and the Middleman Protocol

**Mathematical Guardrails and Channel Removal for LLM Negotiators in Multi-Issue Bargaining**

Thor Matthiasson · Independent Researcher · New Haven, Connecticut, USA

[Paper (PDF, 32 pages)](paper.pdf) · arXiv preprint (link added once live) · [Data packet](data-packet/) · [Cite](#citation)

---

## In short

- **Words make LLM negotiators worse.** Telling a model about its opponent's preferences in plain text consistently lowers what it wins. Giving it a computed numeric target instead recovers the loss and cuts outcome variance 2.3–9.9× (Claude 3.5 Haiku, N = 200 per arm, Holm-corrected permutation tests).
- **Eight models, five labs.** Claude Haiku and Opus 4.6, GPT-5, Gemini 3 Pro and Flash, Grok 3 and 3 Mini, DeepSeek R1 sort into three tiers by how much they respond.
- **Remove the conversation.** When each side submits numbers to a neutral protocol instead of talking, Pareto efficiency reaches 91–96% against a 79% ceiling for the best language-channel setup, and the frontier-model advantage disappears.

## Abstract

Aligned LLMs negotiate cooperatively by default. We test whether providing them with explicit opponent-preference information improves their bargaining outcomes, and find the opposite: text-based strategic advice consistently reduces buyer surplus, an effect we read as alignment-induced cooperation bias though we do not establish it causally. Replacing the advice with a numeric target proposal computed by a small external module (an 11K-parameter NegotiationGRU) recovers and exceeds the unassisted baseline.

Scaled experiments (N = 200, randomized opener, Claude 3.5 Haiku) decompose the effect into anchoring defense (+7.1pp, opener-dependent), a residual GRU-conditioned share edge (+3.0pp, opener-independent), and variance collapse (2.3–9.9×, format-driven); eight of ten pairwise Holm-Bonferroni-corrected comparisons reach p = 0.001. An Oracle ablation shows the cooperative-blend formula mechanically converges to Nash bargaining under accurate weights, yielding the highest Pareto efficiency (79%) but the lowest buyer share (49.5%) — a property of the chosen blend, not a fundamental law. Replication across eight models from five labs (Haiku, Opus 4.6, GPT-5, Gemini 3 Pro/Flash, Grok 3/Mini, DeepSeek R1) yields a three-tier taxonomy — scaffold-responsive (+3–7pp, 2–3× collapse), collapse-only (variance reduction without surplus shift), and no detectable effect at N = 50 for Opus 4.6 and Gemini Pro (power-limited) — with the consistency gap predicted by baseline negotiation stochasticity, not model size. Cross-model: a cheap Haiku-with-scaffold buyer keeps a 5.2pp edge against Grok 3 but loses 0.9–2.1pp to frontier sellers, and the variance-collapse property disappears (±11.3–20.3%).

We then show that the 11,078-parameter GRU reduces, behaviorally, to a 15-line analytical estimator with one tunable scalar: its weight predictions on LLM trajectories are at chance, variance collapse derives entirely from the directive format, and the residual share gain reduces to a single bias parameter. The architecture is best read as a structured numeric sub-agent with a tunable aggression knob, not a learned opponent model in the Theory-of-Mind sense. A stronger architectural intervention removes the language channel entirely: each side submits structured numeric inputs to a neutral middleman protocol. In offline replay, the N = 200 weight pairs routed through Nash bargaining and the cooperative-blend formula reach 96% and 91% Pareto efficiency respectively, breaking the 79% language-channel ceiling of the Oracle condition, and replaying the four cross-model match-ups collapses the 7.3pp buyer-share spread to ≤0.001pp; a live N = 50 Haiku-vs-Haiku LLMIO test confirms the cooperative-blend 50/50 split is structural and robust to translation noise (per-game drift 0.001 vs. Nash's 0.181), with cross-model LLMIO as future work. Appendix A reports a 6.5K-parameter neural policy that reaches 100% deal rate at the lowest variance observed; because it rests on a single RWR temperature and lacks the GRU-mask control, we treat it as hypothesis-generating rather than load-bearing.

## Key Results

### Language channel (Claude 3.5 Haiku, N = 200, randomized opener)

| Condition | Buyer share | Std dev | Pareto eff. |
|---|---|---|---|
| A: No computed target | 52.2% | +/-8.9% | 72% |
| B: Computed target (GRU) | 55.2% | +/-3.8% | 76% |
| B: Computed target (uniform) | 49.4% | +/-1.9% | 75% |
| B: Computed target (oracle) | 49.5% | +/-0.9% | 79% |

*Full results in Table 10 of the paper.*

### Middleman protocols (same N = 200 weight pairs, channel removed)

| Architecture | Buyer share | Std dev | Pareto eff. |
|---|---|---|---|
| LLM with Oracle (language ceiling) | 49.5% | +/-0.9% | 79% |
| **Middleman: NashBargaining2P** | 51.4% | +/-18.7% | **96%** |
| **Middleman: CooperativeBlend2P** | 50.0% | **+/-0.2%** | 91% |

*Full results in Table 15 of the paper.*

### Cross-Model: language channel vs. middleman

| Match-up | Language share (sigma) | Middleman: Nash | Middleman: Coop |
|---|---|---|---|
| Haiku vs Grok 3 | 55.2% (+/-11.3%) | 50.6% | 50.0% |
| Haiku vs Opus 4.6 | 49.1% (+/-16.1%) | 50.6% | 50.0% |
| Haiku vs GPT-5 | 48.1% (+/-19.5%) | 50.6% | 50.0% |
| Haiku vs Gemini 3 Pro | 47.9% (+/-20.3%) | 50.6% | 50.0% |

*The 7.3pp cross-opponent spread under the language channel collapses to <=0.001pp under the middleman. Full results in Table 16 of the paper.*

### Cross-Model Variance Collapse (within-model, language channel)

| Model | Baseline sigma | With targets | Collapse |
|---|---|---|---|
| Grok 3 | +/-11.4% | +/-4.1% | 2.78x |
| Claude 3.5 Haiku | +/-8.9% | +/-3.8% | 2.34x |
| Grok 3 Mini | +/-9.5% | +/-4.7% | 2.02x |
| GPT-5 | +/-7.0% | +/-5.0% | 1.40x |
| Claude Opus 4.6 | +/-4.5% | +/-4.9% | 1.09x |

*Full eight-model panel in Table 9 of the paper.*

## Repository Structure

```
llm-negotiation/
  paper.pdf                    # Full paper (32 pages, 3 figures, 18 tables)
  paper.tex                    # LaTeX source (inline bibliography, builds standalone)
  CITATION.cff                 # Citation metadata (GitHub "Cite this repository")
  AUDIT_REPORT.md              # Record of the pre-submission audit pass
  src/
    negotiation.py             # LLM negotiation engine (OpenRouter API)
    train_negotiation_gru.py   # NegotiationGRU model (11,078 params)
    model.py                   # Model utilities
    policy_net.py              # NegotiationPolicyNet (6,533 params)
    train_policy_net.py        # Policy training (BC + RWR)
    policy_data.py             # Training data extraction from game logs
    policy_eval.py             # Live A/B evaluation
    heuristic_om.py            # Analytical estimator (15-line replacement)
    calibrate_heuristic.py     # Bias scalar calibration
    counterfactual_replay.py   # Offline diagnostic replay
    stats_tests.py             # Statistical testing utilities
    run_alignment_experiment.py # Text-advice backfire experiments
    modal_negotiation.py       # Scaled experiments on Modal
    modal_crossmodel_baseline.py # Cross-model experiments
    arena.py                   # Tournament framework
    agent_loop.py              # Agent control loop
    protocols/                 # Middleman protocols (Phase B)
      nash_bargaining.py       # NashBargaining2P
      vickrey.py               # Vickrey auction protocol
      llm_io_negotiation.py    # LLMIO translation layer
      phase_b_offline.py       # Offline replay of N=200 weight pairs
      phase_b_crossmodel.py    # Cross-model replay through protocols
      PHASE_B_RESULTS.md       # Phase B findings write-up
    auctions/                  # Sealed-bid auction replication (Phase A)
      auction_env.py           # First-price sealed-bid environment
      equilibrium_bidder.py    # Equilibrium / naive / shading bidders
      llm_bidder.py            # LLM bidder conditions A0-A2
    framework/                 # Shared LLM-as-mouth scaffolding
  checkpoints/
    negotiation_gru.pt         # Trained NegotiationGRU
    negotiation_gru_finetuned.pt
    policy_net_bc.pt           # Behavioral cloning policy
    policy_net_rwr.pt          # Reward-weighted regression policy
  data-packet/                 # Companion data packet (v1.0)
    MANIFEST.csv               # sha256 + table mapping, one row per file
    seeds.json                 # Seed registry (base seed, opener seeds)
    checkpoints/               # The two load-bearing checkpoints
    prompts/                   # Verbatim system-prompt templates
    results/                   # Per-game JSON logs, grouped by table
    scripts/verify_packet.py   # Confirm sha256 + JSON-loadability
    scripts/compute_tables.py  # Recompute headline tables from raw logs
  requirements.txt
  LICENSE
```

## Reproducing the tables

Every headline table recomputes from the raw per-game logs in `data-packet/`:

```bash
cd data-packet
python3 scripts/verify_packet.py    # sha256 integrity: expect "All checks passed."
python3 scripts/compute_tables.py   # recompute headline tables from raw JSON
```

`verify_packet.py` checks all 49 files against `MANIFEST.csv`. No API keys or
network access are needed for either script -- they read the logged games only.

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run a negotiation

```bash
export OPENROUTER_API_KEY=your-key-here
python src/negotiation.py
```

All LLM calls go through [OpenRouter](https://openrouter.ai/). You'll need an API key with access to your desired models.

### Train the NegotiationGRU from scratch

```bash
python src/train_negotiation_gru.py
```

Trains on 20,000 heuristic-vs-heuristic matches. No LLM calls required.

### Evaluate the neural policy

```bash
python src/policy_eval.py
```

Runs live A/B test: heuristic formula vs neural policy, 50 games each, Claude 3.5 Haiku.

## Models Tested

All models accessed via OpenRouter:

| Model | OpenRouter ID |
|---|---|
| Claude 3.5 Haiku | `anthropic/claude-3.5-haiku` |
| Claude Opus 4.6 | `anthropic/claude-opus-4-6` |
| GPT-5 | `openai/gpt-5` |
| Gemini 3 Pro | `google/gemini-3-pro-preview` |
| Gemini 3 Flash | `google/gemini-3-flash-preview` |
| Grok 3 | `x-ai/grok-3` |
| Grok 3 Mini | `x-ai/grok-3-mini` |
| DeepSeek R1 | `deepseek/deepseek-r1-0528` |

## Citation

```bibtex
@misc{matthiasson2026computed,
  title  = {Computed Targets and the Middleman Protocol:
            Mathematical Guardrails and Channel Removal
            for {LLM} Negotiators in Multi-Issue Bargaining},
  author = {Matthiasson, Thor},
  year   = {2026},
  note   = {Preprint. Code and data:
            \url{https://github.com/Thormatt/llm-negotiation}}
}
```

The arXiv identifier will be added here once the preprint is live. GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).

## License

MIT
