# Computed Targets as Mathematical Guardrails for LLM Negotiators

**Anchoring Defense, the Oracle Paradox, and Variance Collapse in Multi-Issue Bargaining**

Thor Matthiasson | Independent Researcher | Reykjavik, Iceland

[Paper (PDF)](paper.pdf)

---

## Abstract

We study whether external strategic models can improve LLM negotiation outcomes in multi-issue bilateral bargaining. A NegotiationGRU (11K parameters) trained on heuristic-agent data computes concrete numeric target proposals injected into the LLM's prompt as hard directives. Our central finding is that *text-based strategic advice backfires* -- LLMs interpret opponent-preference information as a cooperation signal due to RLHF alignment -- while computed numeric targets produce three distinct, independently operating mechanisms.

Scaled experiments (N = 200, randomized opener, Claude 3.5 Haiku) decompose the computed-target effect into: (1) **anchoring defense** (+7.1pp), neutralizing the seller's first-mover advantage; (2) a **residual opponent-modelling edge** (+3.0pp), persisting when the anchoring confound is removed; and (3) **variance collapse** (2.3--9.9x), reducing outcome standard deviation from +/-8.9% to +/-0.9--3.8% depending on opponent-model accuracy. Full ablations confirm an **Oracle Paradox**: perfect opponent knowledge yields the highest Pareto efficiency (79%) and tightest variance (+/-0.9%) but the lowest buyer share (49.5%), mechanically converging to the Nash bargaining solution.

Replication across **eight models from five labs** -- Claude Haiku, Claude Opus 4.6, GPT-5, Gemini 3 Pro, Gemini 3 Flash, Grok 3, Grok 3 Mini, and DeepSeek R1 -- reveals a **three-tier taxonomy**: (i) *OM-effective* models (Grok 3, Haiku, Grok 3 Mini) exhibit 2.0--2.8x variance collapse and +3--7pp buyer advantage; (ii) *collapse-only* models (GPT-5, Gemini Flash, DeepSeek R1) show 1.4--1.6x variance reduction without surplus shift; and (iii) *OM-irrelevant* models (Opus 4.6, Gemini 3 Pro) already negotiate with low baseline variance (+/-4.5--8.0%), leaving nothing to compress. The key predictor of computed-target effectiveness is baseline negotiation stochasticity, not model capability.

A **neural policy** (6,533-parameter MLP) trained via behavioral cloning and reward-weighted regression on 660 live LLM games successfully crosses the sim-to-real gap with 100% deal rate. Reward-weighted regression converges to the lowest-variance policy observed (+/-2.6%), suggesting that **offline RL in stochastic multi-agent environments naturally converges toward variance minimization rather than reward maximization**.

## Key Results

| Condition | Buyer share | Std dev | Pareto eff. |
|---|---|---|---|
| A: No computed target | 52.2% | +/-8.9% | 72% |
| B: Computed target (GRU) | 55.2% | +/-3.8% | 76% |
| B: Computed target (uniform) | 49.4% | +/-1.9% | 75% |
| B: Computed target (oracle) | 49.5% | +/-0.9% | 79% |

*N = 200, randomized opener, Claude 3.5 Haiku. Full results in Table 10 of the paper.*

### Cross-Model Variance Collapse

| Model | Baseline sigma | With targets | Collapse |
|---|---|---|---|
| Grok 3 | +/-11.4% | +/-4.1% | 2.78x |
| Claude 3.5 Haiku | +/-8.9% | +/-3.8% | 2.34x |
| Grok 3 Mini | +/-9.5% | +/-4.7% | 2.02x |
| GPT-5 | +/-7.0% | +/-5.0% | 1.40x |
| Claude Opus 4.6 | +/-4.5% | +/-4.9% | 1.09x |

## Repository Structure

```
llm-negotiation/
  paper.pdf                    # Full paper (24 pages)
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
  checkpoints/
    negotiation_gru.pt         # Trained NegotiationGRU
    negotiation_gru_finetuned.pt
    policy_net_bc.pt           # Behavioral cloning policy
    policy_net_rwr.pt          # Reward-weighted regression policy
  requirements.txt
  LICENSE
```

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
@article{matthiasson2025computed,
  title={Computed Targets as Mathematical Guardrails for {LLM} Negotiators:
         Anchoring Defense, the Oracle Paradox, and Variance Collapse
         in Multi-Issue Bargaining},
  author={Matthiasson, Thor},
  year={2025},
  note={Revised April 2026. Available at
        \url{https://github.com/Ravenai307/llm-negotiation}}
}
```

## License

MIT
