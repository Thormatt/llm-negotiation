# LLM Negotiation — Data Packet (v1.0)

Companion data and model checkpoints for:

> **Computed Targets and the Middleman Protocol: Mathematical Guardrails and Channel Removal for LLM Negotiators in Multi-Issue Bargaining**
> Thor Matthiasson, 2026.

This packet is self-contained. It bundles every per-game JSON log, both
load-bearing model checkpoints, the prompt templates, and reproduction
scripts that recompute the headline tables from raw data.

## Layout

```
data-packet/
├── README.md                      this file
├── MANIFEST.csv                   one row per file: sha256 + table mapping
├── LICENSE                        CC-BY-4.0 (data) / MIT (scripts)
├── seeds.json                     seed registry (base seed, opener seeds)
├── checkpoints/                   PyTorch state dicts
│   ├── negotiation_gru.pt         11,078-param NegotiationGRU (§3.2)
│   └── policy_net_rwr_tau0.2.pt   6,533-param NegotiationPolicyNet (Appendix A)
├── prompts/
│   └── negotiator_system_prompt.txt   Verbatim NEGOTIATOR_SYSTEM template + variants
├── results/                       Per-game JSON logs (full transcripts)
│   ├── primary/                   N=200 randomized opener — Tables 7, 8, 10, 14
│   ├── multimodel/                8 models × 2 conditions (N=50) — Table 9
│   ├── crossmodel/                Haiku+OM vs frontier sellers (N=50) — Tables 11, 15
│   ├── seller_first/              Original N=50 seller-first runs — Tables 4, 6
│   ├── randomized_n50/            Smaller-N randomized sanity — Table 7
│   ├── llmio_middleman/           Live LLMIO under translation noise — Table 16
│   ├── policy_ab/                 Neural policy A/B test — Tables 17, 18
│   └── diagnostic/                Heuristic-estimator runs — Tables 12, 13
└── scripts/
    ├── verify_packet.py           Confirm sha256 + JSON-loadability for every file
    └── compute_tables.py          Recompute headline tables from raw JSON
```

Total: 38 result JSONs + 2 checkpoints + 1 prompt file ≈ 16 MB unpacked.

## Quick start

```bash
# unpack
tar xzf llm-negotiation-data-v1.0.tar.gz
cd data-packet

# verify integrity (sha256 + JSON-parseable)
python3 scripts/verify_packet.py

# reproduce headline numbers (no API calls; reads JSONs only)
python3 scripts/compute_tables.py
```

`compute_tables.py` reads only the bundled JSONs and uses numpy. The
output should match the paper PDF to ≤0.5pp on every cell (small drift
where some files are subsamples or earlier seeds).

## What each per-game record contains

Every game in every JSON has this structure (per `results.B[i]` or
`results[i]`, depending on layout):

```json
{
  "deal": true,
  "buyer_utility": 0.608,
  "seller_utility": 0.523,
  "joint_surplus": 1.132,
  "max_joint_surplus": 1.540,
  "pareto_efficiency": 0.735,
  "buyer_surplus_share": 0.538,
  "deal_proposal":  [0.65, 0.72, 0.35, 0.38, 0.28],
  "optimal_proposal":[0.0, 1.0, 1.0, 0.0, 0.0],
  "rounds": 7,
  "buyer_weights":  [...],   "seller_weights": [...],
  "turns": [
    { "role": "seller",
      "action": "offer",
      "proposal": [...],
      "reasoning": "...",
      "had_briefing": false },
    ...
  ]
}
```

The `turns` array contains the full LLM transcript per game including
each agent's `internal_monologue`, public `message`, and `formal_proposal`.
This is the raw material for §6.2 (text-advice failure analysis) and any
downstream qualitative coding work.

## Loading the checkpoints

```python
import torch

# NegotiationGRU (§3.2) — 11,078 params
gru_ckpt = torch.load("checkpoints/negotiation_gru.pt", map_location="cpu")
# state_dict keys live under gru_ckpt["model_state_dict"]

# NegotiationPolicyNet (Appendix A) — 6,533 params, 27D state -> 5D proposal
pol_ckpt = torch.load("checkpoints/policy_net_rwr_tau0.2.pt", map_location="cpu")
```

The architectures are defined in `belief_gru/model.py` and
`belief_gru/policy_net.py` in the source repository
(`https://github.com/thormatthiasson/ml_sandbox/tree/main/belief_gru`).
Use the `load_negotiation_gru()` helper for canonical loading.

## Reproducing tables not covered by `compute_tables.py`

Three tables are computed from the bundled data plus a small replay
script that lives outside the packet (in the source repo):

| Table | Source | Script |
|-------|--------|--------|
| Tab 14 (Middleman offline replay) | `results/primary/n200_B_gru_haiku.json` weight pairs | `protocols/phase_b_offline.py` |
| Tab 15 (Middleman cross-model replay) | `results/crossmodel/n50_cross_*.json` weight pairs | `protocols/phase_b_crossmodel.py` |
| Tab 12 (Predictor accuracy) | All `n200_*` + `n50_*` files | `belief_gru/crossmodel_diagnostic.py` |

These three scripts read weight pairs from the bundled JSONs and apply a
deterministic mapping (Nash bargaining, cooperative blend, or analytical
inversion). No API calls. They are not bundled in this packet because
they import from the larger `ml_sandbox` codebase.

## What is NOT in the packet (and why)

| Excluded | Reason |
|----------|--------|
| GRU training data (20K heuristic-vs-heuristic matches) | Regenerated deterministically by `train_negotiation_gru.py` from seed=42 |
| Policy training tuples (4,341 state-action-rewards) | Derived from `results/*.json` by `policy_data.py:load_all_trajectories()` |
| Auction / SGRM / MoGRU / toy-task results | Belong to other projects in `ml_sandbox`, not this paper |
| Pilot N=2 / N=30 runs | Superseded by N=50 / N=200 final runs |
| Other-version model runs (e.g. `_v4` suffix files) | The packet retains only the version that matches the paper PDF; older runs dropped |

## Provenance and seeds

- **Base seed**: 42 (per-game seed = 42 + game_index)
- **Dirichlet α**: 1.0 for all weight sampling (paper uses Dirichlet(1.0))
- **Opener selection**: under randomized opener, opener role chosen
  uniformly per game using the per-game seed
- **Locked**: nothing in this packet may be re-sampled. New runs must use
  fresh seeds and be reported as such.

See `seeds.json` for the canonical seed registry.

## Citation

If you use this data, please cite:

```bibtex
@misc{matthiasson2026negotiation,
  author       = {Matthiasson, Thor},
  title        = {Computed Targets and the Middleman Protocol:
                  Mathematical Guardrails and Channel Removal for
                  LLM Negotiators in Multi-Issue Bargaining},
  year         = {2026},
  howpublished = {arXiv preprint},
  note         = {Data packet v1.0}
}
```

## License

- **Data** (everything in `results/`, `checkpoints/`, `prompts/`):
  CC-BY-4.0. Attribution required.
- **Scripts** (`scripts/`): MIT License.

See `LICENSE` for the full text.

## Contact

`thor@aiwiththor.com` — New Haven, Connecticut, USA.
