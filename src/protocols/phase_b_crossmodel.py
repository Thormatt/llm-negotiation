"""
Phase B cross-model: does the frontier-model advantage survive under a middleman protocol?

The negotiation paper's Table 9 (Cross-Model Asymmetric) reported that a Haiku
buyer with sub-agent against an unassisted frontier seller takes:

    vs Grok 3:        55.2% buyer share, ± 11.3%
    vs Opus 4.6:      49.1% buyer share, ± 16.1%
    vs GPT-5:         48.1% buyer share, ± 19.5%
    vs Gemini Pro:    47.9% buyer share, ± 20.3%

Stronger sellers extracted 2-5pp from the cheap-with-scaffold buyer in the
language-channel architecture.

The middleman hypothesis says: the language channel is what's leaking
surplus. If both sides submit weights to a deterministic protocol instead,
buyer share should depend only on the relative weights, not on the seller's
identity. Frontier-model advantage should vanish.

We test this by replaying the *exact* weight pairs from the four logged
cross-model runs through NashBargaining2P and CooperativeBlend2P. Same
weights, different channel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from protocols.nash_bargaining import CooperativeBlend2P, NashBargaining2P

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"

BUYER_DIRECTION = np.array([-1, +1, -1, +1, +1], dtype=np.float64)
SELLER_DIRECTION = np.array([+1, -1, +1, -1, -1], dtype=np.float64)

CROSS_MODEL_LOGS = {
    "Haiku vs Grok 3":      RESULTS / "n50_cross_haiku_vs_grok3.json",
    "Haiku vs Opus 4.6":    RESULTS / "n50_cross_haiku_vs_opus46.json",
    "Haiku vs GPT-5":       RESULTS / "n50_cross_haiku_vs_gpt5.json",
    "Haiku vs Gemini Pro":  RESULTS / "n50_cross_haiku_vs_gemini3pro.json",
}

# Paper's logged LLM-channel results (cross-model with sub-agent on buyer side)
PAPER_LLM_CHANNEL = {
    "Haiku vs Grok 3":      {"share": 0.552, "share_sd": 0.113},
    "Haiku vs Opus 4.6":    {"share": 0.491, "share_sd": 0.161},
    "Haiku vs GPT-5":       {"share": 0.481, "share_sd": 0.195},
    "Haiku vs Gemini Pro":  {"share": 0.479, "share_sd": 0.203},
}


def additive_utility(v: np.ndarray, weights: np.ndarray, direction: np.ndarray) -> float:
    tilde = np.where(direction > 0, v, 1.0 - v)
    return float(np.dot(weights, tilde))


def max_joint_surplus(buyer_w: np.ndarray, seller_w: np.ndarray) -> float:
    align_high = (buyer_w * (BUYER_DIRECTION > 0).astype(np.float64) +
                  seller_w * (SELLER_DIRECTION > 0).astype(np.float64))
    align_low = (buyer_w * (BUYER_DIRECTION < 0).astype(np.float64) +
                 seller_w * (SELLER_DIRECTION < 0).astype(np.float64))
    proposal_max = np.where(align_high > align_low, 0.95, 0.05)
    return additive_utility(proposal_max, buyer_w, BUYER_DIRECTION) + \
           additive_utility(proposal_max, seller_w, SELLER_DIRECTION)


def replay_via_protocol(games: list, proto) -> dict:
    shares, joints, paretos = [], [], []
    for g in games:
        wb = np.asarray(g["buyer_weights"], dtype=np.float64)
        ws = np.asarray(g["seller_weights"], dtype=np.float64)
        out = proto.run([
            {"weights": wb, "direction": BUYER_DIRECTION},
            {"weights": ws, "direction": SELLER_DIRECTION},
        ])
        ub, us = out.payoffs
        joint = ub + us
        max_j = max_joint_surplus(wb, ws)
        shares.append(ub / (ub + us) if (ub + us) > 1e-12 else 0.5)
        joints.append(joint)
        paretos.append(joint / max_j if max_j > 1e-12 else 0.0)
    arr = np.array(shares)
    return {
        "n": len(games),
        "share_mean": float(arr.mean()),
        "share_sd": float(arr.std()),
        "joint_mean": float(np.mean(joints)),
        "pareto_mean": float(np.mean(paretos)),
    }


def main() -> None:
    print("Phase B cross-model replay: same weight pairs, different channel.\n")
    print(f"{'Match-up':<24} {'Channel':<28} {'Share':>8} {'σ':>8} {'Pareto':>8}")
    print("-" * 80)

    for label, path in CROSS_MODEL_LOGS.items():
        if not path.exists():
            print(f"{label:<24} (missing {path.name})")
            continue
        log = json.load(open(path))
        games = log["results"]["B"]

        # Paper's LLM-channel result
        paper = PAPER_LLM_CHANNEL[label]
        print(f"{label:<24} {'LLM-LLM with sub-agent':<28} "
              f"{paper['share']:>8.3f} {paper['share_sd']:>8.3f} {' — ':>8}")

        # Middleman: NashBargaining2P
        nash = replay_via_protocol(games, NashBargaining2P())
        print(f"{label:<24} {'Middleman: NashBargaining2P':<28} "
              f"{nash['share_mean']:>8.3f} {nash['share_sd']:>8.3f} "
              f"{nash['pareto_mean']:>7.0%}")

        # Middleman: CooperativeBlend2P
        coop = replay_via_protocol(games, CooperativeBlend2P())
        print(f"{label:<24} {'Middleman: CooperativeBlend2P':<28} "
              f"{coop['share_mean']:>8.3f} {coop['share_sd']:>8.3f} "
              f"{coop['pareto_mean']:>7.0%}")
        print()

    print("Reading:")
    print("  LLM-LLM channel rows: from paper Table 9. Frontier sellers extract")
    print("  surplus from cheap-with-sub-agent buyer through the language channel.")
    print()
    print("  Middleman rows: same weight pairs, no language channel. The protocol")
    print("  takes both sides' weights and computes a deterministic outcome.")
    print()
    print("  If frontier-advantage disappears under middleman, the asymmetry was")
    print("  a property of the language channel, not of any inherent capability gap.")


if __name__ == "__main__":
    main()
