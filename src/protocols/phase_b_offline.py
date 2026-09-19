"""
Phase B offline analysis: replace the LLM-LLM language channel with a
deterministic middleman protocol. No API calls.

Procedure:
  1. Replay the N=200 Dirichlet(α=1) buyer/seller weight seeds used by the
     paper's randomized-opener Condition B run.
  2. For each seed, compute the protocol-mediated outcome under:
        NashBargaining2P     — corner-solution Pareto-optimal allocation
        CooperativeBlend2P   — paper's compute_target_proposal at aggression=1
  3. Aggregate buyer share, joint surplus, Pareto efficiency, variance.
  4. Compare to paper's logged Condition A (unassisted LLM-LLM) and
     Condition B (LLM-with-sub-agent) numbers.

This gives a clean answer to the user's middleman hypothesis: if the
protocol-mediated outcomes match or beat the LLM-channel outcomes on every
metric, the language channel is empirically replaceable for this domain
of structured commerce.

Paper baselines (from paper Tables 1 and tab:ablation_opener):
  Condition A (LLM-LLM, no scaffold):          buyer share 52.2% ± 8.9%, joint 1.021, Pareto 72%
  Condition B + GRU (LLM with sub-agent):      buyer share 55.2% ± 3.8%, joint 1.083, Pareto 76%
  Condition B + Oracle (LLM with true weights): buyer share 49.5% ± 0.9%, joint 1.126, Pareto 79%
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from protocols.nash_bargaining import CooperativeBlend2P, NashBargaining2P

# Negotiation environment direction vectors (from belief_gru/train_negotiation_gru.py)
BUYER_DIRECTION = np.array([-1, +1, -1, +1, +1], dtype=np.float64)
SELLER_DIRECTION = np.array([+1, -1, +1, -1, -1], dtype=np.float64)
N_VARIABLES = 5
N_GAMES = 200
DIRICHLET_ALPHA = 1.0
SEED = 42

# Paper baselines for comparison
PAPER_BASELINES = {
    "A (LLM-LLM, no scaffold)":        {"share": 0.522, "share_sd": 0.089, "joint": 1.021, "pareto": 0.72},
    "B (LLM with GRU sub-agent)":      {"share": 0.552, "share_sd": 0.038, "joint": 1.083, "pareto": 0.76},
    "B (LLM with bias=0.55, N=32)":    {"share": 0.560, "share_sd": 0.038, "joint": None,  "pareto": None},
    "B (LLM with Oracle)":             {"share": 0.495, "share_sd": 0.009, "joint": 1.126, "pareto": 0.79},
    "B (LLM with Uniform)":            {"share": 0.494, "share_sd": 0.019, "joint": 1.069, "pareto": 0.75},
}


def additive_utility(v: np.ndarray, weights: np.ndarray, direction: np.ndarray) -> float:
    tilde = np.where(direction > 0, v, 1.0 - v)
    return float(np.dot(weights, tilde))


def max_joint_surplus(buyer_w: np.ndarray, seller_w: np.ndarray) -> float:
    """Pareto-frontier ceiling — take each variable to whichever side weights it more."""
    direction_alignment = (buyer_w * (BUYER_DIRECTION > 0).astype(np.float64) +
                           seller_w * (SELLER_DIRECTION > 0).astype(np.float64))
    direction_total_low = (buyer_w * (BUYER_DIRECTION < 0).astype(np.float64) +
                           seller_w * (SELLER_DIRECTION < 0).astype(np.float64))
    proposal_max = np.where(direction_alignment > direction_total_low, 0.95, 0.05)
    return additive_utility(proposal_max, buyer_w, BUYER_DIRECTION) + \
           additive_utility(proposal_max, seller_w, SELLER_DIRECTION)


def evaluate_protocol(proto, n_games: int = N_GAMES, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    shares = []
    joints = []
    paretos = []
    buyer_utils = []
    seller_utils = []

    for _ in range(n_games):
        wb = rng.dirichlet(np.ones(N_VARIABLES) * DIRICHLET_ALPHA)
        ws = rng.dirichlet(np.ones(N_VARIABLES) * DIRICHLET_ALPHA)

        inputs = [
            {"weights": wb, "direction": BUYER_DIRECTION},
            {"weights": ws, "direction": SELLER_DIRECTION},
        ]
        out = proto.run(inputs)
        ub, us = out.payoffs

        joint = ub + us
        max_joint = max_joint_surplus(wb, ws)
        share = ub / (ub + us) if (ub + us) > 1e-12 else 0.5

        shares.append(share)
        joints.append(joint)
        paretos.append(joint / max_joint if max_joint > 1e-12 else 0.0)
        buyer_utils.append(ub)
        seller_utils.append(us)

    shares = np.array(shares)
    joints = np.array(joints)
    paretos = np.array(paretos)
    return {
        "protocol": proto.name,
        "n_games": n_games,
        "share_mean": float(shares.mean()),
        "share_sd": float(shares.std()),
        "joint_mean": float(joints.mean()),
        "joint_sd": float(joints.std()),
        "pareto_mean": float(paretos.mean()),
        "buyer_util_mean": float(np.mean(buyer_utils)),
        "seller_util_mean": float(np.mean(seller_utils)),
    }


def main() -> None:
    print(f"Phase B offline analysis  (N={N_GAMES} Dirichlet({DIRICHLET_ALPHA}) "
          f"seeds, seed={SEED})\n")
    print("Protocols evaluated:")
    print("  NashBargaining2P    — corner-solution Pareto-optimal")
    print("  CooperativeBlend2P  — paper's compute_target_proposal equivalent")
    print()

    nash = evaluate_protocol(NashBargaining2P())
    coop = evaluate_protocol(CooperativeBlend2P())

    print(f"{'Condition':<38} {'Share':>9} {'Share σ':>9} {'Joint':>8} {'Pareto':>8}")
    print("-" * 80)
    for label, b in PAPER_BASELINES.items():
        share = f"{b['share']:.3f}"
        share_sd = f"{b['share_sd']:.3f}"
        joint = f"{b['joint']:.3f}" if b['joint'] is not None else "  —  "
        pareto = f"{b['pareto']:.0%}" if b['pareto'] is not None else "  —  "
        print(f"{label:<38} {share:>9} {share_sd:>9} {joint:>8} {pareto:>8}")

    print()
    print("Protocol-mediated (no LLM-LLM channel):")
    print("-" * 80)
    for r in [nash, coop]:
        share = f"{r['share_mean']:.3f}"
        share_sd = f"{r['share_sd']:.3f}"
        joint = f"{r['joint_mean']:.3f}"
        pareto = f"{r['pareto_mean']:.0%}"
        print(f"{('Middleman: ' + r['protocol']):<38} {share:>9} {share_sd:>9} {joint:>8} {pareto:>8}")

    print()
    print("Reading:")
    print("  Share  — buyer's fraction of joint surplus.")
    print("  σ      — standard deviation of buyer share across games (variance collapse metric).")
    print("  Joint  — total surplus, ceiling depends on weights.")
    print("  Pareto — joint / max-possible joint, fraction of Pareto frontier captured.")


if __name__ == "__main__":
    main()
