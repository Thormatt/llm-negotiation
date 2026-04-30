"""
Counterfactual replay: would HeuristicOM have produced different target proposals
than the trained GRU on logged Haiku-vs-Haiku games?

For each turn where the buyer was making an offer (had_briefing=True), we replay
the env up to that point and compute:

  - GRU prediction → compute_target_proposal → target_gru
  - HeuristicOM prediction → compute_target_proposal → target_heur
  - Uniform [0.2]^5 prediction → compute_target_proposal → target_unif

Then we compare:
  - L2 distance between target_gru and target_heur (in proposal space)
  - Per-variable max absolute deviation
  - Buyer utility under each target (using the buyer's true weights)

If GRU vs Heur are nearly identical at the proposal level, swapping them in a
live A/B should give equivalent buyer share.

Free experiment — no API spend, runs in seconds on the existing log file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from heuristic_om import predict_heuristic_om
from negotiation import compute_target_proposal
from train_negotiation_gru import (
    BUYER_DIRECTION,
    NegotiationEnv,
    build_observation_sequence,
    compute_utility,
    load_negotiation_gru,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
CKPT = Path(__file__).resolve().parent / "checkpoints" / "negotiation_gru.pt"
UNIFORM = np.full(5, 0.2, dtype=np.float32)


def replay_partial(game: dict, up_to_turn_idx: int) -> NegotiationEnv:
    """Reconstruct env having processed turns [0..up_to_turn_idx)."""
    env = NegotiationEnv(
        buyer_weights=np.asarray(game["buyer_weights"], dtype=np.float32),
        seller_weights=np.asarray(game["seller_weights"], dtype=np.float32),
        max_rounds=8,
    )
    for turn in game["turns"][:up_to_turn_idx]:
        if turn["action"] != "offer":
            continue
        env.step(turn["role"], "offer", np.asarray(turn["proposal"], dtype=np.float32))
        if env.done:
            break
    return env


def gru_predict(model, env: NegotiationEnv) -> np.ndarray:
    obs, length = build_observation_sequence(env, perspective="buyer")
    length_t = torch.tensor([length], dtype=torch.long)
    with torch.no_grad():
        pred, _, _ = model(obs, lengths=length_t)
    return pred.squeeze(0).cpu().numpy().astype(np.float32)


def main() -> None:
    log_path = RESULTS / "n200_B_gru_haiku.json"
    with open(log_path) as f:
        log = json.load(f)

    model, _ = load_negotiation_gru(str(CKPT))

    deltas_l2 = []      # |target_gru - target_heur|_2
    deltas_unif_l2 = [] # |target_gru - target_unif|_2
    util_gru, util_heur, util_unif = [], [], []

    games = log["results"]["B"]
    for g in games:
        env_full_history_idx = len(g["turns"])
        # Reconstruct opponent's first proposal index — we only score buyer
        # turns where the seller has made at least one move (so the OM has
        # something to predict from).
        for turn_idx, turn in enumerate(g["turns"]):
            if turn["role"] != "buyer" or turn["action"] != "offer":
                continue
            if not turn.get("had_briefing", False):
                continue

            env = replay_partial(g, turn_idx)
            opp_proposals_seen = [r for r, _ in env.history if r == "seller"]
            if not opp_proposals_seen:
                continue

            round_frac = env.round / env.max_rounds
            buyer_w = np.asarray(g["buyer_weights"], dtype=np.float32)

            # Three predictions
            gru_w = gru_predict(model, env)
            heur_w, _ = predict_heuristic_om(env, perspective="buyer")
            unif_w = UNIFORM

            # Three target proposals
            t_gru = compute_target_proposal(buyer_w, gru_w, BUYER_DIRECTION, round_frac)
            t_heur = compute_target_proposal(buyer_w, heur_w, BUYER_DIRECTION, round_frac)
            t_unif = compute_target_proposal(buyer_w, unif_w, BUYER_DIRECTION, round_frac)

            # Distance in proposal space (L2 over 5 dims)
            deltas_l2.append(float(np.linalg.norm(t_gru - t_heur)))
            deltas_unif_l2.append(float(np.linalg.norm(t_gru - t_unif)))

            # Buyer utility evaluated against the *seller's actual response* —
            # not available counterfactually. Use the proposal's self-utility
            # as a proxy: a higher self-utility target is one the buyer would
            # rather have accepted.
            util_gru.append(float(compute_utility(t_gru, buyer_w, "buyer")))
            util_heur.append(float(compute_utility(t_heur, buyer_w, "buyer")))
            util_unif.append(float(compute_utility(t_unif, buyer_w, "buyer")))

    if not deltas_l2:
        print("No buyer-with-briefing turns found in log.")
        return

    d = np.array(deltas_l2)
    du = np.array(deltas_unif_l2)
    ug = np.array(util_gru)
    uh = np.array(util_heur)
    un = np.array(util_unif)

    print(f"Counterfactual replay over {len(games)} games, {len(d)} buyer-with-OM turns.\n")
    print("Target-proposal distance (L2 over 5-dim proposal space, range 0–√5≈2.24):")
    print(f"  GRU vs Heuristic OM:  mean {d.mean():.3f}   median {np.median(d):.3f}   "
          f"max {d.max():.3f}   p90 {np.percentile(d, 90):.3f}")
    print(f"  GRU vs Uniform OM:    mean {du.mean():.3f}   median {np.median(du):.3f}   "
          f"max {du.max():.3f}   p90 {np.percentile(du, 90):.3f}")

    print()
    print("Buyer self-utility of each target proposal (higher = better for buyer):")
    print(f"  GRU target:        mean {ug.mean():.4f} ± {ug.std():.4f}")
    print(f"  Heuristic target:  mean {uh.mean():.4f} ± {uh.std():.4f}    "
          f"(diff vs GRU: {uh.mean()-ug.mean():+.4f})")
    print(f"  Uniform target:    mean {un.mean():.4f} ± {un.std():.4f}    "
          f"(diff vs GRU: {un.mean()-ug.mean():+.4f})")

    # Fraction of turns where GRU and Heur agree closely
    close = (d < 0.05).mean()
    print(f"\nFraction of turns where GRU/Heur targets agree within L2 < 0.05: {close:.1%}")
    very_close = (d < 0.02).mean()
    print(f"Fraction of turns where GRU/Heur targets agree within L2 < 0.02: {very_close:.1%}")


if __name__ == "__main__":
    main()
