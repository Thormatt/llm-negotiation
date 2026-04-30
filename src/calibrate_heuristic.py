"""
Calibrate the heuristic OM's `bias` knob to match the trained GRU's mean
counterfactual buyer-utility on logged Haiku-vs-Haiku turns.

Sweep bias ∈ [0.0, 0.6] and for each:
  - replay every buyer-with-OM turn from n200_B_gru_haiku.json
  - compute target proposal under heuristic-with-bias OM
  - measure mean buyer self-utility of the target

Find the bias level that matches GRU's reference utility (~0.598).
Also report L2 distance between calibrated-heur and GRU targets at that bias.
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

GRU_REFERENCE_UTIL = 0.5985  # from counterfactual_replay.py
BIAS_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def replay_partial(game: dict, up_to: int) -> NegotiationEnv:
    env = NegotiationEnv(
        buyer_weights=np.asarray(game["buyer_weights"], dtype=np.float32),
        seller_weights=np.asarray(game["seller_weights"], dtype=np.float32),
        max_rounds=8,
    )
    for turn in game["turns"][:up_to]:
        if turn["action"] != "offer":
            continue
        env.step(turn["role"], "offer", np.asarray(turn["proposal"], dtype=np.float32))
        if env.done:
            break
    return env


def main() -> None:
    log_path = RESULTS / "n200_B_gru_haiku.json"
    with open(log_path) as f:
        log = json.load(f)
    games = log["results"]["B"]

    model, _ = load_negotiation_gru(str(CKPT))

    # Pre-compute GRU targets and turn data once
    turn_data = []
    for g in games:
        for turn_idx, turn in enumerate(g["turns"]):
            if turn["role"] != "buyer" or turn["action"] != "offer":
                continue
            if not turn.get("had_briefing", False):
                continue
            env = replay_partial(g, turn_idx)
            opp_seen = [r for r, _ in env.history if r == "seller"]
            if not opp_seen:
                continue
            obs, length = build_observation_sequence(env, perspective="buyer")
            length_t = torch.tensor([length], dtype=torch.long)
            with torch.no_grad():
                pred, _, _ = model(obs, lengths=length_t)
            gru_w = pred.squeeze(0).cpu().numpy().astype(np.float32)
            buyer_w = np.asarray(g["buyer_weights"], dtype=np.float32)
            round_frac = env.round / env.max_rounds
            t_gru = compute_target_proposal(buyer_w, gru_w, BUYER_DIRECTION, round_frac)
            turn_data.append({
                "env": env,
                "buyer_w": buyer_w,
                "round_frac": round_frac,
                "t_gru": t_gru,
                "gru_util": compute_utility(t_gru, buyer_w, "buyer"),
            })

    print(f"{len(turn_data)} buyer-with-OM turns from {len(games)} games.\n")
    print(f"GRU reference target utility (mean): {GRU_REFERENCE_UTIL:.4f}\n")

    print(f"{'bias':>6}  {'mean util':>10}  {'Δ vs GRU':>10}  {'L2 to GRU':>11}  {'agree<0.05':>11}")
    print("-" * 60)

    for bias in BIAS_GRID:
        utils = []
        l2s = []
        for d in turn_data:
            heur_w, _ = predict_heuristic_om(
                d["env"], perspective="buyer",
                bias=bias, my_weights=d["buyer_w"],
            )
            t = compute_target_proposal(d["buyer_w"], heur_w, BUYER_DIRECTION, d["round_frac"])
            utils.append(float(compute_utility(t, d["buyer_w"], "buyer")))
            l2s.append(float(np.linalg.norm(t - d["t_gru"])))

        mean_u = float(np.mean(utils))
        l2_arr = np.array(l2s)
        agree = (l2_arr < 0.05).mean()
        print(f"{bias:>6.2f}  {mean_u:>10.4f}  {mean_u - GRU_REFERENCE_UTIL:>+10.4f}  "
              f"{l2_arr.mean():>11.3f}  {agree:>10.1%}")


if __name__ == "__main__":
    main()
