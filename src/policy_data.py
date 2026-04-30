"""
Policy Data Pipeline — Extract state-action-reward tuples from LLM negotiation games.

27D state vector (22D base + 5D GRU opponent weight predictions):
  Dims 0-4:   my_weights (5)           — what am I optimizing for
  Dim 5:      round_frac (1)           — deadline pressure
  Dims 6-10:  opp_latest_proposal (5)  — opponent's current offer
  Dim 11:     my_utility_at_opp (1)    — utility if I accept now
  Dims 12-16: my_latest_proposal (5)   — my last offer
  Dim 17:     opp_utility_at_mine (1)  — estimated opponent utility at my last offer
  Dim 18:     n_opp_proposals (1)      — how many turns (normalized)
  Dim 19:     is_buyer (1)             — role indicator
  Dim 20:     had_briefing (1)         — was WM directive present
  Dim 21:     opp_concession_rate (1)  — mean L2 change per opponent turn
  Dims 22-26: gru_predicted_opp_w (5)  — GRU's predicted opponent weights

Reward: agent_utility + 0.1 * pareto_efficiency for deals, -0.1 for no-deal.

Usage:
  python3 -m belief_gru.policy_data --results-dir results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import glob
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset

from train_negotiation_gru import (
    NegotiationEnv, NegotiationGRU, load_negotiation_gru,
    compute_utility, build_observation_sequence,
    BUYER_DIRECTION, SELLER_DIRECTION, VARIABLES, MAX_ROUNDS,
)

POLICY_STATE_DIM = 27


@dataclass
class StateActionReward:
    """Single turn from a game, from one agent's perspective."""
    state: np.ndarray          # (22,) policy state vector
    action: np.ndarray         # (5,) proposal made by this agent
    reward: float              # terminal reward for this game
    game_idx: int              # which game this came from
    turn_idx: int              # which turn within the game
    total_turns: int           # total turns in game
    perspective: str           # "buyer" or "seller"


def _maybe_mask_gru(state, mask_gru):
    """If mask_gru is True, replace dims 22-26 with the uniform prior [0.2]*5.

    Used for the §5.7 load-bearing test: confirms whether the policy net
    actually uses the GRU's predictions or just the 22 base features.
    """
    if mask_gru:
        state[22:27] = 0.2
    return state


def extract_state_at_turn(game, turn_idx, perspective, max_rounds=MAX_ROUNDS,
                          gru_model=None, mask_gru=False):
    """Extract 27D state vector at the moment just before agent makes turn_idx's proposal.

    Args:
        game: dict with keys: buyer_weights, seller_weights, turns, deal, ...
        turn_idx: index into game["turns"] where perspective agent is about to act
        perspective: "buyer" or "seller"
        max_rounds: max negotiation rounds
        gru_model: optional NegotiationGRU for computing predicted opponent weights
        mask_gru: if True, force dims 22-26 to the uniform prior [0.2]*5 even when
            gru_model is supplied. Used for the §5.7 GRU-feature load-bearing test.

    Returns:
        np.ndarray shape (27,) or None if invalid
    """
    turns = game.get("turns", [])
    if turn_idx >= len(turns):
        return None

    turn = turns[turn_idx]
    if turn["role"] != perspective:
        return None
    if turn["action"] != "offer" or turn.get("proposal") is None:
        return None

    my_weights = np.array(game["buyer_weights" if perspective == "buyer" else "seller_weights"],
                          dtype=np.float32)
    opp_weights = np.array(game["seller_weights" if perspective == "buyer" else "buyer_weights"],
                           dtype=np.float32)
    my_direction = BUYER_DIRECTION if perspective == "buyer" else SELLER_DIRECTION

    # Collect opponent and my proposals up to (but not including) this turn
    opp_proposals = []
    my_proposals = []
    for t in turns[:turn_idx]:
        prop = t.get("proposal")
        if prop is None:
            continue
        if t["role"] != perspective:
            opp_proposals.append(np.array(prop, dtype=np.float32))
        else:
            my_proposals.append(np.array(prop, dtype=np.float32))

    state = np.zeros(POLICY_STATE_DIM, dtype=np.float32)

    # Dims 0-4: my_weights
    state[0:5] = my_weights

    # Dim 5: round_frac — count all preceding turns to estimate round
    n_preceding = len(turns[:turn_idx])
    round_estimate = (n_preceding + 1) / 2.0  # ~rounds elapsed
    state[5] = min(round_estimate / max_rounds, 1.0)

    # Dims 6-10: opp_latest_proposal
    if opp_proposals:
        state[6:11] = opp_proposals[-1]
    else:
        state[6:11] = 0.5  # neutral default

    # Dim 11: my_utility_at_opp — utility if I accept opponent's latest
    if opp_proposals:
        state[11] = compute_utility(opp_proposals[-1], my_weights, perspective)
    else:
        state[11] = 0.0

    # Dims 12-16: my_latest_proposal
    if my_proposals:
        state[12:17] = my_proposals[-1]
    else:
        # First move — use aggressive opening as default
        state[12:17] = np.where(my_direction > 0, 0.80, 0.20).astype(np.float32)

    # Dim 17: opp_utility_at_mine — estimated with uniform weights (we don't know true)
    if my_proposals:
        opp_role = "seller" if perspective == "buyer" else "buyer"
        uniform_w = np.full(5, 0.2, dtype=np.float32)
        state[17] = compute_utility(my_proposals[-1], uniform_w, opp_role)
    else:
        state[17] = 0.5

    # Dim 18: n_opp_proposals (normalized by max_rounds)
    state[18] = len(opp_proposals) / max_rounds

    # Dim 19: is_buyer
    state[19] = 1.0 if perspective == "buyer" else 0.0

    # Dim 20: had_briefing
    state[20] = 1.0 if turn.get("had_briefing", False) else 0.0

    # Dim 21: opp_concession_rate — mean L2 change per opponent turn
    if len(opp_proposals) >= 2:
        deltas = [np.linalg.norm(opp_proposals[i+1] - opp_proposals[i])
                  for i in range(len(opp_proposals) - 1)]
        state[21] = float(np.mean(deltas))
    else:
        state[21] = 0.0

    # Dims 22-26: GRU predicted opponent weights
    if gru_model is not None and opp_proposals:
        # Replay game up to this turn through a fresh env, then run GRU
        buyer_w = np.array(game["buyer_weights"], dtype=np.float32)
        seller_w = np.array(game["seller_weights"], dtype=np.float32)
        replay_env = NegotiationEnv(buyer_w, seller_w, max_rounds=max_rounds)
        for t in turns[:turn_idx]:
            prop = t.get("proposal")
            if t["action"] == "offer" and prop is not None:
                replay_env.step(t["role"], "offer", np.array(prop, dtype=np.float32))
            elif t["action"] == "accept":
                replay_env.step(t["role"], "accept")

        obs, length = build_observation_sequence(replay_env, perspective)
        gru_model.eval()
        with torch.no_grad():
            w_pred, _, _ = gru_model(obs, torch.tensor([length]))
        state[22:27] = w_pred[0].numpy()
    else:
        # No GRU or no opponent data yet — uniform prior
        state[22:27] = 0.2

    return _maybe_mask_gru(state, mask_gru)


def compute_game_reward(game):
    """Compute terminal reward for a game.

    Returns:
        (buyer_reward, seller_reward)
    """
    if game.get("deal", False):
        buyer_util = game.get("buyer_utility", 0.0)
        seller_util = game.get("seller_utility", 0.0)
        pareto_eff = game.get("pareto_efficiency", 0.0)
        buyer_reward = buyer_util + 0.1 * pareto_eff
        seller_reward = seller_util + 0.1 * pareto_eff
        return buyer_reward, seller_reward
    else:
        return -0.1, -0.1


def load_all_trajectories(results_dir, gru_model=None, mask_gru=False):
    """Load state-action-reward tuples from all negotiation game JSONs.

    Scans for JSON files containing negotiation results (must have "results" key
    with condition-keyed game lists, and games must have "turns", "buyer_weights",
    "seller_weights").

    Args:
        results_dir: path to results/ directory
        gru_model: optional NegotiationGRU for computing predicted opponent weights
                   (dims 22-26 of the state vector)

    Returns:
        list of StateActionReward dataclasses
    """
    json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    all_samples = []
    game_counter = 0
    skipped_files = 0

    for fpath in json_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped_files += 1
            continue

        # Must have negotiation-style results
        results = data.get("results")
        if not isinstance(results, dict):
            skipped_files += 1
            continue

        max_rounds = data.get("max_rounds", MAX_ROUNDS)

        for cond, games in results.items():
            if not isinstance(games, list):
                continue

            for game in games:
                # Validate game has required fields
                if not all(k in game for k in ("buyer_weights", "seller_weights", "turns")):
                    continue
                if not isinstance(game["turns"], list) or len(game["turns"]) < 2:
                    continue

                buyer_reward, seller_reward = compute_game_reward(game)
                total_turns = len(game["turns"])

                for perspective in ("buyer", "seller"):
                    reward = buyer_reward if perspective == "buyer" else seller_reward

                    for t_idx, turn in enumerate(game["turns"]):
                        if turn["role"] != perspective:
                            continue
                        if turn["action"] != "offer" or turn.get("proposal") is None:
                            continue

                        state = extract_state_at_turn(game, t_idx, perspective,
                                                       max_rounds, gru_model=gru_model,
                                                       mask_gru=mask_gru)
                        if state is None:
                            continue

                        action = np.array(turn["proposal"], dtype=np.float32)

                        all_samples.append(StateActionReward(
                            state=state,
                            action=action,
                            reward=reward,
                            game_idx=game_counter,
                            turn_idx=t_idx,
                            total_turns=total_turns,
                            perspective=perspective,
                        ))

                game_counter += 1

    return all_samples


def build_policy_dataset(trajectories, val_frac=0.15, test_frac=0.15, seed=42):
    """Build train/val/test TensorDatasets with game-level splitting.

    Game-level splitting prevents data leakage (turns from same game
    won't appear in both train and test).

    Args:
        trajectories: list of StateActionReward
        val_frac: fraction for validation
        test_frac: fraction for test
        seed: random seed

    Returns:
        (train_ds, val_ds, test_ds, split_info)
        Each dataset contains (states, actions, rewards, discount_weights)
    """
    if not trajectories:
        raise ValueError("No trajectories to build dataset from")

    rng = np.random.RandomState(seed)

    # Group by game_idx
    game_ids = sorted(set(t.game_idx for t in trajectories))
    rng.shuffle(game_ids)

    n_games = len(game_ids)
    n_val = max(1, int(n_games * val_frac))
    n_test = max(1, int(n_games * test_frac))
    n_train = n_games - n_val - n_test

    train_games = set(game_ids[:n_train])
    val_games = set(game_ids[n_train:n_train + n_val])
    test_games = set(game_ids[n_train + n_val:])

    def make_dataset(game_set):
        samples = [t for t in trajectories if t.game_idx in game_set]
        if not samples:
            # Return empty dataset
            return TensorDataset(
                torch.zeros(0, POLICY_STATE_DIM),
                torch.zeros(0, 5),
                torch.zeros(0),
                torch.zeros(0),
            )

        states = np.array([s.state for s in samples], dtype=np.float32)
        actions = np.array([s.action for s in samples], dtype=np.float32)
        rewards = np.array([s.reward for s in samples], dtype=np.float32)

        # Discount weights: turn t gets reward * 0.95^(T-t-1)
        discount_weights = np.array([
            0.95 ** (s.total_turns - s.turn_idx - 1) for s in samples
        ], dtype=np.float32)

        return TensorDataset(
            torch.from_numpy(states),
            torch.from_numpy(actions),
            torch.from_numpy(rewards),
            torch.from_numpy(discount_weights),
        )

    train_ds = make_dataset(train_games)
    val_ds = make_dataset(val_games)
    test_ds = make_dataset(test_games)

    split_info = {
        "n_games": n_games,
        "n_train_games": len(train_games),
        "n_val_games": len(val_games),
        "n_test_games": len(test_games),
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_test_samples": len(test_ds),
    }

    return train_ds, val_ds, test_ds, split_info


def print_dataset_stats(trajectories, split_info):
    """Print summary statistics about the extracted dataset."""
    if not trajectories:
        print("No trajectories found!")
        return

    states = np.array([t.state for t in trajectories])
    rewards = np.array([t.reward for t in trajectories])
    perspectives = [t.perspective for t in trajectories]

    n_buyer = sum(1 for p in perspectives if p == "buyer")
    n_seller = sum(1 for p in perspectives if p == "seller")
    n_briefed = sum(1 for t in trajectories if t.state[20] > 0.5)
    n_deals = sum(1 for t in trajectories if t.reward > 0)

    print(f"\n{'='*60}")
    print("POLICY DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Games:       {split_info['n_games']}")
    print(f"  Samples:     {len(trajectories)} ({n_buyer} buyer, {n_seller} seller)")
    print(f"  With WM:     {n_briefed} ({n_briefed/len(trajectories):.0%})")
    print(f"  Deal turns:  {n_deals} ({n_deals/len(trajectories):.0%})")
    print(f"\n  Split: train={split_info['n_train_samples']} | "
          f"val={split_info['n_val_samples']} | "
          f"test={split_info['n_test_samples']}")

    print(f"\n  Reward stats:")
    print(f"    mean={rewards.mean():.3f}  std={rewards.std():.3f}  "
          f"min={rewards.min():.3f}  max={rewards.max():.3f}")

    print(f"\n  Feature stats ({POLICY_STATE_DIM}D state):")
    feature_names = [
        "my_w[price]", "my_w[contract]", "my_w[payment]", "my_w[sla]", "my_w[data]",
        "round_frac",
        "opp[price]", "opp[contract]", "opp[payment]", "opp[sla]", "opp[data]",
        "my_util@opp",
        "my[price]", "my[contract]", "my[payment]", "my[sla]", "my[data]",
        "opp_util@mine",
        "n_opp_props", "is_buyer", "had_briefing", "opp_concession",
        "gru_opp[price]", "gru_opp[contract]", "gru_opp[payment]",
        "gru_opp[sla]", "gru_opp[data]",
    ]
    for i in range(POLICY_STATE_DIM):
        name = feature_names[i] if i < len(feature_names) else f"dim_{i}"
        vals = states[:, i]
        print(f"    {name:18s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"[{vals.min():.3f}, {vals.max():.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Policy Data Pipeline")
    parser.add_argument("--results-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "results"),
                        help="Directory containing game JSON files")
    parser.add_argument("--gru-checkpoint", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "checkpoints", "negotiation_gru.pt"),
                        help="Path to NegotiationGRU checkpoint for opponent weight predictions")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-gru", action="store_true",
                        help="Replace GRU prediction (state[22:27]) with uniform "
                             "[0.2]*5 — used for the §5.7 GRU-feature load-bearing test.")
    args = parser.parse_args()

    # Load GRU for opponent weight predictions
    gru_model = None
    gru_path = os.path.abspath(args.gru_checkpoint)
    if os.path.exists(gru_path):
        gru_model, n_params = load_negotiation_gru(gru_path)
        print(f"Loaded GRU ({n_params:,} params) for opponent weight predictions")
    else:
        print(f"WARNING: GRU checkpoint not found at {gru_path}")
        print("  Dims 22-26 will be uniform (0.2) — no opponent weight predictions")

    results_dir = os.path.abspath(args.results_dir)
    print(f"Loading trajectories from: {results_dir}")

    if args.mask_gru:
        print("[mask-gru] state[22:27] will be forced to [0.2]*5 — load-bearing ablation")
    trajectories = load_all_trajectories(results_dir, gru_model=gru_model,
                                         mask_gru=args.mask_gru)
    if not trajectories:
        print("ERROR: No valid negotiation games found!")
        print("Expected JSON files with 'results' key containing game lists.")
        return

    print(f"Loaded {len(trajectories)} state-action-reward tuples "
          f"from {len(set(t.game_idx for t in trajectories))} games")

    train_ds, val_ds, test_ds, split_info = build_policy_dataset(
        trajectories, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )

    print_dataset_stats(trajectories, split_info)


if __name__ == "__main__":
    main()
