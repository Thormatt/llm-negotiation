"""
Train NegotiationPolicyNet — Behavioral Cloning (BC) and Reward-Weighted Regression (RWR).

Modes:
  bc   — Behavioral cloning on top-quartile games by agent utility
  rwr  — Reward-weighted regression on all data
  eval — Offline evaluation: compare policy vs heuristic utility_at_proposal

Usage:
  python3 -m belief_gru.train_policy_net bc  --results-dir results/
  python3 -m belief_gru.train_policy_net rwr --results-dir results/
  python3 -m belief_gru.train_policy_net eval --checkpoint checkpoints/policy_net_bc.pt --results-dir results/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from policy_data import (
    load_all_trajectories, build_policy_dataset, POLICY_STATE_DIM,
)
from policy_net import NegotiationPolicyNet, load_policy_net, save_policy_net
from train_negotiation_gru import (
    compute_utility, load_negotiation_gru,
    BUYER_DIRECTION, SELLER_DIRECTION, VARIABLES, MAX_ROUNDS,
)
from negotiation import compute_target_proposal


# ===========================================================================
# Behavioral Cloning
# ===========================================================================

def filter_top_quartile(trajectories):
    """Filter for top-quartile games by agent reward.

    Returns trajectories from games where the agent's reward is in the
    top 25% of all game rewards.
    """
    if not trajectories:
        return trajectories

    rewards = np.array([t.reward for t in trajectories])
    threshold = np.percentile(rewards, 75)
    filtered = [t for t in trajectories if t.reward >= threshold]
    return filtered


def build_bc_dataset(trajectories, val_frac=0.15, test_frac=0.15, seed=42):
    """Build BC dataset: filter top quartile, then split by game."""
    filtered = filter_top_quartile(trajectories)
    print(f"  Top-quartile filter: {len(trajectories)} → {len(filtered)} samples "
          f"(reward >= {np.percentile([t.reward for t in trajectories], 75):.3f})")
    return build_policy_dataset(filtered, val_frac=val_frac, test_frac=test_frac, seed=seed)


def train_bc(train_ds, val_ds, epochs=100, lr=3e-4, batch_size=64, seed=42):
    """Train NegotiationPolicyNet with behavioral cloning.

    Weighted MSE loss: w_i * ||pred - actual||^2 where w_i = normalized utility.
    """
    torch.manual_seed(seed)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size) if len(val_ds) > 0 else None

    model = NegotiationPolicyNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNegotiationPolicyNet: {n_params:,} params")
    print(f"Training BC: {len(train_ds)} samples | Val: {len(val_ds) if val_ds else 0}")
    print(f"Epochs: {epochs} | LR: {lr} | Batch: {batch_size}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for states, actions, rewards, disc_w in train_dl:
            pred = model(states)

            # Weighted MSE: weight by normalized reward (shift to positive)
            weights = rewards - rewards.min() + 1e-3
            weights = weights / weights.mean()

            per_sample_loss = ((pred - actions) ** 2).mean(dim=1)  # (B,)
            loss = (weights * per_sample_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * states.shape[0]
            train_n += states.shape[0]

        scheduler.step()
        train_loss = train_loss_sum / max(train_n, 1)

        # Validate
        val_loss = train_loss
        if val_dl is not None:
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for states, actions, rewards, disc_w in val_dl:
                    pred = model(states)
                    per_sample_loss = ((pred - actions) ** 2).mean(dim=1)
                    val_loss_sum += per_sample_loss.sum().item()
                    val_n += states.shape[0]
            val_loss = val_loss_sum / max(val_n, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


# ===========================================================================
# Reward-Weighted Regression (RWR)
# ===========================================================================

def build_rwr_dataset(trajectories, val_frac=0.15, test_frac=0.15, seed=42):
    """Build RWR dataset: use ALL data (no filtering)."""
    return build_policy_dataset(trajectories, val_frac=val_frac, test_frac=test_frac, seed=seed)


def train_rwr(train_ds, val_ds, temperature=1.0, epochs=100, lr=3e-4,
              batch_size=64, seed=42):
    """Train NegotiationPolicyNet with Reward-Weighted Regression.

    Weight each sample by exp(advantage / temperature) where
    advantage = discounted_reward - mean_reward.
    """
    torch.manual_seed(seed)

    # Precompute RWR weights from the dataset
    all_rewards = train_ds.tensors[2]  # (N,)
    all_disc_w = train_ds.tensors[3]   # (N,)
    discounted_rewards = all_rewards * all_disc_w
    mean_reward = discounted_rewards.mean()
    advantages = discounted_rewards - mean_reward
    rwr_weights = torch.exp(advantages / temperature)
    rwr_weights = rwr_weights / rwr_weights.mean()  # normalize

    # Append rwr_weights to dataset
    rwr_ds = TensorDataset(
        train_ds.tensors[0],  # states
        train_ds.tensors[1],  # actions
        rwr_weights,          # rwr weights
    )

    train_dl = DataLoader(rwr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size) if len(val_ds) > 0 else None

    model = NegotiationPolicyNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNegotiationPolicyNet: {n_params:,} params")
    print(f"Training RWR: {len(train_ds)} samples | Val: {len(val_ds) if val_ds else 0}")
    print(f"Temperature: {temperature} | Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")

    # Stats
    print(f"  Reward: mean={all_rewards.mean():.3f} std={all_rewards.std():.3f}")
    print(f"  RWR weight: mean={rwr_weights.mean():.3f} std={rwr_weights.std():.3f} "
          f"max={rwr_weights.max():.3f}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience = 15
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for states, actions, weights in train_dl:
            pred = model(states)
            per_sample_loss = ((pred - actions) ** 2).mean(dim=1)
            loss = (weights * per_sample_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * states.shape[0]
            train_n += states.shape[0]

        scheduler.step()
        train_loss = train_loss_sum / max(train_n, 1)

        # Validate (unweighted MSE on val set)
        val_loss = train_loss
        if val_dl is not None:
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for states, actions, rewards, disc_w in val_dl:
                    pred = model(states)
                    per_sample_loss = ((pred - actions) ** 2).mean(dim=1)
                    val_loss_sum += per_sample_loss.sum().item()
                    val_n += states.shape[0]
            val_loss = val_loss_sum / max(val_n, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_loss


# ===========================================================================
# Offline Evaluation
# ===========================================================================

def evaluate_policy_offline(model, test_ds, trajectories=None):
    """Compare policy proposals vs heuristic proposals on test states.

    For each test state, compute:
      1. Policy proposal → utility for agent
      2. Heuristic proposal (compute_target_proposal) → utility for agent
      3. Actual LLM proposal → utility for agent (ground truth)

    Returns dict of comparison metrics.
    """
    model.eval()

    states = test_ds.tensors[0]  # (N, 22)
    actions = test_ds.tensors[1]  # (N, 5) actual LLM proposals
    rewards = test_ds.tensors[2]  # (N,)

    with torch.no_grad():
        policy_proposals = model(states).numpy()  # (N, 5)

    states_np = states.numpy()
    actions_np = actions.numpy()

    policy_utils = []
    heuristic_utils = []
    actual_utils = []
    policy_mse = []

    for i in range(len(states_np)):
        s = states_np[i]
        my_weights = s[0:5]
        is_buyer = s[19] > 0.5
        perspective = "buyer" if is_buyer else "seller"
        my_direction = BUYER_DIRECTION if is_buyer else SELLER_DIRECTION
        round_frac = s[5]

        # Policy proposal utility
        p_util = compute_utility(policy_proposals[i], my_weights, perspective)
        policy_utils.append(p_util)

        # Heuristic proposal utility — use GRU predicted weights from state dims 22-26
        opp_weights_estimate = s[22:27] if len(s) >= 27 else np.full(5, 0.2, dtype=np.float32)
        heuristic_target = compute_target_proposal(
            my_weights, opp_weights_estimate, my_direction, round_frac,
        )
        h_util = compute_utility(heuristic_target, my_weights, perspective)
        heuristic_utils.append(h_util)

        # Actual LLM proposal utility
        a_util = compute_utility(actions_np[i], my_weights, perspective)
        actual_utils.append(a_util)

        # MSE between policy and actual
        policy_mse.append(float(np.mean((policy_proposals[i] - actions_np[i]) ** 2)))

    policy_utils = np.array(policy_utils)
    heuristic_utils = np.array(heuristic_utils)
    actual_utils = np.array(actual_utils)
    policy_mse = np.array(policy_mse)

    # Policy vs heuristic wins
    policy_wins = np.sum(policy_utils > heuristic_utils)
    heuristic_wins = np.sum(heuristic_utils > policy_utils)
    ties = np.sum(np.abs(policy_utils - heuristic_utils) < 1e-6)

    metrics = {
        "n_samples": len(states_np),
        "policy_utility_mean": float(policy_utils.mean()),
        "policy_utility_std": float(policy_utils.std()),
        "heuristic_utility_mean": float(heuristic_utils.mean()),
        "heuristic_utility_std": float(heuristic_utils.std()),
        "actual_utility_mean": float(actual_utils.mean()),
        "actual_utility_std": float(actual_utils.std()),
        "policy_vs_heuristic_delta": float(policy_utils.mean() - heuristic_utils.mean()),
        "policy_vs_actual_delta": float(policy_utils.mean() - actual_utils.mean()),
        "policy_wins": int(policy_wins),
        "heuristic_wins": int(heuristic_wins),
        "ties": int(ties),
        "policy_mse_mean": float(policy_mse.mean()),
        "policy_mse_std": float(policy_mse.std()),
    }

    return metrics


def print_eval_results(metrics):
    """Print offline evaluation results."""
    print(f"\n{'='*60}")
    print("OFFLINE POLICY EVALUATION")
    print(f"{'='*60}")
    print(f"  Test samples: {metrics['n_samples']}")
    print(f"\n  Utility at proposed action (higher = better):")
    print(f"    Policy net:     {metrics['policy_utility_mean']:.4f} ± {metrics['policy_utility_std']:.4f}")
    print(f"    Heuristic:      {metrics['heuristic_utility_mean']:.4f} ± {metrics['heuristic_utility_std']:.4f}")
    print(f"    Actual (LLM):   {metrics['actual_utility_mean']:.4f} ± {metrics['actual_utility_std']:.4f}")
    print(f"\n  Policy vs heuristic: {metrics['policy_vs_heuristic_delta']:+.4f}")
    print(f"  Policy vs actual:    {metrics['policy_vs_actual_delta']:+.4f}")
    print(f"  Wins: policy={metrics['policy_wins']} | heuristic={metrics['heuristic_wins']} | ties={metrics['ties']}")
    print(f"\n  Policy MSE vs actual: {metrics['policy_mse_mean']:.4f} ± {metrics['policy_mse_std']:.4f}")

    delta = metrics["policy_vs_heuristic_delta"]
    if delta > 0.01:
        print(f"\n  RESULT: Policy beats heuristic by {delta:+.4f} utility")
    elif delta < -0.01:
        print(f"\n  RESULT: Heuristic beats policy by {-delta:+.4f} utility")
    else:
        print(f"\n  RESULT: Policy and heuristic roughly tied ({delta:+.4f})")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate NegotiationPolicyNet")
    sub = parser.add_subparsers(dest="mode", help="Mode")

    gru_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "negotiation_gru.pt")
    results_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "results")

    # BC mode
    bc_p = sub.add_parser("bc", help="Behavioral cloning (top-quartile)")
    bc_p.add_argument("--results-dir", type=str, default=results_default)
    bc_p.add_argument("--gru-checkpoint", type=str, default=gru_default)
    bc_p.add_argument("--epochs", type=int, default=100)
    bc_p.add_argument("--lr", type=float, default=3e-4)
    bc_p.add_argument("--batch-size", type=int, default=64)
    bc_p.add_argument("--seed", type=int, default=42)
    bc_p.add_argument("--mask-gru", action="store_true",
                      help="Force state[22:27] = [0.2]*5 — §5.7 GRU-feature ablation.")

    # RWR mode
    rwr_p = sub.add_parser("rwr", help="Reward-weighted regression (all data)")
    rwr_p.add_argument("--results-dir", type=str, default=results_default)
    rwr_p.add_argument("--gru-checkpoint", type=str, default=gru_default)
    rwr_p.add_argument("--temperature", type=float, default=1.0,
                       help="RWR temperature (lower = more aggressive)")
    rwr_p.add_argument("--epochs", type=int, default=100)
    rwr_p.add_argument("--lr", type=float, default=3e-4)
    rwr_p.add_argument("--batch-size", type=int, default=64)
    rwr_p.add_argument("--seed", type=int, default=42)
    rwr_p.add_argument("--mask-gru", action="store_true",
                       help="Force state[22:27] = [0.2]*5 — §5.7 GRU-feature ablation.")

    # Eval mode
    eval_p = sub.add_parser("eval", help="Offline evaluation of trained policy")
    eval_p.add_argument("--checkpoint", type=str, required=True,
                        help="Path to policy_net checkpoint")
    eval_p.add_argument("--results-dir", type=str, default=results_default)
    eval_p.add_argument("--gru-checkpoint", type=str, default=gru_default)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--mask-gru", action="store_true",
                        help="Force state[22:27] = [0.2]*5 — §5.7 GRU-feature ablation.")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Load GRU for opponent weight predictions in state vector
    gru_model = None
    gru_path = os.path.abspath(args.gru_checkpoint)
    if os.path.exists(gru_path):
        gru_model, n_gru = load_negotiation_gru(gru_path)
        print(f"Loaded GRU ({n_gru:,} params) for state augmentation")
    else:
        print(f"WARNING: GRU not found at {gru_path} — dims 22-26 will be uniform")

    if args.mode == "bc":
        print("=" * 60)
        print("BEHAVIORAL CLONING — NegotiationPolicyNet")
        print("=" * 60)

        results_dir = os.path.abspath(args.results_dir)
        print(f"Loading trajectories from: {results_dir}")
        trajectories = load_all_trajectories(results_dir, gru_model=gru_model, mask_gru=getattr(args, "mask_gru", False))
        print(f"Loaded {len(trajectories)} samples from "
              f"{len(set(t.game_idx for t in trajectories))} games")

        if not trajectories:
            print("ERROR: No trajectories found!")
            return

        train_ds, val_ds, test_ds, split_info = build_bc_dataset(
            trajectories, seed=args.seed,
        )

        model, best_val_loss = train_bc(
            train_ds, val_ds,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        )

        # Offline eval on test set
        if len(test_ds) > 0:
            metrics = evaluate_policy_offline(model, test_ds)
            print_eval_results(metrics)
        else:
            metrics = {"best_val_loss": best_val_loss}
            print("\n  (No test set — skipping offline eval)")

        suffix = "_masked" if getattr(args, "mask_gru", False) else ""
        ckpt_path = os.path.join(ckpt_dir, f"policy_net_bc{suffix}.pt")
        save_policy_net(model, metrics, ckpt_path)
        print(f"\nCheckpoint saved to: {ckpt_path}")

    elif args.mode == "rwr":
        print("=" * 60)
        print("REWARD-WEIGHTED REGRESSION — NegotiationPolicyNet")
        print("=" * 60)

        results_dir = os.path.abspath(args.results_dir)
        print(f"Loading trajectories from: {results_dir}")
        trajectories = load_all_trajectories(results_dir, gru_model=gru_model, mask_gru=getattr(args, "mask_gru", False))
        print(f"Loaded {len(trajectories)} samples from "
              f"{len(set(t.game_idx for t in trajectories))} games")

        if not trajectories:
            print("ERROR: No trajectories found!")
            return

        train_ds, val_ds, test_ds, split_info = build_rwr_dataset(
            trajectories, seed=args.seed,
        )

        model, best_val_loss = train_rwr(
            train_ds, val_ds,
            temperature=args.temperature,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        )

        # Offline eval on test set
        if len(test_ds) > 0:
            metrics = evaluate_policy_offline(model, test_ds)
            print_eval_results(metrics)
        else:
            metrics = {"best_val_loss": best_val_loss}
            print("\n  (No test set — skipping offline eval)")

        suffix_parts = []
        if getattr(args, "mask_gru", False):
            suffix_parts.append("masked")
        if args.temperature != 1.0:
            suffix_parts.append(f"tau{args.temperature}")
        suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
        ckpt_path = os.path.join(ckpt_dir, f"policy_net_rwr{suffix}.pt")
        save_policy_net(model, metrics, ckpt_path)
        print(f"\nCheckpoint saved to: {ckpt_path}")

    elif args.mode == "eval":
        print("=" * 60)
        print("OFFLINE EVALUATION — NegotiationPolicyNet")
        print("=" * 60)

        model, n_params = load_policy_net(args.checkpoint)
        print(f"Loaded policy net ({n_params:,} params) from: {args.checkpoint}")

        results_dir = os.path.abspath(args.results_dir)
        trajectories = load_all_trajectories(results_dir, gru_model=gru_model, mask_gru=getattr(args, "mask_gru", False))
        print(f"Loaded {len(trajectories)} samples from "
              f"{len(set(t.game_idx for t in trajectories))} games")

        _, _, test_ds, split_info = build_policy_dataset(
            trajectories, seed=args.seed,
        )

        if len(test_ds) == 0:
            print("ERROR: No test samples!")
            return

        metrics = evaluate_policy_offline(model, test_ds)
        print_eval_results(metrics)


if __name__ == "__main__":
    main()
