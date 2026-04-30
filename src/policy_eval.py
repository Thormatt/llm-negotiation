"""
Policy Evaluation Suite — Offline and Online A/B testing.

Offline: Compare utility_at_proposal for policy vs heuristic on held-out test states.
Online:  Run live LLM games with policy_net as directive source, compare to heuristic baseline.

Usage:
  python3 -m belief_gru.policy_eval offline --checkpoint checkpoints/policy_net_bc.pt
  python3 -m belief_gru.policy_eval online  --checkpoint checkpoints/policy_net_bc.pt --games 50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import torch

from policy_data import load_all_trajectories, build_policy_dataset, POLICY_STATE_DIM
from policy_net import load_policy_net
from train_negotiation_gru import (
    NegotiationEnv, load_negotiation_gru, compute_utility,
    BUYER_DIRECTION, SELLER_DIRECTION, VARIABLES, MAX_ROUNDS,
)
from negotiation import (
    compute_target_proposal, compute_pareto_frontier, compute_match_metrics,
    LLMNegotiator, run_match, print_condition_summary,
    generate_negotiation_briefing,
)


# ===========================================================================
# Offline Evaluation
# ===========================================================================

def offline_eval(policy_model, results_dir, seed=42, gru_model=None):
    """Run offline evaluation: compare policy vs heuristic on test states."""
    from train_policy_net import evaluate_policy_offline, print_eval_results

    trajectories = load_all_trajectories(results_dir, gru_model=gru_model)
    if not trajectories:
        print("ERROR: No trajectories found!")
        return None

    print(f"Loaded {len(trajectories)} samples from "
          f"{len(set(t.game_idx for t in trajectories))} games")

    _, _, test_ds, split_info = build_policy_dataset(trajectories, seed=seed)
    if len(test_ds) == 0:
        print("ERROR: No test samples!")
        return None

    metrics = evaluate_policy_offline(policy_model, test_ds)
    print_eval_results(metrics)
    return metrics


# ===========================================================================
# Online A/B Test
# ===========================================================================

def run_online_ab_test(policy_model, gru_model, n_games=50, max_rounds=MAX_ROUNDS,
                       dirichlet_alpha=1.0, base_seed=42, model_name="anthropic/claude-3.5-haiku",
                       mask_gru=False, accept_override=True):
    """Run online A/B test: heuristic (B-heuristic) vs policy (B-policy).

    Both conditions use buyer+WM, but:
      - B-heuristic: compute_target_proposal() heuristic generates the directive
      - B-policy:    NegotiationPolicyNet generates the directive

    All games use randomized opener.

    Args:
        policy_model: NegotiationPolicyNet
        gru_model: NegotiationGRU (needed for both conditions — provides weight predictions)
        n_games: games per condition
        max_rounds: rounds per game
        dirichlet_alpha: Dirichlet alpha for weight sampling
        base_seed: random seed
        model_name: LLM model name for OpenRouter

    Returns:
        dict with results for both conditions
    """
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    conditions = {}

    for label, use_policy in [("B-heuristic", False), ("B-policy", True)]:
        print(f"\n{'#'*60}")
        print(f"# CONDITION: {label}")
        print(f"{'#'*60}")

        results = []
        rng = np.random.RandomState(base_seed)

        for i in range(n_games):
            seed = base_seed + i
            game_rng = np.random.RandomState(seed)

            buyer_weights = game_rng.dirichlet(np.ones(5) * dirichlet_alpha)
            seller_weights = game_rng.dirichlet(np.ones(5) * dirichlet_alpha)

            env = NegotiationEnv(buyer_weights, seller_weights,
                                 max_rounds=max_rounds, seed=seed)

            # Buyer always gets WM. The difference is policy_net vs heuristic.
            buyer = LLMNegotiatorWithPolicy(
                client, "buyer", buyer_weights, env, model_name,
                world_model=gru_model,
                policy_net=policy_model if use_policy else None,
                mask_gru_in_policy=mask_gru,
                accept_override=accept_override,
            )
            seller = LLMNegotiator(client, "seller", seller_weights, env, model_name,
                                   accept_override=accept_override)

            opener = game_rng.choice(["buyer", "seller"])

            bw_str = ", ".join(f"{VARIABLES[j]}={buyer_weights[j]:.2f}" for j in range(5))
            print(f"\n  Game {i+1}/{n_games} | Opens: {opener}")
            print(f"  Buyer weights: [{bw_str}]")

            result = run_match(buyer, seller, env, buyer_weights, seller_weights,
                               verbose=False, opener=opener)
            results.append(result)

            if result["deal"]:
                print(f"  DEAL: buyer_util={result['buyer_utility']:.3f} "
                      f"share={result['buyer_surplus_share']:.1%}")
            else:
                print(f"  NO DEAL")

        conditions[label] = results
        print_condition_summary(label, results)

    return conditions


class LLMNegotiatorWithPolicy(LLMNegotiator):
    """LLMNegotiator that optionally uses NegotiationPolicyNet for directives."""

    def __init__(self, client, role, weights, env, model_name,
                 world_model=None, ablation=None, policy_net=None,
                 mask_gru_in_policy=False, accept_override=True):
        super().__init__(client, role, weights, env, model_name,
                         world_model=world_model, ablation=ablation,
                         accept_override=accept_override)
        self.policy_net = policy_net
        self.mask_gru_in_policy = mask_gru_in_policy

    def decide(self, env):
        """Override to pass policy_net to generate_negotiation_briefing."""
        rounds_left = env.max_rounds - env.round
        opp_proposals = env.get_opponent_proposals(self.role)
        if self.accept_override and rounds_left <= 1 and opp_proposals:
            util = compute_utility(opp_proposals[-1], self.weights, self.role)
            if util > 0.15:
                return "accept", None, f"[Final-round auto-accept: utility {util:.3f} > 0.15]", None

        briefing = None
        if self.world_model is not None:
            briefing, _ = generate_negotiation_briefing(
                self.world_model, env, perspective=self.role,
                ablation=self.ablation,
                policy_net=self.policy_net,
                mask_gru_in_policy=self.mask_gru_in_policy,
            )

        user_msg = self._format_state(env)
        if briefing:
            user_msg += f"\n\n{briefing}"
        user_msg += "\n\nYour move. Respond with JSON only."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=400,
            )
            raw = response.choices[0].message.content.strip()

            text = raw
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            decision = json.loads(text)
            action = decision.get("action", "offer").lower()
            reasoning = decision.get("internal_monologue", "")

            if action == "accept":
                opp_proposals = env.get_opponent_proposals(self.role)
                if opp_proposals:
                    util = compute_utility(opp_proposals[-1], self.weights, self.role)
                    if util < 0.10:
                        action = "offer"
                        reasoning += " [Override: utility too low to accept]"
                    else:
                        return "accept", None, reasoning, briefing
                else:
                    action = "offer"
                    reasoning = "Nothing to accept yet"

            if action == "offer":
                fp = decision.get("formal_proposal", {})
                if fp:
                    proposal = np.array([
                        float(fp.get("price", 0.5)),
                        float(fp.get("contract_length", 0.5)),
                        float(fp.get("payment_terms", 0.5)),
                        float(fp.get("sla_tier", 0.5)),
                        float(fp.get("data_rights", 0.5)),
                    ], dtype=np.float32)
                else:
                    proposal = self._fallback_proposal(env)

                proposal = proposal.clip(0.05, 0.95)
                return "offer", proposal, reasoning, briefing

            proposal = self._fallback_proposal(env)
            return "offer", proposal, f"Unknown action '{action}', fallback", briefing

        except Exception as e:
            proposal = self._fallback_proposal(env)
            return "offer", proposal, f"LLM error ({e}), fallback", briefing


# ===========================================================================
# Statistical Tests
# ===========================================================================

def permutation_test(a_values, b_values, n_permutations=10000, seed=42):
    """Two-sided permutation test for difference in means.

    Returns:
        (observed_diff, p_value)
    """
    rng = np.random.RandomState(seed)
    a = np.array(a_values)
    b = np.array(b_values)

    observed_diff = b.mean() - a.mean()
    combined = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = combined[n_a:].mean() - combined[:n_a].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return observed_diff, p_value


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap confidence interval for the mean.

    Returns:
        (mean, ci_low, ci_high)
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    means = np.sort(means)
    alpha = (1 - ci) / 2
    ci_low = means[int(alpha * n_bootstrap)]
    ci_high = means[int((1 - alpha) * n_bootstrap)]
    return float(values.mean()), float(ci_low), float(ci_high)


def print_ab_comparison(conditions):
    """Print statistical comparison of A/B test results."""
    print(f"\n{'='*60}")
    print("A/B TEST STATISTICAL COMPARISON")
    print(f"{'='*60}")

    for label in ("B-heuristic", "B-policy"):
        results = conditions.get(label, [])
        deals = [r for r in results if r["deal"]]
        if not deals:
            print(f"\n  {label}: No deals!")
            continue

        shares = [r["buyer_surplus_share"] for r in deals]
        mean, ci_lo, ci_hi = bootstrap_ci(shares)
        print(f"\n  {label}:")
        print(f"    Deal rate:    {len(deals)}/{len(results)} ({len(deals)/len(results):.0%})")
        print(f"    Buyer share:  {mean:.1%} (95% CI: [{ci_lo:.1%}, {ci_hi:.1%}])")
        print(f"    Buyer share std: {np.std(shares):.1%}")
        print(f"    Pareto eff:   {np.mean([r['pareto_efficiency'] for r in deals]):.0%}")

    # Permutation test
    heur_deals = [r for r in conditions.get("B-heuristic", []) if r["deal"]]
    pol_deals = [r for r in conditions.get("B-policy", []) if r["deal"]]

    if heur_deals and pol_deals:
        heur_shares = [r["buyer_surplus_share"] for r in heur_deals]
        pol_shares = [r["buyer_surplus_share"] for r in pol_deals]

        diff, p_val = permutation_test(heur_shares, pol_shares)
        print(f"\n  Permutation test (policy - heuristic):")
        print(f"    Observed diff: {diff:+.1%}")
        print(f"    p-value:       {p_val:.4f}")
        print(f"    Significant:   {'YES' if p_val < 0.05 else 'NO'} (alpha=0.05)")

        # Success gate
        pol_mean = np.mean(pol_shares)
        pol_std = np.std(pol_shares)
        # +3.4pp means buyer_surplus_share >= 0.534 (50% baseline + 3.4pp)
        gate_share = pol_mean >= 0.534
        gate_std = pol_std <= 0.04

        print(f"\n  SUCCESS GATE:")
        print(f"    Buyer share >= 53.4%: {pol_mean:.1%} → {'PASS' if gate_share else 'FAIL'}")
        print(f"    Buyer share std <= 4%: {pol_std:.1%} → {'PASS' if gate_std else 'FAIL'}")
        if gate_share and gate_std:
            print(f"    >>> GATE PASSED: Policy meets deployment threshold!")
        else:
            print(f"    >>> GATE FAILED: Policy needs improvement")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Policy Evaluation Suite")
    sub = parser.add_subparsers(dest="mode", help="Evaluation mode")

    # Offline eval
    off_p = sub.add_parser("offline", help="Offline evaluation on test states")
    off_p.add_argument("--checkpoint", type=str, required=True)
    off_p.add_argument("--gru-checkpoint", type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "checkpoints", "negotiation_gru.pt"))
    off_p.add_argument("--results-dir", type=str,
                       default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "results"))
    off_p.add_argument("--seed", type=int, default=42)

    # Online A/B test
    on_p = sub.add_parser("online", help="Online A/B test with live LLM games")
    on_p.add_argument("--checkpoint", type=str, required=True,
                      help="Path to policy_net checkpoint")
    on_p.add_argument("--gru-checkpoint", type=str,
                      default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "checkpoints", "negotiation_gru.pt"))
    on_p.add_argument("--games", type=int, default=50,
                      help="Games per condition")
    on_p.add_argument("--rounds", type=int, default=8)
    on_p.add_argument("--model", type=str, default="anthropic/claude-3.5-haiku")
    on_p.add_argument("--seed", type=int, default=42)
    on_p.add_argument("--output", type=str, default=None,
                      help="Output path for results JSON")
    on_p.add_argument("--mask-gru", action="store_true",
                      help="Replace state[22:27] with [0.2]*5 at inference — "
                           "tests whether the policy is using the GRU prediction.")
    on_p.add_argument("--no-accept-override", action="store_true",
                      help="Disable the final-round utility>0.15 auto-accept "
                           "for §5.7 hard-accept-override sensitivity.")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    if args.mode == "offline":
        print("=" * 60)
        print("OFFLINE POLICY EVALUATION")
        print("=" * 60)

        policy_model, n_params = load_policy_net(args.checkpoint)
        print(f"Loaded policy net ({n_params:,} params) from: {args.checkpoint}")

        gru_model, n_gru_params = load_negotiation_gru(args.gru_checkpoint)
        print(f"GRU model: {n_gru_params:,} params from {args.gru_checkpoint}")

        offline_eval(policy_model, os.path.abspath(args.results_dir),
                     seed=args.seed, gru_model=gru_model)

    elif args.mode == "online":
        print("=" * 60)
        print("ONLINE A/B TEST: Policy vs Heuristic")
        print("=" * 60)

        policy_model, n_pol_params = load_policy_net(args.checkpoint)
        print(f"Policy net: {n_pol_params:,} params from {args.checkpoint}")

        gru_model, n_gru_params = load_negotiation_gru(args.gru_checkpoint)
        print(f"GRU model: {n_gru_params:,} params from {args.gru_checkpoint}")

        conditions = run_online_ab_test(
            policy_model, gru_model,
            n_games=args.games, max_rounds=args.rounds,
            base_seed=args.seed, model_name=args.model,
            mask_gru=getattr(args, "mask_gru", False),
            accept_override=not getattr(args, "no_accept_override", False),
        )

        print_ab_comparison(conditions)

        # Save results
        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "results",
                "policy_ab_test.json",
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        serializable = {}
        for label, results in conditions.items():
            serializable[label] = []
            for r in results:
                entry = {k: v for k, v in r.items() if k != "log"}
                entry["turns"] = r.get("log", {}).get("turns", [])
                entry["buyer_weights"] = r.get("log", {}).get("buyer_weights", [])
                entry["seller_weights"] = r.get("log", {}).get("seller_weights", [])
                serializable[label].append(entry)

        with open(output_path, "w") as f:
            json.dump({
                "policy_checkpoint": args.checkpoint,
                "gru_checkpoint": args.gru_checkpoint,
                "model": args.model,
                "games_per_condition": args.games,
                "max_rounds": args.rounds,
                "results": serializable,
            }, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
