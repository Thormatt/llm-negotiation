"""
RLHF vs Base Model Factorial Experiment.

Tests whether text-based advice backfires specifically in RLHF/instruct models
by comparing base vs instruct checkpoints under three strategic input formats.

Design: 2×3 factorial
  Model type:  {base, instruct}
  Format:      {none, text, numeric}

Key hypothesis: Text advice harms instruct models (RLHF cooperation trigger)
but not base models. Numeric directives bypass the trigger in both.

Usage:
  OPENROUTER_API_KEY=sk-... python -m belief_gru.run_alignment_experiment \
    --family llama --games 50

  OPENROUTER_API_KEY=sk-... python -m belief_gru.run_alignment_experiment \
    --family llama --games 2 --quick
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import time
import numpy as np
import torch
from pathlib import Path
from openai import OpenAI

from train_negotiation_gru import (
    NegotiationEnv, load_negotiation_gru, compute_utility,
    VARIABLES, BUYER_DIRECTION, SELLER_DIRECTION,
)
from negotiation import (
    LLMNegotiator, run_match, compute_match_metrics,
)

# Model families: base/instruct pairs
MODEL_FAMILIES = {
    "llama": {
        "base": "meta-llama/llama-3.1-405b",
        "instruct": "meta-llama/llama-3.1-405b-instruct",
        "label": "Llama 3.1 405B",
    },
    "deepseek": {
        "base": "deepseek/deepseek-v3-base",
        "instruct": "deepseek/deepseek-chat",
        "label": "DeepSeek V3",
    },
}

# Strategic format conditions
CONDITIONS = ["none", "text", "numeric"]

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_cell(
    client,
    gru_model,
    model_id: str,
    model_type: str,
    strategic_format: str,
    games: int,
    seed: int = 42,
    max_rounds: int = 8,
    verbose: bool = False,
) -> list[dict]:
    """Run one cell of the 2×3 factorial.

    Args:
        model_type: "base" or "instruct"
        strategic_format: "none", "text", or "numeric"
    """
    results = []

    for game_idx in range(games):
        game_seed = seed + game_idx
        rng = np.random.RandomState(game_seed)

        buyer_weights = rng.dirichlet(np.ones(5)).astype(np.float32)
        seller_weights = rng.dirichlet(np.ones(5)).astype(np.float32)
        opener = rng.choice(["buyer", "seller"])

        env = NegotiationEnv(buyer_weights, seller_weights, max_rounds=max_rounds)

        # Buyer gets the strategic input; seller is always unassisted
        buyer_wm = gru_model if strategic_format != "none" else None
        buyer_format = strategic_format if strategic_format != "none" else None

        buyer = LLMNegotiator(
            client=client,
            role="buyer",
            weights=buyer_weights,
            env=env,
            model_name=model_id,
            world_model=buyer_wm,
            ablation="uniform",  # Clean isolation: format effect only
            strategic_format=buyer_format,
            accept_override=False,  # No forced accept — deal collapse is real
        )
        seller = LLMNegotiator(
            client=client,
            role="seller",
            weights=seller_weights,
            env=env,
            model_name=model_id,
            world_model=None,
            strategic_format=None,
            accept_override=False,
        )

        try:
            result = run_match(buyer, seller, env, buyer_weights, seller_weights,
                               verbose=verbose, opener=opener)
            result["game_seed"] = game_seed
            result["opener"] = opener
            results.append(result)

            status = "DEAL" if result["deal"] else "NO DEAL"
            share = result.get("buyer_surplus_share", 0)
            if verbose or (game_idx + 1) % 10 == 0:
                print(f"  Game {game_idx+1}/{games}: {status} "
                      f"| buyer_share={share:.1%} | rounds={result.get('rounds', '?')}")

        except Exception as e:
            print(f"  Game {game_idx+1}/{games}: ERROR — {e}")
            results.append({
                "deal": False, "buyer_utility": 0.0, "seller_utility": 0.0,
                "joint_surplus": 0.0, "pareto_efficiency": 0.0,
                "buyer_surplus_share": 0.0, "rounds": max_rounds,
                "game_seed": game_seed, "error": str(e),
            })

    return results


def summarize_cell(results: list[dict], label: str) -> dict:
    """Compute summary statistics for one cell."""
    deals = [r for r in results if r.get("deal", False)]
    deal_rate = len(deals) / len(results) if results else 0
    n = len(results)

    if deals:
        shares = [r["buyer_surplus_share"] for r in deals]
        paretos = [r["pareto_efficiency"] for r in deals]
        rounds_list = [r["rounds"] for r in deals]
        joint = [r["joint_surplus"] for r in deals]

        mean_share = np.mean(shares)
        std_share = np.std(shares)
        mean_pareto = np.mean(paretos)
        mean_rounds = np.mean(rounds_list)
        mean_joint = np.mean(joint)
    else:
        mean_share = std_share = mean_pareto = mean_rounds = mean_joint = 0.0

    summary = {
        "label": label,
        "n": n,
        "deal_rate": deal_rate,
        "buyer_share_mean": float(mean_share),
        "buyer_share_std": float(std_share),
        "pareto_efficiency": float(mean_pareto),
        "joint_surplus": float(mean_joint),
        "avg_rounds": float(mean_rounds),
    }

    print(f"\n  {label}:")
    print(f"    Deal rate:        {len(deals)}/{n} ({deal_rate:.0%})")
    print(f"    Avg buyer share:  {mean_share:.1%} +/- {std_share:.1%}")
    print(f"    Pareto efficiency:{mean_pareto:.0%}")
    print(f"    Joint surplus:    {mean_joint:.3f}")
    print(f"    Avg rounds:       {mean_rounds:.1f}")

    return summary


def run_experiment(family_key: str, games: int, seed: int = 42,
                   quick: bool = False, verbose: bool = False):
    """Run full 2×3 factorial for one model family."""
    family = MODEL_FAMILIES[family_key]
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Load GRU (needed for text and numeric conditions)
    checkpoint_path = Path(__file__).parent / "checkpoints" / "negotiation_gru.pt"
    gru_model = load_negotiation_gru(str(checkpoint_path))
    print(f"Loaded NegotiationGRU from {checkpoint_path}")

    if quick:
        games = 2

    print(f"\n{'='*70}")
    print(f"RLHF vs BASE FACTORIAL EXPERIMENT")
    print(f"{'='*70}")
    print(f"Family:     {family['label']}")
    print(f"Base:       {family['base']}")
    print(f"Instruct:   {family['instruct']}")
    print(f"Games/cell: {games}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Accept override: DISABLED")
    print(f"Opener: randomized")
    print(f"{'='*70}\n")

    all_results = {}

    for model_type in ["base", "instruct"]:
        model_id = family[model_type]
        for fmt in CONDITIONS:
            cell_key = f"{model_type}_{fmt}"
            label = f"{model_type.title()} + {fmt.title()}"

            print(f"\n{'#'*70}")
            print(f"# CELL: {label} ({model_id})")
            print(f"{'#'*70}")

            t0 = time.time()
            results = run_cell(
                client=client,
                gru_model=gru_model,
                model_id=model_id,
                model_type=model_type,
                strategic_format=fmt,
                games=games,
                seed=seed,
                verbose=verbose,
            )
            elapsed = time.time() - t0

            summary = summarize_cell(results, label)
            summary["model_id"] = model_id
            summary["model_type"] = model_type
            summary["strategic_format"] = fmt
            summary["elapsed_seconds"] = elapsed

            all_results[cell_key] = {
                "summary": summary,
                "games": results,
            }

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    outfile = RESULTS_DIR / f"alignment_{family_key}_n{games}.json"
    with open(outfile, "w") as f:
        json.dump({
            "family": family,
            "games_per_cell": games,
            "seed": seed,
            "accept_override": False,
            "cells": {k: v["summary"] for k, v in all_results.items()},
            "raw": {k: v["games"] for k, v in all_results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {outfile}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: {family['label']}")
    print(f"{'='*70}")
    print(f"{'Cell':<25} {'Deal%':>6} {'Share':>8} {'σ':>8} {'Pareto':>7} {'Rounds':>7}")
    print("-" * 70)
    for cell_key in sorted(all_results.keys()):
        s = all_results[cell_key]["summary"]
        print(f"{s['label']:<25} {s['deal_rate']:>5.0%} "
              f"{s['buyer_share_mean']:>7.1%} {s['buyer_share_std']:>7.1%} "
              f"{s['pareto_efficiency']:>6.0%} {s['avg_rounds']:>6.1f}")

    # Compute difference-in-differences
    print(f"\n{'='*70}")
    print(f"DIFFERENCE-IN-DIFFERENCES")
    print(f"{'='*70}")
    for fmt in ["text", "numeric"]:
        instruct_none = all_results.get("instruct_none", {}).get("summary", {})
        instruct_fmt = all_results.get(f"instruct_{fmt}", {}).get("summary", {})
        base_none = all_results.get("base_none", {}).get("summary", {})
        base_fmt = all_results.get(f"base_{fmt}", {}).get("summary", {})

        if all([instruct_none, instruct_fmt, base_none, base_fmt]):
            instruct_diff = instruct_fmt.get("buyer_share_mean", 0) - instruct_none.get("buyer_share_mean", 0)
            base_diff = base_fmt.get("buyer_share_mean", 0) - base_none.get("buyer_share_mean", 0)
            did = instruct_diff - base_diff
            print(f"\n  {fmt.title()} effect:")
            print(f"    Instruct ({fmt} - none): {instruct_diff:+.1%}")
            print(f"    Base     ({fmt} - none): {base_diff:+.1%}")
            print(f"    DiD (instruct - base):   {did:+.1%}")
            if fmt == "text" and did < -0.02:
                print(f"    >>> SUPPORTS H1: Text uniquely backfires under RLHF")
            elif fmt == "text" and did >= 0:
                print(f"    >>> DOES NOT SUPPORT H1: Text effect similar across alignment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLHF vs Base factorial experiment")
    parser.add_argument("--family", choices=list(MODEL_FAMILIES.keys()),
                        default="llama", help="Model family")
    parser.add_argument("--games", type=int, default=50, help="Games per cell")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 games/cell)")
    parser.add_argument("--verbose", action="store_true", help="Print every game")
    args = parser.parse_args()

    run_experiment(
        family_key=args.family,
        games=args.games,
        seed=args.seed,
        quick=args.quick,
        verbose=args.verbose,
    )
