"""
Modal runner: Haiku (no ML) vs frontier models.

Fills the missing comparison: how does a cheap model WITHOUT the GRU
fare against frontier models? Paired with existing cross-model data
(Haiku+ML vs frontier), this isolates the ML's contribution.

Setup:
    modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

Usage:
    modal run belief_gru/modal_crossmodel_baseline.py
    modal run belief_gru/modal_crossmodel_baseline.py --quick
"""

import modal
from pathlib import Path

BELIEF_GRU_DIR = str(Path(__file__).resolve().parent)
PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("crossmodel-baseline")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "openai")
    .add_local_dir(BELIEF_GRU_DIR, remote_path="/root/belief_gru")
)

# Haiku (no ML) buyer vs each frontier seller
# Restricted to the two models cited in paper §4.9 #4 (Grok 3, Opus 4.6).
# gemini-3-pro-preview is deprecated on OpenRouter; gpt-5 baseline is not paper-cited.
EXPERIMENTS = {
    "haiku_vs_grok3": {
        "buyer_model": "anthropic/claude-3.5-haiku",
        "seller_model": "x-ai/grok-3",
        "condition": "A",
        "games": 50,
    },
    "haiku_vs_opus": {
        "buyer_model": "anthropic/claude-3.5-haiku",
        "seller_model": "anthropic/claude-opus-4-6",
        "condition": "A",
        "games": 50,
    },
}


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=7200,
    retries=1,
)
def run_experiment(
    experiment_name: str,
    buyer_model: str,
    seller_model: str,
    condition: str,
    games: int,
    seed: int = 42,
) -> dict:
    """Run a single cross-model negotiation experiment."""
    import os
    import sys
    import numpy as np
    import torch

    sys.path.insert(0, "/root")

    from belief_gru.train_negotiation_gru import (
        NegotiationEnv, load_negotiation_gru, compute_utility,
        VARIABLES, BUYER_DIRECTION, SELLER_DIRECTION,
    )
    from belief_gru.negotiation import (
        LLMNegotiator, run_match, compute_pareto_frontier,
    )
    from openai import OpenAI

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  Buyer:  {buyer_model} (condition {condition})")
    print(f"  Seller: {seller_model}")
    print(f"  Games: {games}, randomized opener")
    print(f"{'='*70}\n")

    # Load world model (only used if condition B)
    world_model = None
    if condition in ("B", "C"):
        ckpt_path = "/root/belief_gru/checkpoints/negotiation_gru.pt"
        if os.path.exists(ckpt_path):
            world_model, n_params = load_negotiation_gru(ckpt_path)
            print(f"Loaded NegotiationGRU ({n_params:,} params)")

    api_key = os.environ["OPENROUTER_API_KEY"]
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    for i in range(games):
        game_seed = seed + i
        game_rng = np.random.RandomState(game_seed)

        buyer_weights = game_rng.dirichlet(np.ones(5) * 1.0)
        seller_weights = game_rng.dirichlet(np.ones(5) * 1.0)

        env = NegotiationEnv(buyer_weights, seller_weights,
                             max_rounds=8, seed=game_seed)

        buyer_wm = world_model if condition in ("B", "C") else None
        seller_wm = world_model if condition == "C" else None

        buyer = LLMNegotiator(client, "buyer", buyer_weights, env, buyer_model,
                              world_model=buyer_wm)
        seller = LLMNegotiator(client, "seller", seller_weights, env, seller_model,
                               world_model=seller_wm)

        opener = game_rng.choice(["buyer", "seller"])

        bw_str = ", ".join(f"{VARIABLES[j]}={buyer_weights[j]:.2f}" for j in range(5))
        sw_str = ", ".join(f"{VARIABLES[j]}={seller_weights[j]:.2f}" for j in range(5))
        print(f"\nGame {i+1}/{games} | Opens: {opener}")
        print(f"  Buyer [{buyer_model.split('/')[-1]}]:  [{bw_str}]")
        print(f"  Seller [{seller_model.split('/')[-1]}]: [{sw_str}]")

        try:
            result = run_match(buyer, seller, env, buyer_weights, seller_weights,
                               verbose=True, opener=opener)
            results.append(result)
        except Exception as e:
            print(f"  ERROR in game {i+1}: {e}")
            results.append({
                "deal": False, "buyer_utility": 0.0, "seller_utility": 0.0,
                "joint_surplus": 0.0, "pareto_efficiency": 0.0,
                "buyer_surplus_share": 0.0, "rounds": 0,
                "error": str(e),
            })

    deals = [r for r in results if r["deal"]]
    deal_rate = len(deals) / len(results) if results else 0

    summary = {
        "experiment": experiment_name,
        "buyer_model": buyer_model,
        "seller_model": seller_model,
        "condition": condition,
        "games": games,
        "seed": seed,
        "deal_rate": deal_rate,
        "n_deals": len(deals),
    }

    if deals:
        buyer_shares = [r["buyer_surplus_share"] for r in deals]
        buyer_utils = [r["buyer_utility"] for r in deals]
        seller_utils = [r["seller_utility"] for r in deals]
        joints = [r["joint_surplus"] for r in deals]
        pareto_effs = [r["pareto_efficiency"] for r in deals]
        rounds_used = [r["rounds"] for r in deals]

        summary.update({
            "buyer_share_mean": float(np.mean(buyer_shares)),
            "buyer_share_std": float(np.std(buyer_shares)),
            "buyer_utility_mean": float(np.mean(buyer_utils)),
            "seller_utility_mean": float(np.mean(seller_utils)),
            "joint_surplus_mean": float(np.mean(joints)),
            "pareto_efficiency_mean": float(np.mean(pareto_effs)),
            "avg_rounds": float(np.mean(rounds_used)),
        })

        print(f"\n{'='*70}")
        print(f"SUMMARY: {experiment_name}")
        print(f"  Deal rate:     {deal_rate:.0%} ({len(deals)}/{len(results)})")
        print(f"  Buyer share:   {np.mean(buyer_shares):.1%} +/- {np.std(buyer_shares):.1%}")
        print(f"  Pareto eff:    {np.mean(pareto_effs):.0%}")
        print(f"  Avg rounds:    {np.mean(rounds_used):.1f}")
        print(f"{'='*70}")

    return {"summary": summary}


@app.local_entrypoint()
def main(quick: bool = False):
    import json
    from datetime import datetime, timezone

    override_games = 2 if quick else 0

    print(f"Launching {len(EXPERIMENTS)} cross-model baseline experiments...")
    if quick:
        print("QUICK MODE: 2 games each\n")

    futures = {}
    for name, cfg in EXPERIMENTS.items():
        g = override_games if override_games > 0 else cfg["games"]
        print(f"  {name}: Haiku (no ML) vs {cfg['seller_model'].split('/')[-1]} x{g}")
        futures[name] = run_experiment.spawn(
            experiment_name=name,
            buyer_model=cfg["buyer_model"],
            seller_model=cfg["seller_model"],
            condition=cfg["condition"],
            games=g,
        )

    all_results = {}
    for name, future in futures.items():
        print(f"\nWaiting for {name}...")
        result = future.get()
        all_results[name] = result
        s = result["summary"]
        if "buyer_share_mean" in s:
            print(f"  {name}: buyer_share={s['buyer_share_mean']:.1%} +/- "
                  f"{s['buyer_share_std']:.1%}")

    # Save results
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    summaries = {name: r["summary"] for name, r in all_results.items()}
    outfile = results_dir / f"crossmodel_baseline_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nSaved: {outfile}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("CROSS-MODEL BASELINE: Haiku (no ML) buyer vs frontier sellers")
    print(f"{'='*80}")
    print(f"{'Seller':<20} {'Buyer Share':>12} {'Std':>8} {'Pareto':>8} {'Rounds':>8}")
    print("-" * 60)

    for name in sorted(all_results.keys()):
        s = all_results[name]["summary"]
        seller_short = s["seller_model"].split("/")[-1]
        if "buyer_share_mean" in s:
            print(f"{seller_short:<20} {s['buyer_share_mean']:>11.1%} "
                  f"{s['buyer_share_std']:>7.1%} "
                  f"{s['pareto_efficiency_mean']:>7.0%} "
                  f"{s['avg_rounds']:>7.1f}")

    # Print comparison with existing Haiku+ML cross-model data
    print(f"\n{'='*80}")
    print("COMPARISON: Haiku without ML vs Haiku with ML (vs same opponents)")
    print(f"{'='*80}")
    print(f"{'Seller':<20} {'No ML':>10} {'With ML':>10} {'ML Boost':>10}")
    print("-" * 55)

    # Existing Haiku+ML data from the paper
    haiku_ml_data = {
        "grok-3": 0.552,
        "claude-opus-4-6": 0.491,
        "gpt-5": 0.481,
        "gemini-3-pro-preview": 0.479,
    }

    for name in sorted(all_results.keys()):
        s = all_results[name]["summary"]
        seller_short = s["seller_model"].split("/")[-1]
        if "buyer_share_mean" in s and seller_short in haiku_ml_data:
            no_ml = s["buyer_share_mean"]
            with_ml = haiku_ml_data[seller_short]
            boost = with_ml - no_ml
            print(f"{seller_short:<20} {no_ml:>9.1%} {with_ml:>9.1%} {boost:>+9.1%}")
