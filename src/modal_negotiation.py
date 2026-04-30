"""
Modal runner for LLM negotiation experiments at scale.

Closes the three main paper limitations:
  1. Scale randomized-opener to N=50 (from N=20)
  2. Run randomized-opener ablations (oracle, random, uniform)
  3. GPT-4o-mini with randomized opener control

Each experiment runs in a separate Modal container, all in parallel.
No GPU needed — just LLM API calls via OpenRouter.

Setup:
    modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-...

Usage:
    # Run all 7 gap-closing experiments
    modal run belief_gru/modal_negotiation.py

    # Quick smoke test (2 games each)
    modal run belief_gru/modal_negotiation.py --quick

    # Single experiment
    modal run belief_gru/modal_negotiation.py --experiment opener_A

    # Custom model + games
    modal run belief_gru/modal_negotiation.py --model openai/gpt-4o-mini --games 30
"""

import modal
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
BELIEF_GRU_DIR = str(Path(__file__).resolve().parent)
CHECKPOINT_DIR = str(Path(__file__).resolve().parent / "checkpoints")

app = modal.App("belief-gru-negotiation")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "openai")
    .add_local_dir(BELIEF_GRU_DIR, remote_path="/root/belief_gru")
)

# Experiment definitions: each closes a specific paper limitation
EXPERIMENTS = {
    # Limitation 1: Scale randomized-opener from N=20 → N=50
    "opener_A": {
        "condition": "A",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": None,
        "description": "Condition A, randomized opener, N=50 (scale from N=20)",
    },
    "opener_B": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": None,
        "description": "Condition B, randomized opener, N=50 (scale from N=20)",
    },
    # Limitation 2: Randomized-opener ablations (never tested)
    "opener_B_oracle": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "oracle",
        "description": "Condition B + oracle ablation, randomized opener, N=50",
    },
    "opener_B_random": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "random",
        "description": "Condition B + random ablation, randomized opener, N=50",
    },
    "opener_B_uniform": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "uniform",
        "description": "Condition B + uniform ablation, randomized opener, N=50",
    },
    # Limitation 3: GPT-4o-mini with randomized opener
    "gpt4o_opener_A": {
        "condition": "A",
        "model": "openai/gpt-4o-mini",
        "games": 30,
        "randomize_opener": True,
        "ablation": None,
        "description": "GPT-4o-mini Condition A, randomized opener, N=30",
    },
    "gpt4o_opener_B": {
        "condition": "B",
        "model": "openai/gpt-4o-mini",
        "games": 30,
        "randomize_opener": True,
        "ablation": None,
        "description": "GPT-4o-mini Condition B, randomized opener, N=30",
    },

    # ---------------------------------------------------------------------
    # Phase 2: Limitations follow-up experiments (paper §5.7).
    # Each one closes one of the limitations introduced in the revision.
    # ---------------------------------------------------------------------

    # §5.7 #1: hard-accept-override sensitivity. Re-run A and B at randomized
    # opener without the final-round utility>0.15 auto-accept.
    "no_override_A": {
        "condition": "A",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": None,
        "accept_override": False,
        "description": "A no-override sensitivity (§5.7 #1)",
    },
    "no_override_B": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": None,
        "accept_override": False,
        "description": "B no-override sensitivity (§5.7 #1)",
    },

    # §5.7 #8: aggression-schedule ablation. B+Uniform with a constant
    # aggression parameter — isolates the "format effect" from the specific
    # decay schedule.
    "schedule_const_low": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "uniform",
        "schedule": "low_const",  # a = 0.55 throughout
        "description": "B+Uniform, constant low aggression a=0.55 (§5.7 #8)",
    },
    "schedule_const_mid": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "uniform",
        "schedule": "const",
        "schedule_kwargs": {"a_const": 0.70},
        "description": "B+Uniform, constant mid aggression a=0.70 (§5.7 #8)",
    },
    "schedule_const_high": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "uniform",
        "schedule": "high_const",  # a = 0.80 throughout
        "description": "B+Uniform, constant high aggression a=0.80 (§5.7 #8)",
    },

    # §4.7: bring Tier 3 models to N=200 to match Haiku.
    "tier3_opus_A": {
        "condition": "A",
        "model": "anthropic/claude-opus-4-6",
        "games": 200,
        "randomize_opener": True,
        "ablation": None,
        "description": "Opus 4.6 Condition A, N=200 (§4.7 power upgrade)",
    },
    "tier3_opus_B": {
        "condition": "B",
        "model": "anthropic/claude-opus-4-6",
        "games": 200,
        "randomize_opener": True,
        "ablation": None,
        "description": "Opus 4.6 Condition B, N=200 (§4.7 power upgrade)",
    },
    "tier3_gemini_A": {
        "condition": "A",
        "model": "google/gemini-3-pro-preview",
        "games": 200,
        "randomize_opener": True,
        "ablation": None,
        "description": "Gemini 3 Pro Condition A, N=200 (§4.7 power upgrade)",
    },
    "tier3_gemini_B": {
        "condition": "B",
        "model": "google/gemini-3-pro-preview",
        "games": 200,
        "randomize_opener": True,
        "ablation": None,
        "description": "Gemini 3 Pro Condition B, N=200 (§4.7 power upgrade)",
    },

    # §6 reframe: hand-coded opponent model (no learned params).
    # Tests whether the GRU's contribution can be replaced by a 15-line
    # analytical estimator, with a single tunable bias scalar calibrated
    # offline against logged GRU target utility.
    "heuristic_unbiased_B": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "heuristic",
        "description": "B + analytical heuristic OM, bias=0 (no learned params)",
    },
    "heuristic_calibrated_B": {
        "condition": "B",
        "model": "anthropic/claude-3.5-haiku",
        "games": 50,
        "randomize_opener": True,
        "ablation": "heuristic_calibrated",
        "description": "B + analytical heuristic OM, bias=0.55 (no learned params, calibrated)",
    },
}


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=7200,  # 2 hours per experiment
    retries=1,
)
def run_experiment(
    experiment_name: str,
    condition: str,
    model: str,
    games: int,
    randomize_opener: bool,
    ablation: str | None,
    max_rounds: int = 8,
    seed: int = 42,
    accept_override: bool = True,
    schedule: str = "decay",
    schedule_kwargs: dict | None = None,
) -> dict:
    """Run a single negotiation experiment on Modal."""
    import os
    import sys
    import json
    import numpy as np
    import torch

    sys.path.insert(0, "/root")

    from belief_gru.train_negotiation_gru import (
        NegotiationEnv, load_negotiation_gru, compute_utility,
        VARIABLES, BUYER_DIRECTION, SELLER_DIRECTION,
    )
    from belief_gru.negotiation import (
        LLMNegotiator, run_match, compute_pareto_frontier,
        compute_match_metrics,
    )
    from openai import OpenAI

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"  Condition: {condition} | Model: {model}")
    print(f"  Games: {games} | Opener: {'randomized' if randomize_opener else 'seller-first'}")
    if ablation:
        print(f"  Ablation: {ablation}")
    print(f"{'='*70}\n")

    # Load world model
    ckpt_path = "/root/belief_gru/checkpoints/negotiation_gru.pt"
    if os.path.exists(ckpt_path):
        world_model, n_params = load_negotiation_gru(ckpt_path)
        print(f"Loaded NegotiationGRU ({n_params:,} params)")
    else:
        print("WARNING: No checkpoint found, running without world model")
        world_model = None

    # LLM client
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
                             max_rounds=max_rounds, seed=game_seed)

        buyer_wm = world_model if condition in ("B", "C") else None
        seller_wm = world_model if condition == "C" else None

        buyer = LLMNegotiator(client, "buyer", buyer_weights, env, model,
                              world_model=buyer_wm, ablation=ablation,
                              accept_override=accept_override,
                              schedule=schedule, schedule_kwargs=schedule_kwargs)
        seller = LLMNegotiator(client, "seller", seller_weights, env, model,
                               world_model=seller_wm, ablation=ablation,
                               accept_override=accept_override,
                               schedule=schedule, schedule_kwargs=schedule_kwargs)

        opener = (game_rng.choice(["buyer", "seller"])
                  if randomize_opener else "seller")

        bw_str = ", ".join(f"{VARIABLES[j]}={buyer_weights[j]:.2f}" for j in range(5))
        sw_str = ", ".join(f"{VARIABLES[j]}={seller_weights[j]:.2f}" for j in range(5))
        print(f"\nGame {i+1}/{games} | Opens: {opener}")
        print(f"  Buyer:  [{bw_str}]")
        print(f"  Seller: [{sw_str}]")

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
                "log": {"buyer_weights": buyer_weights.tolist(),
                        "seller_weights": seller_weights.tolist(),
                        "opener": opener, "turns": []},
            })

    # Compute summary stats
    deals = [r for r in results if r["deal"]]
    deal_rate = len(deals) / len(results) if results else 0

    summary = {
        "experiment": experiment_name,
        "condition": condition,
        "model": model,
        "games": games,
        "randomize_opener": randomize_opener,
        "ablation": ablation or "gru",
        "seed": seed,
        "accept_override": accept_override,
        "schedule": schedule,
        "schedule_kwargs": schedule_kwargs or {},
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
        print(f"  Buyer utility: {np.mean(buyer_utils):.3f} +/- {np.std(buyer_utils):.3f}")
        print(f"  Seller utility:{np.mean(seller_utils):.3f} +/- {np.std(seller_utils):.3f}")
        print(f"  Joint surplus: {np.mean(joints):.3f}")
        print(f"  Pareto eff:    {np.mean(pareto_effs):.0%}")
        print(f"  Avg rounds:    {np.mean(rounds_used):.1f}")
        print(f"{'='*70}")

    # Serialize results for return
    serializable_results = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "log"}
        if "log" in r:
            entry["turns"] = r["log"]["turns"]
            entry["buyer_weights"] = r["log"]["buyer_weights"]
            entry["seller_weights"] = r["log"]["seller_weights"]
            entry["opener"] = r["log"]["opener"]
        serializable_results.append(entry)

    return {
        "summary": summary,
        "results": serializable_results,
    }


@app.local_entrypoint()
def main(
    experiment: str = "all",
    games: int = 0,
    model: str = "",
    seed: int = 42,
    max_rounds: int = 8,
    quick: bool = False,
):
    import json
    from datetime import datetime, timezone

    if quick:
        override_games = 2
    elif games > 0:
        override_games = games
    else:
        override_games = 0  # use default per experiment

    # Select experiments
    if experiment == "all":
        to_run = dict(EXPERIMENTS)
    elif experiment in EXPERIMENTS:
        to_run = {experiment: EXPERIMENTS[experiment]}
    else:
        # Support comma-separated list
        names = [e.strip() for e in experiment.split(",")]
        to_run = {n: EXPERIMENTS[n] for n in names if n in EXPERIMENTS}
        if not to_run:
            print(f"Unknown experiment(s): {experiment}")
            print(f"Available: {', '.join(EXPERIMENTS.keys())}")
            return

    print(f"Launching {len(to_run)} experiments on Modal...")
    if quick:
        print("QUICK MODE: 2 games per experiment")
    print()

    for name, cfg in to_run.items():
        g = override_games if override_games > 0 else cfg["games"]
        m = model if model else cfg["model"]
        print(f"  {name}: {cfg['description']}")
        print(f"    model={m}, games={g}, opener={'rand' if cfg['randomize_opener'] else 'seller'}")
        if cfg["ablation"]:
            print(f"    ablation={cfg['ablation']}")

    # Launch all experiments in parallel
    futures = {}
    for name, cfg in to_run.items():
        g = override_games if override_games > 0 else cfg["games"]
        m = model if model else cfg["model"]
        futures[name] = run_experiment.spawn(
            experiment_name=name,
            condition=cfg["condition"],
            model=m,
            games=g,
            randomize_opener=cfg["randomize_opener"],
            ablation=cfg["ablation"],
            max_rounds=max_rounds,
            seed=seed,
            accept_override=cfg.get("accept_override", True),
            schedule=cfg.get("schedule", "decay"),
            schedule_kwargs=cfg.get("schedule_kwargs"),
        )

    # Collect results as they complete
    all_results = {}
    for name, future in futures.items():
        print(f"\nWaiting for {name}...")
        result = future.get()
        all_results[name] = result
        s = result["summary"]
        if "buyer_share_mean" in s:
            print(f"  {name}: buyer_share={s['buyer_share_mean']:.1%} +/- "
                  f"{s['buyer_share_std']:.1%}, deals={s['n_deals']}/{s['games']}")
        else:
            print(f"  {name}: deal_rate={s['deal_rate']:.0%}")

    # Save results locally
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save per-experiment files (consistent with existing naming)
    for name, result in all_results.items():
        outfile = results_dir / f"{name}_n{result['summary']['games']}.json"
        with open(outfile, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved: {outfile.name}")

    # Save combined summary
    summary_file = results_dir / f"modal_negotiation_{ts}.json"
    summaries = {name: r["summary"] for name, r in all_results.items()}
    with open(summary_file, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nCombined summary: {summary_file.name}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'='*80}")
    print(f"{'Experiment':<22} {'Cond':>5} {'Model':<22} {'N':>4} "
          f"{'Deal%':>6} {'BuyerSh':>8} {'Pareto':>7}")
    print("-" * 80)

    for name in sorted(all_results.keys()):
        s = all_results[name]["summary"]
        m_short = s["model"].split("/")[-1][:20]
        bs = f"{s['buyer_share_mean']:.1%}" if "buyer_share_mean" in s else "N/A"
        pe = f"{s['pareto_efficiency_mean']:.0%}" if "pareto_efficiency_mean" in s else "N/A"
        print(f"{name:<22} {s['condition']:>5} {m_short:<22} {s['games']:>4} "
              f"{s['deal_rate']:>5.0%} {bs:>8} {pe:>7}")

    # Key comparisons
    print(f"\n{'='*80}")
    print("KEY TESTS")
    print(f"{'='*80}")

    if "opener_A" in all_results and "opener_B" in all_results:
        a = all_results["opener_A"]["summary"]
        b = all_results["opener_B"]["summary"]
        if "buyer_share_mean" in a and "buyer_share_mean" in b:
            diff = b["buyer_share_mean"] - a["buyer_share_mean"]
            print(f"\nRandomized-opener WM effect (B-A): {diff:+.1%}")
            print(f"  A: {a['buyer_share_mean']:.1%} +/- {a['buyer_share_std']:.1%}")
            print(f"  B: {b['buyer_share_mean']:.1%} +/- {b['buyer_share_std']:.1%}")
            if abs(diff) < 0.03:
                print("  >>> Confirms: WM effect is pure anchoring defense")
            else:
                print(f"  >>> {'WM helps' if diff > 0 else 'WM hurts'} even with randomized opener")

    if "gpt4o_opener_A" in all_results and "gpt4o_opener_B" in all_results:
        a = all_results["gpt4o_opener_A"]["summary"]
        b = all_results["gpt4o_opener_B"]["summary"]
        if "buyer_share_mean" in a and "buyer_share_mean" in b:
            diff = b["buyer_share_mean"] - a["buyer_share_mean"]
            print(f"\nGPT-4o-mini randomized-opener WM effect (B-A): {diff:+.1%}")
            print(f"  A: {a['buyer_share_mean']:.1%}")
            print(f"  B: {b['buyer_share_mean']:.1%}")

    # Compare ablations under randomized opener
    ablation_names = ["opener_B", "opener_B_oracle", "opener_B_random", "opener_B_uniform"]
    available_ablations = [n for n in ablation_names if n in all_results]
    if len(available_ablations) > 1:
        print(f"\nAblation comparison (all randomized opener):")
        for name in available_ablations:
            s = all_results[name]["summary"]
            abl = s["ablation"]
            if "buyer_share_mean" in s:
                print(f"  {abl:>8}: {s['buyer_share_mean']:.1%} +/- {s['buyer_share_std']:.1%}")
