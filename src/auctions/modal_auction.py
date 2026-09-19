"""
Modal runner: live LLM bidding under conditions A0, A1, A2 vs equilibrium A3.

Each experiment runs `n_auctions` auctions with the same valuation seeds across
conditions, so comparisons are paired (variance reduction across conditions
is honest, not driven by different valuation draws).

Usage:
    modal run auctions/modal_auction.py
    modal run auctions/modal_auction.py --experiment a0,a1,a2 --n-auctions 200
    modal run auctions/modal_auction.py --quick                 # 5 auctions per condition
"""

from __future__ import annotations

from pathlib import Path

import modal

AUCTIONS_DIR = str(Path(__file__).resolve().parent)
PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("auctions-fpsb")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "openai")
    .add_local_dir(AUCTIONS_DIR, remote_path="/root/auctions")
)


CONDITIONS = ("A0", "A1", "A2", "A3")
DEFAULT_MODEL = "anthropic/claude-3.5-haiku"


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=7200,
    retries=1,
)
def run_condition(
    condition: str,
    model: str,
    n_auctions: int,
    seed: int = 42,
) -> dict:
    """Run a fixed condition for N=2 sealed-bid first-price auctions.

    Bidder 0 plays the condition; bidder 1 always plays the equilibrium.
    This isolates the focal bidder's strategy effect — the analogue of the
    negotiation paper's Haiku-vs-Haiku setup with one side scaffolded.
    """
    import os
    import sys
    import numpy as np

    sys.path.insert(0, "/root/auctions")

    from auction_env import (
        FirstPriceAuction,
        equilibrium_bid,
        expected_equilibrium_revenue,
        expected_equilibrium_winner_surplus,
    )
    from equilibrium_bidder import EquilibriumBidder

    print(f"\n{'='*70}\nCONDITION: {condition}  model={model}  N={n_auctions}\n{'='*70}\n")

    # Bidder 1 is always the equilibrium opponent.
    opponent = EquilibriumBidder(n=2)

    if condition == "A3":
        focal = EquilibriumBidder(n=2)
        client = None
    else:
        from openai import OpenAI
        from llm_bidder import LLMBidder
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        focal = LLMBidder(client=client, model_name=model, condition=condition, n=2)

    env = FirstPriceAuction(n=2, seed=seed)
    games = []
    api_errors = 0
    parse_errors = 0

    for i in range(n_auctions):
        v = env.sample_valuations()  # (2,) float64

        if condition == "A3":
            b0 = focal.bid(float(v[0]))
            decision = None
        else:
            decision = focal.bid(float(v[0]))
            b0 = decision.bid
            if decision.error == "parse_failure":
                parse_errors += 1
            elif decision.error:
                api_errors += 1

        b1 = opponent.bid(float(v[1]))
        bids = np.array([b0, b1])
        outcome = env.resolve(v, bids)

        eq_b0 = equilibrium_bid(float(v[0]), n=2)
        games.append({
            "i": i,
            "valuations": v.tolist(),
            "bids": bids.tolist(),
            "winner": outcome.winner,
            "revenue": outcome.revenue,
            "payoffs": outcome.payoffs.tolist(),
            "alloc_efficient": outcome.allocation_efficient,
            "focal_bid_minus_equilibrium": float(b0 - eq_b0),
            "focal_reasoning": (decision.reasoning if decision else "[A3]"),
            "focal_error": (decision.error if decision else None),
        })

        if (i + 1) % 10 == 0:
            recent = games[-10:]
            print(f"  Auction {i+1}/{n_auctions}  "
                  f"recent rev={np.mean([g['revenue'] for g in recent]):.3f}  "
                  f"focal_payoff={np.mean([g['payoffs'][0] for g in recent]):.3f}  "
                  f"errors={api_errors}p={parse_errors}")

    # Aggregate metrics
    revenues = np.array([g["revenue"] for g in games])
    focal_pay = np.array([g["payoffs"][0] for g in games])
    opp_pay = np.array([g["payoffs"][1] for g in games])
    eff = np.array([1.0 if g["alloc_efficient"] else 0.0 for g in games])
    deviations = np.array([g["focal_bid_minus_equilibrium"] for g in games])

    summary = {
        "condition": condition,
        "model": model,
        "n_auctions": n_auctions,
        "seed": seed,
        "api_errors": api_errors,
        "parse_errors": parse_errors,
        "revenue_mean": float(revenues.mean()),
        "revenue_std": float(revenues.std()),
        "focal_payoff_mean": float(focal_pay.mean()),
        "focal_payoff_std": float(focal_pay.std()),
        "opponent_payoff_mean": float(opp_pay.mean()),
        "alloc_efficient_rate": float(eff.mean()),
        "focal_bid_dev_mean": float(deviations.mean()),
        "focal_bid_dev_std": float(deviations.std()),
        "theoretical_revenue": expected_equilibrium_revenue(2),
        "theoretical_winner_surplus": expected_equilibrium_winner_surplus(2),
    }

    print(f"\nSUMMARY {condition}:")
    print(f"  Revenue:           {summary['revenue_mean']:.3f} ± {summary['revenue_std']:.3f}  "
          f"(theory ≈ 0.333)")
    print(f"  Focal payoff:      {summary['focal_payoff_mean']:.3f} ± {summary['focal_payoff_std']:.3f}")
    print(f"  Bid deviation σ:   {summary['focal_bid_dev_std']:.3f}  (lower = more consistent)")
    print(f"  Allocation eff:    {summary['alloc_efficient_rate']:.1%}")
    print(f"  Errors:            api={api_errors}  parse={parse_errors}")

    return {"summary": summary, "games": games}


@app.local_entrypoint()
def main(
    experiment: str = "all",
    n_auctions: int = 200,
    model: str = DEFAULT_MODEL,
    seed: int = 42,
    quick: bool = False,
):
    import json
    from datetime import datetime, timezone

    if quick:
        n_auctions = 5

    if experiment == "all":
        to_run = list(CONDITIONS)
    else:
        to_run = [c.strip().upper() for c in experiment.split(",")]
        for c in to_run:
            if c not in CONDITIONS:
                print(f"Unknown condition: {c}. Choose from {CONDITIONS} or 'all'.")
                return

    print(f"Launching {len(to_run)} conditions on Modal: {to_run}")
    print(f"Model: {model}   N: {n_auctions}   seed: {seed}")
    if quick:
        print("QUICK MODE")
    print()

    futures = {}
    for cond in to_run:
        futures[cond] = run_condition.spawn(
            condition=cond,
            model=model,
            n_auctions=n_auctions,
            seed=seed,
        )

    results = {}
    for cond, future in futures.items():
        print(f"\nWaiting for {cond}...")
        results[cond] = future.get()
        s = results[cond]["summary"]
        print(f"  {cond}: rev={s['revenue_mean']:.3f}±{s['revenue_std']:.3f} "
              f"focal_payoff={s['focal_payoff_mean']:.3f}±{s['focal_payoff_std']:.3f} "
              f"σ(bid-eq)={s['focal_bid_dev_std']:.3f}")

    out_dir = PROJECT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outfile = out_dir / f"auction_fpsb_{ts}.json"

    with open(outfile, "w") as f:
        json.dump({c: r["summary"] for c, r in results.items()}, f, indent=2)
    print(f"\nSaved summary: {outfile}")

    # Save full per-game data
    for cond, r in results.items():
        with open(out_dir / f"auction_fpsb_{cond}_{ts}.json", "w") as f:
            json.dump(r, f, indent=2)
    print(f"Saved per-condition logs to {out_dir}/auction_fpsb_*_{ts}.json")

    # Comparison table
    print(f"\n{'='*80}\nCOMPARISON: focal bidder vs equilibrium opponent\n{'='*80}")
    print(f"{'Cond':<6} {'rev mean':>10} {'rev σ':>8} {'focal pay':>10} "
          f"{'focal pay σ':>12} {'σ(bid-eq)':>11} {'eff':>6}")
    print("-" * 80)
    for c in to_run:
        s = results[c]["summary"]
        print(f"{c:<6} {s['revenue_mean']:>10.3f} {s['revenue_std']:>8.3f} "
              f"{s['focal_payoff_mean']:>10.3f} {s['focal_payoff_std']:>12.3f} "
              f"{s['focal_bid_dev_std']:>11.3f} {s['alloc_efficient_rate']:>6.1%}")
