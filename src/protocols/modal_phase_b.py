"""
Phase B live experiment: LLMIO + middleman protocol on the negotiation domain.

For each game:
  1. Sample buyer and seller true weight vectors (Dirichlet(1.0)) using the
     paper's seed protocol so we replay the same draws used in the paper.
  2. Render each side's true weights as a structured natural-language
     preference description.
  3. Each side's LLMIO translates the description back into a weight vector.
  4. Submit both translated weight vectors to NashBargaining2P (or
     CooperativeBlend2P, configurable).
  5. Compute the outcome and the translation-fidelity error.

Measures: outcome distribution, translation accuracy per model, channel
comparison vs paper-logged LLM-LLM Condition A.

Usage:
    modal run protocols/modal_phase_b.py
    modal run protocols/modal_phase_b.py --buyer-model anthropic/claude-3.5-haiku --seller-model anthropic/claude-3.5-haiku --n-games 50
    modal run protocols/modal_phase_b.py --quick
"""

from __future__ import annotations

from pathlib import Path

import modal

PROTOCOLS_DIR = str(Path(__file__).resolve().parent)
FRAMEWORK_DIR = str(Path(__file__).resolve().parent.parent / "framework")
PROJECT_DIR = Path(__file__).resolve().parent.parent

app = modal.App("phase-b-middleman")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "openai")
    .add_local_dir(PROTOCOLS_DIR, remote_path="/root/protocols")
    .add_local_dir(FRAMEWORK_DIR, remote_path="/root/framework")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=7200,
    retries=1,
)
def run_phase_b(
    buyer_model: str,
    seller_model: str,
    n_games: int,
    protocol_name: str = "nash",
    seed: int = 42,
) -> dict:
    """One sweep of N games with the LLMIO + middleman pipeline."""
    import os
    import sys
    import numpy as np

    sys.path.insert(0, "/root")

    from protocols.llm_io_negotiation import (
        VARIABLES,
        render_preferences,
        translate_intent,
    )
    from protocols.nash_bargaining import CooperativeBlend2P, NashBargaining2P
    from openai import OpenAI

    BUYER_DIRECTION = np.array([-1, +1, -1, +1, +1], dtype=np.float64)
    SELLER_DIRECTION = np.array([+1, -1, +1, -1, -1], dtype=np.float64)

    if protocol_name == "nash":
        proto = NashBargaining2P()
    elif protocol_name == "coop":
        proto = CooperativeBlend2P()
    else:
        raise ValueError(f"unknown protocol: {protocol_name}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    rng = np.random.default_rng(seed)
    games = []
    api_errors = 0
    parse_errors = 0

    print(f"\n{'='*70}")
    print(f"Phase B live LLMIO + {proto.name}")
    print(f"  buyer:  {buyer_model}")
    print(f"  seller: {seller_model}")
    print(f"  N={n_games}  seed={seed}")
    print(f"{'='*70}\n")

    for i in range(n_games):
        wb_true = rng.dirichlet(np.ones(5))
        ws_true = rng.dirichlet(np.ones(5))

        b_intent = render_preferences(wb_true, "buyer")
        s_intent = render_preferences(ws_true, "seller")

        b_xlat = translate_intent(client, buyer_model, b_intent)
        s_xlat = translate_intent(client, seller_model, s_intent)

        for x in (b_xlat, s_xlat):
            if x.error == "parse_failure":
                parse_errors += 1
            elif x.error:
                api_errors += 1

        out = proto.run([
            {"weights": b_xlat.weights, "direction": BUYER_DIRECTION},
            {"weights": s_xlat.weights, "direction": SELLER_DIRECTION},
        ])
        ub, us = out.payoffs

        # Translation fidelity = L1 distance between true and LLMIO-output weights
        b_l1 = float(np.abs(wb_true - b_xlat.weights).sum())
        s_l1 = float(np.abs(ws_true - s_xlat.weights).sum())

        # Reference: outcome had we submitted true weights directly
        ref = proto.run([
            {"weights": wb_true, "direction": BUYER_DIRECTION},
            {"weights": ws_true, "direction": SELLER_DIRECTION},
        ])
        ub_ref, us_ref = ref.payoffs

        share = ub / (ub + us) if (ub + us) > 1e-12 else 0.5
        share_ref = ub_ref / (ub_ref + us_ref) if (ub_ref + us_ref) > 1e-12 else 0.5

        games.append({
            "i": i,
            "buyer_true": wb_true.tolist(),
            "seller_true": ws_true.tolist(),
            "buyer_xlat": b_xlat.weights.tolist(),
            "seller_xlat": s_xlat.weights.tolist(),
            "buyer_l1": b_l1,
            "seller_l1": s_l1,
            "proposal": out.allocation["proposal"],
            "buyer_util": ub,
            "seller_util": us,
            "share": share,
            "share_ref": share_ref,
            "buyer_xlat_error": b_xlat.error,
            "seller_xlat_error": s_xlat.error,
        })

        if (i + 1) % 10 == 0:
            recent = games[-10:]
            mean_share = np.mean([g["share"] for g in recent])
            mean_l1 = np.mean([g["buyer_l1"] + g["seller_l1"] for g in recent]) / 2
            print(f"  Game {i+1}/{n_games}  recent share={mean_share:.3f}  "
                  f"avg L1={mean_l1:.3f}  errors api={api_errors} parse={parse_errors}")

    shares = np.array([g["share"] for g in games])
    shares_ref = np.array([g["share_ref"] for g in games])
    l1_buyer = np.array([g["buyer_l1"] for g in games])
    l1_seller = np.array([g["seller_l1"] for g in games])

    summary = {
        "buyer_model": buyer_model,
        "seller_model": seller_model,
        "protocol": proto.name,
        "n_games": n_games,
        "seed": seed,
        "api_errors": api_errors,
        "parse_errors": parse_errors,
        "share_mean": float(shares.mean()),
        "share_sd": float(shares.std()),
        "share_ref_mean": float(shares_ref.mean()),
        "share_ref_sd": float(shares_ref.std()),
        "share_drift_mean": float(np.abs(shares - shares_ref).mean()),
        "buyer_translation_l1_mean": float(l1_buyer.mean()),
        "buyer_translation_l1_sd": float(l1_buyer.std()),
        "seller_translation_l1_mean": float(l1_seller.mean()),
        "seller_translation_l1_sd": float(l1_seller.std()),
    }

    print(f"\nSUMMARY:")
    print(f"  Share (LLMIO):  {summary['share_mean']:.3f} ± {summary['share_sd']:.3f}")
    print(f"  Share (true):   {summary['share_ref_mean']:.3f} ± {summary['share_ref_sd']:.3f}")
    print(f"  Drift |LLMIO−true|: {summary['share_drift_mean']:.3f} per game")
    print(f"  Translation L1: buyer {summary['buyer_translation_l1_mean']:.3f}, "
          f"seller {summary['seller_translation_l1_mean']:.3f}")
    print(f"  Errors: api={api_errors}, parse={parse_errors}")

    return {"summary": summary, "games": games}


@app.local_entrypoint()
def main(
    buyer_model: str = "anthropic/claude-3.5-haiku",
    seller_model: str = "anthropic/claude-3.5-haiku",
    n_games: int = 50,
    protocol: str = "nash",
    seed: int = 42,
    quick: bool = False,
):
    import json
    from datetime import datetime, timezone

    if quick:
        n_games = 5

    print(f"Launching Phase B live: buyer={buyer_model}  seller={seller_model}")
    print(f"  protocol={protocol}  N={n_games}  seed={seed}")
    if quick:
        print("  QUICK MODE")
    print()

    future = run_phase_b.spawn(
        buyer_model=buyer_model,
        seller_model=seller_model,
        n_games=n_games,
        protocol_name=protocol,
        seed=seed,
    )
    result = future.get()

    out_dir = PROJECT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_b = buyer_model.split("/")[-1].replace(".", "")
    safe_s = seller_model.split("/")[-1].replace(".", "")
    outfile = out_dir / f"phase_b_live_{safe_b}_vs_{safe_s}_{protocol}_n{n_games}_{ts}.json"

    with open(outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {outfile}")
