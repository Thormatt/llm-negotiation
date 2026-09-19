"""
Post-hoc analysis on saved auction results.

Inputs: per-condition JSON files written by modal_auction.py
        (e.g. results/auction_fpsb_A0_20260428_xxxxxx.json)

Outputs: a comparison table with the three signatures we expect from the
         negotiation paper applied to FPSB:
            (1) variance collapse: σ(bid - equilibrium) by condition
            (2) payoff approach to A3 / theoretical
            (3) deviation pattern (over- or under-bidding)

Usage:
    python3 analyze_auction.py results/auction_fpsb_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def load_run(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summarize(games: list[dict]) -> dict:
    if not games:
        return {"n": 0}
    arr = lambda k: np.array([g[k] for g in games])
    bids = np.array([g["bids"][0] for g in games])
    valuations = np.array([g["valuations"][0] for g in games])
    eq_bids = valuations * 0.5  # n=2 equilibrium
    revenues = arr("revenue")
    payoffs = np.array([g["payoffs"][0] for g in games])
    eff = np.array([1 if g["alloc_efficient"] else 0 for g in games])
    devs = bids - eq_bids
    return {
        "n": len(games),
        "rev_mean": float(revenues.mean()),
        "rev_std": float(revenues.std()),
        "focal_payoff_mean": float(payoffs.mean()),
        "focal_payoff_std": float(payoffs.std()),
        "bid_dev_mean": float(devs.mean()),
        "bid_dev_std": float(devs.std()),
        "alloc_eff": float(eff.mean()),
        "bid_to_value_mean": float(np.mean(bids / np.maximum(valuations, 1e-9))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="JSON files from modal_auction.py")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files.extend(glob.glob(p))
    files = sorted(set(files))

    if not files:
        print("No files found. Pass paths or globs to per-condition JSON files.")
        return

    rows = []
    for path in files:
        run = load_run(path)
        if "games" in run:
            cond = run.get("summary", {}).get("condition", Path(path).stem)
            stats = summarize(run["games"])
        else:
            # might be a summary-only file
            s = run if "condition" in run else next(iter(run.values()))
            cond = s.get("condition", Path(path).stem)
            stats = {
                "n": s.get("n_auctions", 0),
                "rev_mean": s.get("revenue_mean"),
                "rev_std": s.get("revenue_std"),
                "focal_payoff_mean": s.get("focal_payoff_mean"),
                "focal_payoff_std": s.get("focal_payoff_std"),
                "bid_dev_mean": s.get("focal_bid_dev_mean"),
                "bid_dev_std": s.get("focal_bid_dev_std"),
                "alloc_eff": s.get("alloc_efficient_rate"),
                "bid_to_value_mean": None,
            }
        rows.append((cond, path, stats))

    rows.sort(key=lambda r: r[0])
    print(f"{'Cond':<6} {'N':>5} {'rev mean':>10} {'rev σ':>8} "
          f"{'payoff':>10} {'payoff σ':>10} {'σ(b−eq)':>10} {'eff':>7}")
    print("-" * 80)
    for cond, _, s in rows:
        if s["n"] == 0:
            continue
        print(f"{cond:<6} {s['n']:>5} "
              f"{s['rev_mean']:>10.3f} {s['rev_std']:>8.3f} "
              f"{s['focal_payoff_mean']:>10.3f} {s['focal_payoff_std']:>10.3f} "
              f"{s['bid_dev_std']:>10.3f} {s['alloc_eff']:>6.1%}")

    print()
    print("Reading: lower σ(b−eq) means tighter approach to equilibrium bidding.")
    print("If A2 has σ(b−eq) ≪ A0, A1 with payoff ≈ A3, the architectural")
    print("pattern from the negotiation paper generalises to auctions.")


if __name__ == "__main__":
    main()
