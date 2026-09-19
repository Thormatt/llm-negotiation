#!/usr/bin/env python3
"""
compute_tables.py — recompute the headline numbers from the paper directly
from the bundled JSONs. This is a self-contained reproduction script: no
external dependencies beyond the Python standard library + numpy.

Usage:
    python3 scripts/compute_tables.py

Reproduces:
    Table 4   (N=50 seller-first, conditions A/B/C — only A/B here)
    Table 6   (Ablation A vs B with 4 weight sources, N=50 seller-first)
    Table 7   (Effect of opener randomization)
    Table 8   (Variance Haiku, randomized vs seller-first)
    Table 9   (Multi-model replication, N=50 randomized opener)
    Table 10  (Ablation under randomized opener, N=200)
    Table 11  (Cross-model: Haiku+OM vs frontier seller, N=50)
    Table 16  (Live LLMIO middleman, N=50)
    Table 17  (Live A/B test for neural policy, N=50)

Each table prints to stdout. Discrepancies vs the paper PDF should be
limited to the third decimal (rounding). Larger discrepancies indicate
either (a) you have the wrong source files or (b) the paper PDF is stale.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def load_games(path: Path, condition: str | None = None) -> list[dict]:
    """Return the per-game records inside a result JSON, agnostic to the
    two layouts we use:
      Layout A (n200_*, n50_*): top-level results -> {COND: [game, ...]}
      Layout B (opener_random_*, opener_*_n50): top-level results -> [game, ...]
                or top-level games -> [game, ...]
    """
    with path.open() as f:
        data = json.load(f)
    if "games" in data and isinstance(data["games"], list):
        return data["games"]
    res = data["results"]
    if isinstance(res, list):
        return res
    if isinstance(res, dict):
        if condition and condition in res:
            value = res[condition]
            return value if isinstance(value, list) else list(value.values())
        if len(res) == 1:
            (only_key,) = res.keys()
            value = res[only_key]
            return value if isinstance(value, list) else list(value.values())
    raise ValueError(f"Could not extract games from {path}")


def share_stats(games: list[dict]) -> tuple[float, float, int]:
    shares = [g["buyer_surplus_share"] for g in games if g.get("deal")]
    return (float(np.mean(shares)), float(np.std(shares, ddof=1)), len(shares))


def pareto_mean(games: list[dict]) -> float:
    p = [g["pareto_efficiency"] for g in games if g.get("deal")]
    return float(np.mean(p)) if p else float("nan")


def joint_mean(games: list[dict]) -> float:
    j = [g["joint_surplus"] for g in games if g.get("deal")]
    return float(np.mean(j)) if j else float("nan")


def fmt_row(label: str, games: list[dict]) -> str:
    m, s, n = share_stats(games)
    pe = pareto_mean(games)
    js = joint_mean(games)
    return f"  {label:<28} share={m*100:5.1f}% ± {s*100:4.1f}%   joint={js:.3f}   Pareto={pe*100:4.1f}%   N={n}"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def table_4_6():
    section("Table 4 + Table 6  —  N=50 seller-first (original A/B + ablations)")
    files = {
        "A (No OM)":     ("seller_first/opener_A_n50.json", "A"),
        "B + GRU":       ("seller_first/opener_B_n50.json", "B"),
        "B + Oracle":    ("seller_first/opener_B_oracle_n50.json", "B"),
        "B + Random":    ("seller_first/opener_B_random_n50.json", "B"),
        "B + Uniform":   ("seller_first/opener_B_uniform_n50.json", "B"),
    }
    for label, (rel, cond) in files.items():
        path = RES / rel
        if not path.exists():
            print(f"  {label:<28} [missing: {rel}]")
            continue
        games = load_games(path, cond)
        print(fmt_row(label, games))


def table_7_8_10():
    section("Tables 7 + 8 + 10  —  Randomized opener N=200 ablation gradient")
    files = {
        "A (No OM)":     ("primary/n200_A_haiku.json", "A"),
        "B + GRU":       ("primary/n200_B_gru_haiku.json", "B"),
        "B + Oracle":    ("primary/n200_B_oracle_haiku.json", "B"),
        "B + Random":    ("primary/n200_B_random_haiku.json", "B"),
        "B + Uniform":   ("primary/n200_B_uniform_haiku.json", "B"),
    }
    for label, (rel, cond) in files.items():
        path = RES / rel
        if not path.exists():
            print(f"  {label:<28} [missing: {rel}]")
            continue
        games = load_games(path, cond)
        print(fmt_row(label, games))


def table_9():
    section("Table 9  —  Multi-model replication (N=50 randomized opener per cell)")
    models = ["grok3", "grok3mini", "gemini3flash", "deepseekr1", "gpt5", "opus46", "gemini3pro"]
    print(f"  {'model':<14} {'cond':<6} {'share':>10} {'σ':>8} {'Pareto':>8}")
    for m in models:
        for cond in ("A", "B"):
            rel = f"multimodel/n50_{cond}{'_gru' if cond == 'B' else ''}_{m}.json"
            path = RES / rel
            if not path.exists():
                continue
            games = load_games(path, cond)
            mean, sd, n = share_stats(games)
            pe = pareto_mean(games) * 100
            print(f"  {m:<14} {cond:<6} {mean*100:9.1f}% {sd*100:7.1f}% {pe:7.1f}%")


def table_11():
    section("Table 11  —  Cross-model: Haiku+GRU buyer vs frontier seller (N=50)")
    sellers = ["grok3", "opus46", "gpt5", "gemini3pro"]
    for s in sellers:
        rel = f"crossmodel/n50_cross_haiku_vs_{s}.json"
        path = RES / rel
        if not path.exists():
            print(f"  vs {s:<14} [missing]")
            continue
        games = load_games(path, "B")
        print(fmt_row(f"Haiku+GRU vs {s}", games))


def table_16():
    section("Table 16  —  Live LLMIO middleman (N=50, Haiku-Haiku)")
    files = {
        "CoopBlend2P": "llmio_middleman/phase_b_live_claude-35-haiku_vs_claude-35-haiku_coop_n50_20260428_223059.json",
        "Nash2P":      "llmio_middleman/phase_b_live_claude-35-haiku_vs_claude-35-haiku_nash_n50_20260428_223105.json",
    }
    for label, rel in files.items():
        path = RES / rel
        if not path.exists():
            print(f"  {label:<14} [missing]")
            continue
        with path.open() as f:
            d = json.load(f)
        s = d["summary"]
        print(f"  {label:<14} share={s['share_mean']*100:5.2f}% ± {s['share_sd']*100:5.3f}%   "
              f"drift={s.get('share_drift_mean', float('nan')):.3f}   "
              f"L1_buyer={s.get('buyer_translation_l1_mean', float('nan')):.3f}")


def table_17():
    section("Table 17  —  Live A/B test for neural policy (N=50, randomized opener)")
    path = RES / "policy_ab/policy_ab_test.json"
    if not path.exists():
        print("  [missing policy_ab_test.json]")
        return
    with path.open() as f:
        d = json.load(f)
    for cond_name, games in d["results"].items():
        if not isinstance(games, list):
            continue
        print(fmt_row(cond_name, games))


def main():
    print(f"Reproducing paper headline tables from {RES}")
    table_4_6()
    table_7_8_10()
    table_9()
    table_11()
    table_16()
    table_17()
    print()
    print("Compare to paper PDF Tables 4, 6, 7, 8, 9, 10, 11, 16, 17.")


if __name__ == "__main__":
    main()
