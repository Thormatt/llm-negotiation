"""
Permutation tests + Holm-Bonferroni correction for the N=200 ablation gradient.

Loads per-game buyer_surplus_share from results/n200_*_haiku.json and runs pairwise
two-sided permutation tests across A (no OM) / B+Uniform / B+Random / B+GRU / B+Oracle.

Output: markdown table with mean diffs, raw p-values, and Holm-corrected p-values.
This addresses paper §4.8 / §5.7 multiple-comparisons concern.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CONDITIONS = {
    "A (No OM)":      ("n200_A_haiku.json",        "A"),
    "B + Uniform":    ("n200_B_uniform_haiku.json", "B"),
    "B + Random":     ("n200_B_random_haiku.json",  "B"),
    "B + GRU":        ("n200_B_gru_haiku.json",     "B"),
    "B + Oracle":     ("n200_B_oracle_haiku.json",  "B"),
}

N_PERMUTATIONS = 10_000
RNG_SEED = 42


def load_buyer_share(filename: str, cond_key: str) -> np.ndarray:
    path = RESULTS_DIR / filename
    with path.open() as f:
        data = json.load(f)
    games = data["results"][cond_key]
    return np.array([g["buyer_surplus_share"] for g in games], dtype=np.float64)


def permutation_test(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Two-sided permutation test on the difference of means."""
    observed = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)
    n_total = len(pooled)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_x].mean() - pooled[n_x:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    return (count + 1) / (n_perm + 1)  # +1 smoothing


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction."""
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda t: t[1])
    corrected = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        corrected[orig_idx] = running_max
    return corrected


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    samples = {name: load_buyer_share(fname, key) for name, (fname, key) in CONDITIONS.items()}

    print(f"Loaded {len(samples)} conditions, n={len(next(iter(samples.values())))} games each.\n")
    print("Sample means and stds:")
    for name, arr in samples.items():
        print(f"  {name:<14} mean={arr.mean()*100:.2f}%  sd={arr.std(ddof=1)*100:.2f}%  n={len(arr)}")

    pairs = list(combinations(samples.keys(), 2))
    rows = []
    for a, b in pairs:
        x, y = samples[a], samples[b]
        diff = (x.mean() - y.mean()) * 100  # percentage points
        p = permutation_test(x, y, N_PERMUTATIONS, rng)
        rows.append((a, b, diff, p))

    raw_p = [r[3] for r in rows]
    holm_p = holm_correction(raw_p)

    print(f"\n## Pairwise permutation tests ({N_PERMUTATIONS:,} resamples, two-sided)\n")
    print("| Comparison | Δ buyer share (pp) | p (raw) | p (Holm) | sig (Holm < .05) |")
    print("|---|---:|---:|---:|:---:|")
    for (a, b, diff, p_raw), p_h in zip(rows, holm_p):
        sig = "**yes**" if p_h < 0.05 else "no"
        print(f"| {a} vs {b} | {diff:+.2f} | {p_raw:.4f} | {p_h:.4f} | {sig} |")

    print("\n## Headline conclusions for paper §4.8")
    keys = ["B + GRU", "B + Random", "B + Uniform", "B + Oracle", "A (No OM)"]

    def lookup(a, b):
        for ra, rb, d, p in rows:
            if (ra, rb) == (a, b): return d, p
            if (ra, rb) == (b, a): return -d, p
        raise KeyError((a, b))

    for a, b in [
        ("B + GRU", "B + Random"),
        ("B + GRU", "B + Uniform"),
        ("B + Random", "B + Uniform"),
        ("B + GRU", "A (No OM)"),
        ("B + Oracle", "B + GRU"),
    ]:
        d, p_raw = lookup(a, b)
        idx = pairs.index(tuple(sorted([a, b], key=lambda k: list(samples.keys()).index(k))))
        p_h = holm_p[idx]
        print(f"  {a} - {b}: Δ={d:+.2f}pp  p_raw={p_raw:.4f}  p_holm={p_h:.4f}")


if __name__ == "__main__":
    main()
