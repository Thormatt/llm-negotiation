"""
Hand-coded opponent model. No learned parameters.

Inverts the analytical concession formula of HeuristicNegotiator:

    concession_speed_i ∝ (1 - w_i)^2 * aggression * base_rate

So observed cumulative concession c_i ∝ (1 - w_i)^2 (up to a per-game scaling
factor that's identical across variables within one opponent). Solving for w:

    1 - w_i ∝ sqrt(c_i)
    w_i ∝ -sqrt(c_i) (after a softmax normalization to a probability simplex)

The threshold output mirrors HeuristicNegotiator's rising threshold:

    threshold = 0.35 + 0.15 * (round / max_rounds)

This module is a drop-in functional replacement for NegotiationGRU.predict at
the level of "given an env state, return predicted opponent weights + accept
threshold." It uses zero learned parameters.
"""

from __future__ import annotations

import numpy as np


def predict_heuristic_om(env, perspective: str, scale: float = 6.0,
                         bias: float = 0.0, my_weights=None) -> tuple[np.ndarray, float]:
    """Return (predicted_opponent_weights, predicted_acceptance_threshold).

    Args:
        env: NegotiationEnv after some history has accumulated.
        perspective: "buyer" or "seller" — whose opponent we're modelling.
        scale: temperature on the softmax over -sqrt(concession). Higher =
               more confident / sharper predictions. 6.0 chosen by inspection
               on heuristic-vs-heuristic data; small effect across [4, 10].
        bias: convex blend toward "1 - my_weights / sum". When bias=0 we
              return the unbiased analytical inversion. When bias=1 we
              predict the opponent cares about exactly what we don't —
              biasing the cooperative blend in our favor.
        my_weights: required when bias > 0. The agent's own utility weights.

    Returns:
        weights: np.ndarray shape (5,), simplex (sums to 1).
        threshold: float in [0.35, 0.50].
    """
    opponent = "seller" if perspective == "buyer" else "buyer"
    opp_proposals = [p for r, p in env.history if r == opponent]

    if len(opp_proposals) < 2:
        weights = np.full(5, 0.2, dtype=np.float32)
    else:
        cumulative = np.abs(opp_proposals[-1] - opp_proposals[0]).astype(np.float64)
        cumulative = cumulative + 1e-3
        logits = -np.sqrt(cumulative) * scale
        ex = np.exp(logits - logits.max())
        weights = (ex / ex.sum()).astype(np.float32)

    if bias > 0.0:
        if my_weights is None:
            raise ValueError("my_weights required when bias > 0")
        my_w = np.asarray(my_weights, dtype=np.float32)
        # Multiplicative down-weight on the agent's own priority variables.
        # bias=0 returns the unbiased prediction. Larger bias more aggressively
        # suppresses opp_w on variables the agent cares about, which inflates
        # my_share in compute_target_proposal and pulls the target toward the
        # agent's ideal on those variables.
        weights = weights * np.exp(-bias * my_w * 5.0)
        weights = (weights / weights.sum()).astype(np.float32)

    round_frac = (env.round / env.max_rounds) if env.max_rounds > 0 else 0.0
    threshold = float(0.35 + 0.15 * round_frac)
    return weights, threshold
