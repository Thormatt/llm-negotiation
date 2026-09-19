"""
Nash bargaining solution for 2-party multi-issue negotiation with
additive separable utilities and box constraints.

Each party submits a utility-weight vector w_i over k issues plus a direction
vector d_i ∈ {+1, -1}^k indicating which way each issue increases its utility.
We require directions to be opposite (d_A = -d_B): each issue is a pure
distributive issue, not a shared one.

Utility:
    U_i(v) = sum_k w_{i,k} * tilde{v}_{i,k}
    where tilde{v}_{i,k} = v_k if d_{i,k} = +1 else 1 - v_k.

The Nash bargaining solution maximises (U_A - d_A) * (U_B - d_B), with
disagreement payoffs d_A = d_B = 0 (no deal yields 0 utility on both sides).
Equivalently, it maximises log(U_A) + log(U_B).

Because the utility functions are additive linear and the feasible set is the
box [v_min, v_max]^k, the KKT conditions give a corner-solution optimum: each
v_k is at v_min or v_max in the generic case. The choice of corner depends on
the sign of the marginal-utility ratio at the optimum:

    push v_k to high end if    w_{A,k}/U_A * d_{A,k} + w_{B,k}/U_B * d_{B,k} > 0
    push v_k to low end  otherwise.

Since U_A and U_B depend on which corners are chosen, this is a fixed-point
problem; we solve by simple iteration. Convergence in a few steps in practice.

NashBargaining2P     — the corner-solution Nash bargaining protocol.
CooperativeBlend2P   — the paper's compute_target-style proportional split,
                       useful as a baseline contrast (smoother but not Pareto-optimal).
"""

from __future__ import annotations

import numpy as np

from framework.protocol import Outcome, Protocol


def _additive_utility(v: np.ndarray, weights: np.ndarray, direction: np.ndarray) -> float:
    tilde = np.where(direction > 0, v, 1.0 - v)
    return float(np.dot(weights, tilde))


def _validate_two_party(inputs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(inputs) != 2:
        raise ValueError("Protocol requires exactly 2 participants.")
    a, b = inputs
    wa = np.asarray(a["weights"], dtype=np.float64)
    wb = np.asarray(b["weights"], dtype=np.float64)
    da = np.asarray(a["direction"], dtype=np.float64)
    db = np.asarray(b["direction"], dtype=np.float64)
    if wa.shape != wb.shape or da.shape != wa.shape or db.shape != wb.shape:
        raise ValueError("Weight and direction arrays must share shape.")
    if not np.allclose(da, -db):
        raise ValueError("Directions must be opposite for a 2-party bargaining problem.")
    return wa, wb, da, db


class NashBargaining2P(Protocol):
    """Two-party Nash bargaining for additive multi-issue utility.

    Returns a corner-solution optimum on a box [v_min, v_max]^k.

    inputs:
        [{"weights": np.ndarray, "direction": np.ndarray (+1 or -1)},
         {"weights": np.ndarray, "direction": np.ndarray (opposite)}]
    """

    name = "NashBargaining2P"

    def __init__(self, v_min: float = 0.05, v_max: float = 0.95, max_iter: int = 100):
        self.v_min = v_min
        self.v_max = v_max
        self.max_iter = max_iter

    def run(self, inputs: list[dict]) -> Outcome:
        wa, wb, da, db = _validate_two_party(inputs)
        k = wa.size

        # Initialise at the box midpoint
        v = np.full(k, 0.5 * (self.v_min + self.v_max), dtype=np.float64)
        for _ in range(self.max_iter):
            ua = max(_additive_utility(v, wa, da), 1e-12)
            ub = max(_additive_utility(v, wb, db), 1e-12)
            # Marginal of log(U_A) + log(U_B) w.r.t. v_k
            grad = (wa / ua) * da + (wb / ub) * db
            new_v = np.where(grad > 0, self.v_max, self.v_min)
            if np.array_equal(new_v, v):
                break
            v = new_v

        ua = float(_additive_utility(v, wa, da))
        ub = float(_additive_utility(v, wb, db))
        return Outcome(
            allocation={"proposal": v.tolist()},
            payoffs=(ua, ub),
            metadata={
                "weights_a": wa.tolist(),
                "weights_b": wb.tolist(),
                "direction_a": da.tolist(),
                "direction_b": db.tolist(),
                "nash_product": ua * ub,
                "v_min": self.v_min,
                "v_max": self.v_max,
            },
        )


class CooperativeBlend2P(Protocol):
    """Paper's compute_target-style protocol: each issue split proportionally
    to relative weights.

    Smoother (interior) than NashBargaining2P, NOT Pareto-optimal in general.
    Included so we can A/B compare middleman protocols within Phase B and so
    the bias-scalar bias=0.55 calibration from the negotiation paper has a
    direct protocol analogue.
    """

    name = "CooperativeBlend2P"

    def __init__(self, v_min: float = 0.05, v_max: float = 0.95):
        self.v_min = v_min
        self.v_max = v_max

    def run(self, inputs: list[dict]) -> Outcome:
        wa, wb, da, db = _validate_two_party(inputs)
        share_a = wa / (wa + wb + 1e-12)
        # If A prefers high (d_a=+1), proposal sits at share_a; if A prefers
        # low (d_a=-1), proposal sits at 1 - share_a.
        proposal = np.where(da > 0, share_a, 1.0 - share_a)
        proposal = proposal.clip(self.v_min, self.v_max)

        ua = float(_additive_utility(proposal, wa, da))
        ub = float(_additive_utility(proposal, wb, db))
        return Outcome(
            allocation={"proposal": proposal.tolist()},
            payoffs=(ua, ub),
            metadata={
                "weights_a": wa.tolist(),
                "weights_b": wb.tolist(),
                "direction_a": da.tolist(),
                "direction_b": db.tolist(),
                "nash_product": ua * ub,
            },
        )
