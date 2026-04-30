"""
Multi-Variable Bilateral LLM Negotiation with NegotiationGRU.

LLM agents negotiate 5-dimensional B2B SaaS deals (Price, Contract Length,
Payment Terms, SLA Tier, Data Rights) with hidden asymmetric utility weights.
A trained NegotiationGRU predicts the opponent's weights from behavioral
patterns, enabling strategic exploitation of the Pareto frontier.

Conditions:
  A. Symmetric:  LLM vs LLM (no world model)
  B. Asymmetric: LLM+WM vs LLM (WM on buyer only)
  C. Arms race:  LLM+WM vs LLM+WM (both have world model)

Usage:
  OPENROUTER_API_KEY=key python3 -m belief_gru.negotiation --condition B --games 1 --rounds 4
  OPENROUTER_API_KEY=key python3 -m belief_gru.negotiation --condition all --games 5 --quiet
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import torch
from openai import OpenAI

from train_negotiation_gru import (
    NegotiationEnv, NegotiationGRU, load_negotiation_gru, compute_utility,
    extract_features, build_observation_sequence, HeuristicNegotiator,
    VARIABLES, BUYER_DIRECTION, SELLER_DIRECTION, MAX_ROUNDS,
)


# ===========================================================================
# WM Briefing Generator
# ===========================================================================

def aggression_schedule(round_frac, kind="decay", a_const=0.70, a_max=0.80, a_min=0.55, decay_rate=0.30):
    """Aggression parameter `a` as a function of round fraction.

    kind:
      - "decay" (default): a = max(a_min, a_max - decay_rate * round_frac)  [paper Eq.3]
      - "const": a = a_const (no schedule; isolates "decay" from "format")
      - "low_const": a = a_min
      - "high_const": a = a_max
    """
    if kind == "decay":
        return max(a_min, a_max - decay_rate * round_frac)
    if kind == "const":
        return a_const
    if kind == "low_const":
        return a_min
    if kind == "high_const":
        return a_max
    raise ValueError(f"unknown schedule kind: {kind}")


def compute_target_proposal(my_weights, predicted_opp_weights, my_direction, round_frac,
                            schedule="decay", schedule_kwargs=None):
    """Compute an optimal target proposal given predicted opponent weights.

    For each variable, blend between agent's ideal and opponent's ideal
    based on relative weight importance. Variables where we care more:
    push toward our direction. Variables where opponent cares more:
    concede to grease the deal.

    Aggressiveness follows `schedule` (see aggression_schedule). Default reproduces
    paper Eq.3; alternatives support the §5.7 schedule-ablation limitation.
    """
    my_w = np.asarray(my_weights, dtype=np.float32)
    opp_w = np.asarray(predicted_opp_weights, dtype=np.float32)
    direction = np.asarray(my_direction, dtype=np.float32)

    target = np.zeros(5, dtype=np.float32)
    aggression = aggression_schedule(round_frac, kind=schedule, **(schedule_kwargs or {}))

    for i in range(5):
        my_ideal = 0.90 if direction[i] > 0 else 0.10
        opp_ideal = 0.10 if direction[i] > 0 else 0.90

        total = my_w[i] + opp_w[i] + 1e-8
        my_share = my_w[i] / total

        blend = my_share * aggression + (1 - aggression) * 0.5
        target[i] = blend * my_ideal + (1 - blend) * opp_ideal

    return target.clip(0.05, 0.95)


def _build_policy_state(env, perspective, gru_predicted_opp_w=None):
    """Build 27D policy state vector from current env state.

    Mirrors extract_state_at_turn() from policy_data.py but works on live env
    without requiring a game dict. Used at inference time.

    Args:
        env: NegotiationEnv
        perspective: "buyer" or "seller"
        gru_predicted_opp_w: (5,) GRU-predicted opponent weights (from the
            GRU model that's already running in generate_negotiation_briefing)
    """
    my_weights = env.buyer_weights if perspective == "buyer" else env.seller_weights
    my_direction = BUYER_DIRECTION if perspective == "buyer" else SELLER_DIRECTION

    opp_proposals = [p for r, p in env.history if r != perspective]
    my_proposals = [p for r, p in env.history if r == perspective]

    state = np.zeros(27, dtype=np.float32)

    # Dims 0-4: my_weights
    state[0:5] = my_weights

    # Dim 5: round_frac
    state[5] = min(env.round / env.max_rounds, 1.0)

    # Dims 6-10: opp_latest_proposal
    if opp_proposals:
        state[6:11] = opp_proposals[-1]
    else:
        state[6:11] = 0.5

    # Dim 11: my_utility_at_opp
    if opp_proposals:
        state[11] = compute_utility(opp_proposals[-1], my_weights, perspective)
    else:
        state[11] = 0.0

    # Dims 12-16: my_latest_proposal
    if my_proposals:
        state[12:17] = my_proposals[-1]
    else:
        state[12:17] = np.where(my_direction > 0, 0.80, 0.20).astype(np.float32)

    # Dim 17: opp_utility_at_mine (uniform approximation)
    if my_proposals:
        opp_role = "seller" if perspective == "buyer" else "buyer"
        uniform_w = np.full(5, 0.2, dtype=np.float32)
        state[17] = compute_utility(my_proposals[-1], uniform_w, opp_role)
    else:
        state[17] = 0.5

    # Dim 18: n_opp_proposals (normalized)
    state[18] = len(opp_proposals) / env.max_rounds

    # Dim 19: is_buyer
    state[19] = 1.0 if perspective == "buyer" else 0.0

    # Dim 20: had_briefing (at inference, we always have briefing if this code runs)
    state[20] = 1.0

    # Dim 21: opp_concession_rate
    if len(opp_proposals) >= 2:
        deltas = [float(np.linalg.norm(opp_proposals[i+1] - opp_proposals[i]))
                  for i in range(len(opp_proposals) - 1)]
        state[21] = float(np.mean(deltas))
    else:
        state[21] = 0.0

    # Dims 22-26: GRU predicted opponent weights
    if gru_predicted_opp_w is not None:
        state[22:27] = gru_predicted_opp_w
    else:
        state[22:27] = 0.2  # uniform prior

    return state


def generate_negotiation_briefing(model, env, perspective, ablation=None, policy_net=None,
                                  schedule="decay", schedule_kwargs=None,
                                  mask_gru_in_policy=False):
    """Generate computed-target briefing from NegotiationGRU or PolicyNet.

    Instead of text advice (which LLMs misinterpret), computes a concrete
    target proposal that maximizes the agent's utility given predicted
    opponent weights. The LLM's job is to negotiate TOWARD this target.

    Args:
        model: NegotiationGRU model (used for GRU-based predictions)
        env: NegotiationEnv instance
        perspective: "buyer" or "seller"
        ablation: None = use GRU predictions (default)
                  "oracle" = use true opponent weights (upper bound)
                  "random" = use random Dirichlet(1.0) weights (control)
                  "uniform" = use uniform weights [0.2, 0.2, 0.2, 0.2, 0.2]
        policy_net: NegotiationPolicyNet (if provided, bypasses GRU+heuristic
                    and directly outputs 5D proposal from 22D state)

    Returns:
        (briefing_text, diagnostics_dict)
    """
    obs, length = build_observation_sequence(env, perspective)

    model.eval()
    with torch.no_grad():
        w_pred, t_pred, beliefs = model(obs, torch.tensor([length]))

    gru_predicted = w_pred[0].numpy()  # (5,)
    threshold = t_pred[0].item()

    my_weights = env.buyer_weights if perspective == "buyer" else env.seller_weights
    my_direction = BUYER_DIRECTION if perspective == "buyer" else SELLER_DIRECTION
    round_frac = env.round / env.max_rounds

    # Ablation: override predicted weights
    if ablation == "oracle":
        predicted_opp_weights = (env.seller_weights if perspective == "buyer"
                                 else env.buyer_weights).copy()
    elif ablation == "random":
        predicted_opp_weights = np.random.dirichlet(np.ones(5)).astype(np.float32)
    elif ablation == "uniform":
        predicted_opp_weights = np.full(5, 0.2, dtype=np.float32)
    elif ablation == "heuristic":
        from heuristic_om import predict_heuristic_om
        predicted_opp_weights, threshold = predict_heuristic_om(
            env, perspective, bias=0.0,
        )
    elif ablation == "heuristic_calibrated":
        from heuristic_om import predict_heuristic_om
        predicted_opp_weights, threshold = predict_heuristic_om(
            env, perspective, bias=0.55, my_weights=my_weights,
        )
    else:
        predicted_opp_weights = gru_predicted

    # Policy net path: build 27D state (includes GRU predictions), forward pass
    if policy_net is not None:
        gru_for_state = (np.full(5, 0.2, dtype=np.float32) if mask_gru_in_policy
                         else gru_predicted)
        state = _build_policy_state(env, perspective, gru_predicted_opp_w=gru_for_state)
        policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target = policy_net(state_t)[0].numpy()
    else:
        # Heuristic path: GRU predicts weights → formula computes proposal
        target = compute_target_proposal(
            my_weights, predicted_opp_weights, my_direction, round_frac,
            schedule=schedule, schedule_kwargs=schedule_kwargs,
        )

    my_utility_at_target = compute_utility(target, my_weights, perspective)

    # Format as a concrete, actionable directive
    target_lines = []
    for i, v in enumerate(VARIABLES):
        target_lines.append(f"  {v:20s}: {target[i]:.2f}")

    briefing = f"""=== COMPUTED OPTIMAL PROPOSAL ===

Your model-recommended proposal (maximizes YOUR utility given opponent behavior):
{chr(10).join(target_lines)}
  Expected utility: {my_utility_at_target:.3f}

INSTRUCTIONS:
- Offer this proposal or something very close to it.
- This is calculated to exploit the opponent's weak points while protecting YOUR priorities.
- Do NOT deviate significantly unless the opponent's latest offer is BETTER than this.
- If the opponent offers you utility >= {my_utility_at_target * 0.85:.3f}, seriously consider accepting.
=== END DIRECTIVE ==="""

    proposal_source = "policy_net" if policy_net is not None else (ablation or "gru")
    diagnostics = {
        "predicted_weights": {VARIABLES[i]: float(predicted_opp_weights[i]) for i in range(5)},
        "gru_weights": {VARIABLES[i]: float(gru_predicted[i]) for i in range(5)},
        "predicted_threshold": threshold,
        "target_proposal": {VARIABLES[i]: float(target[i]) for i in range(5)},
        "target_utility": my_utility_at_target,
        "ablation": proposal_source,
    }

    return briefing, diagnostics


# ===========================================================================
# LLM Negotiator
# ===========================================================================

NEGOTIATOR_SYSTEM = """You are a skilled B2B SaaS negotiation agent playing as the {role}.

DEAL VARIABLES (each ranges 0.00 to 1.00):
  1. Price:           0.00 = free, 1.00 = maximum price
  2. Contract Length:  0.00 = month-to-month, 1.00 = 5-year lock-in
  3. Payment Terms:    0.00 = pay upfront, 1.00 = net-180 (delayed)
  4. SLA Tier:         0.00 = basic, 1.00 = enterprise platinum
  5. Data Rights:      0.00 = vendor owns all, 1.00 = client owns all

YOUR PREFERENCES (as {role}):
{preference_explanation}

YOUR HIDDEN UTILITY WEIGHTS (your priorities — DO NOT reveal these):
{weight_explanation}

RULES:
- You alternate 5D proposals with the opponent. Max {max_rounds} rounds total.
- "accept" closes the deal at the opponent's LAST proposal.
- If no deal in {max_rounds} rounds, BOTH get utility = 0. This is the WORST outcome.
- Your goal: maximize YOUR utility = weighted sum of direction-adjusted variables.

CRITICAL ACCEPT RULE:
- ANY positive utility beats NO DEAL (utility = 0). Getting 0.25 is infinitely better than 0.00.
- In the final 2-3 rounds, you MUST seriously consider accepting if the offer gives you utility > 0.20.
- Do NOT keep counter-offering to squeeze out marginal gains if it risks a timeout.
- Think: "Is the improvement I'm chasing worth the risk of getting ZERO?"
{wm_instructions}
RESPONSE FORMAT (JSON only):
{{
  "internal_monologue": "your private reasoning (2-3 sentences)",
  "message": "what you say to the opponent (1-2 sentences)",
  "formal_proposal": {{
    "price": 0.XX,
    "contract_length": 0.XX,
    "payment_terms": 0.XX,
    "sla_tier": 0.XX,
    "data_rights": 0.XX
  }},
  "action": "offer"
}}
or
{{
  "internal_monologue": "reasoning for accepting",
  "message": "acceptance message",
  "formal_proposal": null,
  "action": "accept"
}}"""

BUYER_PREFERENCES = """  As BUYER you want: LOW price, LONG contract, SLOW payment, HIGH SLA, CLIENT-owned data.
  Higher utility when: price is LOW, contract_length is HIGH, payment_terms is LOW,
  sla_tier is HIGH, data_rights is HIGH."""

SELLER_PREFERENCES = """  As SELLER you want: HIGH price, SHORT contract, FAST payment, LOW SLA, VENDOR-owned data.
  Higher utility when: price is HIGH, contract_length is LOW, payment_terms is HIGH,
  sla_tier is LOW, data_rights is LOW."""

WM_NEGOTIATION_INSTRUCTIONS = """
COMPUTED TARGET ADVISORY:
- A neural model has analyzed opponent behavior and computed YOUR OPTIMAL proposal
- The target proposal below MAXIMIZES YOUR UTILITY given what we know about the opponent
- You MUST offer this proposal or something very close to it — do NOT freelance
- The model has already done the strategic thinking — your job is to EXECUTE the target
- Treat the target as your anchor. Only deviate toward the opponent if needed to close a deal
"""

TEXT_ADVICE_INSTRUCTIONS = """
STRATEGIC ADVISORY:
- Our opponent-modeling system has analyzed the other party's negotiation behavior
- The analysis below estimates their priorities and recommends a strategy
- Use this intelligence to guide your proposals — push hardest where they are weak
- Remember: your goal is to maximize YOUR utility, not to find a fair compromise
"""


def generate_text_advice(model, env, perspective, ablation=None):
    """Generate text-based strategic advice encoding the same info as numeric targets.

    Same information content as generate_negotiation_briefing(), but delivered
    as natural-language strategy advice rather than a numeric directive.
    """
    obs, length = build_observation_sequence(env, perspective)

    model.eval()
    with torch.no_grad():
        w_pred, t_pred, beliefs = model(obs, torch.tensor([length]))

    gru_predicted = w_pred[0].numpy()

    if ablation == "oracle":
        opp_weights = (env.seller_weights if perspective == "buyer"
                       else env.buyer_weights).copy()
    elif ablation == "random":
        opp_weights = np.random.dirichlet(np.ones(5)).astype(np.float32)
    elif ablation == "uniform":
        opp_weights = np.full(5, 0.2, dtype=np.float32)
    else:
        opp_weights = gru_predicted

    my_weights = env.buyer_weights if perspective == "buyer" else env.seller_weights
    my_direction = BUYER_DIRECTION if perspective == "buyer" else SELLER_DIRECTION

    # Rank opponent priorities
    sorted_indices = np.argsort(-opp_weights)
    priority_lines = []
    labels = ["highest", "high", "moderate", "low", "lowest"]
    for rank, idx in enumerate(sorted_indices):
        priority_lines.append(
            f"  {VARIABLES[idx]:20s}: {labels[rank]} priority (weight ~{opp_weights[idx]:.2f})"
        )

    # Generate per-variable strategy
    strategy_lines = []
    for i in range(5):
        my_w = my_weights[i]
        opp_w = opp_weights[i]
        if my_w > opp_w + 0.05:
            strategy_lines.append(f"  {VARIABLES[i]:20s}: HOLD FIRM — you care more than they do")
        elif opp_w > my_w + 0.05:
            strategy_lines.append(f"  {VARIABLES[i]:20s}: CONCEDE here — opponent values this highly, use it as leverage")
        else:
            strategy_lines.append(f"  {VARIABLES[i]:20s}: Moderate — roughly equal importance to both sides")

    top_var = VARIABLES[sorted_indices[0]]

    advice = f"""=== STRATEGIC ADVISORY ===

OPPONENT'S ESTIMATED PRIORITIES:
{chr(10).join(priority_lines)}

RECOMMENDED STRATEGY PER VARIABLE:
{chr(10).join(strategy_lines)}

KEY INSIGHT: The opponent cares most about {top_var}. Consider conceding on
{top_var} to extract maximum value on YOUR priority variables. But do NOT
over-accommodate — your goal is to maximize YOUR utility, not theirs.
=== END ADVISORY ==="""

    diagnostics = {
        "predicted_weights": {VARIABLES[i]: float(opp_weights[i]) for i in range(5)},
        "format": "text",
        "ablation": ablation or "gru",
    }

    return advice, diagnostics


class LLMNegotiator:
    """LLM-powered multi-variable negotiation agent with optional NegotiationGRU."""

    def __init__(self, client, role, weights, env, model_name, world_model=None,
                 ablation=None, strategic_format=None, accept_override=True,
                 schedule="decay", schedule_kwargs=None, mask_gru_in_policy=False,
                 policy_net=None):
        """
        Args:
            strategic_format: None = no advisory, "numeric" = computed target,
                              "text" = text advice. If world_model is provided
                              and strategic_format is None, defaults to "numeric".
            accept_override: If True, force accept in final round if utility > 0.15.
                Set False for the §5.7 hard-accept-override sensitivity ablation.
            schedule: aggression-schedule kind for compute_target_proposal
                ("decay" reproduces paper Eq.3; alternatives ablate the schedule).
            schedule_kwargs: optional dict of schedule parameters (e.g., a_const).
            mask_gru_in_policy: if True, replace the 5D GRU prediction in the
                policy state with uniform [0.2,0.2,0.2,0.2,0.2] — load-bearing
                test for §5.7 GRU-feature ablation.
            policy_net: NegotiationPolicyNet for the neural-policy path.
        """
        self.client = client
        self.role = role
        self.weights = np.asarray(weights, dtype=np.float32)
        self.model_name = model_name
        self.world_model = world_model
        self.ablation = ablation
        self.accept_override = accept_override
        self.schedule = schedule
        self.schedule_kwargs = schedule_kwargs
        self.mask_gru_in_policy = mask_gru_in_policy
        self.policy_net = policy_net
        self.env = env

        # Determine strategic format
        if strategic_format is not None:
            self.strategic_format = strategic_format
        elif world_model is not None:
            self.strategic_format = "numeric"
        else:
            self.strategic_format = None

        pref_text = BUYER_PREFERENCES if role == "buyer" else SELLER_PREFERENCES
        if self.strategic_format == "numeric":
            wm_inst = WM_NEGOTIATION_INSTRUCTIONS
        elif self.strategic_format == "text":
            wm_inst = TEXT_ADVICE_INSTRUCTIONS
        else:
            wm_inst = ""

        # Explain weights to the LLM (it knows its OWN weights)
        weight_lines = []
        for i, v in enumerate(VARIABLES):
            weight_lines.append(f"  {v:20s}: {weights[i]:.2f}")
        weight_text = "\n".join(weight_lines)

        self.system_prompt = NEGOTIATOR_SYSTEM.format(
            role=role.upper(),
            preference_explanation=pref_text,
            weight_explanation=weight_text,
            max_rounds=env.max_rounds,
            wm_instructions=wm_inst,
        )

    def decide(self, env):
        """Produce action and proposal.

        Returns:
            (action, proposal_5d_or_None, reasoning, briefing_text_or_None)
        """
        # Hard accept override: final round, positive utility -> force accept
        rounds_left = env.max_rounds - env.round
        opp_proposals = env.get_opponent_proposals(self.role)
        if self.accept_override and rounds_left <= 1 and opp_proposals:
            util = compute_utility(opp_proposals[-1], self.weights, self.role)
            if util > 0.15:
                return "accept", None, f"[Final-round auto-accept: utility {util:.3f} > 0.15]", None

        # Generate strategic advisory based on format
        briefing = None
        if self.world_model is not None and self.strategic_format == "numeric":
            briefing, _ = generate_negotiation_briefing(
                self.world_model, env, perspective=self.role,
                ablation=self.ablation,
                policy_net=self.policy_net,
                schedule=self.schedule,
                schedule_kwargs=self.schedule_kwargs,
                mask_gru_in_policy=self.mask_gru_in_policy,
            )
        elif self.world_model is not None and self.strategic_format == "text":
            briefing, _ = generate_text_advice(
                self.world_model, env, perspective=self.role,
                ablation=self.ablation,
            )

        user_msg = self._format_state(env)
        if briefing:
            user_msg += f"\n\n{briefing}"
        user_msg += "\n\nYour move. Respond with JSON only."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            raw = response.choices[0].message.content or ""
            # Strip <think>...</think> blocks from reasoning models
            if "<think>" in raw:
                think_end = raw.rfind("</think>")
                if think_end != -1:
                    raw = raw[think_end + 8:]
                else:
                    # Unclosed <think> — take everything after last </think> or after <think>
                    raw = raw.split("<think>")[-1]
            raw = raw.strip()

            text = raw
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            # Fallback: extract first JSON object by finding outermost { }
            if not text or text[0] != "{":
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end > start:
                    text = raw[start : end + 1]

            # Allow control chars in strings (Gemini multi-line values)
            decision = json.loads(text, strict=False)
            action = decision.get("action", "offer").lower()
            reasoning = decision.get("internal_monologue", "")
            message = decision.get("message", "")

            if action == "accept":
                # Validate: don't accept if utility < 0.10 (soft floor)
                opp_proposals = env.get_opponent_proposals(self.role)
                if opp_proposals:
                    util = compute_utility(opp_proposals[-1], self.weights, self.role)
                    if util < 0.10:
                        action = "offer"
                        reasoning += " [Override: utility too low to accept]"
                    else:
                        return "accept", None, reasoning, briefing
                else:
                    action = "offer"
                    reasoning = "Nothing to accept yet"

            if action == "offer":
                fp = decision.get("formal_proposal", {})
                if fp:
                    proposal = np.array([
                        float(fp.get("price", 0.5)),
                        float(fp.get("contract_length", 0.5)),
                        float(fp.get("payment_terms", 0.5)),
                        float(fp.get("sla_tier", 0.5)),
                        float(fp.get("data_rights", 0.5)),
                    ], dtype=np.float32)
                else:
                    proposal = self._fallback_proposal(env)

                # Clamp to [0.05, 0.95]
                proposal = proposal.clip(0.05, 0.95)
                return "offer", proposal, reasoning, briefing

            # Unknown action fallback
            proposal = self._fallback_proposal(env)
            return "offer", proposal, f"Unknown action '{action}', fallback", briefing

        except Exception as e:
            proposal = self._fallback_proposal(env)
            return "offer", proposal, f"LLM error ({e}), fallback", briefing

    def _fallback_proposal(self, env):
        """Small concession from last proposal, or opening offer."""
        last = env.get_last_proposal(self.role)
        if last is not None:
            direction = BUYER_DIRECTION if self.role == "buyer" else SELLER_DIRECTION
            concession = np.where(direction > 0, -0.03, 0.03).astype(np.float32)
            return (last + concession).clip(0.05, 0.95)
        # Opening offer: aggressive
        direction = BUYER_DIRECTION if self.role == "buyer" else SELLER_DIRECTION
        return np.where(direction > 0, 0.80, 0.20).astype(np.float32)

    def _format_state(self, env):
        rounds_left = env.max_rounds - env.round
        lines = [f"Round {env.round}/{env.max_rounds} ({rounds_left} rounds remaining)"]

        opp_role = "seller" if self.role == "buyer" else "buyer"
        opp_proposals = env.get_opponent_proposals(self.role)
        my_proposals = [p for r, p in env.history if r == self.role]

        if opp_proposals:
            latest = opp_proposals[-1]
            util = compute_utility(latest, self.weights, self.role)
            lines.append(f"\n{opp_role.title()}'s latest proposal:")
            for i, v in enumerate(VARIABLES):
                lines.append(f"  {v:20s}: {latest[i]:.3f}")
            lines.append(f"\n  >>> YOUR UTILITY IF YOU ACCEPT NOW: {util:.3f} <<<")

            # Deadline pressure warnings
            if rounds_left <= 2 and util > 0.15:
                lines.append(f"\n  *** DEADLINE WARNING: Only {rounds_left} round(s) left! ***")
                lines.append(f"  *** Accepting gives you {util:.3f}. NO DEAL gives you 0.000. ***")
                lines.append(f"  *** A bird in the hand: {util:.3f} > 0.000. Strongly consider accepting. ***")
            elif rounds_left <= 3 and util > 0.20:
                lines.append(f"\n  DEADLINE APPROACHING: {rounds_left} rounds left. "
                             f"Current offer gives you {util:.3f} vs 0.000 if no deal.")

        if my_proposals:
            my_last = my_proposals[-1]
            lines.append(f"\nYour last proposal:")
            for i, v in enumerate(VARIABLES):
                lines.append(f"  {v:20s}: {my_last[i]:.3f}")

        lines.append(f"\nHistory ({len(env.history)} turns):")
        for turn_idx, (role, prop) in enumerate(env.history):
            label = "You" if role == self.role else opp_role.title()
            vals = ", ".join(f"{VARIABLES[i]}={prop[i]:.2f}" for i in range(5))
            lines.append(f"  Turn {turn_idx+1}: {label}: [{vals}]")

        if not env.history:
            lines.append("  (No proposals yet — you open!)" if self.role == "seller" else "  (Waiting for seller's opening)")

        return "\n".join(lines)


# ===========================================================================
# Pareto Efficiency + Metrics
# ===========================================================================

def compute_pareto_frontier(buyer_weights, seller_weights):
    """Compute max joint surplus and optimal proposal.

    For each variable, the Pareto-optimal assignment gives it to
    whichever role weights it more (in effective terms).

    Returns:
        (max_joint_surplus, optimal_proposal)
    """
    buyer_w = np.asarray(buyer_weights, dtype=np.float32)
    seller_w = np.asarray(seller_weights, dtype=np.float32)

    # For each variable, find the value that maximizes joint utility.
    # buyer utility contribution for var i: buyer_w[i] * (val if buyer_dir > 0 else 1-val)
    # seller utility contribution for var i: seller_w[i] * (val if seller_dir > 0 else 1-val)
    optimal = np.zeros(5, dtype=np.float32)
    for i in range(5):
        # buyer's marginal utility for increasing var i
        buyer_marginal = buyer_w[i] * BUYER_DIRECTION[i]
        seller_marginal = seller_w[i] * SELLER_DIRECTION[i]
        joint_marginal = buyer_marginal + seller_marginal

        # If joint marginal > 0: set var to 1.0 (both benefit or net benefit from high)
        # If joint marginal < 0: set var to 0.0
        # If == 0: indifferent, use 0.5
        if joint_marginal > 0:
            optimal[i] = 1.0
        elif joint_marginal < 0:
            optimal[i] = 0.0
        else:
            optimal[i] = 0.5

    max_buyer_util = compute_utility(optimal, buyer_w, "buyer")
    max_seller_util = compute_utility(optimal, seller_w, "seller")
    max_joint = max_buyer_util + max_seller_util

    return max_joint, optimal


def compute_match_metrics(env, buyer_weights, seller_weights):
    """Compute comprehensive metrics for a completed match.

    Returns dict with: deal, buyer_utility, seller_utility, joint_surplus,
    pareto_efficiency, buyer_surplus_share.
    """
    buyer_w = np.asarray(buyer_weights, dtype=np.float32)
    seller_w = np.asarray(seller_weights, dtype=np.float32)

    max_joint, optimal = compute_pareto_frontier(buyer_w, seller_w)

    if env.deal_proposal is not None:
        buyer_util = compute_utility(env.deal_proposal, buyer_w, "buyer")
        seller_util = compute_utility(env.deal_proposal, seller_w, "seller")
        joint = buyer_util + seller_util
        pareto_eff = joint / max_joint if max_joint > 0 else 0.0
        buyer_share = buyer_util / joint if joint > 0 else 0.5

        return {
            "deal": True,
            "buyer_utility": buyer_util,
            "seller_utility": seller_util,
            "joint_surplus": joint,
            "max_joint_surplus": max_joint,
            "pareto_efficiency": pareto_eff,
            "buyer_surplus_share": buyer_share,
            "deal_proposal": env.deal_proposal.tolist(),
            "optimal_proposal": optimal.tolist(),
            "rounds": env.round,
        }
    else:
        return {
            "deal": False,
            "buyer_utility": 0.0,
            "seller_utility": 0.0,
            "joint_surplus": 0.0,
            "max_joint_surplus": max_joint,
            "pareto_efficiency": 0.0,
            "buyer_surplus_share": 0.0,
            "deal_proposal": None,
            "optimal_proposal": optimal.tolist(),
            "rounds": env.round,
        }


# ===========================================================================
# Match Runner
# ===========================================================================

def run_match(buyer, seller, env, buyer_weights, seller_weights, verbose=True,
              opener="seller"):
    """Run a single multi-variable negotiation match.

    Alternating offers. Max max_rounds rounds.

    Args:
        opener: "seller" (default) or "buyer" — who makes the first proposal.

    Returns:
        result dict with metrics, logs, proposals
    """
    match_log = {
        "buyer_weights": buyer_weights.tolist(),
        "seller_weights": seller_weights.tolist(),
        "opener": opener,
        "turns": [],
    }

    turn_order = [opener, "buyer" if opener == "seller" else "seller"]
    players = {"buyer": buyer, "seller": seller}
    turn_idx = 0

    while not env.done:
        role = turn_order[turn_idx % 2]
        player = players[role]

        action, proposal, reasoning, briefing = player.decide(env)

        if action == "accept":
            env.step(role, "accept")
        else:
            env.step(role, "offer", proposal)

        turn_log = {
            "role": role,
            "action": action,
            "proposal": proposal.tolist() if proposal is not None else None,
            "reasoning": reasoning,
            "had_briefing": briefing is not None,
        }
        match_log["turns"].append(turn_log)

        if verbose:
            wm_tag = " [+WM]" if briefing else ""
            if action == "accept":
                print(f"  {role.title()}{wm_tag} ACCEPTS opponent's last proposal")
                print(f"    Reasoning: {reasoning[:120]}")
            else:
                prop_str = ", ".join(f"{VARIABLES[i]}={proposal[i]:.2f}" for i in range(5))
                print(f"  {role.title()}{wm_tag} offers: [{prop_str}]")
                print(f"    Reasoning: {reasoning[:120]}")

        turn_idx += 1
        if turn_idx > env.max_rounds * 2 + 2:
            env.done = True
            break

    # Compute metrics
    metrics = compute_match_metrics(env, buyer_weights, seller_weights)
    metrics["log"] = match_log

    if verbose:
        if metrics["deal"]:
            prop = env.deal_proposal
            prop_str = ", ".join(f"{VARIABLES[i]}={prop[i]:.2f}" for i in range(5))
            print(f"\n  DEAL at round {metrics['rounds']}: [{prop_str}]")
            print(f"  Buyer utility: {metrics['buyer_utility']:.3f} | "
                  f"Seller utility: {metrics['seller_utility']:.3f}")
            print(f"  Joint surplus: {metrics['joint_surplus']:.3f} / {metrics['max_joint_surplus']:.3f} "
                  f"(Pareto efficiency: {metrics['pareto_efficiency']:.0%})")
            print(f"  Buyer share: {metrics['buyer_surplus_share']:.0%}")
        else:
            print(f"\n  NO DEAL — both get utility 0")
            print(f"  Wasted joint surplus: {metrics['max_joint_surplus']:.3f}")

    return metrics


# ===========================================================================
# Tournament Runner
# ===========================================================================

def run_condition(client, condition, model_name, world_model, n_games,
                  max_rounds, dirichlet_alpha, base_seed, verbose=True,
                  ablation=None, randomize_opener=False,
                  seller_model_name=None):
    """Run one experimental condition across n_games.

    Conditions:
      A: no WM on either side
      B: WM on buyer only
      C: WM on both sides

    Ablation modes (applied to WM-equipped agents):
      None: use GRU predictions (default)
      "oracle": use true opponent weights
      "random": use random Dirichlet(1.0) weights
      "uniform": use uniform [0.2]*5 weights
    """
    results = []
    rng = np.random.RandomState(base_seed)

    for i in range(n_games):
        seed = base_seed + i
        game_rng = np.random.RandomState(seed)

        buyer_weights = game_rng.dirichlet(np.ones(5) * dirichlet_alpha)
        seller_weights = game_rng.dirichlet(np.ones(5) * dirichlet_alpha)

        env = NegotiationEnv(buyer_weights, seller_weights, max_rounds=max_rounds, seed=seed)

        buyer_wm = world_model if condition in ("B", "C") else None
        seller_wm = world_model if condition == "C" else None

        s_model = seller_model_name or model_name
        buyer = LLMNegotiator(client, "buyer", buyer_weights, env, model_name,
                              world_model=buyer_wm, ablation=ablation)
        seller = LLMNegotiator(client, "seller", seller_weights, env, s_model,
                               world_model=seller_wm, ablation=ablation)

        opener = (game_rng.choice(["buyer", "seller"])
                  if randomize_opener else "seller")

        if verbose:
            buyer_tag = "+WM" if buyer_wm else "   "
            seller_tag = "+WM" if seller_wm else "   "
            bw_str = ", ".join(f"{VARIABLES[j]}={buyer_weights[j]:.2f}" for j in range(5))
            sw_str = ", ".join(f"{VARIABLES[j]}={seller_weights[j]:.2f}" for j in range(5))
            print(f"\n{'='*70}")
            print(f"Game {i+1}/{n_games} | Condition {condition} | "
                  f"Buyer[{buyer_tag}] vs Seller[{seller_tag}] | Opens: {opener}")
            print(f"Buyer weights:  [{bw_str}]")
            print(f"Seller weights: [{sw_str}]")
            max_joint, _ = compute_pareto_frontier(buyer_weights, seller_weights)
            print(f"Max joint surplus: {max_joint:.3f}")
            print(f"{'='*70}")

        result = run_match(buyer, seller, env, buyer_weights, seller_weights,
                           verbose=verbose, opener=opener)
        results.append(result)

    return results


def print_condition_summary(condition, results):
    """Print summary statistics for one condition."""
    deals = [r for r in results if r["deal"]]
    deal_rate = len(deals) / len(results)

    print(f"\n  Condition {condition}:")
    print(f"    Deal rate:        {len(deals)}/{len(results)} ({deal_rate:.0%})")

    if deals:
        buyer_utils = [r["buyer_utility"] for r in deals]
        seller_utils = [r["seller_utility"] for r in deals]
        joints = [r["joint_surplus"] for r in deals]
        pareto_effs = [r["pareto_efficiency"] for r in deals]
        buyer_shares = [r["buyer_surplus_share"] for r in deals]
        rounds_used = [r["rounds"] for r in deals]

        print(f"    Avg buyer utility:  {np.mean(buyer_utils):.3f} +/- {np.std(buyer_utils):.3f}")
        print(f"    Avg seller utility: {np.mean(seller_utils):.3f} +/- {np.std(seller_utils):.3f}")
        print(f"    Avg joint surplus:  {np.mean(joints):.3f}")
        print(f"    Pareto efficiency:  {np.mean(pareto_effs):.0%} +/- {np.std(pareto_effs):.0%}")
        print(f"    Buyer share:        {np.mean(buyer_shares):.1%} +/- {np.std(buyer_shares):.1%}")
        print(f"    Avg rounds:         {np.mean(rounds_used):.1f}")


# ===========================================================================
# CLI + Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Variable LLM Negotiation with NegotiationGRU")
    parser.add_argument("--games", type=int, default=5, help="Games per condition")
    parser.add_argument("--rounds", type=int, default=8, help="Max rounds per game")
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-haiku",
                        help="OpenRouter model name (used for buyer, or both if --seller-model not set)")
    parser.add_argument("--seller-model", type=str, default=None,
                        help="Separate model for seller (enables cross-model experiments)")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "negotiation_gru.pt",
                        ))
    parser.add_argument("--condition", type=str, default="all",
                        choices=["A", "B", "C", "all"])
    parser.add_argument("--dirichlet", type=float, default=1.0,
                        help="Dirichlet alpha for weight generation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["oracle", "random", "uniform"],
                        help="Ablation mode for WM weight source "
                             "(oracle=true weights, random=Dirichlet(1.0), uniform=0.2 each)")
    parser.add_argument("--randomize-opener", action="store_true",
                        help="Randomly assign buyer or seller as opener each game "
                             "(default: seller always opens)")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path for results JSON (default: results/negotiation_results.json)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load NegotiationGRU
    checkpoint_path = os.path.abspath(args.checkpoint)
    if os.path.exists(checkpoint_path):
        world_model, n_params = load_negotiation_gru(checkpoint_path)
        wm_label = f"NegotiationGRU ({n_params:,} params)"
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")
        print("Run `python3 -m belief_gru.train_negotiation_gru` first.")
        print("Continuing without world model (conditions B/C identical to A)\n")
        world_model = None
        wm_label = "None (checkpoint not found)"

    # LLM client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print("=" * 70)
    print("MULTI-VARIABLE BILATERAL LLM NEGOTIATION")
    print("=" * 70)
    if args.seller_model:
        print(f"Buyer LLM:  {args.model}")
        print(f"Seller LLM: {args.seller_model}")
    else:
        print(f"LLM: {args.model}")
    print(f"World model: {wm_label}")
    print(f"Variables: {', '.join(VARIABLES)}")
    print(f"Games per condition: {args.games} | Max rounds: {args.rounds}")
    print(f"Dirichlet alpha: {args.dirichlet}")
    if args.ablation:
        print(f"Ablation mode: {args.ablation}")
    if args.randomize_opener:
        print(f"Opener: randomized (buyer or seller each game)")

    conditions = ["A", "B", "C"] if args.condition == "all" else [args.condition]
    all_results = {}

    for cond in conditions:
        label = {"A": "Symmetric (no WM)", "B": "Asymmetric (buyer+WM)",
                 "C": "Arms race (both+WM)"}[cond]
        print(f"\n{'#'*70}")
        print(f"# CONDITION {cond}: {label}")
        print(f"{'#'*70}")

        results = run_condition(
            client, cond, args.model, world_model,
            n_games=args.games, max_rounds=args.rounds,
            dirichlet_alpha=args.dirichlet,
            base_seed=args.seed, verbose=not args.quiet,
            ablation=args.ablation,
            randomize_opener=args.randomize_opener,
            seller_model_name=args.seller_model,
        )
        all_results[cond] = results
        print_condition_summary(cond, results)

    # Cross-condition comparison
    if len(conditions) > 1:
        print("\n" + "=" * 70)
        print("CROSS-CONDITION COMPARISON")
        print("=" * 70)

        header = (f"{'Condition':<30} {'Deal%':>7} {'Pareto%':>9} "
                  f"{'Joint':>7} {'BuyerSh':>9} {'Rounds':>8}")
        print(f"\n{header}")
        print("-" * 70)

        for cond in conditions:
            results = all_results[cond]
            deals = [r for r in results if r["deal"]]
            deal_pct = len(deals) / len(results) * 100

            if deals:
                pareto = np.mean([r["pareto_efficiency"] for r in deals]) * 100
                joint = np.mean([r["joint_surplus"] for r in deals])
                b_share = np.mean([r["buyer_surplus_share"] for r in deals]) * 100
                rounds = np.mean([r["rounds"] for r in deals])
            else:
                pareto = joint = b_share = rounds = 0.0

            label = {"A": "Symmetric (no WM)", "B": "Buyer has WM",
                     "C": "Both have WM"}[cond]
            print(f"{label:<30} {deal_pct:>6.0f}% {pareto:>8.0f}% "
                  f"{joint:>7.3f} {b_share:>8.0f}% {rounds:>8.1f}")

        # Key test: WM effect
        if "A" in all_results and "B" in all_results:
            a_deals = [r for r in all_results["A"] if r["deal"]]
            b_deals = [r for r in all_results["B"] if r["deal"]]
            if a_deals and b_deals:
                a_share = np.mean([r["buyer_surplus_share"] for r in a_deals])
                b_share = np.mean([r["buyer_surplus_share"] for r in b_deals])
                shift = b_share - a_share
                print(f"\nWM effect on buyer surplus share: {shift:+.1%}")

                a_pareto = np.mean([r["pareto_efficiency"] for r in a_deals])
                b_pareto = np.mean([r["pareto_efficiency"] for r in b_deals])
                print(f"WM effect on Pareto efficiency: {b_pareto - a_pareto:+.1%}")

                if shift > 0.03:
                    print(">>> World model gives buyer a measurable advantage!")
                elif shift < -0.03:
                    print(">>> World model hurts buyer (unexpected)")
                else:
                    print(">>> No significant buyer advantage (need more games or check briefing)")

        if "B" in all_results and "C" in all_results:
            b_deals = [r for r in all_results["B"] if r["deal"]]
            c_deals = [r for r in all_results["C"] if r["deal"]]
            if b_deals and c_deals:
                b_pareto = np.mean([r["pareto_efficiency"] for r in b_deals])
                c_pareto = np.mean([r["pareto_efficiency"] for r in c_deals])
                c_share = np.mean([r["buyer_surplus_share"] for r in c_deals])
                print(f"\nArms race effect (C vs B):")
                print(f"  Pareto efficiency: {c_pareto:.0%} vs {b_pareto:.0%} ({c_pareto - b_pareto:+.1%})")
                print(f"  Buyer share in C: {c_share:.0%} (expect ~50% if arms race neutralizes)")

    # Save results
    if args.output:
        log_path = args.output
    else:
        log_path = os.path.join(
            os.path.dirname(__file__), "..", "results", "negotiation_results.json"
        )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    serializable = {}
    for cond, results in all_results.items():
        serializable[cond] = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "log"}
            entry["turns"] = r["log"]["turns"] if "log" in r else []
            entry["buyer_weights"] = r["log"]["buyer_weights"] if "log" in r else []
            entry["seller_weights"] = r["log"]["seller_weights"] if "log" in r else []
            serializable[cond].append(entry)

    with open(log_path, "w") as f:
        json.dump({
            "model": args.model,
            "seller_model": args.seller_model,
            "world_model": wm_label,
            "games_per_condition": args.games,
            "max_rounds": args.rounds,
            "dirichlet_alpha": args.dirichlet,
            "conditions_run": conditions,
            "results": serializable,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
