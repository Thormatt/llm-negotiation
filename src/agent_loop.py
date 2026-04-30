"""
Strategic Bargaining Agent Loop — LLM + BeliefGRU World Model

Architecture:
  1. Environment: Alternating-offers bargaining (buyer vs seller)
  2. Text Bridge: BeliefGRU v2 → "Strategic Briefing" for the LLM
  3. LLM Client: OpenRouter API (OpenAI-compatible) → action decision
  4. Evaluation: LLM agent vs heuristic opponent, with full logging

The LLM receives a structured briefing containing:
  - World model's predicted opponent response
  - Estimated surplus (value remaining on the table)
  - Surprise score (how predictable the opponent is being)
  - Trend analysis (is opponent softening or hardening?)

Usage:
  OPENROUTER_API_KEY=your_key python3 -m belief_gru.agent_loop
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

from model import BeliefGRU


# ===========================================================================
# 1. Alternating-Offers Bargaining Environment
# ===========================================================================

class BargainingEnv:
    """Simple alternating-offers bargaining game.

    A buyer and seller negotiate over a price for an item.
    - Seller's reserve (minimum acceptable): drawn from [0.2, 0.4]
    - Buyer's valuation (maximum willingness to pay): drawn from [0.6, 0.9]
    - The "zone of possible agreement" (ZOPA) is the gap between them.
    - Each round, the active player proposes a price.
    - The other player accepts or counters.
    - If no deal after max_rounds, both get 0 (disagreement).

    Observations are 6D to match our world model:
      [proposed_price, round_frac, my_surplus, opp_last_price, price_delta, deal_pressure]
    """

    def __init__(self, seller_reserve=None, buyer_value=None, max_rounds=10, seed=None):
        rng = np.random.RandomState(seed)
        self.seller_reserve = seller_reserve or rng.uniform(0.2, 0.4)
        self.buyer_value = buyer_value or rng.uniform(0.6, 0.9)
        self.max_rounds = max_rounds
        self.round = 0
        self.history = []  # list of (player, price) tuples
        self.done = False
        self.deal_price = None

    @property
    def zopa(self):
        return self.buyer_value - self.seller_reserve

    def get_observation(self, perspective="buyer"):
        """Build 6D observation from the current game state."""
        round_frac = self.round / self.max_rounds

        if not self.history:
            return np.array([0.5, round_frac, 0.0, 0.5, 0.0, round_frac], dtype=np.float32)

        last_player, last_price = self.history[-1]
        opp_last = last_price

        if perspective == "buyer":
            my_surplus = max(0, self.buyer_value - last_price)
        else:
            my_surplus = max(0, last_price - self.seller_reserve)

        # Price delta: how much opponent moved from their previous offer
        opp_prices = [p for pl, p in self.history if pl != perspective]
        if len(opp_prices) >= 2:
            price_delta = opp_prices[-1] - opp_prices[-2]
        else:
            price_delta = 0.0

        deal_pressure = round_frac  # increases as deadline approaches

        return np.array([
            last_price,
            round_frac,
            np.clip(my_surplus, 0, 1),
            opp_last,
            np.clip(price_delta + 0.5, 0, 1),  # centered around 0.5
            deal_pressure,
        ], dtype=np.float32)

    def step(self, player, action, price=None):
        """Process a player's action.

        Args:
            player: "buyer" or "seller"
            action: "offer" or "accept"
            price: float in [0, 1] if action is "offer"

        Returns:
            obs, reward, done, info
        """
        if self.done:
            return self.get_observation(player), 0.0, True, {"reason": "already_done"}

        if action == "accept" and self.history:
            _, last_price = self.history[-1]
            self.done = True
            self.deal_price = last_price
            if player == "buyer":
                reward = self.buyer_value - last_price
            else:
                reward = last_price - self.seller_reserve
            return self.get_observation(player), reward, True, {"reason": "accepted", "price": last_price}

        if action == "offer":
            self.history.append((player, price))
            self.round += 1

            if self.round >= self.max_rounds:
                self.done = True
                return self.get_observation(player), 0.0, True, {"reason": "timeout"}

            return self.get_observation(player), 0.0, False, {"reason": "offer_made", "price": price}

        return self.get_observation(player), 0.0, False, {"reason": "invalid"}


# ===========================================================================
# 2. Heuristic Opponent (Seller Bot)
# ===========================================================================

class HeuristicSeller:
    """A seller that starts high and slowly concedes.

    Strategy: Start at 0.85, concede ~5% of remaining gap each round,
    with some noise. Accepts if offer >= reserve + small margin.
    """

    def __init__(self, reserve, concession_rate=0.05, noise=0.02, seed=None):
        self.reserve = reserve
        self.concession_rate = concession_rate
        self.noise = noise
        self.rng = np.random.RandomState(seed)
        self.initial_ask = 0.85
        self.current_ask = self.initial_ask

    def decide(self, env):
        """Return (action, price) tuple."""
        # Check if we should accept the buyer's last offer
        buyer_offers = [p for pl, p in env.history if pl == "buyer"]
        if buyer_offers:
            last_buyer_offer = buyer_offers[-1]
            accept_threshold = self.reserve + 0.03  # small margin above reserve
            if last_buyer_offer >= accept_threshold:
                return "accept", None

        # Concede: move current ask toward reserve
        gap = self.current_ask - self.reserve
        concession = gap * self.concession_rate + self.rng.normal(0, self.noise)
        self.current_ask = max(self.reserve + 0.02, self.current_ask - abs(concession))

        # Add deadline pressure: concede faster near the end
        round_frac = env.round / env.max_rounds
        if round_frac > 0.7:
            pressure_concession = gap * 0.1 * (round_frac - 0.7) / 0.3
            self.current_ask -= abs(pressure_concession)
            self.current_ask = max(self.reserve + 0.01, self.current_ask)

        return "offer", round(float(self.current_ask), 3)


# ===========================================================================
# 3. Text Bridge: BeliefGRU → Strategic Briefing
# ===========================================================================

def build_observation_tensor(env, perspective="buyer"):
    """Build observation sequence tensor from game history."""
    obs_list = []
    # Reconstruct observations at each timestep
    temp_env = BargainingEnv(
        seller_reserve=env.seller_reserve,
        buyer_value=env.buyer_value,
        max_rounds=env.max_rounds,
    )
    obs_list.append(temp_env.get_observation(perspective))

    for player, price in env.history:
        temp_env.history.append((player, price))
        temp_env.round += 1
        obs_list.append(temp_env.get_observation(perspective))

    # Pad to minimum length for multi-timepoint extraction (need at least 4 steps)
    while len(obs_list) < 8:
        obs_list.append(obs_list[-1])

    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).unsqueeze(0)  # (1, T, 6)
    return obs_tensor


def generate_strategic_briefing(model, env, perspective="buyer"):
    """Run BeliefGRU on game history, produce a text briefing.

    Returns:
        briefing: str — formatted text for LLM consumption
        diagnostics: dict — raw numbers for logging
    """
    obs_tensor = build_observation_tensor(env, perspective)
    T = obs_tensor.shape[1]

    model.eval()
    with torch.no_grad():
        out = model(obs_tensor)

    beliefs = out["beliefs"]          # (1, T, 64)
    surprise = out["mean_surprise"]
    surplus_pred = out["surplus_pred"].item()

    # Get per-step surprise from re-running forward
    # Use the last few gate values to assess opponent predictability
    # Run a simplified version to get gate values
    B = 1
    device = obs_tensor.device
    x_flat = obs_tensor.reshape(B * T, -1)
    z_online = model.obs_encoder(x_flat).reshape(B, T, -1)
    z_target = model.target_encoder(x_flat).reshape(B, T, -1)
    h_all, _ = model.gru(z_online)

    gate_values = []
    b_prev = torch.zeros(B, model.latent_dim, device=device)
    for t in range(T):
        h_t = h_all[:, t]
        pred_z = model.predict_k_step(b_prev, 1)
        target_z = z_target[:, t]
        surprise_vec = torch.abs(pred_z - target_z)
        g_t = torch.sigmoid(model.surprise_mlp(surprise_vec))
        b_t = g_t * h_t + (1 - g_t) * b_prev
        gate_values.append(g_t.item())
        b_prev = b_t

    # Analyze recent gates (last 3 steps)
    recent_gates = gate_values[-3:] if len(gate_values) >= 3 else gate_values
    avg_recent_surprise = np.mean(recent_gates)

    # Trend: is opponent getting more or less predictable?
    if len(gate_values) >= 4:
        early_gates = gate_values[:len(gate_values)//2]
        late_gates = gate_values[len(gate_values)//2:]
        surprise_trend = np.mean(late_gates) - np.mean(early_gates)
    else:
        surprise_trend = 0.0

    # Opponent price trend
    opp_prices = [p for pl, p in env.history if pl != perspective]
    if len(opp_prices) >= 2:
        price_trend = opp_prices[-1] - opp_prices[-2]
        price_direction = "dropping" if price_trend < -0.01 else "rising" if price_trend > 0.01 else "stable"
    elif len(opp_prices) == 1:
        price_direction = "initial offer"
        price_trend = 0.0
    else:
        price_direction = "no data yet"
        price_trend = 0.0

    # Build briefing
    diagnostics = {
        "surplus_pred": surplus_pred,
        "avg_surprise": avg_recent_surprise,
        "surprise_trend": surprise_trend,
        "price_trend": price_trend,
        "gate_values": gate_values,
    }

    # Interpret surprise level
    if avg_recent_surprise > 0.8:
        surprise_label = "HIGH (opponent is unpredictable — they may be bluffing or changing strategy)"
    elif avg_recent_surprise > 0.5:
        surprise_label = "MODERATE (some uncertainty in opponent's behavior)"
    else:
        surprise_label = "LOW (opponent is predictable — their pattern is clear)"

    # Interpret surplus
    if surplus_pred > 0.15:
        surplus_label = "SIGNIFICANT value remaining — room to negotiate harder"
    elif surplus_pred > 0.05:
        surplus_label = "MODERATE value remaining — balanced approach recommended"
    else:
        surplus_label = "THIN margins — consider accepting soon"

    briefing = f"""=== WORLD MODEL STRATEGIC BRIEFING ===
Round: {env.round}/{env.max_rounds}

OPPONENT ANALYSIS:
  Price trend: {price_direction} (last delta: {price_trend:+.3f})
  Surprise level: {surprise_label}
  Surprise trend: {"increasing" if surprise_trend > 0.05 else "decreasing" if surprise_trend < -0.05 else "stable"} (opponent becoming {"less" if surprise_trend < -0.05 else "more" if surprise_trend > 0.05 else "equally"} predictable)

VALUE ASSESSMENT:
  Predicted surplus: {surplus_pred:.3f} — {surplus_label}

RECOMMENDATION:
  Deadline pressure: {"HIGH" if env.round/env.max_rounds > 0.7 else "MODERATE" if env.round/env.max_rounds > 0.4 else "LOW"} ({env.max_rounds - env.round} rounds remaining)
=== END BRIEFING ==="""

    return briefing, diagnostics


# ===========================================================================
# 4. LLM Client (OpenRouter)
# ===========================================================================

def create_llm_client():
    """Create OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


SYSTEM_PROMPT = """You are a skilled negotiation agent playing the role of BUYER in a bargaining game.

GAME RULES:
- You are negotiating with a seller over a price (range: 0.00 to 1.00)
- Your private valuation (maximum you'd pay): {buyer_value:.2f}
- You and the seller take turns making offers
- You want to buy as LOW as possible to maximize your surplus (valuation - price)
- If no deal is reached in {max_rounds} rounds, BOTH players get nothing
- You should NEVER offer more than your valuation

STRATEGY GUIDELINES:
- Use the World Model's Strategic Briefing to inform your decisions
- If the briefing shows the opponent is conceding, be patient
- If surprise is high, the opponent may be bluffing — test with a moderate counter
- As deadline pressure rises, consider accepting reasonable offers
- A deal that gives you SOME surplus is always better than no deal (0)

RESPONSE FORMAT:
You must respond with a JSON object (no other text):
{{"action": "offer", "price": 0.XX, "reasoning": "brief explanation"}}
or
{{"action": "accept", "reasoning": "brief explanation"}}"""


def llm_decide(client, env, briefing, model_name="anthropic/claude-3.5-haiku"):
    """Call LLM to decide next action.

    Args:
        client: OpenAI client configured for OpenRouter
        env: BargainingEnv
        briefing: str from generate_strategic_briefing
        model_name: OpenRouter model identifier

    Returns:
        action: str ("offer" or "accept")
        price: float or None
        reasoning: str
        raw_response: str
    """
    # Build game state summary
    buyer_offers = [p for pl, p in env.history if pl == "buyer"]
    seller_offers = [p for pl, p in env.history if pl == "seller"]

    game_state = f"Round {env.round}/{env.max_rounds}\n"
    if seller_offers:
        game_state += f"Seller's latest ask: {seller_offers[-1]:.3f}\n"
    if buyer_offers:
        game_state += f"Your last offer: {buyer_offers[-1]:.3f}\n"
    game_state += f"\nFull history:\n"
    for i, (pl, pr) in enumerate(env.history):
        game_state += f"  Round {i+1}: {pl} offered {pr:.3f}\n"

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                buyer_value=env.buyer_value,
                max_rounds=env.max_rounds,
            ),
        },
        {
            "role": "user",
            "content": f"CURRENT GAME STATE:\n{game_state}\n\n{briefing}\n\nWhat is your next move? Respond with JSON only.",
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        text = raw
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        decision = json.loads(text)
        action = decision.get("action", "offer")
        price = decision.get("price")
        reasoning = decision.get("reasoning", "")

        # Safety: clamp price to valid range
        if action == "offer" and price is not None:
            price = round(float(np.clip(price, 0.01, env.buyer_value)), 3)

        return action, price, reasoning, raw

    except Exception as e:
        # Fallback: make a reasonable offer
        if seller_offers:
            fallback_price = round(seller_offers[-1] * 0.9, 3)
        else:
            fallback_price = round(env.buyer_value * 0.5, 3)
        return "offer", fallback_price, f"LLM error: {e}", str(e)


# ===========================================================================
# 5. LLM-Free Baseline Buyer (for comparison)
# ===========================================================================

class HeuristicBuyer:
    """A buyer that starts low and slowly concedes. No world model, no LLM."""

    def __init__(self, valuation, concession_rate=0.08, seed=None):
        self.valuation = valuation
        self.concession_rate = concession_rate
        self.rng = np.random.RandomState(seed)
        self.current_bid = valuation * 0.3

    def decide(self, env):
        seller_offers = [p for pl, p in env.history if pl == "seller"]
        if seller_offers and seller_offers[-1] <= self.valuation * 0.55:
            return "accept", None, "good enough price"

        gap = self.valuation - self.current_bid
        concession = gap * self.concession_rate + self.rng.normal(0, 0.01)
        self.current_bid = min(self.valuation - 0.02, self.current_bid + abs(concession))

        round_frac = env.round / env.max_rounds
        if round_frac > 0.7:
            self.current_bid += gap * 0.1 * (round_frac - 0.7) / 0.3

        return "offer", round(float(self.current_bid), 3), "heuristic concession"


# ===========================================================================
# 6. Evaluation Loop
# ===========================================================================

def run_game(client, model, model_name, seed=42, max_rounds=10, verbose=True):
    """Run one full bargaining game: LLM buyer vs heuristic seller."""
    env = BargainingEnv(max_rounds=max_rounds, seed=seed)
    seller = HeuristicSeller(env.seller_reserve, seed=seed + 1)

    if verbose:
        print(f"\n{'='*60}")
        print(f"NEW GAME | Buyer value: {env.buyer_value:.3f} | "
              f"Seller reserve: {env.seller_reserve:.3f} | ZOPA: {env.zopa:.3f}")
        print(f"{'='*60}")

    game_log = {
        "buyer_value": env.buyer_value,
        "seller_reserve": env.seller_reserve,
        "zopa": env.zopa,
        "rounds": [],
        "outcome": None,
    }

    while not env.done:
        current_round = env.round

        # Seller's turn (odd rounds or first move)
        if current_round == 0 or (env.history and env.history[-1][0] == "buyer"):
            action, price = seller.decide(env)
            obs, reward, done, info = env.step("seller", action, price)

            round_log = {"seller": {"action": action, "price": price}}

            if verbose:
                if action == "accept":
                    print(f"  Seller ACCEPTS buyer's offer!")
                else:
                    print(f"  Seller offers: {price:.3f}")

            if done:
                game_log["rounds"].append(round_log)
                break

        # Buyer's turn (LLM with world model)
        briefing, diagnostics = generate_strategic_briefing(model, env, "buyer")
        action, price, reasoning, raw = llm_decide(client, env, briefing, model_name)
        obs, reward, done, info = env.step("buyer", action, price)

        round_log["buyer"] = {
            "action": action,
            "price": price,
            "reasoning": reasoning,
            "surprise": diagnostics["avg_surprise"],
            "surplus_pred": diagnostics["surplus_pred"],
        }
        game_log["rounds"].append(round_log)

        if verbose:
            if action == "accept":
                print(f"  LLM Buyer ACCEPTS at {env.history[-2][1]:.3f}")
            else:
                print(f"  LLM Buyer offers: {price:.3f}")
            print(f"    Reasoning: {reasoning}")
            print(f"    World model: surprise={diagnostics['avg_surprise']:.3f}, "
                  f"surplus_pred={diagnostics['surplus_pred']:.3f}")

    # Game outcome
    if env.deal_price is not None:
        buyer_surplus = env.buyer_value - env.deal_price
        seller_surplus = env.deal_price - env.seller_reserve
        efficiency = (buyer_surplus + seller_surplus) / env.zopa
        game_log["outcome"] = {
            "deal": True,
            "price": env.deal_price,
            "buyer_surplus": buyer_surplus,
            "seller_surplus": seller_surplus,
            "efficiency": efficiency,
            "rounds_used": env.round,
        }
        if verbose:
            print(f"\n  DEAL at {env.deal_price:.3f} (round {env.round})")
            print(f"  Buyer surplus: {buyer_surplus:.3f} | Seller surplus: {seller_surplus:.3f}")
            print(f"  Efficiency: {efficiency:.1%} of ZOPA captured")
    else:
        game_log["outcome"] = {"deal": False, "rounds_used": env.round}
        if verbose:
            print(f"\n  NO DEAL — both get 0 (disagreement)")

    return game_log


def run_baseline_game(seed=42, max_rounds=10, verbose=True):
    """Run one game: heuristic buyer vs heuristic seller (no LLM, no world model)."""
    env = BargainingEnv(max_rounds=max_rounds, seed=seed)
    seller = HeuristicSeller(env.seller_reserve, seed=seed + 1)
    buyer = HeuristicBuyer(env.buyer_value, seed=seed + 2)

    if verbose:
        print(f"\n  Baseline game | value={env.buyer_value:.3f} reserve={env.seller_reserve:.3f}")

    while not env.done:
        # Seller
        if env.round == 0 or (env.history and env.history[-1][0] == "buyer"):
            action, price = seller.decide(env)
            env.step("seller", action, price)
            if env.done:
                break

        # Buyer
        action, price, reasoning = buyer.decide(env)
        env.step("buyer", action, price)

    if env.deal_price is not None:
        surplus = env.buyer_value - env.deal_price
        if verbose:
            print(f"  Baseline: deal at {env.deal_price:.3f}, surplus={surplus:.3f}, round={env.round}")
        return surplus, env.round
    else:
        if verbose:
            print(f"  Baseline: no deal")
        return 0.0, env.round


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM + BeliefGRU Bargaining Agent")
    parser.add_argument("--games", type=int, default=5, help="Number of games to play")
    parser.add_argument("--max-rounds", type=int, default=10, help="Max rounds per game")
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-haiku",
                        help="OpenRouter model name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load world model
    world_model = BeliefGRU(obs_dim=6)
    params = sum(p.numel() for p in world_model.parameters())
    print(f"World model loaded: BeliefGRU v2 ({params:,} params)")

    if args.baseline_only:
        print("\n" + "=" * 60)
        print("BASELINE: Heuristic Buyer vs Heuristic Seller")
        print("=" * 60)
        surpluses = []
        for i in range(args.games):
            surplus, rounds = run_baseline_game(seed=args.seed + i, max_rounds=args.max_rounds)
            surpluses.append(surplus)
        avg = np.mean(surpluses)
        deals = sum(1 for s in surpluses if s > 0)
        print(f"\nBaseline: {deals}/{args.games} deals, avg surplus: {avg:.3f}")
        return

    # Create LLM client
    client = create_llm_client()
    print(f"LLM: {args.model} via OpenRouter")

    # Run games
    print(f"\nRunning {args.games} games (LLM buyer vs heuristic seller)...")

    all_logs = []
    llm_surpluses = []
    baseline_surpluses = []

    for i in range(args.games):
        seed = args.seed + i

        # LLM game
        log = run_game(client, world_model, args.model, seed=seed,
                       max_rounds=args.max_rounds, verbose=True)
        all_logs.append(log)

        if log["outcome"]["deal"]:
            llm_surpluses.append(log["outcome"]["buyer_surplus"])
        else:
            llm_surpluses.append(0.0)

        # Baseline game (same seed for fair comparison)
        surplus, _ = run_baseline_game(seed=seed, max_rounds=args.max_rounds, verbose=False)
        baseline_surpluses.append(surplus)

    # Summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    llm_deals = sum(1 for s in llm_surpluses if s > 0)
    base_deals = sum(1 for s in baseline_surpluses if s > 0)

    print(f"\n{'Metric':<30} {'LLM+WorldModel':>15} {'Heuristic':>15}")
    print("-" * 60)
    print(f"{'Deal rate':<30} {llm_deals}/{args.games:>13} {base_deals}/{args.games:>13}")
    print(f"{'Avg buyer surplus':<30} {np.mean(llm_surpluses):>15.3f} {np.mean(baseline_surpluses):>15.3f}")

    if llm_surpluses and baseline_surpluses:
        llm_avg = np.mean(llm_surpluses)
        base_avg = np.mean(baseline_surpluses)
        if base_avg > 0:
            improvement = (llm_avg - base_avg) / base_avg * 100
            print(f"{'Improvement':<30} {improvement:>+14.1f}%")

    # Per-game comparison
    print(f"\n{'Game':<8} {'LLM Surplus':>12} {'Base Surplus':>12} {'Winner':>10}")
    print("-" * 42)
    llm_wins = 0
    for i in range(args.games):
        winner = "LLM" if llm_surpluses[i] > baseline_surpluses[i] else "Baseline" if baseline_surpluses[i] > llm_surpluses[i] else "Tie"
        if winner == "LLM":
            llm_wins += 1
        print(f"{i+1:<8} {llm_surpluses[i]:>12.3f} {baseline_surpluses[i]:>12.3f} {winner:>10}")

    print(f"\nLLM wins: {llm_wins}/{args.games}")

    # Save logs
    log_path = os.path.join(os.path.dirname(__file__), "..", "results", "agent_loop_results.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "model": args.model,
            "games": args.games,
            "llm_surpluses": llm_surpluses,
            "baseline_surpluses": baseline_surpluses,
            "logs": all_logs,
        }, f, indent=2, default=str)
    print(f"\nFull logs saved to: {log_path}")


if __name__ == "__main__":
    main()
