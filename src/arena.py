"""
LLM vs LLM Bargaining Arena — with optional World Model advantage.

Three experimental conditions:
  A. Symmetric:  Opus vs Opus (no world model) → expect ~50/50 surplus split
  B. Asymmetric: Opus+WM vs Opus (world model on buyer only) → does WM help?
  C. Arms race:  Opus+WM vs Opus+WM (both have world model) → back to 50/50?

Usage:
  OPENROUTER_API_KEY=your_key python3 -m belief_gru.arena
  OPENROUTER_API_KEY=your_key python3 -m belief_gru.arena --condition B --games 10
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
from agent_loop import BargainingEnv, build_observation_tensor, generate_strategic_briefing


# ===========================================================================
# LLM Player
# ===========================================================================

BUYER_SYSTEM = """You are a skilled negotiation agent playing as the BUYER.

GAME RULES:
- You negotiate with a seller over a price (range: 0.00 to 1.00)
- Your private valuation (max you'd pay): {valuation:.2f}
- You take turns making offers. You want to buy as LOW as possible.
- Your surplus = valuation - deal_price. Higher is better.
- If no deal in {max_rounds} rounds, BOTH get nothing.
- NEVER offer more than your valuation.

STRATEGY:
- Start with a low anchor offer (~40-50% of your valuation)
- Concede slowly — small increments each round
- Use the World Model briefing (if provided) to read opponent patterns
- A deal with SOME surplus beats no deal (0)
{wm_instructions}
RESPONSE FORMAT (JSON only, no other text):
{{"action": "offer", "price": 0.XX, "reasoning": "brief explanation"}}
or
{{"action": "accept", "reasoning": "brief explanation"}}"""

SELLER_SYSTEM = """You are a skilled negotiation agent playing as the SELLER.

GAME RULES:
- You negotiate with a buyer over a price (range: 0.00 to 1.00)
- Your private reserve price (minimum you'd accept): {valuation:.2f}
- You take turns making offers. You want to sell as HIGH as possible.
- Your surplus = deal_price - reserve. Higher is better.
- If no deal in {max_rounds} rounds, BOTH get nothing.
- NEVER accept less than your reserve price.

STRATEGY:
- Start with a high anchor offer (~80-90% of the price range)
- Concede slowly — small decrements each round
- Use the World Model briefing (if provided) to read opponent patterns
- A deal with SOME surplus beats no deal (0)
{wm_instructions}
RESPONSE FORMAT (JSON only, no other text):
{{"action": "offer", "price": 0.XX, "reasoning": "brief explanation"}}
or
{{"action": "accept", "reasoning": "brief explanation"}}"""

WM_INSTRUCTIONS = """
WORLD MODEL ADVISORY:
- You have a neural world model analyzing the opponent's behavior in real-time
- Pay close attention to the Strategic Briefing — it contains predictions about
  the opponent's likely next move, their predictability (surprise score), and
  estimated remaining value on the table
- When surprise is HIGH, the opponent is deviating from their pattern — be cautious
- When surprise is LOW, you can predict their next move — use that to your advantage
"""

NO_WM_INSTRUCTIONS = ""


class LLMPlayer:
    """An LLM-powered negotiation agent with optional world model."""

    def __init__(self, client, role, valuation, max_rounds, model_name,
                 world_model=None):
        self.client = client
        self.role = role  # "buyer" or "seller"
        self.valuation = valuation
        self.max_rounds = max_rounds
        self.model_name = model_name
        self.world_model = world_model

        has_wm = world_model is not None
        wm_inst = WM_INSTRUCTIONS if has_wm else NO_WM_INSTRUCTIONS

        if role == "buyer":
            self.system_prompt = BUYER_SYSTEM.format(
                valuation=valuation, max_rounds=max_rounds,
                wm_instructions=wm_inst,
            )
        else:
            self.system_prompt = SELLER_SYSTEM.format(
                valuation=valuation, max_rounds=max_rounds,
                wm_instructions=wm_inst,
            )

    def decide(self, env):
        """Produce an action given the current game state.

        Returns:
            action: "offer" or "accept"
            price: float or None
            reasoning: str
            briefing: str or None
        """
        # Build world model briefing if available
        briefing = None
        if self.world_model is not None:
            briefing, _ = generate_strategic_briefing(
                self.world_model, env, perspective=self.role
            )

        # Build game state text
        game_state = self._format_game_state(env)

        user_msg = f"CURRENT GAME STATE:\n{game_state}"
        if briefing:
            user_msg += f"\n\n{briefing}"
        user_msg += "\n\nWhat is your next move? Respond with JSON only."

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON
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

            # Safety clamps
            if action == "offer" and price is not None:
                if self.role == "buyer":
                    price = round(float(np.clip(price, 0.01, self.valuation)), 3)
                else:
                    price = round(float(np.clip(price, self.valuation, 0.99)), 3)

            # Prevent accepting when no offer has been made
            if action == "accept" and not env.history:
                action = "offer"
                price = 0.85 if self.role == "seller" else round(self.valuation * 0.4, 3)
                reasoning = "No offer to accept yet, making opening offer"

            return action, price, reasoning, briefing

        except Exception as e:
            # Fallback
            if self.role == "buyer":
                opp_prices = [p for pl, p in env.history if pl == "seller"]
                price = round(opp_prices[-1] * 0.85, 3) if opp_prices else round(self.valuation * 0.4, 3)
            else:
                opp_prices = [p for pl, p in env.history if pl == "buyer"]
                price = round(opp_prices[-1] * 1.1, 3) if opp_prices else 0.85
                price = min(price, 0.99)
            return "offer", price, f"LLM error: {e}", briefing

    def _format_game_state(self, env):
        lines = [f"Round {env.round}/{env.max_rounds}"]

        my_offers = [p for pl, p in env.history if pl == self.role]
        opp_offers = [p for pl, p in env.history if pl != self.role]
        opp_name = "seller" if self.role == "buyer" else "buyer"

        if opp_offers:
            lines.append(f"{opp_name.title()}'s latest offer: {opp_offers[-1]:.3f}")
        if my_offers:
            lines.append(f"Your last offer: {my_offers[-1]:.3f}")

        lines.append(f"\nFull history:")
        for i, (pl, pr) in enumerate(env.history):
            label = "You" if pl == self.role else opp_name.title()
            lines.append(f"  Turn {i+1}: {label} offered {pr:.3f}")

        if not env.history:
            lines.append("  (No offers yet)")

        return "\n".join(lines)


# ===========================================================================
# Match Runner
# ===========================================================================

def run_match(buyer, seller, env, verbose=True):
    """Run a single bargaining match between two LLM players.

    Returns:
        result dict with deal info, surplus split, round count, logs
    """
    match_log = {
        "buyer_value": env.buyer_value,
        "seller_reserve": env.seller_reserve,
        "zopa": env.zopa,
        "turns": [],
    }

    # Seller opens
    turn_order = ["seller", "buyer"]
    players = {"buyer": buyer, "seller": seller}
    turn_idx = 0

    while not env.done:
        role = turn_order[turn_idx % 2]
        player = players[role]

        action, price, reasoning, briefing = player.decide(env)
        obs, reward, done, info = env.step(role, action, price)

        turn_log = {
            "role": role,
            "action": action,
            "price": price,
            "reasoning": reasoning,
            "had_briefing": briefing is not None,
        }
        match_log["turns"].append(turn_log)

        if verbose:
            wm_tag = " [+WM]" if briefing else ""
            if action == "accept":
                last_price = env.history[-2][1] if len(env.history) >= 2 else "?"
                print(f"  {role.title()}{wm_tag} ACCEPTS at {last_price}")
            else:
                print(f"  {role.title()}{wm_tag} offers: {price:.3f}")
            print(f"    Reasoning: {reasoning}")

        turn_idx += 1

        # Safety: prevent infinite loops
        if turn_idx > env.max_rounds * 2 + 2:
            env.done = True
            break

    # Compute outcome
    if env.deal_price is not None:
        buyer_surplus = env.buyer_value - env.deal_price
        seller_surplus = env.deal_price - env.seller_reserve
        total_surplus = buyer_surplus + seller_surplus
        buyer_share = buyer_surplus / total_surplus if total_surplus > 0 else 0.5

        result = {
            "deal": True,
            "price": env.deal_price,
            "buyer_surplus": buyer_surplus,
            "seller_surplus": seller_surplus,
            "buyer_share": buyer_share,
            "rounds": env.round,
            "log": match_log,
        }

        if verbose:
            print(f"\n  DEAL at {env.deal_price:.3f} (round {env.round})")
            print(f"  Buyer: {buyer_surplus:.3f} ({buyer_share:.0%}) | "
                  f"Seller: {seller_surplus:.3f} ({1-buyer_share:.0%})")
    else:
        result = {
            "deal": False,
            "price": None,
            "buyer_surplus": 0.0,
            "seller_surplus": 0.0,
            "buyer_share": 0.0,
            "rounds": env.round,
            "log": match_log,
        }
        if verbose:
            print(f"\n  NO DEAL — both get 0")

    return result


# ===========================================================================
# Tournament Runner
# ===========================================================================

def run_condition(client, condition, model_name, world_model, n_games,
                  max_rounds, base_seed, verbose=True):
    """Run one experimental condition across n_games.

    Conditions:
      A: no WM on either side
      B: WM on buyer only
      C: WM on both sides
    """
    results = []

    for i in range(n_games):
        seed = base_seed + i
        env = BargainingEnv(max_rounds=max_rounds, seed=seed)

        buyer_wm = world_model if condition in ("B", "C") else None
        seller_wm = world_model if condition == "C" else None

        buyer = LLMPlayer(
            client, "buyer", env.buyer_value, max_rounds,
            model_name, world_model=buyer_wm,
        )
        seller = LLMPlayer(
            client, "seller", env.seller_reserve, max_rounds,
            model_name, world_model=seller_wm,
        )

        if verbose:
            buyer_tag = "+WM" if buyer_wm else "   "
            seller_tag = "+WM" if seller_wm else "   "
            print(f"\n{'='*60}")
            print(f"Game {i+1}/{n_games} | Condition {condition} | "
                  f"Buyer[{buyer_tag}] vs Seller[{seller_tag}]")
            print(f"Value={env.buyer_value:.3f} Reserve={env.seller_reserve:.3f} "
                  f"ZOPA={env.zopa:.3f}")
            print(f"{'='*60}")

        result = run_match(buyer, seller, env, verbose=verbose)
        results.append(result)

    return results


def print_condition_summary(condition, results):
    """Print summary statistics for one condition."""
    deals = [r for r in results if r["deal"]]
    deal_rate = len(deals) / len(results)

    buyer_surps = [r["buyer_surplus"] for r in results]
    seller_surps = [r["seller_surplus"] for r in results]
    buyer_shares = [r["buyer_share"] for r in results if r["deal"]]
    rounds_used = [r["rounds"] for r in results if r["deal"]]

    print(f"\n  Condition {condition}:")
    print(f"    Deal rate:      {len(deals)}/{len(results)} ({deal_rate:.0%})")
    print(f"    Avg buyer surp: {np.mean(buyer_surps):.3f}")
    print(f"    Avg seller surp:{np.mean(seller_surps):.3f}")
    if buyer_shares:
        print(f"    Buyer share:    {np.mean(buyer_shares):.1%} ± {np.std(buyer_shares):.1%}")
        print(f"    Avg rounds:     {np.mean(rounds_used):.1f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM vs LLM Bargaining Arena")
    parser.add_argument("--games", type=int, default=5, help="Games per condition")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model", type=str, default="anthropic/claude-3.5-haiku",
                        help="OpenRouter model name")
    parser.add_argument("--condition", type=str, default="all",
                        choices=["A", "B", "C", "all"],
                        help="Which condition(s) to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load world model (untrained for now — swap checkpoint later)
    world_model = BeliefGRU(obs_dim=6)
    world_model.eval()
    params = sum(p.numel() for p in world_model.parameters())

    # Create LLM client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Set OPENROUTER_API_KEY environment variable")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print("=" * 60)
    print("LLM vs LLM BARGAINING ARENA")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"World model: BeliefGRU v2 ({params:,} params)")
    print(f"Games per condition: {args.games}")
    print(f"Max rounds: {args.max_rounds}")

    conditions = ["A", "B", "C"] if args.condition == "all" else [args.condition]
    all_results = {}

    for cond in conditions:
        label = {"A": "Symmetric (no WM)", "B": "Asymmetric (buyer+WM)",
                 "C": "Arms race (both+WM)"}[cond]
        print(f"\n{'#'*60}")
        print(f"# CONDITION {cond}: {label}")
        print(f"{'#'*60}")

        results = run_condition(
            client, cond, args.model, world_model,
            n_games=args.games, max_rounds=args.max_rounds,
            base_seed=args.seed, verbose=not args.quiet,
        )
        all_results[cond] = results
        print_condition_summary(cond, results)

    # Cross-condition comparison
    if len(conditions) > 1:
        print("\n" + "=" * 60)
        print("CROSS-CONDITION COMPARISON")
        print("=" * 60)
        print(f"\n{'Condition':<30} {'Deal%':>8} {'Buyer Share':>12} {'Avg Rounds':>12}")
        print("-" * 62)

        for cond in conditions:
            results = all_results[cond]
            deals = [r for r in results if r["deal"]]
            deal_pct = len(deals) / len(results) * 100
            shares = [r["buyer_share"] for r in deals] if deals else [0]
            rounds = [r["rounds"] for r in deals] if deals else [0]
            label = {"A": "Symmetric (no WM)", "B": "Buyer has WM",
                     "C": "Both have WM"}[cond]
            print(f"{label:<30} {deal_pct:>7.0f}% {np.mean(shares):>11.1%} {np.mean(rounds):>12.1f}")

        # Key test: does WM shift the buyer's share?
        if "A" in all_results and "B" in all_results:
            a_shares = [r["buyer_share"] for r in all_results["A"] if r["deal"]]
            b_shares = [r["buyer_share"] for r in all_results["B"] if r["deal"]]
            if a_shares and b_shares:
                shift = np.mean(b_shares) - np.mean(a_shares)
                print(f"\nWM effect on buyer share: {shift:+.1%}")
                if shift > 0.03:
                    print(">>> World model gives buyer a measurable advantage!")
                elif shift < -0.03:
                    print(">>> World model hurts buyer (unexpected)")
                else:
                    print(">>> No significant effect (need more games or trained model)")

    # Save results
    log_path = os.path.join(os.path.dirname(__file__), "..", "results", "arena_results.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    serializable = {}
    for cond, results in all_results.items():
        serializable[cond] = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "log"}
            entry["turns"] = r["log"]["turns"] if "log" in r else []
            serializable[cond].append(entry)

    with open(log_path, "w") as f:
        json.dump({
            "model": args.model,
            "games_per_condition": args.games,
            "conditions_run": conditions,
            "results": serializable,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {log_path}")


if __name__ == "__main__":
    main()
