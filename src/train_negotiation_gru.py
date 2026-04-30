"""
NegotiationGRU — Train a GRU to predict opponent utility weights from
multi-variable bilateral negotiation behavior.

5 continuous deal variables [0,1]: Price, Contract Length, Payment Terms,
SLA Tier, Data Rights. Asymmetric utility weights from Dirichlet(1.0).
Heuristic agents concede inversely proportional to their weights, creating
a learnable signal the GRU exploits.

Trained entirely on heuristic-vs-heuristic matches (no API calls needed).

Usage:
  python3 -m belief_gru.train_negotiation_gru                    # train + eval
  python3 -m belief_gru.train_negotiation_gru --matches 20000    # more data
  python3 -m belief_gru.train_negotiation_gru --dirichlet 0.5    # sharper weights
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ===========================================================================
# Deal Variables & Utility Model
# ===========================================================================

VARIABLES = ["price", "contract_length", "payment_terms", "sla_tier", "data_rights"]

# Buyer wants: low price, long contract, slow payment, high SLA, client owns data
BUYER_DIRECTION = np.array([-1, +1, -1, +1, +1], dtype=np.float32)
# Seller wants the opposite
SELLER_DIRECTION = np.array([+1, -1, +1, -1, -1], dtype=np.float32)

MAX_ROUNDS = 8


def compute_utility(proposal, weights, role):
    """Compute utility of a 5D proposal given weights and role.

    Each variable in [0,1]. Direction adjusts so higher = better for that role.
    Utility = weighted sum of direction-adjusted values.
    """
    direction = BUYER_DIRECTION if role == "buyer" else SELLER_DIRECTION
    proposal = np.asarray(proposal, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    adjusted = np.where(direction > 0, proposal, 1.0 - proposal)
    return float(np.dot(weights, adjusted))


# ===========================================================================
# NegotiationEnv
# ===========================================================================

class NegotiationEnv:
    """Multi-variable bilateral negotiation environment.

    Two agents alternate 5D proposals over max_rounds rounds.
    Seller opens. Accept = deal closes at opponent's last proposal.
    BATNA: utility = 0 for both if no deal.
    """

    def __init__(self, buyer_weights, seller_weights, max_rounds=MAX_ROUNDS, seed=None):
        self.buyer_weights = np.asarray(buyer_weights, dtype=np.float32)
        self.seller_weights = np.asarray(seller_weights, dtype=np.float32)
        self.max_rounds = max_rounds
        self.rng = np.random.RandomState(seed)

        self.round = 0
        self.history = []  # [(role, proposal_5d), ...]
        self.done = False
        self.deal_proposal = None  # 5D proposal of the accepted deal

    def step(self, role, action, proposal=None):
        """Process one turn.

        Args:
            role: "buyer" or "seller"
            action: "offer" or "accept"
            proposal: np.ndarray shape (5,) if action == "offer"

        Returns:
            (done, info) where info has deal details if done
        """
        if self.done:
            return True, {"reason": "already_done"}

        if action == "accept":
            # Accept opponent's last proposal
            opp_proposals = [p for r, p in self.history if r != role]
            if not opp_proposals:
                # Nothing to accept — treat as offer
                action = "offer"
            else:
                self.deal_proposal = opp_proposals[-1]
                self.done = True
                return True, {
                    "reason": "accepted",
                    "deal": self.deal_proposal,
                    "buyer_utility": compute_utility(self.deal_proposal, self.buyer_weights, "buyer"),
                    "seller_utility": compute_utility(self.deal_proposal, self.seller_weights, "seller"),
                }

        if action == "offer":
            proposal = np.asarray(proposal, dtype=np.float32).clip(0.0, 1.0)
            self.history.append((role, proposal.copy()))
            self.round += 1

            if self.round >= self.max_rounds:
                self.done = True
                return True, {"reason": "timeout"}

            return False, {"reason": "continue"}

        return False, {"reason": "unknown_action"}

    def get_last_proposal(self, role):
        """Get the last proposal made BY this role, or None."""
        for r, p in reversed(self.history):
            if r == role:
                return p.copy()
        return None

    def get_opponent_proposals(self, role):
        """Get all proposals made by the opponent."""
        return [p.copy() for r, p in self.history if r != role]


# ===========================================================================
# Feature Extraction
# ===========================================================================

OBS_DIM = 11  # 5 per-variable concession + 6 aggregate features


def extract_features(env, perspective):
    """Extract an 11D feature vector from the current state.

    Dims 0-4: per-variable cumulative concession (absolute change from opening).
              Directly encodes which variables opponent concedes on most/least.
    Dims 5-10: aggregate features (centroid, round_frac, utility_delta,
              concession_magnitude, variable_spread, max_concession_channel).

    Args:
        env: NegotiationEnv
        perspective: "buyer" or "seller" — who is observing

    Returns:
        np.ndarray shape (11,) or None if no opponent data yet
    """
    opponent = "seller" if perspective == "buyer" else "buyer"
    opp_proposals = [p for r, p in env.history if r == opponent]

    if not opp_proposals:
        return None

    latest = opp_proposals[-1]
    features = np.zeros(OBS_DIM, dtype=np.float32)

    # Dims 0-4: per-variable cumulative concession (absolute)
    if len(opp_proposals) >= 2:
        cumulative = np.abs(opp_proposals[-1] - opp_proposals[0])
        features[0:5] = cumulative
    else:
        features[0:5] = 0.0

    # Dim 5: proposal centroid — mean of opponent's latest 5D proposal
    features[5] = np.mean(latest)

    # Dim 6: round fraction — deadline pressure context
    features[6] = env.round / env.max_rounds

    # Dim 7: utility delta — change in uniform-weight utility from last 2 proposals
    if len(opp_proposals) >= 2:
        uniform_w = np.ones(5, dtype=np.float32) / 5.0
        u_now = compute_utility(opp_proposals[-1], uniform_w, perspective)
        u_prev = compute_utility(opp_proposals[-2], uniform_w, perspective)
        features[7] = u_now - u_prev
    else:
        features[7] = 0.0

    # Dim 8: concession magnitude — L2 norm of proposal delta
    if len(opp_proposals) >= 2:
        delta = opp_proposals[-1] - opp_proposals[-2]
        features[8] = float(np.linalg.norm(delta))
    else:
        features[8] = 0.0

    # Dim 9: variable spread — std of per-variable cumulative concession rates
    if len(opp_proposals) >= 2:
        total_delta = opp_proposals[-1] - opp_proposals[0]
        features[9] = float(np.std(total_delta))
    else:
        features[9] = 0.0

    # Dim 10: max concession channel — argmax of cumulative concession / 4.0
    if len(opp_proposals) >= 2:
        total_delta = np.abs(opp_proposals[-1] - opp_proposals[0])
        features[10] = float(np.argmax(total_delta)) / 4.0
    else:
        features[10] = 0.0

    return features


def build_observation_sequence(env, perspective):
    """Build padded observation tensor (1, T, 6) from full negotiation history.

    Replays history step by step, extracting features at each point where
    the opponent has made at least one proposal. Pads to max_rounds.

    Returns:
        obs: torch.Tensor shape (1, T, 6) where T = max_rounds
        length: int — actual number of valid timesteps
    """
    opponent = "seller" if perspective == "buyer" else "buyer"
    max_t = env.max_rounds
    obs = np.zeros((max_t, OBS_DIM), dtype=np.float32)

    # Replay environment to extract features at each timestep
    replay = NegotiationEnv(env.buyer_weights, env.seller_weights, env.max_rounds)
    t = 0
    for role, proposal in env.history:
        replay.step(role, "offer", proposal)
        if role == opponent:
            feats = extract_features(replay, perspective)
            if feats is not None and t < max_t:
                obs[t] = feats
                t += 1

    length = max(t, 1)

    # Pad remaining with last valid observation
    if t > 0 and t < max_t:
        obs[t:] = obs[t - 1]
    elif t == 0:
        obs[:] = 0.0

    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0), length


# ===========================================================================
# HeuristicNegotiator
# ===========================================================================

class HeuristicNegotiator:
    """Heuristic agent whose concession rate leaks its utility weights.

    Key behavior: concession per variable = (1.0 - weight_i)^2 * base_rate.
    Squared mapping amplifies asymmetry: high-weight variables barely move,
    low-weight variables concede aggressively. Noise is low (0.01) to keep
    the weight-dependent signal clean.
    """

    def __init__(self, role, weights, aggression=0.5, noise=0.01, seed=None):
        self.role = role
        self.weights = np.asarray(weights, dtype=np.float32)
        self.aggression = aggression
        self.noise = noise
        self.rng = np.random.RandomState(seed)

        direction = BUYER_DIRECTION if role == "buyer" else SELLER_DIRECTION
        # Start with aggressive opening: values that maximize own utility
        self.current_proposal = np.where(direction > 0, 0.85, 0.15).astype(np.float32)
        self.rounds_offered = 0

    def decide(self, env):
        """Decide action and proposal.

        Returns:
            (action, proposal_5d)
        """
        # Check if should accept opponent's last proposal
        opp_proposals = env.get_opponent_proposals(self.role)
        if opp_proposals:
            opp_latest = opp_proposals[-1]
            my_utility = compute_utility(opp_latest, self.weights, self.role)
            # Rising threshold: more willing to accept as deadline approaches
            round_frac = env.round / env.max_rounds
            accept_threshold = 0.35 + 0.15 * round_frac
            if my_utility >= accept_threshold:
                return "accept", opp_latest

        # Concede: each variable moves proportional to (1 - weight)^2
        # Squared mapping: w=0.4 -> 0.36*base, w=0.1 -> 0.81*base (2.25x ratio)
        direction = BUYER_DIRECTION if self.role == "buyer" else SELLER_DIRECTION
        base_rate = 0.04 + 0.08 * self.aggression  # [0.04, 0.12]

        concessions = np.zeros(5, dtype=np.float32)
        for i in range(5):
            concession_speed = (1.0 - self.weights[i]) ** 2 * base_rate
            noise = self.rng.normal(0, self.noise)
            concessions[i] = concession_speed + noise

        # Deadline pressure: accelerate concessions in final 30%
        round_frac = env.round / env.max_rounds
        if round_frac > 0.7:
            pressure = 1.0 + 2.0 * (round_frac - 0.7) / 0.3
            concessions *= pressure

        # Apply concessions: move toward opponent's preferred direction
        for i in range(5):
            if direction[i] > 0:
                self.current_proposal[i] -= abs(concessions[i])
            else:
                self.current_proposal[i] += abs(concessions[i])

        self.current_proposal = self.current_proposal.clip(0.05, 0.95)
        self.rounds_offered += 1

        return "offer", self.current_proposal.copy()


# ===========================================================================
# NegotiationGRU
# ===========================================================================

class NegotiationGRU(nn.Module):
    """GRU that predicts opponent utility weights from negotiation behavior.

    Architecture: CollusionGRU encoder+GRU body + dual head
      - Weight head: 5-way softmax (predicted opponent weights)
      - Threshold head: sigmoid (predicted acceptance threshold)

    11,078 params (encoder 2,976; gru 6,336; weight_head 1,221; threshold_head 545).
    """

    def __init__(self, obs_dim=OBS_DIM, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: same as CollusionGRU
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Temporal integrator
        self.gru = nn.GRU(latent_dim, latent_dim, batch_first=True)

        # Weight prediction head: 5-way softmax
        self.weight_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 5),
        )

        # Acceptance threshold head: sigmoid
        self.threshold_head = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs, lengths=None):
        """
        Args:
            obs: (B, T, 6) observation sequences
            lengths: (B,) actual sequence lengths

        Returns:
            weight_pred: (B, 5) predicted opponent weights (softmax, sums to 1)
            threshold_pred: (B,) predicted acceptance threshold
            beliefs: (B, T, latent_dim) hidden state trajectory
        """
        B, T, D = obs.shape
        encoded = self.encoder(obs)       # (B, T, latent_dim)
        beliefs, _ = self.gru(encoded)    # (B, T, latent_dim)

        # Read at 3T//4 — more data = better prediction
        if lengths is not None:
            read_idx = (3 * lengths // 4).clamp(min=0, max=T - 1)
            pooled = beliefs[torch.arange(B), read_idx]
        else:
            pooled = beliefs[:, 3 * T // 4]

        weight_logits = self.weight_head(pooled)       # (B, 5)
        weight_pred = F.softmax(weight_logits, dim=-1)  # (B, 5)

        threshold_pred = self.threshold_head(pooled).squeeze(-1)  # (B,)

        return weight_pred, threshold_pred, beliefs


def load_negotiation_gru(path):
    """Load trained NegotiationGRU from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NegotiationGRU(**ckpt["hyperparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    return model, params


# ===========================================================================
# Data Generation
# ===========================================================================

def generate_training_data(n_matches=10000, max_rounds=MAX_ROUNDS, dirichlet_alpha=1.0, seed=42):
    """Generate training data from heuristic-vs-heuristic negotiation matches.

    Each match: Dirichlet(alpha) weights for each side, random aggression.
    Extract features from BOTH perspectives -> 2 samples per match.

    Returns:
        obs: (2*n_matches, T, 6) observation sequences
        target_weights: (2*n_matches, 5) opponent's true weights
        target_thresholds: (2*n_matches,) opponent's effective accept threshold
        lengths: (2*n_matches,) actual sequence lengths
    """
    rng = np.random.RandomState(seed)

    all_obs = []
    all_weights = []
    all_thresholds = []
    all_lengths = []

    for i in range(n_matches):
        buyer_w = rng.dirichlet(np.ones(5) * dirichlet_alpha)
        seller_w = rng.dirichlet(np.ones(5) * dirichlet_alpha)

        buyer_aggression = rng.uniform(0.2, 0.8)
        seller_aggression = rng.uniform(0.2, 0.8)

        env = NegotiationEnv(buyer_w, seller_w, max_rounds=max_rounds, seed=seed + i)
        buyer = HeuristicNegotiator("buyer", buyer_w, buyer_aggression, seed=seed + i * 2)
        seller = HeuristicNegotiator("seller", seller_w, seller_aggression, seed=seed + i * 2 + 1)

        # Run match
        turn_order = ["seller", "buyer"]
        turn_idx = 0
        while not env.done:
            role = turn_order[turn_idx % 2]
            agent = buyer if role == "buyer" else seller
            action, proposal = agent.decide(env)
            env.step(role, action, proposal)
            turn_idx += 1
            if turn_idx > max_rounds * 2 + 2:
                break

        # Extract from buyer's perspective: target = seller's weights
        obs_b, len_b = build_observation_sequence(env, "buyer")
        all_obs.append(obs_b.squeeze(0).numpy())
        all_weights.append(seller_w)
        all_thresholds.append(0.35 + 0.15 * (env.round / max_rounds))
        all_lengths.append(len_b)

        # Extract from seller's perspective: target = buyer's weights
        obs_s, len_s = build_observation_sequence(env, "seller")
        all_obs.append(obs_s.squeeze(0).numpy())
        all_weights.append(buyer_w)
        all_thresholds.append(0.35 + 0.15 * (env.round / max_rounds))
        all_lengths.append(len_s)

    obs = torch.tensor(np.array(all_obs), dtype=torch.float32)
    target_weights = torch.tensor(np.array(all_weights), dtype=torch.float32)
    target_thresholds = torch.tensor(np.array(all_thresholds), dtype=torch.float32)
    lengths = torch.tensor(all_lengths, dtype=torch.long)

    return obs, target_weights, target_thresholds, lengths


# ===========================================================================
# Training
# ===========================================================================

def train_negotiation_gru(obs, target_weights, target_thresholds, lengths,
                          epochs=50, lr=1e-3, batch_size=128, seed=42):
    """Train NegotiationGRU on heuristic match data.

    Loss: KL divergence on weights + MSE on threshold.
    Adam optimizer, CosineAnnealing scheduler, early stopping on val KL.
    """
    torch.manual_seed(seed)
    n_total = obs.shape[0]
    n_train = int(0.8 * n_total)

    # Shuffle
    perm = torch.randperm(n_total)
    obs, target_weights, target_thresholds, lengths = (
        obs[perm], target_weights[perm], target_thresholds[perm], lengths[perm]
    )

    train_ds = TensorDataset(obs[:n_train], target_weights[:n_train],
                             target_thresholds[:n_train], lengths[:n_train])
    val_ds = TensorDataset(obs[n_train:], target_weights[n_train:],
                           target_thresholds[n_train:], lengths[n_train:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = NegotiationGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNegotiationGRU: {n_params:,} params")
    print(f"Training: {n_train} samples | Validation: {n_total - n_train} samples")
    print(f"Epochs: {epochs} | LR: {lr} | Batch: {batch_size}\n")

    best_val_kl = float("inf")
    patience = 8
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_kl_sum = 0.0
        train_mse_sum = 0.0
        train_n = 0

        for batch_obs, batch_w, batch_t, batch_l in train_dl:
            w_pred, t_pred, _ = model(batch_obs, batch_l)

            # KL divergence: D_KL(target || pred) for weight distributions
            kl_loss = F.kl_div(
                w_pred.log().clamp(min=-10),
                batch_w,
                reduction="batchmean",
                log_target=False,
            )
            mse_loss = F.mse_loss(t_pred, batch_t)
            loss = kl_loss + 0.5 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = batch_obs.shape[0]
            train_kl_sum += kl_loss.item() * bs
            train_mse_sum += mse_loss.item() * bs
            train_n += bs

        scheduler.step()

        # Validate
        model.eval()
        val_kl_sum = 0.0
        val_mse_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for batch_obs, batch_w, batch_t, batch_l in val_dl:
                w_pred, t_pred, _ = model(batch_obs, batch_l)

                kl_loss = F.kl_div(
                    w_pred.log().clamp(min=-10),
                    batch_w,
                    reduction="batchmean",
                    log_target=False,
                )
                mse_loss = F.mse_loss(t_pred, batch_t)

                bs = batch_obs.shape[0]
                val_kl_sum += kl_loss.item() * bs
                val_mse_sum += mse_loss.item() * bs
                val_n += bs

        train_kl = train_kl_sum / train_n
        val_kl = val_kl_sum / val_n
        val_mse = val_mse_sum / val_n

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train KL: {train_kl:.4f} | "
                  f"Val KL: {val_kl:.4f} | Val MSE: {val_mse:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Early stopping
        if val_kl < best_val_kl:
            best_val_kl = val_kl
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (best val KL: {best_val_kl:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_kl


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_weight_prediction(model, obs, target_weights, lengths):
    """Evaluate quality of weight predictions.

    Metrics:
      - Cosine similarity (target > 0.8)
      - Top-1 accuracy: correct most-valued variable (target > 60%, baseline 20%)
      - Per-variable MAE
    """
    model.eval()
    with torch.no_grad():
        w_pred, t_pred, _ = model(obs, lengths)

    w_pred_np = w_pred.numpy()
    w_true_np = target_weights.numpy()

    # Cosine similarity
    cos_sims = []
    for i in range(len(w_pred_np)):
        dot = np.dot(w_pred_np[i], w_true_np[i])
        norm_pred = np.linalg.norm(w_pred_np[i])
        norm_true = np.linalg.norm(w_true_np[i])
        if norm_pred > 0 and norm_true > 0:
            cos_sims.append(dot / (norm_pred * norm_true))
    cos_sim = float(np.mean(cos_sims))

    # Top-1 accuracy
    pred_top1 = np.argmax(w_pred_np, axis=1)
    true_top1 = np.argmax(w_true_np, axis=1)
    top1_acc = float(np.mean(pred_top1 == true_top1))

    # Per-variable MAE
    mae_per_var = np.mean(np.abs(w_pred_np - w_true_np), axis=0)

    return {
        "cosine_similarity": cos_sim,
        "top1_accuracy": top1_acc,
        "per_variable_mae": {VARIABLES[i]: float(mae_per_var[i]) for i in range(5)},
        "mean_mae": float(np.mean(mae_per_var)),
    }


# ===========================================================================
# LLM Telemetry Extraction (Sim-to-Real)
# ===========================================================================

def extract_telemetry_from_results(results_path):
    """Extract GRU training data from LLM negotiation results JSON.

    Replays each game's proposal history through NegotiationEnv to extract
    the same 11D features used during heuristic training, but mapped to
    the actual LLM concession patterns.

    Returns:
        obs, target_weights, target_thresholds, lengths (same format as generate_training_data)
    """
    import json as _json
    with open(results_path) as f:
        data = _json.load(f)

    all_obs = []
    all_weights = []
    all_thresholds = []
    all_lengths = []

    for cond, games in data.get("results", {}).items():
        for game in games:
            buyer_w = np.array(game["buyer_weights"], dtype=np.float32)
            seller_w = np.array(game["seller_weights"], dtype=np.float32)
            turns = game.get("turns", [])
            max_rounds = data.get("max_rounds", MAX_ROUNDS)

            if len(turns) < 2:
                continue

            # Replay the game
            env = NegotiationEnv(buyer_w, seller_w, max_rounds=max_rounds)
            for turn in turns:
                role = turn["role"]
                action = turn["action"]
                proposal = turn.get("proposal")
                if action == "offer" and proposal is not None:
                    env.step(role, "offer", np.array(proposal, dtype=np.float32))
                elif action == "accept":
                    env.step(role, "accept")

            # Extract from buyer perspective: target = seller weights
            obs_b, len_b = build_observation_sequence(env, "buyer")
            if len_b >= 2:
                all_obs.append(obs_b.squeeze(0).numpy())
                all_weights.append(seller_w)
                all_thresholds.append(0.35 + 0.15 * (env.round / max_rounds))
                all_lengths.append(len_b)

            # Extract from seller perspective: target = buyer weights
            obs_s, len_s = build_observation_sequence(env, "seller")
            if len_s >= 2:
                all_obs.append(obs_s.squeeze(0).numpy())
                all_weights.append(buyer_w)
                all_thresholds.append(0.35 + 0.15 * (env.round / max_rounds))
                all_lengths.append(len_s)

    if not all_obs:
        raise ValueError(f"No valid games found in {results_path}")

    obs = torch.tensor(np.array(all_obs), dtype=torch.float32)
    target_weights = torch.tensor(np.array(all_weights), dtype=torch.float32)
    target_thresholds = torch.tensor(np.array(all_thresholds), dtype=torch.float32)
    lengths = torch.tensor(all_lengths, dtype=torch.long)

    return obs, target_weights, target_thresholds, lengths


def finetune_negotiation_gru(checkpoint_path, obs, target_weights, target_thresholds,
                             lengths, epochs=30, lr=1e-4, batch_size=32, seed=42):
    """Fine-tune existing NegotiationGRU on LLM telemetry data.

    Low LR + small batches to prevent catastrophic forgetting of the heuristic
    "physics" while learning LLM "psychology". Classic Sim-to-Real transfer.
    """
    torch.manual_seed(seed)

    # Load pretrained checkpoint
    model, n_params = load_negotiation_gru(checkpoint_path)
    print(f"\nFine-tuning NegotiationGRU ({n_params:,} params)")
    print(f"Loaded pretrained weights from: {checkpoint_path}")

    n_total = obs.shape[0]
    n_train = max(int(0.8 * n_total), 1)

    perm = torch.randperm(n_total)
    obs, target_weights, target_thresholds, lengths = (
        obs[perm], target_weights[perm], target_thresholds[perm], lengths[perm]
    )

    train_ds = TensorDataset(obs[:n_train], target_weights[:n_train],
                             target_thresholds[:n_train], lengths[:n_train])
    val_ds = TensorDataset(obs[n_train:], target_weights[n_train:],
                           target_thresholds[n_train:], lengths[n_train:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size) if len(val_ds) > 0 else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Train: {n_train} samples | Val: {n_total - n_train} samples")
    print(f"Epochs: {epochs} | LR: {lr} | Batch: {batch_size}\n")

    best_val_kl = float("inf")
    patience = 10
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_kl_sum = 0.0
        train_n = 0

        for batch_obs, batch_w, batch_t, batch_l in train_dl:
            w_pred, t_pred, _ = model(batch_obs, batch_l)

            kl_loss = F.kl_div(
                w_pred.log().clamp(min=-10), batch_w,
                reduction="batchmean", log_target=False,
            )
            mse_loss = F.mse_loss(t_pred, batch_t)
            loss = kl_loss + 0.5 * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            bs = batch_obs.shape[0]
            train_kl_sum += kl_loss.item() * bs
            train_n += bs

        scheduler.step()
        train_kl = train_kl_sum / train_n

        # Validate
        if val_dl is not None:
            model.eval()
            val_kl_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch_obs, batch_w, batch_t, batch_l in val_dl:
                    w_pred, t_pred, _ = model(batch_obs, batch_l)
                    kl_loss = F.kl_div(
                        w_pred.log().clamp(min=-10), batch_w,
                        reduction="batchmean", log_target=False,
                    )
                    val_kl_sum += kl_loss.item() * batch_obs.shape[0]
                    val_n += batch_obs.shape[0]
            val_kl = val_kl_sum / val_n if val_n > 0 else train_kl
        else:
            val_kl = train_kl

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train KL: {train_kl:.4f} | Val KL: {val_kl:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_kl < best_val_kl:
            best_val_kl = val_kl
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (best val KL: {best_val_kl:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_kl


# ===========================================================================
# CLI + Main
# ===========================================================================

def _save_checkpoint(model, metrics, ckpt_path):
    """Save model checkpoint with hyperparams and metrics."""
    torch.save({
        "hyperparams": {
            "obs_dim": model.obs_dim,
            "hidden_dim": model.hidden_dim,
            "latent_dim": model.latent_dim,
        },
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }, ckpt_path)
    print(f"\nCheckpoint saved to: {ckpt_path}")


def _print_eval(metrics):
    """Print evaluation metrics."""
    print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}  (target > 0.80)")
    print(f"  Top-1 accuracy:    {metrics['top1_accuracy']:.1%}  (target > 60%, baseline 20%)")
    print(f"  Mean MAE:          {metrics['mean_mae']:.4f}")
    print(f"  Per-variable MAE:")
    for var, mae in metrics["per_variable_mae"].items():
        print(f"    {var:20s}: {mae:.4f}")

    cos_ok = metrics["cosine_similarity"] > 0.80
    top1_ok = metrics["top1_accuracy"] > 0.60
    print(f"\n{'PASS' if cos_ok else 'FAIL'}: Cosine similarity {metrics['cosine_similarity']:.4f} {'>' if cos_ok else '<='} 0.80")
    print(f"{'PASS' if top1_ok else 'FAIL'}: Top-1 accuracy {metrics['top1_accuracy']:.1%} {'>' if top1_ok else '<='} 60%")
    return cos_ok and top1_ok


def main():
    parser = argparse.ArgumentParser(description="Train / fine-tune NegotiationGRU")
    sub = parser.add_subparsers(dest="mode", help="Mode")

    # Train from scratch on heuristic data
    train_p = sub.add_parser("train", help="Train from scratch on heuristic matches")
    train_p.add_argument("--matches", type=int, default=10000)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--batch-size", type=int, default=128)
    train_p.add_argument("--dirichlet", type=float, default=1.0)
    train_p.add_argument("--seed", type=int, default=42)

    # Evaluate pretrained GRU on LLM game trajectories (NO fine-tuning)
    eval_p = sub.add_parser("eval-llm", help="Evaluate pretrained GRU on LLM game logs")
    eval_p.add_argument("--results", type=str, required=True,
                        help="Path to negotiation_results.json with LLM game logs")
    eval_p.add_argument("--checkpoint", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "negotiation_gru.pt",
                        ))
    eval_p.add_argument("--condition", type=str, default=None,
                        help="Only evaluate on a specific condition (A, B, C)")

    # Fine-tune on LLM telemetry (Sim-to-Real)
    ft_p = sub.add_parser("finetune", help="Fine-tune pretrained GRU on LLM game logs")
    ft_p.add_argument("--results", type=str, required=True,
                      help="Path to negotiation_results.json with LLM game logs")
    ft_p.add_argument("--checkpoint", type=str,
                      default=os.path.join(
                          os.path.dirname(os.path.abspath(__file__)),
                          "checkpoints", "negotiation_gru.pt",
                      ))
    ft_p.add_argument("--epochs", type=int, default=30)
    ft_p.add_argument("--lr", type=float, default=1e-4)
    ft_p.add_argument("--batch-size", type=int, default=32)
    ft_p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Default to train if no subcommand
    if args.mode is None:
        args.mode = "train"
        args.matches = 10000
        args.epochs = 50
        args.lr = 1e-3
        args.batch_size = 128
        args.dirichlet = 1.0
        args.seed = 42

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.mode == "train":
        print("=" * 60)
        print("NEGOTIATION GRU TRAINING (from scratch)")
        print("=" * 60)
        print(f"Matches: {args.matches} | Dirichlet alpha: {args.dirichlet}")
        print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")

        print("\nGenerating training data...")
        obs, target_weights, target_thresholds, lengths = generate_training_data(
            n_matches=args.matches, dirichlet_alpha=args.dirichlet, seed=args.seed,
        )
        print(f"Generated {obs.shape[0]} samples (obs shape: {obs.shape})")

        print("\nTraining...")
        model, _ = train_negotiation_gru(
            obs, target_weights, target_thresholds, lengths,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        )

        print("\n" + "=" * 60)
        print("EVALUATION (on heuristic data)")
        print("=" * 60)
        metrics = evaluate_weight_prediction(model, obs, target_weights, lengths)
        _print_eval(metrics)
        _save_checkpoint(model, metrics, os.path.join(ckpt_dir, "negotiation_gru.pt"))

    elif args.mode == "eval-llm":
        print("=" * 60)
        print("GRU EVALUATION ON LLM TRAJECTORIES (pretrained, no fine-tuning)")
        print("=" * 60)

        model, n_params = load_negotiation_gru(args.checkpoint)
        print(f"Loaded model ({n_params:,} params) from: {args.checkpoint}")

        print(f"\nExtracting telemetry from: {args.results}")
        # Filter by condition if specified
        if args.condition:
            import json as _json
            with open(args.results) as f:
                raw = _json.load(f)
            filtered = {k: v for k, v in raw.get("results", {}).items()
                        if k == args.condition}
            if not filtered:
                print(f"No games found for condition {args.condition}")
                return
            raw["results"] = filtered
            # Write temp file for extraction
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                _json.dump(raw, tf, default=str)
                tmp_path = tf.name
            obs, target_weights, target_thresholds, lengths = extract_telemetry_from_results(tmp_path)
            os.unlink(tmp_path)
        else:
            obs, target_weights, target_thresholds, lengths = extract_telemetry_from_results(args.results)

        print(f"Extracted {obs.shape[0]} samples from LLM games")

        print("\n" + "=" * 60)
        print("PRETRAINED GRU ACCURACY ON LLM BEHAVIOR")
        print("=" * 60)
        metrics = evaluate_weight_prediction(model, obs, target_weights, lengths)
        _print_eval(metrics)

        # Also report Spearman rank correlation (ordinal accuracy)
        from scipy.stats import spearmanr
        model.eval()
        with torch.no_grad():
            w_pred, _, _ = model(obs, lengths)
        w_pred_np = w_pred.numpy()
        w_true_np = target_weights.numpy()

        rank_corrs = []
        for i in range(len(w_pred_np)):
            corr, _ = spearmanr(w_pred_np[i], w_true_np[i])
            if not np.isnan(corr):
                rank_corrs.append(corr)
        mean_rank_corr = float(np.mean(rank_corrs)) if rank_corrs else 0.0

        # Top-2 accuracy: are both top-2 predicted variables in the true top-2?
        pred_top2 = set(np.argsort(w_pred_np, axis=1)[:, -2:].flatten().reshape(-1, 2).tolist()[0]) if len(w_pred_np) == 1 else None
        top2_accs = []
        for i in range(len(w_pred_np)):
            pred_t2 = set(np.argsort(w_pred_np[i])[-2:])
            true_t2 = set(np.argsort(w_true_np[i])[-2:])
            top2_accs.append(len(pred_t2 & true_t2) / 2.0)
        top2_acc = float(np.mean(top2_accs))

        print(f"\n  Spearman rank corr: {mean_rank_corr:.4f}  (ordinal accuracy)")
        print(f"  Top-2 accuracy:     {top2_acc:.1%}  (both top-2 vars correct, baseline 10%)")

    elif args.mode == "finetune":
        print("=" * 60)
        print("NEGOTIATION GRU FINE-TUNING (Sim-to-Real)")
        print("=" * 60)

        print(f"\nExtracting telemetry from: {args.results}")
        obs, target_weights, target_thresholds, lengths = extract_telemetry_from_results(args.results)
        print(f"Extracted {obs.shape[0]} samples from LLM games")

        print("\nFine-tuning...")
        model, _ = finetune_negotiation_gru(
            args.checkpoint, obs, target_weights, target_thresholds, lengths,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        )

        print("\n" + "=" * 60)
        print("EVALUATION (on LLM telemetry)")
        print("=" * 60)
        metrics = evaluate_weight_prediction(model, obs, target_weights, lengths)
        _print_eval(metrics)

        ft_path = os.path.join(ckpt_dir, "negotiation_gru_finetuned.pt")
        _save_checkpoint(model, metrics, ft_path)


if __name__ == "__main__":
    main()
