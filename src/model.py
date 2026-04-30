"""
Belief GRU v2 — auction/strategic-transfer world model.

NOTE FOR REVIEWERS: this is NOT the model described in the negotiation paper.
The paper's model is `NegotiationGRU` defined in `train_negotiation_gru.py`
(11,078 params, 11D feature vector, dual softmax+sigmoid heads). This file
contains a separate VICReg+EMA self-supervised model used by the auction
benchmark (auction.py, sequential_auction.py, benchmark.py) and the early
agent_loop.py / arena.py prototypes — it predates the negotiation experiments.

Architecture (revised to match winning patterns from StrategicTransferModel):
  - obs_encoder: MLP(6 → 128 → 64) with ReLU + LayerNorm
  - target_encoder: EMA copy (τ=0.996, no grad)
  - gru: GRU(64, 64) — full sequence temporal integrator
  - surprise_gate: |pred - target| → MLP → sigmoid → modulates belief OUTPUT
    b_t = g_t * h_t + (1 - g_t) * b_{t-1}  (output gating, not input gating)
  - predictor: MLP(64+1, 32, 64) — K-step autoregressive from beliefs
  - surplus_head: Linear(64, 1) — supervised surplus prediction

Extraction:
  Multi-timepoint beliefs at T//4, T//2, 3T//4 → concatenated → Ridge probe
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefGRU(nn.Module):
    def __init__(self, obs_dim=6, hidden_dim=128, latent_dim=64, K=5, ema_tau=0.996):
        super().__init__()
        self.latent_dim = latent_dim
        self.K = K
        self.ema_tau = ema_tau

        # Observation encoder: MLP (obs_dim → hidden → latent)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )

        # Target encoder (EMA copy, no grad)
        self.target_encoder = copy.deepcopy(self.obs_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Temporal integrator: full GRU over latent sequence
        self.gru = nn.GRU(latent_dim, latent_dim, num_layers=1, batch_first=True)

        # Surprise gate: prediction error → gate value
        self.surprise_mlp = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # K-step predictor: (belief + step_index) → predicted latent
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # Surplus prediction head (downstream probe)
        self.surplus_head = nn.Linear(latent_dim, 1)

    @torch.no_grad()
    def update_target_encoder(self):
        tau = self.ema_tau
        for p_online, p_target in zip(
            self.obs_encoder.parameters(), self.target_encoder.parameters()
        ):
            p_target.data.mul_(tau).add_(p_online.data, alpha=1 - tau)

    def predict_k_step(self, belief, k):
        """Predict latent k steps ahead from a belief state.

        Args:
            belief: (B, latent_dim)
            k: int or (B, 1) tensor — step index

        Returns:
            predicted latent: (B, latent_dim)
        """
        B = belief.size(0)
        if isinstance(k, int):
            step_k = torch.full((B, 1), k, device=belief.device, dtype=torch.float)
        else:
            step_k = k
        cat = torch.cat([belief, step_k], dim=-1)
        return self.predictor(cat)

    def forward(self, obs):
        """Full forward pass with joint VICReg + probe loss.

        Args:
            obs: (B, T, obs_dim) raw observations

        Returns:
            dict with vicreg_loss, surplus_pred, beliefs, gate stats
        """
        B, T, _ = obs.shape
        device = obs.device

        # Encode all timesteps
        x_flat = obs.reshape(B * T, -1)
        z_online = self.obs_encoder(x_flat).reshape(B, T, -1)

        with torch.no_grad():
            z_target = self.target_encoder(x_flat).reshape(B, T, -1)

        # Run temporal integrator over full sequence
        h_all, _ = self.gru(z_online)  # (B, T, latent_dim)

        # Surprise-gated belief update (OUTPUT gating)
        beliefs = []
        b_prev = torch.zeros(B, self.latent_dim, device=device)
        total_surprise = 0.0

        for t in range(T):
            h_t = h_all[:, t]

            # One-step prediction from previous belief
            pred_z = self.predict_k_step(b_prev, 1)
            target_z = z_target[:, t]

            # Surprise = prediction error → gate
            surprise = torch.abs(pred_z - target_z.detach())
            g_t = torch.sigmoid(self.surprise_mlp(surprise.detach()))  # (B, 1)

            # Output gating: blend GRU output with previous belief
            b_t = g_t * h_t + (1 - g_t) * b_prev
            beliefs.append(b_t)
            total_surprise += g_t.mean().item()
            b_prev = b_t

        beliefs = torch.stack(beliefs, dim=1)  # (B, T, latent_dim)

        # K-step autoregressive VICReg loss
        vic_loss = torch.tensor(0.0, device=device)
        n_predictions = 0

        for t in range(T - self.K):
            b_t = beliefs[:, t]
            for k in range(1, self.K + 1):
                if t + k >= T:
                    break
                pred = self.predict_k_step(b_t, k)
                target = z_target[:, t + k]
                vic_loss = vic_loss + self._vicreg_pair(pred, target)
                n_predictions += 1

        if n_predictions > 0:
            vic_loss = vic_loss / n_predictions

        # Surplus prediction from final belief
        surplus_pred = self.surplus_head(beliefs[:, -1]).squeeze(-1)

        return {
            "vicreg_loss": vic_loss,
            "surplus_pred": surplus_pred,
            "beliefs": beliefs,
            "z_target": z_target,
            "mean_surprise": total_surprise / T,
        }

    def _vicreg_pair(self, z1, z2, lam=25.0, mu=25.0, nu=1.0):
        """VICReg loss for a single prediction-target pair."""
        # Invariance
        inv_loss = F.mse_loss(z1, z2)

        # Variance
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = (torch.relu(1 - std_z1).mean() + torch.relu(1 - std_z2).mean()) / 2

        # Covariance
        z1_c = z1 - z1.mean(dim=0)
        z2_c = z2 - z2.mean(dim=0)
        N = z1.size(0)
        D = z1.size(1)
        cov1 = (z1_c.T @ z1_c) / (N - 1)
        cov2 = (z2_c.T @ z2_c) / (N - 1)
        cov_loss = (cov1.fill_diagonal_(0).pow(2).sum() / D +
                    cov2.fill_diagonal_(0).pow(2).sum() / D) / 2

        return lam * inv_loss + mu * var_loss + nu * cov_loss

    def extract_multi_timepoint(self, obs):
        """Extract multi-timepoint belief features for downstream probe.

        Args:
            obs: (B, T, obs_dim)

        Returns:
            features: (B, 3*latent_dim)
        """
        with torch.no_grad():
            out = self.forward(obs)
        beliefs = out["beliefs"]
        T = beliefs.shape[1]
        t1, t2, t3 = T // 4, T // 2, 3 * T // 4
        return torch.cat([beliefs[:, t1], beliefs[:, t2], beliefs[:, t3]], dim=-1)

    def extract_raw_features(self, obs):
        """Baseline: raw observation features at midpoint + mean."""
        T = obs.shape[1]
        mid = obs[:, T // 2]                    # (B, obs_dim)
        mean = obs[:, : T // 2].mean(dim=1)     # (B, obs_dim)
        return torch.cat([mid, mean], dim=-1)
