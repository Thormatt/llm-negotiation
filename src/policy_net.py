"""
NegotiationPolicyNet — MLP that maps 27D state to 5D proposal.

Architecture:
  state (27D) → Linear(27, 64) → LayerNorm → GELU
              → Linear(64, 64) → LayerNorm → GELU
              → Linear(64, 5)  → Sigmoid   → clamp [0.05, 0.95]

State includes 22D base features + 5D GRU-predicted opponent weights.
MLP not RNN: the policy sees full state each turn. Single-step decision function.
"""

import torch
import torch.nn as nn

from policy_data import POLICY_STATE_DIM


class NegotiationPolicyNet(nn.Module):
    """MLP policy network: 27D state → 5D proposal."""

    def __init__(self, state_dim=POLICY_STATE_DIM, hidden_dim=64, action_dim=5):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state):
        """
        Args:
            state: (B, 27) state tensor

        Returns:
            proposal: (B, 5) clamped to [0.05, 0.95]
        """
        raw = self.net(state)
        return raw.clamp(0.05, 0.95)


def load_policy_net(path):
    """Load trained NegotiationPolicyNet from checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = NegotiationPolicyNet(**ckpt["hyperparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def save_policy_net(model, metrics, path):
    """Save model checkpoint with hyperparams and metrics."""
    torch.save({
        "hyperparams": {
            "state_dim": model.state_dim,
            "hidden_dim": model.hidden_dim,
            "action_dim": model.action_dim,
        },
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }, path)
