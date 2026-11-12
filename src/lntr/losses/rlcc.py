import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseCycleConsistencyLoss(nn.Module):
    """Module B: Reverse Latent Cycle-Consistency (RLCC) over K rollout steps."""
    def __init__(self, k_steps: int = 4):
        super().__init__()
        self.k_steps = k_steps

    def forward(self, model_forward, model_reverse, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, 1024]
        B, T, D = z.shape
        if T <= self.k_steps:
            return torch.tensor(0.0, device=z.device)

        z_anchor = z[:, 0]  # First frame anchor: [B, 1024]
        rollout_states = []
        curr_state = z_anchor.unsqueeze(1)  # [B, 1, 1024]

        for _ in range(self.k_steps):
            next_pred = model_forward(curr_state)[:, -1:]
            rollout_states.append(next_pred)
            curr_state = torch.cat([curr_state, next_pred], dim=1)

        rollout_tensor = torch.cat(rollout_states, dim=1)  # [B, K, 1024]
        z_reconstructed = model_reverse(rollout_tensor)    # [B, 1024]

        return F.mse_loss(z_reconstructed, z_anchor)