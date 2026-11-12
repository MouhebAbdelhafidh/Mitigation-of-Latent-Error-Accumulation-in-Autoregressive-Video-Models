import torch
import torch.nn as nn
import torch.nn.functional as F
from .lsni import LatentNoiseScheduler
from .rlcc import ReverseCycleConsistencyLoss

class UnifiedLNTRLoss(nn.Module):
    """Composite optimization loss: L_total = L_MSE + λ_noise * L_noisy_pred + λ_cycle * L_cycle."""
    def __init__(self, lambda_noise: float = 0.5, lambda_cycle: float = 0.25, alpha_min: float = 0.7, alpha_max: float = 1.0, max_epochs: int = 50, k_steps: int = 4):
        super().__init__()
        self.lambda_noise = lambda_noise
        self.lambda_cycle = lambda_cycle
        self.lsni_scheduler = LatentNoiseScheduler(alpha_min, alpha_max, max_epochs)
        self.rlcc_loss = ReverseCycleConsistencyLoss(k_steps)

    def forward(self, model_forward, model_reverse, z: torch.Tensor, epoch: int):
        # 1. Standard Teacher-Forcing Loss (L_MSE)
        z_input = z[:, :-1]
        z_target = z[:, 1:]
        z_pred_clean = model_forward(z_input)
        l_mse = F.mse_loss(z_pred_clean, z_target)

        # 2. LSNI Noisy Prediction Loss (L_noisy_pred)
        alpha = self.lsni_scheduler.get_alpha(epoch)
        z_input_noisy = self.lsni_scheduler.inject_noise(z_input, alpha)
        z_pred_noisy = model_forward(z_input_noisy)
        l_noisy_pred = F.mse_loss(z_pred_noisy, z_target)

        # 3. RLCC Reverse Cycle Loss (L_cycle)
        l_cycle = torch.tensor(0.0, device=z.device)
        if self.lambda_cycle > 0:
            l_cycle = self.rlcc_loss(model_forward, model_reverse, z)

        l_total = l_mse + (self.lambda_noise * l_noisy_pred) + (self.lambda_cycle * l_cycle)

        return l_total, {
            "l_total": l_total.item(),
            "l_mse": l_mse.item(),
            "l_noisy_pred": l_noisy_pred.item(),
            "l_cycle": l_cycle.item(),
            "alpha": alpha
        }