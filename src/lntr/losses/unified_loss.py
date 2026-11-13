import torch
import torch.nn as nn
import torch.nn.functional as F
from .lsni import LatentNoiseScheduler
from .rlcc import ReverseCycleConsistencyLoss


class UnifiedLNTRLoss(nn.Module):
    """L_total = L_MSE + lambda_noise * L_noisy_pred + lambda_cycle * L_cycle."""

    def __init__(self, lambda_noise: float = 0.5, lambda_cycle: float = 0.25,
                 alpha_min: float = 0.7, alpha_max: float = 1.0,
                 max_epochs: int = 50, k_steps: int = 4):
        super().__init__()
        self.lambda_noise = lambda_noise
        self.lambda_cycle = lambda_cycle
        self.lsni_scheduler = LatentNoiseScheduler(alpha_min, alpha_max, max_epochs)
        self.rlcc_loss = ReverseCycleConsistencyLoss(k_steps)

    def forward(self, model_forward, model_reverse, z: torch.Tensor, epoch: int):
        z_input, z_target = z[:, :-1], z[:, 1:]

        # 1. Clean teacher-forcing loss
        l_mse = F.mse_loss(model_forward(z_input), z_target)

        # 2. LSNI noisy-input loss (skipped entirely when disabled)
        alpha = self.lsni_scheduler.get_alpha(epoch)
        l_noisy_pred = torch.zeros((), device=z.device)
        if self.lambda_noise > 0:
            z_input_noisy = self.lsni_scheduler.inject_noise(z_input, alpha)
            l_noisy_pred = F.mse_loss(model_forward(z_input_noisy), z_target)

        # 3. RLCC reverse cycle loss
        l_cycle = torch.zeros((), device=z.device)
        if self.lambda_cycle > 0:
            l_cycle = self.rlcc_loss(model_forward, model_reverse, z)

        l_total = l_mse + self.lambda_noise * l_noisy_pred + self.lambda_cycle * l_cycle

        return l_total, {
            "l_total": l_total.item(),
            "l_mse": l_mse.item(),
            "l_noisy_pred": l_noisy_pred.item(),
            "l_cycle": l_cycle.item(),
            "alpha": alpha,
        }