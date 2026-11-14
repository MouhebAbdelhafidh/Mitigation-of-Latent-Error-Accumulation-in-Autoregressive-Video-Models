# The loss that caused Regression Shrinkage

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .lsni import LatentNoiseScheduler
# from .rlcc import ReverseCycleConsistencyLoss


# class UnifiedLNTRLoss(nn.Module):
#     """L_total = L_MSE + lambda_noise * L_noisy_pred + lambda_cycle * L_cycle."""

#     def __init__(self, lambda_noise: float = 0.5, lambda_cycle: float = 0.25,
#                  alpha_min: float = 0.7, alpha_max: float = 1.0,
#                  max_epochs: int = 50, k_steps: int = 4):
#         super().__init__()
#         self.lambda_noise = lambda_noise
#         self.lambda_cycle = lambda_cycle
#         self.lsni_scheduler = LatentNoiseScheduler(alpha_min, alpha_max, max_epochs)
#         self.rlcc_loss = ReverseCycleConsistencyLoss(k_steps)

#     def forward(self, model_forward, model_reverse, z: torch.Tensor, epoch: int):
#         z_input, z_target = z[:, :-1], z[:, 1:]

#         # 1. Clean teacher-forcing loss
#         l_mse = F.mse_loss(model_forward(z_input), z_target)

#         # 2. LSNI noisy-input loss (skipped entirely when disabled)
#         alpha = self.lsni_scheduler.get_alpha(epoch)
#         l_noisy_pred = torch.zeros((), device=z.device)
#         if self.lambda_noise > 0:
#             z_input_noisy = self.lsni_scheduler.inject_noise(z_input, alpha)
#             l_noisy_pred = F.mse_loss(model_forward(z_input_noisy), z_target)

#         # 3. RLCC reverse cycle loss
#         l_cycle = torch.zeros((), device=z.device)
#         if self.lambda_cycle > 0:
#             l_cycle = self.rlcc_loss(model_forward, model_reverse, z)

#         l_total = l_mse + self.lambda_noise * l_noisy_pred + self.lambda_cycle * l_cycle

#         return l_total, {
#             "l_total": l_total.item(),
#             "l_mse": l_mse.item(),
#             "l_noisy_pred": l_noisy_pred.item(),
#             "l_cycle": l_cycle.item(),
#             "alpha": alpha,
#         }

# The loss for the rollout training, which is the main contribution of this work. It includes the previous losses as well as a rollout loss that uses the model's own predictions as input for a number of steps and compares them to the ground truth.
import torch
import torch.nn as nn
import torch.nn.functional as F
from .lsni import LatentNoiseScheduler
from .rlcc import ReverseCycleConsistencyLoss


class UnifiedLNTRLoss(nn.Module):
    """L = L_mse + lambda_noise*L_noisy + lambda_cycle*L_cycle + lambda_roll*L_roll."""

    def __init__(self, lambda_noise: float = 0.5, lambda_cycle: float = 0.25,
                 alpha_min: float = 0.7, alpha_max: float = 1.0,
                 max_epochs: int = 50, k_steps: int = 4,
                 lambda_roll: float = 0.0, roll_k: int = 8,
                 roll_ctx_min: int = 2, roll_detach: bool = True):
        super().__init__()
        self.lambda_noise = lambda_noise
        self.lambda_cycle = lambda_cycle
        self.lambda_roll = lambda_roll
        self.roll_k = roll_k
        self.roll_ctx_min = roll_ctx_min
        self.roll_detach = roll_detach
        self.lsni_scheduler = LatentNoiseScheduler(alpha_min, alpha_max, max_epochs)
        self.rlcc_loss = ReverseCycleConsistencyLoss(k_steps)

    def _rollout_loss(self, model_forward, z: torch.Tensor) -> torch.Tensor:
        """Free-running loss: context from ground truth, then feed back the model's own
        predictions for k steps and score each step against the true latent.
        With roll_detach=True the fed-back predictions are detached (pushforward trick)."""
        T = z.shape[1]
        k = min(self.roll_k, T - self.roll_ctx_min)
        if k < 1:
            return z.new_zeros(())
        ctx = int(torch.randint(self.roll_ctx_min, T - k + 1, (1,)).item())

        cur = z[:, :ctx]
        loss = z.new_zeros(())
        for step in range(k):
            nxt = model_forward(cur)[:, -1:]
            loss = loss + F.mse_loss(nxt, z[:, ctx + step: ctx + step + 1])
            cur = torch.cat([cur, nxt.detach() if self.roll_detach else nxt], dim=1)
        return loss / k

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

        # 4. Rollout (own-prediction) loss
        l_roll = torch.zeros((), device=z.device)
        if self.lambda_roll > 0:
            l_roll = self._rollout_loss(model_forward, z)

        l_total = (l_mse + self.lambda_noise * l_noisy_pred
                   + self.lambda_cycle * l_cycle + self.lambda_roll * l_roll)

        return l_total, {
            "l_total": l_total.item(),
            "l_mse": l_mse.item(),
            "l_noisy_pred": l_noisy_pred.item(),
            "l_cycle": l_cycle.item(),
            "l_roll": l_roll.item(),
            "alpha": alpha,
        }