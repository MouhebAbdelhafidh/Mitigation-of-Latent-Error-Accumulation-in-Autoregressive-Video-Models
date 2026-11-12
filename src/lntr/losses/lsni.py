import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentNoiseScheduler(nn.Module):
    """Module A: Latent Scheduled Noise Injection (LSNI)."""
    def __init__(self, alpha_min: float = 0.7, alpha_max: float = 1.0, max_epochs: int = 50):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.max_epochs = max_epochs

    def get_alpha(self, current_epoch: int) -> float:
        """Cosine annealing noise factor α(e)."""
        e = min(current_epoch, self.max_epochs)
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (1.0 + math.cos(math.pi * e / self.max_epochs))
        return float(alpha)

    def inject_noise(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        """Applies variance-preserving Gaussian noise: z' = α*z + √(1-α²)*ε."""
        noise = torch.randn_like(z)
        noisy_z = alpha * z + math.sqrt(1.0 - alpha**2) * noise
        return F.normalize(noisy_z, p=2, dim=-1)