import math
import torch
import torch.nn as nn


class LatentNoiseScheduler(nn.Module):
    """Module A: Latent Scheduled Noise Injection (LSNI)."""

    def __init__(self, alpha_min: float = 0.7, alpha_max: float = 1.0, max_epochs: int = 50):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.max_epochs = max_epochs

    def get_alpha(self, current_epoch: int) -> float:
        """Cosine schedule for the noise factor alpha(e)."""
        e = min(current_epoch, self.max_epochs)
        alpha = self.alpha_min + 0.5 * (self.alpha_max - self.alpha_min) * (
            1.0 + math.cos(math.pi * e / self.max_epochs)
        )
        return float(alpha)

    def inject_noise(self, z: torch.Tensor, alpha: float) -> torch.Tensor:
        """Variance-preserving noise z' = a*z + sqrt(1-a^2)*eps.

        Valid because latents are standardized to ~unit variance per channel.
        No re-normalization, so clean and noisy inputs share the same scale.
        """
        noise = torch.randn_like(z)
        return alpha * z + math.sqrt(max(0.0, 1.0 - alpha**2)) * noise