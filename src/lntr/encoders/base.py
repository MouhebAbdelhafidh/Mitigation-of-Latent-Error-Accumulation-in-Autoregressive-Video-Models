from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseVisionEncoder(nn.Module, ABC):
    """Abstract base class for all vision feature extractors."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim

    @abstractmethod
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representations for input frames.
        Args:
            frames: Tensor of shape [B, T, 3, H, W] or [T, 3, H, W]
        Returns:
            latents: Tensor of shape [B, T, feature_dim] or [T, feature_dim]
        """
        pass