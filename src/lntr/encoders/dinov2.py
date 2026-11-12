import torch
from .base import BaseVisionEncoder

class DINOv2Encoder(BaseVisionEncoder):
    """DINOv2 ViT-S/14 Frame Extractor wrapper (d=384)."""
    def __init__(self, model_name: str = "dinov2_vits14"):
        super().__init__(feature_dim=384)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, 3, H, W]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.model(frames)  # [T, 384]
        return feats.float()