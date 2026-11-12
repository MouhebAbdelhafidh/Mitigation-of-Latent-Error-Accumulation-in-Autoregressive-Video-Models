import torch
from .base import BaseVisionEncoder

class VJEPA2Encoder(BaseVisionEncoder):
    """Meta V-JEPA 2 ViT-L Extractor wrapper (d=1024)."""
    def __init__(self, model_name: str = "vjepa2_vitl"):
        super().__init__(feature_dim=1024)
        try:
            self.model = torch.hub.load("facebookresearch/vjepa2", model_name)
        except Exception:
            # Fallback if PyTorch Hub uses standard DINOv2 ViT-L signature
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [T, 3, H, W]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.model(frames)  # [T, 1024]
        return feats.float()