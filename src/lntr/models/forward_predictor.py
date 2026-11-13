import torch
import torch.nn as nn
from .position_embed import PositionalEncoding


class CausalLatentPredictor(nn.Module):
    """Causal Transformer Predictor g_phi mapping z_{1:t} -> z_hat_{t+1} (pre-LN)."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: much more stable than the post-LN default
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)  # final norm required for pre-LN stacks
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, d] -> [B, T, d]
        B, T, D = z.shape
        h = self.pos_encoder(self.in_proj(z))
        causal_mask = self._generate_causal_mask(T, z.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        return self.out_proj(self.norm(h))