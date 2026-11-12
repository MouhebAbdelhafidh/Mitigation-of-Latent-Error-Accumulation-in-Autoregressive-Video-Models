import torch
import torch.nn as nn
from .position_embed import PositionalEncoding

class CausalLatentPredictor(nn.Module):
    """Causal Transformer Predictor g_ϕ mapping z_{1:t} -> ẑ_{t+1} (~14M parameters)."""
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
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
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, 1024] -> returns z_pred: [B, T, 1024]
        B, T, D = z.shape
        h = self.in_proj(z)
        h = self.pos_encoder(h)
        
        causal_mask = self._generate_causal_mask(T, z.device)
        h_out = self.transformer(h, mask=causal_mask, is_causal=True)
        h_out = self.norm(h_out)
        
        return self.out_proj(h_out)