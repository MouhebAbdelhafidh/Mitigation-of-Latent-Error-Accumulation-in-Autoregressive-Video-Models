import torch
import torch.nn as nn
from .position_embed import PositionalEncoding


class CausalLatentPredictor(nn.Module):
    """Causal Transformer predictor g_phi: z_{1:t} -> z_hat_{t+1} (pre-LN).

    residual=True: z_hat_{t+1} = z_t + g(z_{1:t}) with the output layer zero-initialized,
    so training starts exactly at the copy-last baseline and only learns the change.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, dropout: float = 0.1, residual: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        # Saved in the checkpoint so evaluation scripts can tell which variant it is.
        self.register_buffer("is_residual", torch.tensor(1 if residual else 0))

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

        if residual:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, d] -> [B, T, d]; position t holds the prediction of z_{t+1}
        B, T, D = z.shape
        h = self.pos_encoder(self.in_proj(z))
        causal_mask = self._generate_causal_mask(T, z.device)
        h = self.transformer(h, mask=causal_mask, is_causal=True)
        delta = self.out_proj(self.norm(h))
        return z + delta if self.residual else delta