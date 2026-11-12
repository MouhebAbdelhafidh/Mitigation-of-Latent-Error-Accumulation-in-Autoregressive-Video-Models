import torch
import torch.nn as nn

class ReverseCycleHead(nn.Module):
    """Reverse Temporal Head g_Ψ^rev mapping multi-step rollouts back to past anchor z_t."""
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)])
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        # z_future: [B, K, 1024] -> returns ẑ_anchor: [B, 1024]
        pooled = z_future.mean(dim=1)
        return self.net(pooled)