import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int):
        super().__init__()
        hidden = max(in_dim, proj_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["ProjectionHead"]
