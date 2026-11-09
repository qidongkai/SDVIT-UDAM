import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionAlign(nn.Module):
    def __init__(self, dim, orig_grid, hidden_ratio=1.0):
        super().__init__()
        self.orig_grid = orig_grid
        hidden = int(dim * hidden_ratio)
        self.w1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, f_mid, target_grid):
        B, Np, D = f_mid.shape
        hp = int(round(Np ** 0.5))
        assert hp * hp == Np, f"Np={Np} not perfect square"
        x = self.w1(f_mid)
        x = self.act(x)
        x = self.norm1(x)
        x2d = x.transpose(1, 2).reshape(B, -1, hp, hp)
        x2d = F.interpolate(x2d, size=(target_grid, target_grid),
                            mode="bilinear", align_corners=False)
        x = x2d.flatten(2).transpose(1, 2)
        x = self.w2(x)
        x = self.norm2(x)
        return self.alpha * x