import torch
import torch.nn as nn
import torch.nn.functional as F


class UperNetHead(nn.Module):
    def __init__(self, in_dim, num_classes, mid_dim=256, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, mid_dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, mid_dim)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_dim, hidden_dim, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, hidden_dim)
        self.act2 = nn.GELU()
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(hidden_dim, num_classes, 1)
        self.aux = nn.Conv2d(mid_dim, num_classes, 1)

    def forward(self, x, return_aux=False):
        x1 = self.act1(self.gn1(self.conv1(x)))
        x2 = self.act2(self.gn2(self.conv2(x1)))
        x2 = self.drop(x2)
        logits = self.cls(x2)
        if return_aux:
            return logits, self.aux(x1)
        return logits