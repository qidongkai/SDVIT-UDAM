"""
Static-Dynamic collaborative Vision Transformer (SDViT).

Implements the architecture described in Section 3.2.1 of the thesis:
    - the deepest layer is never pruned,
    - intermediate layers are pruned layer by layer with an adaptive
      threshold,
    - the self-distillation branch aligns each pruned intermediate feature
      to the unpruned deep feature.

The scene complexity proxy follows Figure 4-5 of the thesis: complexity
is the fraction of pixels belonging to foreground categories (anything
other than sky, road, building, vegetation, and terrain). Because
ground-truth labels are not available at inference time, the proxy is
computed from a lightweight semantic energy map derived from the patch
embeddings, which correlates with the foreground fraction without
requiring labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .vit_backbone import PatchEmbed, Block
from .dynamic_pruner import DynamicPruneBlock
from .projector import ProjectionAlign
from .seg_head import UperNetHead


# Background categories in the unified label space, following the
# definition of "foreground" in Figure 4-5:
#   2 Building, 3 Wall, 4 Fence, 8 Vegetation, 9 Terrain, 10 Sky, 0 Road
BACKGROUND_CATEGORIES = {0, 2, 3, 4, 8, 9, 10}


class SDViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.img_size = cfg["img_size"]
        self.patch_size = cfg["patch_size"]
        self.grid = self.img_size // self.patch_size
        self.embed_dim = cfg["embed_dim"]
        self.depth = cfg["depth"]
        self.num_classes = cfg["num_classes"]

        self.patch_embed = PatchEmbed(
            self.img_size, self.patch_size, 3, self.embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            Block(self.embed_dim, cfg["num_heads"], cfg["mlp_ratio"])
            for _ in range(self.depth)
        ])

        self.prune_layers = set(cfg["prune_layers"])
        self.pruners = nn.ModuleDict({
            str(i): DynamicPruneBlock(
                self.embed_dim,
                cfg["alpha_attn"],
                cfg["min_keep_ratio"],
                cfg["max_keep_ratio"],
            )
            for i in self.prune_layers
        })
        self.projectors = nn.ModuleDict({
            str(i): ProjectionAlign(self.embed_dim, self.grid)
            for i in self.prune_layers
        })

        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = UperNetHead(self.embed_dim, self.num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _compute_complexity(self, patch_tokens):
        """
        Scene complexity proxy following Figure 4-5 of the thesis.

        The thesis defines complexity as the fraction of pixels belonging
        to foreground categories. Because labels are unavailable at
        inference time, the proxy is computed from the semantic energy of
        the patch embeddings: patches whose features are far from the
        background prototype are counted as foreground.

        Args:
            patch_tokens: [B, N, D] patch embeddings.

        Returns:
            [B, 1] scene complexity in [0, 1].
        """
        B, N, D = patch_tokens.shape
        # Project each token to a scalar foreground score via its
        # distance from the mean background direction. The mean token is
        # a reasonable background prototype because background regions
        # dominate AR scenes (Figure 4-5 shows foreground fraction
        # typically below 0.5).
        mean_token = patch_tokens.mean(dim=1, keepdim=True)      # [B, 1, D]
        diff = (patch_tokens - mean_token).pow(2).sum(-1)         # [B, N]
        foreground = (diff > diff.mean(dim=1, keepdim=True)).float()
        complexity = foreground.mean(dim=1, keepdim=True)         # [B, 1]
        return complexity

    def forward(self, x, return_aux=False):
        B = x.shape[0]
        x = self.patch_embed(x) + self.pos_embed
        x = self.pos_drop(x)

        key_mask = torch.ones(B, x.shape[1], device=x.device, dtype=torch.bool)
        complexity = self._compute_complexity(x)

        mid_feats = []
        f_deep = None

        for i, blk in enumerate(self.blocks):
            x, attn_w = blk(x, key_mask)

            if i in self.prune_layers and i != self.depth - 1:
                # Save the pre-pruning feature and mask for the
                # self-distillation branch.
                mid_feats.append((i, x.clone(), key_mask.clone(), attn_w.clone()))
                x, key_mask, idx, score = self.pruners[str(i)](
                    x, key_mask, attn_w, complexity)

            if i == self.depth - 1:
                f_deep = x

        out = self.norm(f_deep)

        if return_aux:
            return out, mid_feats, f_deep

        B2 = out.shape[0]
        grid = int(out.shape[1] ** 0.5)
        f2d = out.transpose(1, 2).reshape(B2, -1, grid, grid)
        logits = F.interpolate(
            self.head(f2d),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def forward_features(self, x):
        out, _, _ = self.forward(x, return_aux=True)
        return out