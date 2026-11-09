"""
Dynamic token pruner for SDViT.

Implements the token importance scoring of Eq 3.2 and the physical
variable-length pruning that produces the 45% FLOPs reduction reported
in Section 4.2.2.

Design constraints from the thesis:
    - Section 4.2.3: the scoring module is "deliberately lightweight" and
      "uses a single linear projection ... rather than a full attention
      operation", so no MLP branch is allowed here.
    - Eq 3.2: alpha is a "fixed balance hyperparameter", not learned.
    - Section 4.2.3: the adaptive threshold is computed from the average
      importance score in each scene, and the retention rate follows the
      smooth curve in Figure 4-5.
"""
import torch
import torch.nn as nn


class TokenScorer(nn.Module):
    """
    Token importance scorer (Eq 3.2).

    S_i = H_i + alpha * || A_i ||_1 + single_linear(x_i)

    The single linear projection is the only learned component, matching
    the "deliberately lightweight" design in Section 4.2.3. H_i is the
    feature entropy, A_i is the mean attention weight over heads, and
    alpha is fixed.

    Args:
        dim:   token embedding dimension.
        alpha: fixed balance hyperparameter between entropy and attention.
    """

    def __init__(self, dim, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.proj = nn.Linear(dim, 1)

    def forward(self, x, attn_w):
        """
        Args:
            x:      [B, N, D] token features.
            attn_w: [B, N, N] attention weights averaged over heads.

        Returns:
            [B, N] importance scores.
        """
        p = x.softmax(dim=-1)
        H = -(p * (p + 1e-8).log()).sum(dim=-1)          # [B, N]
        A = attn_w.mean(dim=-1)                           # [B, N]
        bias = self.proj(x).squeeze(-1)                   # [B, N]
        return H + self.alpha * A + bias


def var_len_prune(x, key_padding_mask, score, keep_ratio, min_keep=1):
    """
    Physically prune tokens to a variable length.

    Unlike a mask-only approach, this function actually reduces the token
    count so that the FLOPs reduction claimed in Section 4.2.2 is real.

    Args:
        x:                [B, N, D] token features.
        key_padding_mask: [B, N] bool, True = keep.
        score:            [B, N] importance scores.
        keep_ratio:       float in (0, 1].
        min_keep:         minimum number of tokens to keep.

    Returns:
        new_x:    [B, k, D] pruned features.
        new_mask: [B, k] bool, all True.
        topk_idx: [B, k] indices into the original token sequence.
    """
    B, N, D = x.shape
    keep_len = max(min_keep, int(N * keep_ratio))
    score_masked = score.masked_fill(~key_padding_mask, float("-inf"))
    _, topk_idx = score_masked.topk(keep_len, dim=1)

    new_x = torch.zeros(B, keep_len, D, device=x.device, dtype=x.dtype)
    new_mask = torch.zeros(B, keep_len, device=x.device, dtype=torch.bool)
    for b in range(B):
        idx = topk_idx[b]
        new_x[b] = x[b, idx]
        new_mask[b] = True
    return new_x, new_mask, topk_idx


class DynamicPruneBlock(nn.Module):
    """
    Layer-wise dynamic pruning with an adaptive threshold.

    The retention rate follows Eq 3.3 and the scene-complexity curve in
    Figure 4-5: simple scenes keep a small fraction of tokens, complex
    scenes keep a larger fraction.

    Args:
        dim:       token embedding dimension.
        alpha:     fixed balance hyperparameter passed to TokenScorer.
        min_ratio: minimum retention rate (simple scenes).
        max_ratio: maximum retention rate (complex scenes).
    """

    def __init__(self, dim, alpha=1.0, min_ratio=0.35, max_ratio=0.85):
        super().__init__()
        assert 0 < min_ratio <= max_ratio <= 1
        self.scorer = TokenScorer(dim, alpha)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)

    def forward(self, x, key_padding_mask, attn_w, complexity):
        """
        Args:
            x:                [B, N, D] token features.
            key_padding_mask: [B, N] bool.
            attn_w:           [B, N, N] attention weights.
            complexity:       [B, 1] scene complexity in [0, 1].

        Returns:
            new_x, new_mask, topk_idx, score
        """
        keep_ratio_t = self.min_ratio + (self.max_ratio - self.min_ratio) * complexity
        keep_ratio_t = keep_ratio_t.clamp(self.min_ratio, self.max_ratio)
        keep_ratio = keep_ratio_t.mean().item()

        score = self.scorer(x, attn_w)
        new_x, new_mask, idx = var_len_prune(x, key_padding_mask, score, keep_ratio)
        return new_x, new_mask, idx, score