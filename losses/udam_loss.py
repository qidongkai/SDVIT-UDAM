"""
UDAM unsupervised domain adaptation loss module.

Corresponds to Section 3.2.2 of the thesis:
    - Eq 3.8:  L_ss   = 1 - cos(F_t, F_t^aug)          self-supervised
                                                        consistency
    - Eq 3.9:  L_UDAM = rho * (gamma * L_kd
                               + (1 - gamma) * L_ss)
    - Eq 3.10: gamma  = sigmoid(-alpha * MMD)          domain-gap
                                                        coefficient
    - Eq 3.11: MMD RBF kernel                           domain distance
    - Eq 3.19: KL(P_s || P_t)                           shared-label-space
                                                        KL divergence

Overall logic:
    1. The adaptive fallback mask is used to compute the KL divergence
       (teacher soft labels -> student prediction).
    2. The MMD between the stored source feature bank and the current
       target features yields gamma.
    3. The multi-view consistency loss L_ss supplies the target-domain
       self-supervised signal.
    4. Terms are weighted by gamma and scaled by rho.
"""
import torch
import torch.nn.functional as F

from .reliability import adaptive_fallback, multi_view_consistency


def mmd_rbf(x, y, bandwidth=None):
    """
    Maximum Mean Discrepancy with an RBF kernel (Eq 3.11).

    The bandwidth is selected by the median heuristic, i.e. the median of
    all pairwise squared distances.

    Args:
        x:         [N, D] source features.
        y:         [M, D] target features.
        bandwidth: optional fixed RBF bandwidth.

    Returns:
        Scalar MMD value, clamped to be non-negative.
    """
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    if bandwidth is None:
        all_d = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
        bandwidth = all_d.median().detach().clamp(min=1e-6)

    Kxx = torch.exp(-xx / bandwidth).mean()
    Kyy = torch.exp(-yy / bandwidth).mean()
    Kxy = torch.exp(-xy / bandwidth).mean()
    return (Kxx + Kyy - 2.0 * Kxy).clamp(min=0.0)


def adaptive_gamma(mmd, alpha=1.0, floor=0.2):
    """
    Adaptive domain-gap coefficient gamma
    (Eq 3.10 plus the gamma floor in Section 4.5.3).

    gamma = sigmoid(-alpha * MMD). A larger MMD yields a smaller gamma,
    down-weighting the source-domain supervision. To prevent gamma from
    collapsing to 0 under a deliberately mismatched source (which would
    reduce UDAM to target-only self-supervision and cause negative
    transfer), a lower bound `floor` (default 0.2) is applied.

    Args:
        mmd:   scalar MMD.
        alpha: scaling coefficient.
        floor: minimum value of gamma.

    Returns:
        Scalar gamma.
    """
    g = torch.sigmoid(-alpha * mmd)
    return g.clamp(min=floor)


def kl_shared(p_s, p_t, mask):
    """
    KL divergence on the shared label space (Eq 3.19).

    Args:
        p_s:  [B, C, H, W] teacher soft labels.
        p_t:  [B, C, H, W] student softmax predictions.
        mask: [B, H, W] valid-pixel mask.

    Returns:
        Scalar KL loss averaged over valid pixels.
    """
    kl = (p_s * (torch.log(p_s + 1e-8) - torch.log(p_t + 1e-8))).sum(dim=1)
    num = (kl * mask).sum()
    den = mask.sum() + 1e-8
    return num / den


def udam_total_loss(p_s, p_t, f_t, f_aug_list, feat_bank, cfg):
    """
    Total UDAM loss (Eq 3.9).

    Args:
        p_s:        [B, C, H, W] teacher soft labels (source prior).
        p_t:        [B, C, H, W] student predictions.
        f_t:        [B, D] student original-view features.
        f_aug_list: list of [B, D] student augmented-view features.
        feat_bank:  [N, D] frozen source feature bank (800 vectors).
        cfg:        configuration dictionary.

    Returns:
        loss: scalar.
        info: diagnostics dictionary with keys
              gamma, rho, l_kd, l_ss, mmd, thr.
    """
    mask, rho, thr = adaptive_fallback(
        p_t,
        cfg["udam"]["tau_p"],
        cfg["udam"]["tau_H"],
        cfg["udam"]["rho_min"],
        cfg["udam"]["fallback_floor"],
    )

    l_kd = kl_shared(p_s, p_t, mask)

    n = min(cfg["udam"]["mmd_subset"], f_t.shape[0])
    idx = torch.randint(0, feat_bank.shape[0], (n,), device=f_t.device)
    f_bank = feat_bank[idx]
    mmd = mmd_rbf(f_bank.flatten(1), f_t.flatten(1))
    gamma = adaptive_gamma(
        mmd, cfg["udam"]["alpha_mmd"], cfg["udam"]["gamma_floor"]
    )

    if len(f_aug_list) == 0:
        l_ss = torch.tensor(0.0, device=f_t.device)
    else:
        l_ss = sum(
            multi_view_consistency(f_t, fa) for fa in f_aug_list
        ) / len(f_aug_list)

    loss = rho * (gamma * l_kd + (1.0 - gamma) * l_ss)

    return loss, {
        "gamma": gamma.item(),
        "rho": rho,
        "l_kd": l_kd.item(),
        "l_ss": l_ss.item(),
        "mmd": mmd.item(),
        "thr": thr,
    }