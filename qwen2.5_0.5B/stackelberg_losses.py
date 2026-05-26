"""
Stackelberg losses (Qwen2.5-0.5B).

Port du module Pythia (pythia160M/stackelberg_losses.py). Les losses opèrent
sur des tenseurs (attention weights, head outputs) → indépendantes de
l'architecture. Seules les fonctions head_interaction_matrix / ldb_loss
dépendent du modèle (via get_output_projection_weights de train_utils.py).

Note vs Pythia : les utilitaires get_attention_maps / get_attention_outputs
(qui reconstruisent A,Q,K,V depuis qkv_module fusionné) ne sont pas portés
car ils s'appuient sur la projection query_key_value fusionnée de Pythia.
exp1 n'en a pas besoin (seules ldb_loss + head_interaction_matrix sont
importées). À ajouter ultérieurement avec une implémentation Qwen séparée
si nécessaire (variantes attention-diversity).

Normalization :
  - Leader-follower : mean sur |F| × |L| pairs.
  - Peer-peer       : mean sur |F| × (|F|−1) off-diagonal pairs.
  - Confidence      : mean sur |L| heads.
"""

import torch
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Helper: build follower / leader index tensors
# ---------------------------------------------------------------------------


def _fl_indices(n_heads: int, leader_indices: List[int], device):
    """Return (fi, li, fi_t, li_t) — follower / leader head indices."""
    leader_set = set(leader_indices)
    fi = [i for i in range(n_heads) if i not in leader_set]
    li = list(leader_indices)
    fi_t = torch.tensor(fi, device=device)
    li_t = torch.tensor(li, device=device)
    return fi, li, fi_t, li_t


# ---------------------------------------------------------------------------
# Follower diversity losses
# ---------------------------------------------------------------------------


def follower_diversity_loss(
    attn_weights: torch.Tensor,
    n_heads: int,
    leader_indices: List[int],
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """
    L_div = λ_lead · mean_{i∈F,j∈L} cos(A_i,A_j)
           + λ_peer · mean_{i≠j∈F}   cos(A_i,A_j)
    """
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)

    fi, li, fi_t, li_t = _fl_indices(H, leader_indices, S.device)
    loss = S.new_zeros(())

    if lambda_lead > 0 and len(fi) > 0 and len(li) > 0:
        lf = S[fi_t][:, li_t].mean()
        loss = loss + lambda_lead * lf

    if lambda_peer > 0 and len(fi) > 1:
        S_peer = S[fi_t][:, fi_t]
        mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
        pp = S_peer[mask].mean()
        loss = loss + lambda_peer * pp

    return loss


def follower_diversity_loss_sq(
    attn_weights: torch.Tensor,
    n_heads: int,
    leader_indices: List[int],
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """Squared cosine similarity variant."""
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)
    S_sq = S ** 2

    fi, li, fi_t, li_t = _fl_indices(H, leader_indices, S.device)
    loss = S.new_zeros(())

    if lambda_lead > 0 and len(fi) > 0 and len(li) > 0:
        lf = S_sq[fi_t][:, li_t].mean()
        loss = loss + lambda_lead * lf

    if lambda_peer > 0 and len(fi) > 1:
        S_peer = S_sq[fi_t][:, fi_t]
        mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
        pp = S_peer[mask].mean()
        loss = loss + lambda_peer * pp

    return loss


def follower_diversity_loss_hadamard(
    attn_weights: torch.Tensor,
    n_heads: int,
    leader_indices: List[int],
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """Hadamard (element-wise) dot product variant."""
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)

    fi, li, fi_t, li_t = _fl_indices(H, leader_indices, A_flat.device)
    loss = A_flat.new_zeros(())

    if lambda_lead > 0 and len(fi) > 0 and len(li) > 0:
        A_followers = A_flat[:, fi_t, :]
        A_leaders   = A_flat[:, li_t, :]
        lf = torch.einsum('bfx,blx->bfl', A_followers, A_leaders).mean()
        loss = loss + lambda_lead * lf

    if lambda_peer > 0 and len(fi) > 1:
        A_followers = A_flat[:, fi_t, :]
        M = torch.bmm(A_followers, A_followers.transpose(1, 2)).mean(0)
        n_f = len(fi)
        mask = ~torch.eye(n_f, dtype=torch.bool, device=A_flat.device)
        pp = M[mask].mean()
        loss = loss + lambda_peer * pp

    return loss


def _cka_matrix(Z_heads: torch.Tensor) -> torch.Tensor:
    """
    H×H matrix of linear CKA similarities between head outputs.

    Z_heads : (H, n, d) — n = B*L samples.
    """
    Z_c = Z_heads - Z_heads.mean(1, keepdim=True)
    C = torch.einsum('hnd,gne->hgde', Z_c, Z_c)
    hsic = (C * C).sum(dim=(-2, -1))
    diag = hsic.diagonal()
    norm = (diag.unsqueeze(1) * diag.unsqueeze(0)).sqrt()
    return hsic / (norm + 1e-8)


def follower_diversity_loss_cka(
    head_outputs: torch.Tensor,
    n_heads: int,
    leader_indices: List[int],
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """Linear CKA on head outputs. head_outputs : (B, L, n_heads, d_head)."""
    B, L, H, d_h = head_outputs.shape
    Z = head_outputs.permute(2, 0, 1, 3).reshape(H, B * L, d_h).float()
    S = _cka_matrix(Z)

    fi, li, fi_t, li_t = _fl_indices(H, leader_indices, S.device)
    loss = head_outputs.new_tensor(0.0)

    if lambda_lead > 0 and len(fi) > 0 and len(li) > 0:
        lf = S[fi_t][:, li_t].mean()
        loss = loss + lambda_lead * lf

    if lambda_peer > 0 and len(fi) > 1:
        S_peer = S[fi_t][:, fi_t]
        mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
        pp = S_peer[mask].mean()
        loss = loss + lambda_peer * pp

    return loss


def follower_output_diversity_loss(
    head_outputs: torch.Tensor,
    n_heads: int,
    leader_indices: List[int],
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """Squared cosine similarity on head outputs Z_i = A_i @ V_i."""
    B, L, H, d_h = head_outputs.shape
    Z = head_outputs.permute(0, 2, 1, 3).float()
    Z_norm = F.normalize(Z, dim=-1)
    S = torch.einsum('bhld,bgld->bhg', Z_norm, Z_norm) / L
    S = S.mean(0)
    S_sq = S ** 2

    fi, li, fi_t, li_t = _fl_indices(H, leader_indices, S.device)
    loss = S.new_zeros(())

    if lambda_lead > 0 and len(fi) > 0 and len(li) > 0:
        lf = S_sq[fi_t][:, li_t].mean()
        loss = loss + lambda_lead * lf

    if lambda_peer > 0 and len(fi) > 1:
        S_peer = S_sq[fi_t][:, fi_t]
        mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
        pp = S_peer[mask].mean()
        loss = loss + lambda_peer * pp

    return loss


def follower_erank_loss(
    head_outputs: torch.Tensor,
    n_heads: int,
    leader_indices: List[int],
    lambda_rank: float,
) -> torch.Tensor:
    """L_rank = -lambda_rank · erank(Z) for follower head outputs."""
    B, L, H, d_h = head_outputs.shape
    leader_set = set(leader_indices)
    fi = [i for i in range(H) if i not in leader_set]
    fi_t = torch.tensor(fi, device=head_outputs.device)

    Z = head_outputs.index_select(2, fi_t)
    n_f = len(fi)
    Z = Z.reshape(B * L, n_f * d_h).float()
    Z = Z - Z.mean(dim=0, keepdim=True)

    G = Z.transpose(0, 1) @ Z
    G = 0.5 * (G + G.transpose(0, 1))
    eigs = torch.linalg.eigvalsh(G).clamp(min=0)
    p = eigs / (eigs.sum() + 1e-12)
    entropy = -(p * (p + 1e-12).log()).sum()
    erank = entropy.exp()

    return -lambda_rank * erank


# ---------------------------------------------------------------------------
# Leader confidence losses
# ---------------------------------------------------------------------------


def leader_confidence_loss(
    attn_weights: torch.Tensor,
    leader_indices,
) -> torch.Tensor:
    """L_conf = -1/(|L|·B·L) · Σ_{k∈L} Σ_{b,l} max_{l'} A_k[b,l,l']"""
    if isinstance(leader_indices, int):
        leader_indices = [leader_indices]
    per_head = []
    for k in leader_indices:
        A_k = attn_weights[:, k, :, :]
        per_head.append(-A_k.max(dim=-1).values.mean())
    return torch.stack(per_head).mean()


def leader_confidence_loss_smooth(
    attn_weights: torch.Tensor,
    leader_indices,
) -> torch.Tensor:
    """L_conf = -1/(|L|·B·L) · Σ_{k∈L} Σ_{b,l,l'} A_k[b,l,l']²"""
    if isinstance(leader_indices, int):
        leader_indices = [leader_indices]
    per_head = []
    for k in leader_indices:
        A_k = attn_weights[:, k, :, :]
        per_head.append(-(A_k ** 2).sum(dim=-1).mean())
    return torch.stack(per_head).mean()


def minus_entropy_head(
    attn_weights: torch.Tensor,
    leader_indices,
) -> torch.Tensor:
    """Mean negative-entropy on leader heads."""
    if isinstance(leader_indices, int):
        leader_indices = [leader_indices]
    per_head = []
    for k in leader_indices:
        A_k = attn_weights[:, k, :, :]
        log_attn = torch.log(A_k + 1e-12)
        per_head.append((A_k * log_attn).sum(dim=-1).mean())
    return torch.stack(per_head).mean()


def entropy_heads(attn_weights: torch.Tensor) -> torch.Tensor:
    """Mean entropy per head. attn_weights : (B, n_heads, L, L)."""
    log_attn = torch.log(attn_weights + 1e-12)
    entropies = -(attn_weights * log_attn).sum(dim=-1).mean(dim=(0, 2))
    assert entropies.shape == (attn_weights.shape[1],)
    return entropies


# ---------------------------------------------------------------------------
# Log-determinant barrier  (GAME-LoRA, Chakrabarti & Balachundar 2026)
# ---------------------------------------------------------------------------


def head_interaction_matrix(
    model,
    logits: torch.Tensor,
    labels: torch.Tensor,
    design_layer: int,
    head_dim: int,
) -> torch.Tensor:
    """
    G = ω ⊙ ρ.

      ω_ij = cosine(W_O^(i), W_O^(j))  — couplage des slices par tête de la
             projection de sortie (o_proj). W_O passé avec autograd → L_LDB
             régularise effectivement ces poids.
      ρ_ij = cosine(g_i, g_j) avec g_i = W_O^(i)^T · η_mean.

    Returns: G ∈ R^{H×H}, grad flow only through ω. H = 14 pour Qwen-0.5B.
    """
    from train_utils import HeadInteractionMatrix, get_output_projection_weights

    W_O = get_output_projection_weights(model, design_layer, detach=False)
    omega = HeadInteractionMatrix.compute_weight_coupling(W_O)

    with torch.no_grad():
        W_O_det = get_output_projection_weights(model, design_layer, detach=True)
        rho = HeadInteractionMatrix.compute_gradient_coupling(
            model, logits, labels, W_O_det, head_dim
        )

    return omega * rho


def ldb_loss(G: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """
    L_LDB = -log det(G + εI).

    G : (H, H) head interaction matrix.
    epsilon : régularisation pour définie-positivité stricte.
    """
    H = G.shape[0]
    G_reg = G.float() + epsilon * torch.eye(H, device=G.device, dtype=torch.float32)
    eigvals = torch.linalg.eigvalsh(G_reg).clamp(min=epsilon)
    return -eigvals.log().sum()
