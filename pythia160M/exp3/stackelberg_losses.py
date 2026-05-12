"""
Stackelberg losses — exp3.

Identical to pythia160M/stackelberg_losses.py except that all functions
accept leader_indices (list[int]) instead of leader_idx (int).

Normalization convention (new in exp3):
  - Leader-follower term  : mean over |F| × |L| pairs.
  - Peer-peer term        : mean over |F| × (|F|−1) ordered off-diagonal pairs.
  → λ_lead / λ_peer have the same "cost per pair" semantics regardless of the
    number of leader / follower heads.

Confidence loss:
  - Averaged over all leader heads, so λ_conf also scales independently of |L|.
"""

import torch
import torch.nn.functional as F
from typing import List


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    Q: torch.Tensor,
    K: torch.Tensor,
    rotary_emb,
    rotary_ndims: int,
) -> tuple:
    """
    Applique RoPE aux rotary_ndims premières dimensions de Q et K.
    Q, K : (B, n_heads, L, d_head)
    """
    B, _, L, _ = Q.shape
    position_ids = torch.arange(L, device=Q.device).unsqueeze(0).expand(B, -1)
    cos, sin = rotary_emb(Q, position_ids)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rot = Q[..., :rotary_ndims] * cos + _rotate_half(Q[..., :rotary_ndims]) * sin
    k_rot = K[..., :rotary_ndims] * cos + _rotate_half(K[..., :rotary_ndims]) * sin
    Q = torch.cat([q_rot, Q[..., rotary_ndims:]], dim=-1)
    K = torch.cat([k_rot, K[..., rotary_ndims:]], dim=-1)
    return Q, K


def get_attention_maps(
    hidden: torch.Tensor,
    qkv_module,
    n_heads: int,
    d_head: int,
    rotary_emb=None,
    rotary_ndims: int = 0,
    input_layernorm=None,
) -> torch.Tensor:
    """
    Calcule les cartes d'attention par tête sans eager attention ni output_attentions=True.

    Retourne A : (B, n_heads, L, L) — attention maps post-softmax, float32.
    """
    if hidden.ndim != 3:
        raise ValueError(
            f"Expected hidden (B, L, d_model) got shape {hidden.shape}. "
            "Check HiddenStateCapture.hook_fn."
        )

    if input_layernorm is not None:
        hidden = input_layernorm(hidden)

    W_base = qkv_module.weight
    lora_A = qkv_module.lora_A["default"].weight
    lora_B = qkv_module.lora_B["default"].weight
    scale = qkv_module.scaling["default"]
    W_eff = W_base + lora_B @ lora_A * scale

    bias = getattr(qkv_module, "bias", None)
    qkv_out = F.linear(hidden, W_eff, bias)

    B, L, _ = qkv_out.shape
    qkv_out = qkv_out.view(B, L, n_heads, 3 * d_head).transpose(1, 2)
    Q, K, _ = qkv_out.chunk(3, dim=-1)

    if rotary_emb is not None and rotary_ndims > 0:
        Q, K = _apply_rope(Q, K, rotary_emb, rotary_ndims)

    scores = Q @ K.transpose(-2, -1) * (d_head**-0.5)
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1)  # (B, n_heads, L, L)


def get_attention_outputs(
    hidden: torch.Tensor,
    qkv_module,
    n_heads: int,
    d_head: int,
    rotary_emb=None,
    rotary_ndims: int = 0,
    input_layernorm=None,
) -> tuple:
    """
    Comme get_attention_maps mais retourne aussi Z = A @ V (sorties par tête).

    Returns:
        A : (B, n_heads, L, L)
        Z : (B, L, n_heads, d_head)
    """
    if hidden.ndim != 3:
        raise ValueError(f"Expected hidden (B, L, d_model) got shape {hidden.shape}.")

    if input_layernorm is not None:
        hidden = input_layernorm(hidden)

    W_base = qkv_module.weight
    lora_A = qkv_module.lora_A["default"].weight
    lora_B = qkv_module.lora_B["default"].weight
    scale = qkv_module.scaling["default"]
    W_eff = W_base + lora_B @ lora_A * scale

    bias = getattr(qkv_module, "bias", None)
    qkv_out = F.linear(hidden, W_eff, bias)

    B, L, _ = qkv_out.shape
    qkv_out = qkv_out.view(B, L, n_heads, 3 * d_head).transpose(1, 2)
    Q, K, V = qkv_out.chunk(3, dim=-1)

    if rotary_emb is not None and rotary_ndims > 0:
        Q, K = _apply_rope(Q, K, rotary_emb, rotary_ndims)

    scores = Q @ K.transpose(-2, -1) * (d_head**-0.5)
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    A = torch.softmax(scores.float(), dim=-1)

    Z = A @ V.float()
    Z = Z.transpose(1, 2).contiguous()  # (B, L, n_heads, d_head)
    return A, Z


# ---------------------------------------------------------------------------
# Helper: build follower / leader index tensors
# ---------------------------------------------------------------------------


def _fl_indices(n_heads: int, leader_indices: List[int], device):
    """Return (fi_t, li_t) tensors of follower / leader head indices."""
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

    Normalized by pair count so λ values are independent of |L| and |F|.
    """
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)  # (H, H)

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
    """
    Squared cosine similarity variant (avoids anticorrelation gradients).

    L_div = λ_lead · mean_{i∈F,j∈L} cos²(A_i,A_j)
           + λ_peer · mean_{i≠j∈F}   cos²(A_i,A_j)
    """
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
    """
    Hadamard (element-wise) dot product variant.

    L_div = λ_lead · mean_{i∈F,j∈L} <A_i,A_j>
           + λ_peer · mean_{i≠j∈F}   <A_i,A_j>

    (attention weights are non-negative so |A_i ⊙ A_j| = <A_i,A_j>)
    """
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)

    fi, li, fi_t, li_t = _fl_indices(H, leader_indices, A_flat.device)
    loss = A_flat.new_zeros(())

    if lambda_lead > 0 and len(fi) > 0 and len(li) > 0:
        A_followers = A_flat[:, fi_t, :]   # (B, n_F, L²)
        A_leaders   = A_flat[:, li_t, :]   # (B, n_L, L²)
        # (B, n_F, n_L) inner products, then mean over B, n_F, n_L
        lf = torch.einsum('bfx,blx->bfl', A_followers, A_leaders).mean()
        loss = loss + lambda_lead * lf

    if lambda_peer > 0 and len(fi) > 1:
        A_followers = A_flat[:, fi_t, :]
        M = torch.bmm(A_followers, A_followers.transpose(1, 2)).mean(0)  # (n_F, n_F)
        n_f = len(fi)
        mask = ~torch.eye(n_f, dtype=torch.bool, device=A_flat.device)
        pp = M[mask].mean()
        loss = loss + lambda_peer * pp

    return loss


def _cka_matrix(Z_heads: torch.Tensor) -> torch.Tensor:
    """
    H×H matrix of linear CKA similarities between head outputs.

    Z_heads : (H, n, d) — n = B*L samples.
    Returns S_cka : (H, H) ∈ [0, 1].
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
    """
    Linear CKA on head outputs Z_i = A_i @ V_i.

    L_div = λ_lead · mean_{i∈F,j∈L} CKA(Z_i,Z_j)
           + λ_peer · mean_{i≠j∈F}   CKA(Z_i,Z_j)

    head_outputs : (B, L, n_heads, d_head).
    """
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
    """
    Squared cosine similarity on head outputs Z_i = A_i @ V_i.

    L_div = λ_lead · mean_{i∈F,j∈L} cos²(Z_i,Z_j)
           + λ_peer · mean_{i≠j∈F}   cos²(Z_i,Z_j)

    head_outputs : (B, L, n_heads, d_head).
    """
    B, L, H, d_h = head_outputs.shape
    Z = head_outputs.permute(0, 2, 1, 3)
    Z_flat = Z.reshape(B, H, L * d_h).float()
    Z_norm = F.normalize(Z_flat, dim=-1)
    S = torch.bmm(Z_norm, Z_norm.transpose(1, 2)).mean(0)
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
    """
    L_rank = -lambda_rank · erank(Z)  where Z concatenates all follower head outputs.

    All heads in leader_indices are excluded.
    head_outputs : (B, L, n_heads, d_head).
    """
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
# Leader confidence losses  (averaged over all leader heads)
# ---------------------------------------------------------------------------


def leader_confidence_loss(
    attn_weights: torch.Tensor,
    leader_indices,
) -> torch.Tensor:
    """
    L_conf = -1/(|L|·B·L) · Σ_{k∈L} Σ_{b,l} max_{l'} A_k[b,l,l']

    Averaged over leader heads so λ_conf is independent of |L|.
    attn_weights : (B, n_heads, L, L).
    """
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
    """
    L_conf = -1/(|L|·B·L) · Σ_{k∈L} Σ_{b,l,l'} A_k[b,l,l']²

    Encourages sharper attention per leader head.
    """
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
    """
    Returns mean entropy (positive) averaged over leader heads — minimizing
    this loss maximises attention concentration.

    L_conf = 1/(|L|·B·L) · Σ_{k∈L} Σ_{b,l,l'} A_k log A_k
    """
    if isinstance(leader_indices, int):
        leader_indices = [leader_indices]
    per_head = []
    for k in leader_indices:
        A_k = attn_weights[:, k, :, :]
        log_attn = torch.log(A_k + 1e-12)
        per_head.append((A_k * log_attn).sum(dim=-1).mean())
    return torch.stack(per_head).mean()


def entropy_heads(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Mean entropy per head.

    attn_weights : (B, n_heads, L, L)
    Returns      : (n_heads,) — positive entropy averaged over B and L.
    """
    log_attn = torch.log(attn_weights + 1e-12)
    entropies = -(attn_weights * log_attn).sum(dim=-1).mean(dim=(0, 2))
    assert entropies.shape == (attn_weights.shape[1],)
    return entropies
