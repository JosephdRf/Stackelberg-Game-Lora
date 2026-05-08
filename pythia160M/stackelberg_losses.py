import torch
import torch.nn.functional as F


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
    Retourne Q, K avec la rotation appliquée — même transformation que GPTNeoXAttention.
    """
    B, _, L, _ = Q.shape
    position_ids = torch.arange(L, device=Q.device).unsqueeze(0).expand(B, -1)
    cos, sin = rotary_emb(Q, position_ids)  # (B, L, rotary_ndims) — rotary_emb sur GPTNeoXModel
    cos = cos.unsqueeze(1)                  # (B, 1, L, rotary_ndims) — broadcast sur H
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

    hidden          : (B, L, d_model) — sortie de la couche design_layer-1.
    qkv_module      : PEFT-wrapped query_key_value Linear du design layer.
    input_layernorm : LayerNorm du design layer (appliqué avant QKV, comme dans le modèle).
    rotary_emb      : GPTNeoXRotaryEmbedding (sur gpt_neox, partagé entre couches).
    rotary_ndims    : dims rotées (16 pour Pythia-160M, = rotary_pct * d_head).

    Retourne A : (B, n_heads, L, L) — attention maps post-softmax, float32.
    W_eff = W_frozen + lora_B @ lora_A * scale — gradients via lora_A/B uniquement.
    """
    if hidden.ndim != 3:
        raise ValueError(
            f"Expected hidden (B, L, d_model) got shape {hidden.shape}. "
            "Check HiddenStateCapture.hook_fn — the layer may return a tensor, not a tuple."
        )

    if input_layernorm is not None:
        hidden = input_layernorm(hidden)

    W_base = qkv_module.weight
    lora_A = qkv_module.lora_A["default"].weight
    lora_B = qkv_module.lora_B["default"].weight
    scale = qkv_module.scaling["default"]
    W_eff = W_base + lora_B @ lora_A * scale  # (3*d_model, d_model)

    bias = getattr(qkv_module, "bias", None)
    qkv_out = F.linear(hidden, W_eff, bias)  # (B, L, 3*d_model)

    B, L, _ = qkv_out.shape
    qkv_out = qkv_out.view(B, L, n_heads, 3 * d_head).transpose(1, 2)  # (B, n_heads, L, 3*d_head)
    Q, K, _ = qkv_out.chunk(3, dim=-1)  # each (B, n_heads, L, d_head)

    if rotary_emb is not None and rotary_ndims > 0:
        Q, K = _apply_rope(Q, K, rotary_emb, rotary_ndims)

    scores = Q @ K.transpose(-2, -1) * (d_head**-0.5)  # (B, n_heads, L, L)
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1)        # (B, n_heads, L, L)


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
    Comme get_attention_maps mais retourne aussi Z = A @ V (sorties par tête, pré-projection).

    Returns:
        A : (B, n_heads, L, L)        attention causale post-softmax (float32)
        Z : (B, L, n_heads, d_head)   sorties par tête, axe heads en 3e position pour
                                      faciliter la concaténation des followers
    """
    if hidden.ndim != 3:
        raise ValueError(
            f"Expected hidden (B, L, d_model) got shape {hidden.shape}."
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
    Q, K, V = qkv_out.chunk(3, dim=-1)  # (B, n_heads, L, d_head)

    if rotary_emb is not None and rotary_ndims > 0:
        Q, K = _apply_rope(Q, K, rotary_emb, rotary_ndims)

    scores = Q @ K.transpose(-2, -1) * (d_head**-0.5)
    causal_mask = torch.triu(
        torch.ones(L, L, dtype=torch.bool, device=scores.device),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    A = torch.softmax(scores.float(), dim=-1)  # (B, n_heads, L, L) fp32

    Z = A @ V.float()                          # (B, n_heads, L, d_head)
    Z = Z.transpose(1, 2).contiguous()         # (B, L, n_heads, d_head)
    return A, Z



def follower_diversity_loss(
    attn_weights: torch.Tensor,
    n_heads: int,
    leader_idx: int,
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """
    L_div = λ_lead·Σ_i sim(A_i, A_leader) + λ_peer·Σ_{i≠j, i,j≠leader} sim(A_i, A_j)

    attn_weights : (B, n_heads, L, L) — sortie de get_attention_maps.
    """
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)  # (n_heads, n_heads)

    fi = [i for i in range(n_heads) if i != leader_idx]
    fi_t = torch.tensor(fi, device=S.device)

    lf = S[fi_t, leader_idx].sum()
    S_peer = S[fi_t][:, fi_t]
    mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
    pp = S_peer[mask].sum()

    return lambda_lead * lf + lambda_peer * pp


def follower_diversity_loss_sq(
    attn_weights: torch.Tensor,
    n_heads: int,
    leader_idx: int,
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """
    L_div = λ_lead·Σ_i cos(A_i, A_leader)² + λ_peer·Σ_{i≠j, i,j≠leader} cos(A_i, A_j)²

    Squared cosine similarity avoids anticorrelation gradients (Exp2_5).
    """
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)  # (n_heads, n_heads)
    S_sq = S ** 2

    fi = [i for i in range(n_heads) if i != leader_idx]
    fi_t = torch.tensor(fi, device=S.device)

    lf = S_sq[fi_t, leader_idx].sum()
    S_peer = S_sq[fi_t][:, fi_t]
    mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
    pp = S_peer[mask].sum()

    return lambda_lead * lf + lambda_peer * pp


def follower_diversity_loss_hadamard(
    attn_weights: torch.Tensor,
    n_heads: int,
    leader_idx: int,
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """
    L_div = λ_lead·Σ_i |A_i ⊙ A_leader| + λ_peer·Σ_{i≠j, i,j≠leader} |A_i ⊙ A_j|

    Attention weights are non-negative (softmax), so |A_i ⊙ A_j| = <A_i, A_j> (Exp2_6).
    """
    B, H, L, _ = attn_weights.shape
    A_flat = attn_weights.view(B, H, L * L)  # (B, H, L²)

    fi = [i for i in range(n_heads) if i != leader_idx]
    fi_t = torch.tensor(fi, device=A_flat.device)

    A_leader = A_flat[:, leader_idx, :]   # (B, L²)
    A_followers = A_flat[:, fi_t, :]      # (B, n_f, L²)

    lf = (A_followers * A_leader.unsqueeze(1)).sum(dim=-1).mean(dim=0).sum()

    M = torch.bmm(A_followers, A_followers.transpose(1, 2)).mean(0)  # (n_f, n_f)
    n_f = len(fi)
    mask = ~torch.eye(n_f, dtype=torch.bool, device=A_flat.device)
    pp = M[mask].sum()

    return lambda_lead * lf + lambda_peer * pp


def follower_output_diversity_loss(
    head_outputs: torch.Tensor,
    n_heads: int,
    leader_idx: int,
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """
    Même formule que follower_diversity_loss (cosine similarity) mais sur les vecteurs
    de sortie Z_i = (A @ V)_i au lieu des cartes d'attention A_i.

    L_div = λ_lead·Σ_i sim(Z_i, Z_leader) + λ_peer·Σ_{i≠j, i,j≠leader} sim(Z_i, Z_j)

    head_outputs : (B, L, n_heads, d_head) — sortie de get_attention_outputs.
    """
    B, L, H, d_h = head_outputs.shape
    Z = head_outputs.permute(0, 2, 1, 3)          # (B, n_heads, L, d_head)
    Z_flat = Z.reshape(B, H, L * d_h).float()     # (B, n_heads, L*d_head)
    Z_norm = F.normalize(Z_flat, dim=-1)

    S = torch.bmm(Z_norm, Z_norm.transpose(1, 2)).mean(0)  # (n_heads, n_heads)

    fi = [i for i in range(n_heads) if i != leader_idx]
    fi_t = torch.tensor(fi, device=S.device)

    lf = S[fi_t, leader_idx].sum()
    S_peer = S[fi_t][:, fi_t]
    mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
    pp = S_peer[mask].sum()

    return lambda_lead * lf + lambda_peer * pp


def follower_erank_loss(
    head_outputs: torch.Tensor,
    n_heads: int,
    leader_idx: int,
    lambda_rank: float,
) -> torch.Tensor:
    """
    L_rank = -lambda_rank · erank(Z) avec Z = [Z_1, ..., Z_{n_f}] (followers concat, leader exclu).

    erank(Z) = exp(-Σ p_i log p_i) où p_i = σ_i² / Σ σ_j² et σ_i² sont les v.p. de Z^T Z.

    head_outputs : (B, L, n_heads, d_head) — sortie de get_attention_outputs.
    """
    B, L, H, d_h = head_outputs.shape
    fi = [i for i in range(H) if i != leader_idx]
    fi_t = torch.tensor(fi, device=head_outputs.device)

    Z = head_outputs.index_select(2, fi_t)                # (B, L, n_f, d_h)
    n_f = len(fi)
    Z = Z.reshape(B * L, n_f * d_h).float()               # (B·L, 704)

    Z = Z - Z.mean(dim=0, keepdim=True)                   # centrage colonnes

    G = Z.transpose(0, 1) @ Z                             # (704, 704) symétrique PSD
    G = 0.5 * (G + G.transpose(0, 1))                     # symétrisation explicite

    eigs = torch.linalg.eigvalsh(G).clamp(min=0)          # (704,) ≥ 0
    p = eigs / (eigs.sum() + 1e-12)
    entropy = -(p * (p + 1e-12).log()).sum()
    erank = entropy.exp()

    return -lambda_rank * erank


def leader_confidence_loss(attn_weights: torch.Tensor, leader_idx: int = 0) -> torch.Tensor:
    """
    L_conf = -1/(BL) · Σ_{b,l} max_{l'} A_leader[b, l, l']

    attn_weights : (B, n_heads, L, L) — sortie de get_attention_maps.
    """
    A_leader = attn_weights[:, leader_idx, :, :]  # (B, L, L)
    max_attn = A_leader.max(dim=-1).values         # (B, L)
    return -max_attn.mean()



def leader_confidence_loss_smooth(attn_weights: torch.Tensor, leader_idx: int = 0) -> torch.Tensor:
    """
    Variante de L_conf qui maximise la somme des carrés des poids d'attention du leader, encourageant une distribution plus pointue.
    L_conf = -1/(BL) · Σ_{b,l,l'} A_leader[b, l, l']^2
    """
    A_leader = attn_weights[:, leader_idx, :, :]  # (B, L, L)
    return -(A_leader ** 2).sum(dim=-1).mean()    # -||A_0||^2 moyenné



def minus_entropy_head(attn_weights: torch.Tensor, leader_idx: int = 0) -> torch.Tensor:
    """
    Calcule l'entropie moyenne des poids d'attention pour chaque tête.
    H = -1/(BL) · Σ_{b,l} Σ_{l'} A[head_idx][b, l, l'] log A[head_idx][b, l']

    attn_weights : (B, n_heads, L, L)
    """
    A_leader = attn_weights[:, leader_idx, :, :]  # (B, L, L)
    log_attn = torch.log(A_leader + 1e-12)  # éviter log(0)
    minus_entropies = (A_leader * log_attn).sum(dim=-1).mean() # H moyenné sur B et L
    return minus_entropies


def entropy_heads(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Calcule l'entropie moyenne des poids d'attention pour chaque tête.
    H = -1/(BL) · Σ_{b,l} Σ_{l'} A[head_idx][b, l, l'] log A[head_idx][b, l, l']

    attn_weights : (B, n_heads, L, L)
    Returns      : (n_heads,) — entropie positive moyennée sur B et L
    """
    log_attn = torch.log(attn_weights + 1e-12)
    entropies = -(attn_weights * log_attn).sum(dim=-1).mean(dim=(0, 2))
    assert entropies.shape == (attn_weights.shape[1],), f"Expected entropy shape ({attn_weights.shape[1]},) got {entropies.shape}"
    return entropies