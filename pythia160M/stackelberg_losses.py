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
    B, H, L, _ = Q.shape
    position_ids = torch.arange(L, device=Q.device).unsqueeze(0).expand(B, -1)
    cos, sin = rotary_emb(Q, position_ids)  # (B, L, rotary_ndims) — rotary_emb sur GPTNeoXModel
    cos = cos.unsqueeze(1)                  # (B, 1, L, rotary_ndims) — broadcast sur H
    sin = sin.unsqueeze(1)

    q_rot = Q[..., :rotary_ndims] * cos + _rotate_half(Q[..., :rotary_ndims]) * sin
    k_rot = K[..., :rotary_ndims] * cos + _rotate_half(K[..., :rotary_ndims]) * sin
    Q = torch.cat([q_rot, Q[..., rotary_ndims:]], dim=-1)
    K = torch.cat([k_rot, K[..., rotary_ndims:]], dim=-1)
    return Q, K


def compute_diversity_loss(
    hidden: torch.Tensor,
    qkv_module,
    n_heads: int,
    d_head: int,
    leader_idx: int,
    lambda_lead: float,
    lambda_peer: float,
    rotary_emb=None,
    rotary_ndims: int = 0,
    input_layernorm=None,
) -> torch.Tensor:
    """
    L_div = λ_lead·Σ_i sim(A_i, A_leader) + λ_peer·Σ_{i≠j, i,j≠leader} sim(A_i, A_j)

    hidden          : (B, L, d_model) — sortie de la couche design_layer-1.
    qkv_module      : PEFT-wrapped query_key_value Linear du design layer.
    input_layernorm : LayerNorm du design layer (appliqué à hidden avant QKV, comme dans le modèle).
    rotary_emb      : module RotaryEmbedding du design layer.
    rotary_ndims    : nombre de dims rotées (= rotary_pct * d_head, 16 pour Pythia-160M).

    W_eff = W_frozen + lora_B @ lora_A * scale — gradients flow through lora_A/B only.
    No eager attention, no output_attentions=True required.
    """
    if hidden.ndim != 3:
        raise ValueError(
            f"Expected hidden (B, L, d_model) got shape {hidden.shape}. "
            "Check HiddenStateCapture.hook_fn — the layer may return a tensor, not a tuple."
        )

    if input_layernorm is not None:
        hidden = input_layernorm(hidden)  # identique à GPTNeoXAttention.forward

    W_base = qkv_module.weight  # (3*d_model, d_model), frozen
    lora_A = qkv_module.lora_A["default"].weight  # (r, d_model)
    lora_B = qkv_module.lora_B["default"].weight  # (3*d_model, r)
    scale = qkv_module.scaling["default"]
    W_eff = W_base + lora_B @ lora_A * scale  # (3*d_model, d_model)

    bias = getattr(qkv_module, "bias", None)
    qkv_out = F.linear(hidden, W_eff, bias)  # (B, L, 3*d_model)

    B, L, _ = qkv_out.shape
    # Same reshape as GPTNeoXAttention.forward (interleaved QKV layout)
    qkv_out = qkv_out.view(B, L, n_heads, 3 * d_head).transpose(
        1, 2
    )  # (B, n_heads, L, 3*d_head)
    Q, K, _ = qkv_out.chunk(3, dim=-1)  # each (B, n_heads, L, d_head)

    if rotary_emb is not None and rotary_ndims > 0:
        Q, K = _apply_rope(Q, K, rotary_emb, rotary_ndims)

    scores = Q @ K.transpose(-2, -1) * (d_head**-0.5)  # (B, n_heads, L, L)
    A = torch.softmax(scores.float(), dim=-1)  # float32 for numerical stability

    # Cosine similarity between per-head attention maps, averaged over batch
    A_flat = A.view(B, n_heads, L * L)  # (B, n_heads, L²)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)  # (n_heads, n_heads)

    fi = [i for i in range(n_heads) if i != leader_idx]
    fi_t = torch.tensor(fi, device=S.device)

    lf = S[fi_t, leader_idx].sum()
    S_peer = S[fi_t][:, fi_t]
    mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
    pp = S_peer[mask].sum()

    return lambda_lead * lf + lambda_peer * pp


def leader_confidence_loss(attn_weights: torch.Tensor, leader_idx: int = 0) -> torch.Tensor:
    """
    L_conf = -1/(BL) · Σ_{b,l} max_{l'} A_leader[b, l, l']

    attn_weights : (B, H, L, L) — per-head attention maps (post-softmax).
                   Obtenu via compute_diversity_loss (A) ou output_attentions=True.
    """
    A_leader = attn_weights[:, leader_idx, :, :]  # (B, L, L)
    max_attn = A_leader.max(dim=-1).values         # (B, L)
    return -max_attn.mean()
