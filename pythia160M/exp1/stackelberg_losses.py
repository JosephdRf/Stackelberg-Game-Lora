"""
Stackelberg Attention Diversity — loss functions and utilities (Pythia-160M / GPT-NeoX).

Multi-head attention heads are modeled as Stackelberg game players:
  - Leader   : dense LoRA (output projection) — optimizes L_CE only
  - Followers : query_key_value LoRA           — optimize L_CE + diversity penalty

Follower i's diversity loss:
  L_div = λ_lead · sim(A_i, A_0) + λ_peer · Σ_{j≠i, j≠0} sim(A_i, A_j)

where sim(A_i, A_j) = ⟨A_i, A_j⟩_F / (‖A_i‖_F · ‖A_j‖_F)

Contains:
  - compute_diversity_loss    : vectorized diversity penalty from attention weights
  - split_leader_follower_params : separate LoRA params into leader / follower
  - AttentionWeightCapture   : hook-based extraction of attention matrices (B, H, L, L)

Note: Pythia/GPT-NeoX specifics vs Qwen version:
  - Leader param key : "dense"             (vs "o_proj")
  - Follower param key : "query_key_value" (vs "q/k/v_proj")
  - Attn module path : "gpt_neox.layers.{l}.attention"
  - Attn weights at output[2] when output_attentions=True
    (GPTNeoXAttention returns (attn_output, present, attn_weights))
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Attention similarity & diversity loss
# ---------------------------------------------------------------------------


def compute_diversity_loss(
    attn_weights: torch.Tensor,
    leader_idx: int = 0,
    lambda_lead: float = 0.1,
    lambda_peer: float = 0.01,
) -> torch.Tensor:
    """
    Vectorized diversity loss over all follower heads.

    Args:
        attn_weights: (B, H, L, L) — attention weights from one layer
        leader_idx:  which head is the leader
        lambda_lead: penalty weight for similarity with leader
        lambda_peer: penalty weight for similarity between followers
    Returns:
        Scalar diversity loss
    """
    B, H, L, _ = attn_weights.shape

    # Flatten & normalise per head: (B, H, L²)
    A = attn_weights.reshape(B, H, -1).float()
    norms = A.norm(dim=2, keepdim=True).clamp(min=1e-8) 
    A_n = A / norms

    # Pairwise cosine similarity: (B, H, H) → mean over batch → (H, H)
    S = torch.bmm(A_n, A_n.transpose(1, 2)).mean(0)

    # Follower indices
    fi = [i for i in range(H) if i != leader_idx]
    fi_t = torch.tensor(fi, device=attn_weights.device)

    # Leader-follower similarities: Σ_i sim(A_i, A_leader)
    lf = S[fi_t, leader_idx].sum()

    # Peer similarities (off-diagonal among followers)
    S_peer = S[fi_t][:, fi_t]
    mask = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
    pp = S_peer[mask].sum()

    return lambda_lead * lf + lambda_peer * pp


# ---------------------------------------------------------------------------
# Parameter splitting: leader (leader_idx) vs followers (rest) with gradient hooks
# ---------------------------------------------------------------------------


class GradMaskMode:
    """
    Shared mutable flag for mode-aware gradient hooks on design-layer params.

    The same LoRA parameter is shared by both the leader and follower optimizer.
    The gradient hook reads this flag to decide which head slice to keep/zero:
      - FOLLOWER mode (Phase 1): zero out the leader slice → follower sees only its heads
      - LEADER mode   (Phase 2): zero out follower slices  → leader sees only its head

    Usage in the training loop:
        mask_mode.set_leader()   # before leader_ce.backward()
        ...
        mask_mode.set_follower() # after leader_optimizer.step()
    """
    FOLLOWER = "follower"
    LEADER   = "leader"

    def __init__(self):
        self.value = self.FOLLOWER

    def set_follower(self):
        self.value = self.FOLLOWER

    def set_leader(self):
        self.value = self.LEADER


def split_leader_follower_params(
    model,
    leader_idx: int = 0,
    design_layer: int = 9,
    d_model: int = 768,
    n_heads: int = 12,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], list, "GradMaskMode"]:
    """
    Sépare les paramètres LoRA entraînables en deux groupes :
      - leader_params  : paramètres dont seule la tranche head=leader_idx reçoit un gradient
      - follower_params: paramètres dont les tranches head≠leader_idx reçoivent un gradient

    Pour Pythia (GPT-NeoX), Q/K/V sont fusionnées en un seul tenseur 'query_key_value'
    de shape (d_model, 3*d_model). On ne peut pas mettre ce tenseur dans deux groupes
    d'optimiseur distincts — on enregistre donc un gradient hook à état partagé qui
    masque les tranches indésirables selon la phase (follower ou leader).

    Layout de query_key_value (lora_B shape: (3*d_model, r)) :
        Q : lignes [h*d_head : (h+1)*d_head]
        K : lignes [d_model + h*d_head : d_model + (h+1)*d_head]
        V : lignes [2*d_model + h*d_head : 2*d_model + (h+1)*d_head]

    Layout de dense (lora_A shape: (r, d_model)) :
        Tête h : colonnes [h*d_head : (h+1)*d_head]

    Returns:
        (leader_params, follower_params, hooks, mask_mode)
        Appeler mask_mode.set_leader() avant leader_ce.backward(),
        mask_mode.set_follower() après leader_optimizer.step().
    """
    d_head = d_model // n_heads  # 64

    leader_q = slice(leader_idx * d_head, (leader_idx + 1) * d_head)
    leader_k = slice(d_model   + leader_idx * d_head, d_model   + (leader_idx + 1) * d_head)
    leader_v = slice(2*d_model + leader_idx * d_head, 2*d_model + (leader_idx + 1) * d_head)
    leader_o = slice(leader_idx * d_head, (leader_idx + 1) * d_head)

    layer_prefix = f"layers.{design_layer}."
    mask_mode = GradMaskMode()

    leader_params   = []
    follower_params = []
    _hooks = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_design_layer = layer_prefix in name
        is_qkv   = "query_key_value" in name
        is_dense  = "dense" in name and "dense_h_to_4h" not in name

        if not is_design_layer or (not is_qkv and not is_dense):
            follower_params.append(p)
            continue

        # Design-layer QKV / dense: shared parameter, mode-aware hook handles slicing.
        leader_params.append(p)
        follower_params.append(p)

        def make_mode_hook(param_name, qkv, dense):
            def hook(grad):
                g = grad.clone()
                if mask_mode.value == GradMaskMode.FOLLOWER:
                    # Phase 1: zero the leader slice so the follower doesn't update it.
                    if qkv and "lora_B" in param_name:
                        g[leader_q, :] = 0
                        g[leader_k, :] = 0
                        g[leader_v, :] = 0
                    elif dense and "lora_A" in param_name:
                        g[:, leader_o] = 0
                else:
                    # Phase 2: keep only the leader slice, zero everything else.
                    mask = torch.zeros_like(g)
                    if qkv and "lora_B" in param_name:
                        mask[leader_q, :] = 1
                        mask[leader_k, :] = 1
                        mask[leader_v, :] = 1
                    else:
                        # lora_A (QKV or dense lora_B): shared projection, keep all.
                        mask[:] = 1
                    if dense and "lora_A" in param_name:
                        mask = torch.zeros_like(g)
                        mask[:, leader_o] = 1
                    g = g * mask
                return g
            return hook

        h = p.register_hook(make_mode_hook(name, is_qkv, is_dense))
        _hooks.append(h)

    # Note sur lora_A :
    # lora_A projette (in_features → r) : elle s'applique AVANT la séparation par tête.
    # Son gradient est un mélange de toutes les têtes — on ne peut pas la découper
    # proprement sans refactoriser le modèle. On laisse lora_A partagée entre leader
    # et follower (gradient complet des deux côtés). Seul lora_B, qui projette
    # (r → out_features), est découpable proprement par tranche de tête.

    return leader_params, follower_params, _hooks, mask_mode



# ---------------------------------------------------------------------------
# Attention weight capture (hook-based, single layer)
# ---------------------------------------------------------------------------


class AttentionWeightCapture:
    """
    Hook-based capture of attention weights from a single attention layer.

    Registers a forward hook on ``model.gpt_neox.layers[layer_idx].attention``.

    Important:
    - The model must be called with ``output_attentions=True`` to materialise weights.
    - GPTNeoXAttention returns (attn_output, present, attn_weights) when
      output_attentions=True — weights are at index 2.
    - PEFT-wrapped models expose gpt_neox via attribute delegation.
    """

    def __init__(self):
        self._weights: Optional[torch.Tensor] = None
        self._hooks: list = []

    def register(self, model, layer_idx: int):
        """Register a forward hook on the attention module at *layer_idx*."""
        target_suffix = f"gpt_neox.layers.{layer_idx}.attention"
        attn_module = None
        for name, module in model.named_modules():
            if name.endswith(target_suffix):
                attn_module = module
                break
        if attn_module is None:
            raise RuntimeError(
                f"Could not find attention module ending with '{target_suffix}'. "
                "Check that layer_idx is valid for Pythia-160M (0–11)."
            )

        def hook_fn(module, args, output):
            # GPTNeoXAttention returns (attn_output, attn_weights) — 2-tuple.
            # (Older transformers had a 3-tuple with present_key_value at index 1.)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self._weights = output[1]  # (B, H, L, L)

        self._hooks.append(attn_module.register_forward_hook(hook_fn))

    def get(self) -> Optional[torch.Tensor]:
        """Return captured attention weights (B, H, L, L) and reset buffer."""
        w = self._weights
        self._weights = None
        return w

    def remove(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
