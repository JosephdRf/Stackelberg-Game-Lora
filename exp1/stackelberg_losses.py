"""
Stackelberg Attention Diversity — loss functions and utilities.

Multi-head attention heads are modeled as Stackelberg game players:
  - Head 0 (leader): optimizes L_CE only
  - Heads 1..H-1 (followers): optimize L_CE + diversity penalty

Follower i's diversity loss:
  L_div = λ_lead · sim(A_i, A_0) + λ_peer · Σ_{j≠i, j≠0} sim(A_i, A_j)

where sim(A_i, A_j) = ⟨A_i, A_j⟩_F / (‖A_i‖_F · ‖A_j‖_F)

Contains:
  - compute_diversity_loss : vectorized diversity penalty from attention weights
  - split_leader_follower_params : separate LoRA params into leader / follower
  - AttentionWeightCapture : hook-based extraction of attention matrices (B, H, L, L)
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
# Parameter splitting: leader (o_proj) vs followers (q/k/v_proj)
# ---------------------------------------------------------------------------


def split_leader_follower_params(
    model,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Split trainable (LoRA) parameters into leader and follower groups.

    Leader:   o_proj LoRA — controls how head outputs are mixed
    Follower: q_proj, k_proj, v_proj LoRA — controls what each head attends to

    Returns:
        (leader_params, follower_params)
    """
    leader, follower = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "o_proj" in name:
            leader.append(p)
        else:
            follower.append(p)
    return leader, follower


# ---------------------------------------------------------------------------
# Attention weight capture (hook-based, single layer)
# ---------------------------------------------------------------------------


class AttentionWeightCapture:
    """
    Hook-based capture of attention weights from a single attention layer.

    Registers a forward hook on ``model.layers[layer_idx].self_attn``.
    The model **must** be loaded with ``attn_implementation='eager'``
    (SDPA / Flash Attention do not materialise weight matrices).
    """

    def __init__(self):
        self._weights: Optional[torch.Tensor] = None
        self._hooks: list = []

    def register(self, model, layer_idx: int):
        """Register a forward hook on the attention module at *layer_idx*."""
        target_suffix = f"layers.{layer_idx}.self_attn"
        attn_module = None
        for name, module in model.named_modules():
            if name.endswith(target_suffix):
                attn_module = module
                break
        if attn_module is None:
            raise RuntimeError(
                f"Could not find attention module ending with '{target_suffix}'"
            )

        # Sanity check: eager attention required
        if hasattr(attn_module, "config"):
            impl = getattr(attn_module.config, "_attn_implementation", "unknown")
            if impl != "eager":
                raise RuntimeError(
                    f"Attention must use 'eager' implementation, got '{impl}'. "
                    "Pass attn_implementation='eager' when loading the model."
                )

        def hook_fn(module, args, output):
            # Qwen2Attention.forward → (attn_output, attn_weights)
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
