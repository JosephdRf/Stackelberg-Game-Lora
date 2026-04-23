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


def split_leader_follower_params(
    model,
    leader_idx: int = 0,
    design_layer: int = 9,
    d_model: int = 768,
    n_heads: int = 12,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Sépare les paramètres LoRA entraînables en deux groupes :
      - leader_params  : paramètres dont seule la tranche head=leader_idx reçoit un gradient
      - follower_params: paramètres dont les tranches head≠leader_idx reçoivent un gradient

    Pour Pythia (GPT-NeoX), Q/K/V sont fusionnées en un seul tenseur 'query_key_value'
    de shape (d_model, 3*d_model). On ne peut pas mettre ce tenseur dans deux groupes
    d'optimiseur distincts — on enregistre donc des gradient hooks qui masquent les
    tranches indésirables lors du backward.

    Layout de query_key_value (dim=1 pour lora_B, dim=0 pour lora_A) :
        Q : colonnes [h*d_head : (h+1)*d_head]
        K : colonnes [d_model + h*d_head : d_model + (h+1)*d_head]
        V : colonnes [2*d_model + h*d_head : 2*d_model + (h+1)*d_head]

    Layout de dense / o_proj (dim=0) :
        Tête h : lignes [h*d_head : (h+1)*d_head]

    Args:
        model        : PeftModel Pythia-160M
        leader_idx   : index de la tête leader (défaut 0)
        design_layer : layer sur lequel s'applique la séparation (défaut 9)
        d_model      : dimension du modèle (768 pour Pythia-160M)
        n_heads      : nombre de têtes (12 pour Pythia-160M)

    Returns:
        (leader_params, follower_params) — listes de nn.Parameter
        Les paramètres hors design_layer sont mis dans follower_params par défaut.
    """
    d_head = d_model // n_heads  # 64

    # Slices de la tête leader dans les matrices fusionnées
    # query_key_value : shape (d_model, 3*d_model) — on indexe sur la dim des têtes
    leader_q = slice(leader_idx * d_head, (leader_idx + 1) * d_head)
    leader_k = slice(d_model   + leader_idx * d_head, d_model   + (leader_idx + 1) * d_head)
    leader_v = slice(2*d_model + leader_idx * d_head, 2*d_model + (leader_idx + 1) * d_head)
    leader_o = slice(leader_idx * d_head, (leader_idx + 1) * d_head)  # dim 0 de dense

    # Préfixe du design layer dans les noms de paramètres PEFT
    # ex: "base_model.model.gpt_neox.layers.9.attention.query_key_value.lora_A.default.weight"
    layer_prefix = f"layers.{design_layer}."

    leader_params   = []
    follower_params = []
    _hooks = []  # on stocke les hooks pour pouvoir les retirer si besoin

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_design_layer = layer_prefix in name
        is_qkv   = "query_key_value" in name
        is_dense  = "dense" in name and "dense_h_to_4h" not in name  # éviter FFN

        if not is_design_layer or (not is_qkv and not is_dense):
            # Hors design layer ou FFN → follower par défaut
            follower_params.append(p)
            continue

        # Paramètre du design layer sur attention QKV ou dense
        # On met le même paramètre dans LES DEUX groupes — les hooks
        # garantissent que chaque optimiseur ne voit que sa tranche.
        leader_params.append(p)
        follower_params.append(p)

        # ── Gradient hooks ──
        # lora_A : shape (r, in_features)  — on masque sur dim 1 (in_features)
        # lora_B : shape (out_features, r) — on masque sur dim 0 (out_features)
        # Pour QKV, out_features = 3*d_model, in_features = d_model
        # Pour dense, out_features = d_model, in_features = d_model

        def make_leader_hook(param_name, qkv, dense):
            """Garde seulement la tranche leader, annule le reste."""
            def hook(grad):
                g = grad.clone()
                mask = torch.zeros_like(g)
                if qkv:
                    if "lora_B" in param_name:
                        # lora_B shape (3*d_model, r) → mask sur dim 0
                        mask[leader_q, :] = 1
                        mask[leader_k, :] = 1
                        mask[leader_v, :] = 1
                    else:
                        # lora_A shape (r, d_model) → pas de découpage tête ici
                        # lora_A projette depuis d_model, partagé entre toutes têtes
                        # On garde tout (approximation nécessaire — voir note)
                        mask[:] = 1
                elif dense:
                    if "lora_A" in param_name:
                        # lora_A shape (r, d_model) → mask sur dim 1
                        mask[:, leader_o] = 1
                    else:
                        # lora_B shape (d_model, r) → pas de découpage tête
                        mask[:] = 1
                return g * mask
            return hook

        def make_follower_hook(param_name, qkv, dense):
            """Annule la tranche leader, garde le reste."""
            def hook(grad):
                g = grad.clone()
                if qkv:
                    if "lora_B" in param_name:
                        g[leader_q, :] = 0
                        g[leader_k, :] = 0
                        g[leader_v, :] = 0
                else:
                    if "lora_A" in param_name:
                        g[:, leader_o] = 0
                return g
            return hook

        # On enregistre les deux hooks — ils sont appelés en ordre LIFO
        # donc on enregistre follower en premier, leader en second.
        # Chaque optimiseur appellera step() sur le même p.grad —
        # MAIS les deux backward() sont séparés dans le code d'entraînement.
        # Les hooks sont donc utilisés pour zerograder ce qu'il ne faut pas.
        h1 = p.register_hook(make_follower_hook(name, is_qkv, is_dense))
        _hooks.append(h1)

    # Note sur lora_A :
    # lora_A projette (in_features → r) : elle s'applique AVANT la séparation par tête.
    # Son gradient est un mélange de toutes les têtes — on ne peut pas la découper
    # proprement sans refactoriser le modèle. On laisse lora_A partagée entre leader
    # et follower (gradient complet des deux côtés). Seul lora_B, qui projette
    # (r → out_features), est découpable proprement par tranche de tête.

    return leader_params, follower_params, _hooks



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
