"""
Stackelberg — gradient utilities (Pythia-160M / GPT-NeoX).

Multi-head attention heads are modeled as Stackelberg game players:
  - Leader   : dense LoRA (output projection) — optimises L_CE
  - Followers : query_key_value LoRA          — optimise L_CE + optional L_div

Contains:
  - collect_lora_params   : gather all trainable LoRA params for a single optimizer
  - assemble_gradients    : combine follower + leader gradients before optimizer.step()
  - HiddenStateCapture    : forward hook on a GPTNeoXLayer to capture hidden states
  - compute_diversity_loss: L_div from hidden states, no eager / no output_attentions

Gradient assembly rules per param kind:
  - lora_B of query_key_value (design layer):
      rows are disjoint between leader (h0) and followers (h1-11)
      → g_final = g_follower (follower rows) + g_leader (leader rows)
  - lora_A of query_key_value, lora_B of dense (shared):
      → g_final = (g_follower + g_leader) / 2   (avoid double lr)
  - lora_A of dense (design layer, decomposable by columns):
      cols are disjoint → g_final = g_follower + g_leader
  - all other params (non design-layer):
      → g_final = g_follower   (leader does not own these)

Note: Pythia/GPT-NeoX specifics:
  - Leader param key   : "dense"             (output projection)
  - Follower param key : "query_key_value"   (fused QKV)
  - QKV layout         : interleaved [Q0,K0,V0,Q1,K1,V1,...] — same as modeling_gpt_neox.py
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Gradient masking helpers (pure functions, no hooks)
# ---------------------------------------------------------------------------


def _zero_leader_rows(grad: torch.Tensor, leader_q, leader_k, leader_v) -> torch.Tensor:
    """Zero the leader head rows in qkv lora_B gradient (follower phase)."""
    g = grad.clone()
    g[leader_q, :] = 0
    g[leader_k, :] = 0
    g[leader_v, :] = 0
    return g


def _keep_leader_rows(grad: torch.Tensor, leader_q, leader_k, leader_v) -> torch.Tensor:
    """Keep only leader head rows in qkv lora_B gradient (leader phase)."""
    g = torch.zeros_like(grad)
    g[leader_q, :] = grad[leader_q, :]
    g[leader_k, :] = grad[leader_k, :]
    g[leader_v, :] = grad[leader_v, :]
    return g


def _zero_leader_cols(grad: torch.Tensor, leader_o) -> torch.Tensor:
    """Zero the leader head cols in dense lora_A gradient (follower phase)."""
    g = grad.clone()
    g[:, leader_o] = 0
    return g


def _keep_leader_cols(grad: torch.Tensor, leader_o) -> torch.Tensor:
    """Keep only leader head cols in dense lora_A gradient (leader phase)."""
    g = torch.zeros_like(grad)
    g[:, leader_o] = grad[:, leader_o]
    return g


# ---------------------------------------------------------------------------
# Param collection & grad assembly descriptor
# ---------------------------------------------------------------------------


@dataclass
class ParamRole:
    """Describes how the gradient of a single param should be assembled."""
    param: torch.nn.Parameter
    name: str
    kind: str  # "qkv_lora_B" | "qkv_lora_A" | "dense_lora_A" | "dense_lora_B" | "other"


@dataclass
class GradAssembly:
    """
    Holds all ParamRole descriptors and the slice info needed for assembly.
    Built once before training, reused every step.
    """
    roles: List[ParamRole]
    leader_q: slice
    leader_k: slice
    leader_v: slice
    leader_o: slice


def collect_lora_params(
    model,
    design_layer: int = 9,
    d_model: int = 768,
    n_heads: int = 12,
    leader_idx: int = 0,
) -> Tuple[List[torch.nn.Parameter], "GradAssembly"]:
    """
    Returns (all_trainable_params, grad_assembly).

    all_trainable_params : flat list of all requires_grad params → pass to one AdamW.
    grad_assembly        : descriptor used by assemble_gradients() every step.
    """
    d_head = d_model // n_heads  # 64

    leader_q = slice(leader_idx * d_head,           (leader_idx + 1) * d_head)
    leader_k = slice(d_model   + leader_idx * d_head, d_model   + (leader_idx + 1) * d_head)
    leader_v = slice(2*d_model + leader_idx * d_head, 2*d_model + (leader_idx + 1) * d_head)
    leader_o = slice(leader_idx * d_head,           (leader_idx + 1) * d_head)

    layer_prefix = f"layers.{design_layer}."

    roles: List[ParamRole] = []
    seen_ids = set()
    all_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        all_params.append(p)

        is_design = layer_prefix in name
        is_qkv    = "query_key_value" in name
        is_dense  = "dense" in name and "dense_h_to_4h" not in name

        if is_design and is_qkv and "lora_B" in name:
            kind = "qkv_lora_B"
        elif is_design and is_qkv and "lora_A" in name:
            kind = "qkv_lora_A"
        elif is_design and is_dense and "lora_A" in name:
            kind = "dense_lora_A"
        elif is_design and is_dense and "lora_B" in name:
            kind = "dense_lora_B"
        else:
            kind = "other"

        roles.append(ParamRole(param=p, name=name, kind=kind))

    assembly = GradAssembly(
        roles=roles,
        leader_q=leader_q,
        leader_k=leader_k,
        leader_v=leader_v,
        leader_o=leader_o,
    )
    return all_params, assembly


def mask_follower_grad(assembly: GradAssembly) -> None:
    """
    Called right after follower_loss.backward().
    Masks the leader slices out of .grad for design-layer params so that
    the saved follower gradient contains zero on the leader slice.
    Works in-place on p.grad.
    """
    lq, lk, lv, lo = assembly.leader_q, assembly.leader_k, assembly.leader_v, assembly.leader_o
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_lora_B":
            p.grad[lq, :] = 0
            p.grad[lk, :] = 0
            p.grad[lv, :] = 0
        elif role.kind == "dense_lora_A":
            p.grad[:, lo] = 0


def mask_leader_grad(assembly: GradAssembly) -> None:
    """
    Called right after leader_loss.backward() (accumulates into p.grad).
    Zeros everything except the leader slice for design-layer params,
    and zeros the entire grad for "other" params (leader doesn't own them).
    Works in-place on p.grad.
    """
    lq, lk, lv, lo = assembly.leader_q, assembly.leader_k, assembly.leader_v, assembly.leader_o
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_lora_B":
            mask = torch.zeros_like(p.grad)
            mask[lq, :] = 1; mask[lk, :] = 1; mask[lv, :] = 1
            p.grad.mul_(mask)
        elif role.kind == "dense_lora_A":
            mask = torch.zeros_like(p.grad)
            mask[:, lo] = 1
            p.grad.mul_(mask)
        elif role.kind == "other":
            p.grad.zero_()
        # qkv_lora_A and dense_lora_B: shared, keep grad as-is (will be averaged)


def assemble_gradients(
    assembly: GradAssembly,
    g_follower: Dict[int, torch.Tensor],
    g_leader:   Dict[int, torch.Tensor],
) -> None:
    """
    Writes the final assembled gradient into p.grad for optimizer.step().

    g_follower, g_leader : {id(p): grad_tensor} — already masked by
        mask_follower_grad / mask_leader_grad respectively.

    Assembly rules:
      qkv_lora_B  : disjoint slices → add  (g_F + g_L)
      dense_lora_A: disjoint slices → add  (g_F + g_L)
      qkv_lora_A  : shared          → mean (g_F + g_L) / 2
      dense_lora_B: shared          → mean (g_F + g_L) / 2
      other       : follower only   → g_F
    """
    for role in assembly.roles:
        p   = role.param
        pid = id(p)
        gf  = g_follower.get(pid)
        gl  = g_leader.get(pid)

        if gf is None and gl is None:
            p.grad = None
            continue

        gf = gf if gf is not None else torch.zeros_like(p)
        gl = gl if gl is not None else torch.zeros_like(p)

        if role.kind in ("qkv_lora_B", "dense_lora_A"):
            # slices are disjoint — simple addition is correct
            p.grad = gf + gl
        elif role.kind in ("qkv_lora_A", "dense_lora_B"):
            # shared param, seen by both phases — average to avoid double lr
            p.grad = (gf + gl) * 0.5
        else:
            # "other": only followers own these params
            p.grad = gf


# ---------------------------------------------------------------------------
# Hidden state capture + diversity loss (no eager, no output_attentions)
# ---------------------------------------------------------------------------


class HiddenStateCapture:
    """
    Forward hook on a GPTNeoXLayer to capture hidden states (B, L, d_model).

    Register on layer design_layer-1 to get the input of the design layer.
    GPTNeoXLayer output is a tuple; output[0] is the hidden state tensor.
    The captured tensor stays in the autograd graph — gradients flow through it.
    """

    def __init__(self):
        self._hidden: Optional[torch.Tensor] = None
        self._hook = None

    def register(self, model, layer_idx: int):
        target = f"gpt_neox.layers.{layer_idx}"
        module = None
        for name, mod in model.named_modules():
            if name == target:
                module = mod
                break
        if module is None:
            raise RuntimeError(
                f"Layer '{target}' not found. "
                "Check that layer_idx is valid for Pythia-160M (0–11)."
            )

        def hook_fn(mod, args, output):
            self._hidden = output[0]   # (B, L, d_model), in computation graph

        self._hook = module.register_forward_hook(hook_fn)

    def get(self) -> Optional[torch.Tensor]:
        """Return captured hidden states and reset buffer."""
        h, self._hidden = self._hidden, None
        return h

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def compute_diversity_loss(
    hidden: torch.Tensor,
    qkv_module,
    n_heads: int,
    d_head: int,
    leader_idx: int,
    lambda_lead: float,
    lambda_peer: float,
) -> torch.Tensor:
    """
    L_div = λ_lead·Σ_i sim(A_i, A_leader) + λ_peer·Σ_{i≠j, i,j≠leader} sim(A_i, A_j)

    hidden     : (B, L, d_model) captured from the layer before the design layer.
    qkv_module : PEFT-wrapped query_key_value Linear of the design layer.

    W_eff = W_frozen + lora_B @ lora_A * scale — gradients flow through lora_A/B only.
    RoPE not applied: same rotation for all heads, relative diversity is preserved.
    No eager attention, no output_attentions=True required.
    """
    W_base = qkv_module.weight                        # (3*d_model, d_model), frozen
    lora_A = qkv_module.lora_A['default'].weight      # (r, d_model)
    lora_B = qkv_module.lora_B['default'].weight      # (3*d_model, r)
    scale  = qkv_module.scaling['default']
    W_eff  = W_base + lora_B @ lora_A * scale         # (3*d_model, d_model)

    bias = getattr(qkv_module, 'bias', None)
    qkv_out = F.linear(hidden, W_eff, bias)           # (B, L, 3*d_model)

    B, L, _ = qkv_out.shape
    # Same reshape as GPTNeoXAttention.forward (interleaved QKV layout)
    qkv_out = qkv_out.view(B, L, n_heads, 3 * d_head).transpose(1, 2)  # (B, n_heads, L, 3*d_head)
    Q, K, _ = qkv_out.chunk(3, dim=-1)               # each (B, n_heads, L, d_head)

    scores = Q @ K.transpose(-2, -1) * (d_head ** -0.5)   # (B, n_heads, L, L)
    A = torch.softmax(scores.float(), dim=-1)              # float32 for numerical stability

    # Cosine similarity between per-head attention maps, averaged over batch
    A_flat = A.view(B, n_heads, L * L)                     # (B, n_heads, L²)
    A_norm = F.normalize(A_flat, dim=-1)
    S = torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0)  # (n_heads, n_heads)

    fi   = [i for i in range(n_heads) if i != leader_idx]
    fi_t = torch.tensor(fi, device=S.device)

    lf     = S[fi_t, leader_idx].sum()
    S_peer = S[fi_t][:, fi_t]
    mask   = ~torch.eye(len(fi_t), dtype=torch.bool, device=S.device)
    pp     = S_peer[mask].sum()

    return lambda_lead * lf + lambda_peer * pp