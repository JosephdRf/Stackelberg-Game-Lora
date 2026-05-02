"""
Stackelberg — gradient utilities (Pythia-160M / GPT-NeoX).

One attention layer (design_layer) is treated as a Stackelberg game:
  - Leader  (head h0) : commits first by anticipating the followers' best response.
  - Followers (h1–11) : best-respond to the leader's announced strategy.

LoRA matrices of the design layer and their decomposability:
  - qkv_lora_B  ∈ ℝ^(3d×r) : decomposable by rows — Q/K/V block per head.
                               Leader owns rows lq, lk, lv ; followers own the rest.
  - qkv_lora_A  ∈ ℝ^(r×d)  : shared (input dimension has no per-head structure).
  - dense_lora_A ∈ ℝ^(r×d) : decomposable by columns — cols lo correspond to the
                               d_head input features from head h0's output in the
                               concatenated attention output.
                               Leader owns cols lo ; followers own the rest.
  - dense_lora_B ∈ ℝ^(d×r) : shared (output dimension not decomposable by head).

Gradient assembly rules:
  qkv_lora_B, dense_lora_A : disjoint slices (θ_L ∪ θ_F) → g_final = g_F + g_L
  qkv_lora_A, dense_lora_B : θ_S — g_L zeroed in mask_leader_grad → g_final = g_F
  other (non-design-layer)  : θ_S — follower only → g_final = g_F

Contents:
  collect_lora_params    : classify all trainable LoRA params; return flat list + GradAssembly
  mask_follower_grad     : zero leader slices (θ_L) in p.grad after follower backward
  mask_leader_grad       : keep only leader slices (θ_L) in p.grad; zero everything else
  assemble_gradients     : write final p.grad from g_F + g_L
  HiddenStateCapture     : forward hook on a GPTNeoXLayer to capture hidden states
  compute_diversity_loss : L_div from per-head attention maps; no eager / no output_attentions

GPT-NeoX specifics:
  QKV layout    : interleaved [Q0,K0,V0, Q1,K1,V1, …] — same as modeling_gpt_neox.py
  dense_lora_A cols lo = [leader_idx·d_head : (leader_idx+1)·d_head]
    (the d_head input features supplied by head h0's output in the concatenated input)
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

    leader_q = slice(leader_idx * d_head, (leader_idx + 1) * d_head)
    leader_k = slice(d_model + leader_idx * d_head, d_model + (leader_idx + 1) * d_head)
    leader_v = slice(
        2 * d_model + leader_idx * d_head, 2 * d_model + (leader_idx + 1) * d_head
    )
    leader_o = slice(leader_idx * d_head, (leader_idx + 1) * d_head)

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
        is_qkv = "query_key_value" in name
        is_dense = "dense" in name and "dense_h_to_4h" not in name

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
    lq, lk, lv, lo = (
        assembly.leader_q,
        assembly.leader_k,
        assembly.leader_v,
        assembly.leader_o,
    )
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
    lq, lk, lv, lo = (
        assembly.leader_q,
        assembly.leader_k,
        assembly.leader_v,
        assembly.leader_o,
    )
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_lora_B":
            mask = torch.zeros_like(p.grad)
            mask[lq, :] = 1
            mask[lk, :] = 1
            mask[lv, :] = 1
            p.grad.mul_(mask)
        elif role.kind == "dense_lora_A":
            mask = torch.zeros_like(p.grad)
            mask[:, lo] = 1
            p.grad.mul_(mask)
        elif role.kind in ("qkv_lora_A", "dense_lora_B"):
            p.grad.zero_()  # θ_S: g_L contribution is zero, update comes from g_F only
        elif role.kind == "other":
            p.grad.zero_()


def assemble_gradients(
    assembly: GradAssembly,
    g_follower: Dict[int, torch.Tensor],
    g_leader: Dict[int, torch.Tensor],
) -> None:
    """
    Writes the final assembled gradient into p.grad for optimizer.step().

    g_follower, g_leader : {id(p): grad_tensor} — already masked by
        mask_follower_grad / mask_leader_grad respectively.

    Assembly rules:
      qkv_lora_B, dense_lora_A : disjoint slices → g_F + g_L
      qkv_lora_A, dense_lora_B : θ_S, g_L zeroed by mask_leader_grad → g_F
      other                     : follower only → g_F
    """
    for role in assembly.roles:
        p = role.param
        pid = id(p)
        gf = g_follower.get(pid)
        gl = g_leader.get(pid)

        if gf is None and gl is None:
            p.grad = None
            continue

        gf = gf if gf is not None else torch.zeros_like(p)
        gl = gl if gl is not None else torch.zeros_like(p)

        if role.kind in ("qkv_lora_B", "dense_lora_A"):
            # slices are disjoint — simple addition is correct
            p.grad = gf + gl
        elif role.kind in ("qkv_lora_A", "dense_lora_B"):
            # θ_S: updated only with g_F (g_L is zero after mask_leader_grad)
            p.grad = gf
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
            if name == target or name.endswith("." + target):
                module = mod
                break
        if module is None:
            raise RuntimeError(
                f"Layer '{target}' not found. "
                "Check that layer_idx is valid for Pythia-160M (0–11)."
            )

        def hook_fn(mod, args, output):
            # GPTNeoXLayer returns a tuple (hidden, ...) in most cases, but
            # some transformers versions / PEFT builds return the tensor directly.
            self._hidden = output[0] if isinstance(output, (tuple, list)) else output

        self._hook = module.register_forward_hook(hook_fn)

    def get(self) -> Optional[torch.Tensor]:
        """Return captured hidden states and reset buffer."""
        h, self._hidden = self._hidden, None
        return h

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
