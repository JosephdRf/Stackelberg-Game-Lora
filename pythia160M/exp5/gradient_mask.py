"""
Stackelberg — gradient utilities for full fine-tuning (Process 2).

Unlike the LoRA version (gradient_mask.py at the root), θ_L and θ_F are
slices of the actual pretrained weight matrices W_QKV and W_dense at the
design layer(s), not LoRA adapter matrices.

Parameter kinds:
  "qkv_weight"  — gpt_neox.layers.{dl}.attention.query_key_value.weight  (3d×d)
  "dense_weight" — gpt_neox.layers.{dl}.attention.dense.weight           (d×d)
  "other"        — all other parameters (other layers, layernorms, FFN)

Gradient assembly rules (same logic as LoRA version, different tensors):
  qkv_weight, dense_weight : disjoint slices (θ_L ∪ θ_F) → g_final = g_F + g_L
  other                    : θ_S — follower only → g_final = g_F

HiddenStateCapture is shared with the root gradient_mask — import from there.
"""

import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# Re-export HiddenStateCapture from root to avoid duplication
from gradient_mask import HiddenStateCapture  # noqa: F401


# ---------------------------------------------------------------------------
# Param collection & grad assembly descriptor
# ---------------------------------------------------------------------------


@dataclass
class ParamRole:
    """Describes how the gradient of a single param should be assembled."""
    param: torch.nn.Parameter
    name: str
    kind: str  # "qkv_weight" | "dense_weight" | "other"


@dataclass
class GradAssembly:
    """
    Holds all ParamRole descriptors and the slice lists needed for assembly.

    leader_q / leader_k / leader_v : list of slices, one per leader head,
        indexing ROWS in W_QKV (layout: [Q_h0…, K_h0…, V_h0…, Q_h1…, …]).
    leader_o : list of slices, one per leader head, indexing COLS in W_dense.
    """
    roles: List[ParamRole]
    leader_q: List[slice]
    leader_k: List[slice]
    leader_v: List[slice]
    leader_o: List[slice]


def collect_fullft_params(
    model,
    design_layers: List[int] = None,
    d_model: int = 768,
    n_heads: int = 12,
    leader_indices: List[int] = None,
) -> Tuple[List[torch.nn.Parameter], GradAssembly]:
    """
    Returns (all_trainable_params, grad_assembly) for full fine-tuning.

    design_layers  : layer indices treated as Stackelberg design layers (default [9]).
    leader_indices : head indices acting as leaders (default [0]).

    Assumes all model parameters already have requires_grad=True before calling.
    """
    if design_layers is None:
        design_layers = [9]
    if leader_indices is None:
        leader_indices = [0]

    d_head = d_model // n_heads  # 64 for Pythia-160M

    # Row slices in W_QKV (shape 3d×d): Q block then K block then V block
    leader_q = [slice(i * d_head, (i + 1) * d_head) for i in leader_indices]
    leader_k = [slice(d_model + i * d_head, d_model + (i + 1) * d_head) for i in leader_indices]
    leader_v = [slice(2 * d_model + i * d_head, 2 * d_model + (i + 1) * d_head) for i in leader_indices]
    # Column slices in W_dense (shape d×d)
    leader_o = [slice(i * d_head, (i + 1) * d_head) for i in leader_indices]

    _layer_prefixes = {f"layers.{dl}." for dl in design_layers}

    roles: List[ParamRole] = []
    seen_ids: set = set()
    all_params: List[torch.nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        all_params.append(p)

        is_design = any(prefix in name for prefix in _layer_prefixes)
        # Match only the attention QKV and output projection, not FFN dense
        is_qkv   = "attention.query_key_value" in name and name.endswith(".weight")
        is_dense = "attention.dense" in name and name.endswith(".weight")

        if is_design and is_qkv:
            kind = "qkv_weight"
        elif is_design and is_dense:
            kind = "dense_weight"
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
    Zeros the leader row/col slices in p.grad so that g_follower is zero
    on all leader regions.  Works in-place on p.grad.
    """
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_weight":
            for lq, lk, lv in zip(assembly.leader_q, assembly.leader_k, assembly.leader_v):
                p.grad[lq, :] = 0
                p.grad[lk, :] = 0
                p.grad[lv, :] = 0
        elif role.kind == "dense_weight":
            for lo in assembly.leader_o:
                p.grad[:, lo] = 0


def mask_leader_grad(assembly: GradAssembly) -> None:
    """
    Called right after leader_loss.backward().
    Keeps only the leader row/col slices; zeros everything else.
    Works in-place on p.grad.
    """
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_weight":
            mask = torch.zeros_like(p.grad)
            for lq, lk, lv in zip(assembly.leader_q, assembly.leader_k, assembly.leader_v):
                mask[lq, :] = 1
                mask[lk, :] = 1
                mask[lv, :] = 1
            p.grad.mul_(mask)
        elif role.kind == "dense_weight":
            mask = torch.zeros_like(p.grad)
            for lo in assembly.leader_o:
                mask[:, lo] = 1
            p.grad.mul_(mask)
        elif role.kind == "other":
            p.grad.zero_()


def assemble_gradients(
    assembly: GradAssembly,
    g_follower: Dict[int, torch.Tensor],
    g_leader: Dict[int, torch.Tensor],
) -> None:
    """
    Writes the final assembled gradient into p.grad for optimizer.step().

    Assembly rules:
      qkv_weight, dense_weight : disjoint slices → g_F + g_L
      other                    : follower only → g_F
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

        if role.kind in ("qkv_weight", "dense_weight"):
            # slices are disjoint — simple addition is correct
            p.grad = gf + gl
        else:
            # "other": θ_S updated only by follower gradient
            p.grad = gf
