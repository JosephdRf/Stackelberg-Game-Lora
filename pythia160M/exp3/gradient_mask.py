"""
Stackelberg — gradient utilities (Pythia-160M / GPT-NeoX) — exp3.

Identical to exp2/gradient_mask.py except that leader_idx is replaced by
leader_indices (list[int]), allowing an arbitrary subset of heads to act
as leaders.  All others are followers.

LoRA matrices of the design layer and their decomposability:
  - qkv_lora_B  ∈ ℝ^(3d×r) : decomposable by rows — Q/K/V block per head.
                               Leader heads own rows lq_k, lk_k, lv_k for each k in L.
  - qkv_lora_A  ∈ ℝ^(r×d)  : shared.
  - dense_lora_A ∈ ℝ^(r×d) : decomposable by columns — cols lo_k per leader head k.
  - dense_lora_B ∈ ℝ^(d×r) : shared.

Gradient assembly rules (unchanged from exp2):
  qkv_lora_B, dense_lora_A : disjoint slices (θ_L ∪ θ_F) → g_final = g_F + g_L
  qkv_lora_A, dense_lora_B : θ_S — g_L zeroed → g_final = g_F
  other                     : θ_S — follower only → g_final = g_F
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


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
    Holds all ParamRole descriptors and the slice lists needed for assembly.
    Built once before training, reused every step.

    leader_q / leader_k / leader_v : list of slices, one per leader head,
        indexing rows in qkv_lora_B (layout: [Q_h0, K_h0, V_h0, Q_h1, ...]).
    leader_o : list of slices, one per leader head, indexing cols in dense_lora_A.
    """

    roles: List[ParamRole]
    leader_q: List[slice]
    leader_k: List[slice]
    leader_v: List[slice]
    leader_o: List[slice]


def collect_lora_params(
    model,
    design_layers: List[int] = None,
    d_model: int = 768,
    n_heads: int = 12,
    leader_indices: List[int] = None,
) -> Tuple[List[torch.nn.Parameter], "GradAssembly"]:
    """
    Returns (all_trainable_params, grad_assembly).

    design_layers  : list of layer indices treated as Stackelberg design layers (default [9]).
    leader_indices : list of head indices that act as leaders (default [0]).
    """
    if design_layers is None:
        design_layers = [9]
    if leader_indices is None:
        leader_indices = [0]

    d_head = d_model // n_heads  # 64 for Pythia-160M

    # One slice per leader head
    leader_q = [slice(i * d_head, (i + 1) * d_head) for i in leader_indices]
    leader_k = [slice(d_model + i * d_head, d_model + (i + 1) * d_head) for i in leader_indices]
    leader_v = [slice(2 * d_model + i * d_head, 2 * d_model + (i + 1) * d_head) for i in leader_indices]
    leader_o = [slice(i * d_head, (i + 1) * d_head) for i in leader_indices]

    _layer_prefixes = {f"layers.{dl}." for dl in design_layers}

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

        is_design = any(prefix in name for prefix in _layer_prefixes)
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
    Zeros the leader slices (all leader heads) in p.grad so that the saved
    follower gradient contains zero on all leader slices.
    Works in-place on p.grad.
    """
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_lora_B":
            for lq, lk, lv in zip(assembly.leader_q, assembly.leader_k, assembly.leader_v):
                p.grad[lq, :] = 0
                p.grad[lk, :] = 0
                p.grad[lv, :] = 0
        elif role.kind == "dense_lora_A":
            for lo in assembly.leader_o:
                p.grad[:, lo] = 0


def mask_leader_grad(assembly: GradAssembly) -> None:
    """
    Called right after leader_loss.backward().
    Keeps only the leader slices (all leader heads); zeros everything else.
    Works in-place on p.grad.
    """
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind == "qkv_lora_B":
            mask = torch.zeros_like(p.grad)
            for lq, lk, lv in zip(assembly.leader_q, assembly.leader_k, assembly.leader_v):
                mask[lq, :] = 1
                mask[lk, :] = 1
                mask[lv, :] = 1
            p.grad.mul_(mask)
        elif role.kind == "dense_lora_A":
            mask = torch.zeros_like(p.grad)
            for lo in assembly.leader_o:
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
# Hidden state capture
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
