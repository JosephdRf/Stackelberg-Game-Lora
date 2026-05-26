"""
Stackelberg — gradient utilities (Qwen2.5-0.5B / Qwen2 architecture).

Port du module Pythia (pythia160M/gradient_mask.py) adapté à Qwen2 :
  - Projections séparées q_proj / k_proj / v_proj / o_proj (vs query_key_value fusionné Pythia)
  - GQA : 14 Q-heads regroupées en 2 KV-heads (7 Q-heads par groupe)

Two training modes are supported:

LoRA fine-tuning (collect_lora_params):
  - q_proj.lora_B  ∈ ℝ^(896×r) : decomposable by rows — Q-head h occupies rows [h*64, (h+1)*64).
  - k_proj.lora_B  ∈ ℝ^(128×r) : decomposable by rows — KV-head k occupies rows [k*64, (k+1)*64).
  - v_proj.lora_B  ∈ ℝ^(128×r) : same as k_proj.
  - o_proj.lora_A  ∈ ℝ^(r×896) : decomposable by columns — Q-head h occupies cols [h*64, (h+1)*64).
  - q_proj.lora_A, k_proj.lora_A, v_proj.lora_A ∈ ℝ^(r×896) : shared (follower-only).
  - o_proj.lora_B  ∈ ℝ^(896×r) : shared (follower-only).

Full fine-tuning (collect_fullft_params):
  - q_proj.weight  ∈ ℝ^(896×896) : decomposable by rows.
  - k_proj.weight  ∈ ℝ^(128×896) : decomposable by rows.
  - v_proj.weight  ∈ ℝ^(128×896) : decomposable by rows.
  - o_proj.weight  ∈ ℝ^(896×896) : decomposable by columns.

Leader convention (default) :
  leader_q_heads  = [0, 1, 2, 3, 4, 5, 6]  → 7 Q-heads = 1 groupe GQA complet.
  leader_kv_heads = [0]                     → la KV-head partagée par ce groupe.
  Q-head h ↔ KV-head h // 7 (groupement contigu via repeat_kv expand+reshape).

Gradient assembly rules (both modes):
  row-decomposable {q,k,v}_lora_B / weight   : disjoint slices → g_final = g_F + g_L
  col-decomposable o_lora_A / weight          : disjoint slices → g_final = g_F + g_L
  shared LoRA  {q,k,v}_lora_A, o_lora_B       : g_L zeroed     → g_final = g_F
  other                                        : follower only  → g_final = g_F
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# Kinds that own disjoint row/col slices of the design weight matrix.
_LEADER_Q_ROW_KINDS  = {"q_lora_B", "q_weight"}
_LEADER_KV_ROW_KINDS = {"k_lora_B", "v_lora_B", "k_weight", "v_weight"}
_LEADER_COL_KINDS    = {"o_lora_A", "o_weight"}
_ADDITIVE_KINDS = _LEADER_Q_ROW_KINDS | _LEADER_KV_ROW_KINDS | _LEADER_COL_KINDS
# Shared LoRA params: g_L is zeroed — update comes from g_F only.
_SHARED_KINDS = {"q_lora_A", "k_lora_A", "v_lora_A", "o_lora_B"}


# ---------------------------------------------------------------------------
# Param collection & grad assembly descriptor
# ---------------------------------------------------------------------------


@dataclass
class ParamRole:
    """How the gradient of a single param should be assembled."""

    param: torch.nn.Parameter
    name: str
    kind: str


@dataclass
class GradAssembly:
    """
    Built once before training, reused every step.

    leader_q  : list of slices (rows in q_proj weight/lora_B), one per leader Q-head.
    leader_kv : list of slices (rows in k_proj / v_proj), one per leader KV-head.
    leader_o  : list of slices (cols in o_proj weight/lora_A), one per leader Q-head.
    """

    roles: List[ParamRole]
    leader_q: List[slice]
    leader_kv: List[slice]
    leader_o: List[slice]


def _classify_qwen_param(name: str, design_layer_prefixes: set, is_lora: bool) -> str:
    """Renvoie le kind ('q_lora_B', 'q_weight', 'other', etc.)."""
    is_design = any(prefix in name for prefix in design_layer_prefixes)
    if not is_design:
        return "other"

    # Détection du module (qkv/o sont sous self_attn)
    if "self_attn.q_proj" in name:
        proj = "q"
    elif "self_attn.k_proj" in name:
        proj = "k"
    elif "self_attn.v_proj" in name:
        proj = "v"
    elif "self_attn.o_proj" in name:
        proj = "o"
    else:
        return "other"

    if is_lora:
        if "lora_A" in name:
            return f"{proj}_lora_A"
        elif "lora_B" in name:
            return f"{proj}_lora_B"
        else:
            return "other"
    else:
        # Full FT : on cherche le poids brut .weight
        if name.endswith(".weight"):
            return f"{proj}_weight"
        else:
            return "other"


def collect_lora_params(
    model,
    design_layers: List[int] = None,
    leader_q_heads: List[int] = None,
    leader_kv_heads: List[int] = None,
    n_q_heads: int = 14,
    n_kv_heads: int = 2,
    d_head: int = 64,
    hidden_size: int = 896,
) -> Tuple[List[torch.nn.Parameter], "GradAssembly"]:
    """
    Returns (all_trainable_params, grad_assembly).

    design_layers   : list of layer indices treated as Stackelberg design layers (default [19]).
    leader_q_heads  : Q-head indices acting as leaders (default list(range(7))).
    leader_kv_heads : KV-head indices acting as leaders (default [0]).
    """
    if design_layers is None:
        design_layers = [19]
    if leader_q_heads is None:
        leader_q_heads = list(range(7))
    if leader_kv_heads is None:
        leader_kv_heads = [0]

    # Sanity : chaque leader Q-head doit pointer vers une KV-head leader (groupe contigu)
    group_size = n_q_heads // n_kv_heads
    inferred_kv = {h // group_size for h in leader_q_heads}
    if not inferred_kv.issubset(set(leader_kv_heads)):
        raise ValueError(
            f"leader_q_heads={leader_q_heads} → groupes KV {sorted(inferred_kv)}, "
            f"mais leader_kv_heads={leader_kv_heads}. "
            "Les Q-heads leader doivent appartenir à des groupes GQA leader."
        )

    # One slice per leader Q-head (rows in q_proj.lora_B / cols in o_proj.lora_A)
    leader_q = [slice(h * d_head, (h + 1) * d_head) for h in leader_q_heads]
    # One slice per leader KV-head (rows in k_proj / v_proj lora_B)
    leader_kv = [slice(k * d_head, (k + 1) * d_head) for k in leader_kv_heads]
    # o_proj input cols : Q-head h occupies cols [h*d_head, (h+1)*d_head)
    leader_o = [slice(h * d_head, (h + 1) * d_head) for h in leader_q_heads]

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

        kind = _classify_qwen_param(name, _layer_prefixes, is_lora=True)
        roles.append(ParamRole(param=p, name=name, kind=kind))

    assembly = GradAssembly(
        roles=roles,
        leader_q=leader_q,
        leader_kv=leader_kv,
        leader_o=leader_o,
    )
    return all_params, assembly


def collect_fullft_params(
    model,
    design_layers: List[int] = None,
    leader_q_heads: List[int] = None,
    leader_kv_heads: List[int] = None,
    n_q_heads: int = 14,
    n_kv_heads: int = 2,
    d_head: int = 64,
    hidden_size: int = 896,
) -> Tuple[List[torch.nn.Parameter], "GradAssembly"]:
    """Comme collect_lora_params mais sur les poids bruts (full fine-tuning)."""
    if design_layers is None:
        design_layers = [19]
    if leader_q_heads is None:
        leader_q_heads = list(range(7))
    if leader_kv_heads is None:
        leader_kv_heads = [0]

    group_size = n_q_heads // n_kv_heads
    inferred_kv = {h // group_size for h in leader_q_heads}
    if not inferred_kv.issubset(set(leader_kv_heads)):
        raise ValueError(
            f"leader_q_heads={leader_q_heads} → groupes KV {sorted(inferred_kv)}, "
            f"mais leader_kv_heads={leader_kv_heads}."
        )

    leader_q = [slice(h * d_head, (h + 1) * d_head) for h in leader_q_heads]
    leader_kv = [slice(k * d_head, (k + 1) * d_head) for k in leader_kv_heads]
    leader_o = [slice(h * d_head, (h + 1) * d_head) for h in leader_q_heads]

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

        kind = _classify_qwen_param(name, _layer_prefixes, is_lora=False)
        roles.append(ParamRole(param=p, name=name, kind=kind))

    assembly = GradAssembly(
        roles=roles,
        leader_q=leader_q,
        leader_kv=leader_kv,
        leader_o=leader_o,
    )
    return all_params, assembly


def mask_follower_grad(assembly: GradAssembly) -> None:
    """
    Appelée après follower_loss.backward().
    Zéro les tranches leader dans p.grad → g_follower = 0 sur toutes les régions leader.
    In-place sur p.grad.
    """
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind in _LEADER_Q_ROW_KINDS:
            for lq in assembly.leader_q:
                p.grad[lq, :] = 0
        elif role.kind in _LEADER_KV_ROW_KINDS:
            for lkv in assembly.leader_kv:
                p.grad[lkv, :] = 0
        elif role.kind in _LEADER_COL_KINDS:
            for lo in assembly.leader_o:
                p.grad[:, lo] = 0


def mask_leader_grad(assembly: GradAssembly) -> None:
    """
    Appelée après leader_loss.backward().
    Garde uniquement les tranches leader ; zéro tout le reste (y compris θ_S et other).
    In-place sur p.grad.
    """
    for role in assembly.roles:
        p = role.param
        if p.grad is None:
            continue
        if role.kind in _LEADER_Q_ROW_KINDS:
            mask = torch.zeros_like(p.grad)
            for lq in assembly.leader_q:
                mask[lq, :] = 1
            p.grad.mul_(mask)
        elif role.kind in _LEADER_KV_ROW_KINDS:
            mask = torch.zeros_like(p.grad)
            for lkv in assembly.leader_kv:
                mask[lkv, :] = 1
            p.grad.mul_(mask)
        elif role.kind in _LEADER_COL_KINDS:
            mask = torch.zeros_like(p.grad)
            for lo in assembly.leader_o:
                mask[:, lo] = 1
            p.grad.mul_(mask)
        elif role.kind in _SHARED_KINDS or role.kind == "other":
            p.grad.zero_()


def assemble_gradients(
    assembly: GradAssembly,
    g_follower: Dict[int, torch.Tensor],
    g_leader: Dict[int, torch.Tensor],
) -> None:
    """
    Écrit le gradient assemblé final dans p.grad pour optimizer.step().

    g_follower, g_leader : {id(p): grad_tensor} — déjà masqués par
        mask_follower_grad / mask_leader_grad respectivement.

    Règles (LoRA et full-FT) :
      row/col-decomposable : tranches disjointes → g_F + g_L
      shared / other       : g_L = 0 après mask  → g_F
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

        if role.kind in _ADDITIVE_KINDS:
            p.grad = gf + gl
        else:
            p.grad = gf


# ---------------------------------------------------------------------------
# Hidden state capture
# ---------------------------------------------------------------------------


class HiddenStateCapture:
    """
    Forward hook sur une Qwen2DecoderLayer pour capturer hidden states (B, L, hidden_size).

    Register sur layer design_layer-1 pour obtenir l'input du design layer.
    Le tensor capturé reste dans le graphe autograd.
    """

    def __init__(self):
        self._hidden: Optional[torch.Tensor] = None
        self._hook = None

    def register(self, model, layer_idx: int):
        # Cherche le module Qwen layer ; gère PeftModel wrapper
        target = f"model.layers.{layer_idx}"
        module = None
        for name, mod in model.named_modules():
            if name == target or name.endswith("." + target):
                module = mod
                break
        if module is None:
            raise RuntimeError(
                f"Layer '{target}' not found. "
                "Check that layer_idx is valid for Qwen2.5-0.5B (0–23)."
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
