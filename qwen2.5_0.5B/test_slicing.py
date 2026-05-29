"""
Smoke test : valide les slices leader/follower pour Qwen2.5-0.5B + GQA.

À lancer sur un nœud de calcul (besoin du modèle base Qwen + PEFT) :
  python qwen2.5_0.5B/test_slicing.py
"""
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from gradient_mask import (
    collect_lora_params,
    mask_follower_grad,
    mask_leader_grad,
    assemble_gradients,
)

print("[test] Chargement Qwen2.5-0.5B + LoRA r=32 ...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
print(f"  config.num_hidden_layers     = {base.config.num_hidden_layers}")
print(f"  config.num_attention_heads   = {base.config.num_attention_heads}")
print(f"  config.num_key_value_heads   = {base.config.num_key_value_heads}")
print(f"  config.hidden_size           = {base.config.hidden_size}")
print(f"  num_key_value_groups (Q/KV)  = "
      f"{base.config.num_attention_heads // base.config.num_key_value_heads}")

assert base.config.num_hidden_layers == 24
assert base.config.num_attention_heads == 14
assert base.config.num_key_value_heads == 2

model = get_peft_model(base, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
))

print("[test] Collect leader/follower params @ layer 19 ...")
params, asm = collect_lora_params(
    model, design_layers=[19],
    leader_q_heads=list(range(7)),
    leader_kv_heads=[0],
)
print(f"  ParamRoles total           : {len(asm.roles)}")
print(f"  Trainable params           : {sum(p.numel() for p in params):,}")

kinds_layer19 = {}
for r in asm.roles:
    if "layers.19." in r.name and r.kind != "other":
        kinds_layer19[r.kind] = (r.param.shape, r.name)

print("\n[test] Layer 19 LoRA params:")
for kind, (shape, name) in sorted(kinds_layer19.items()):
    print(f"  {kind:<12} : shape {tuple(shape)}  ← {name}")

# Assertions de forme
expected = {
    "q_lora_B": (896, 32),
    "k_lora_B": (128, 32),
    "v_lora_B": (128, 32),
    "o_lora_A": (32, 896),
    "q_lora_A": (32, 896),
    "k_lora_A": (32, 896),
    "v_lora_A": (32, 896),
    "o_lora_B": (896, 32),
}
ok = True
for kind, exp_shape in expected.items():
    if kind not in kinds_layer19:
        print(f"  ✗ MANQUE : {kind}")
        ok = False
        continue
    got = tuple(kinds_layer19[kind][0])
    sym = "✓" if got == exp_shape else "✗"
    print(f"  {sym} {kind:<12} attendu {exp_shape} → got {got}")
    if got != exp_shape:
        ok = False

if not ok:
    raise AssertionError("Shapes inattendues. Vérifie PEFT/transformers versions.")

# Test mask_follower_grad
print("\n[test] mask_follower_grad : leader slices zeroed, follower untouched ...")
for p in params:
    p.grad = torch.ones_like(p)
mask_follower_grad(asm)

for r in asm.roles:
    if "layers.19." not in r.name or r.kind == "other":
        continue
    g = r.param.grad
    if r.kind == "q_lora_B":   # rows [0:448] zero, [448:896] one
        assert (g[0:448, :] == 0).all(),  f"q_lora_B leader rows not zeroed"
        assert (g[448:896, :] == 1).all(), f"q_lora_B follower rows clobbered"
    elif r.kind == "k_lora_B":
        assert (g[0:64, :] == 0).all()
        assert (g[64:128, :] == 1).all()
    elif r.kind == "v_lora_B":
        assert (g[0:64, :] == 0).all()
        assert (g[64:128, :] == 1).all()
    elif r.kind == "o_lora_A":
        assert (g[:, 0:448] == 0).all()
        assert (g[:, 448:896] == 1).all()
    elif r.kind in ("q_lora_A", "k_lora_A", "v_lora_A", "o_lora_B"):
        assert (g == 1).all(), f"{r.kind} : follower-only must be untouched by mask_follower_grad"
print("  ✓ mask_follower_grad : tranches leader = 0, follower intact")

# Test mask_leader_grad
print("[test] mask_leader_grad : keep only leader, zero everything else ...")
for p in params:
    p.grad = torch.ones_like(p)
mask_leader_grad(asm)

for r in asm.roles:
    if "layers.19." not in r.name:
        if r.kind == "other":
            # other (autres layers) doit être tout-zéro après mask_leader
            if r.param.grad is not None:
                assert (r.param.grad == 0).all(), f"other (layer != 19) {r.name} not zeroed"
        continue
    if r.kind == "other":
        continue
    g = r.param.grad
    if r.kind == "q_lora_B":
        assert (g[0:448, :] == 1).all()
        assert (g[448:896, :] == 0).all()
    elif r.kind == "k_lora_B":
        assert (g[0:64, :] == 1).all()
        assert (g[64:128, :] == 0).all()
    elif r.kind == "v_lora_B":
        assert (g[0:64, :] == 1).all()
        assert (g[64:128, :] == 0).all()
    elif r.kind == "o_lora_A":
        assert (g[:, 0:448] == 1).all()
        assert (g[:, 448:896] == 0).all()
    elif r.kind in ("q_lora_A", "k_lora_A", "v_lora_A", "o_lora_B"):
        assert (g == 0).all(), f"{r.kind} : shared param must be zeroed by mask_leader_grad"
print("  ✓ mask_leader_grad : tranches leader gardées, tout le reste = 0")

# Test GQA sanity : group_size = 7, Q-head 0..6 → KV-head 0
n_q = base.config.num_attention_heads
n_kv = base.config.num_key_value_heads
group_size = n_q // n_kv
print(f"\n[test] GQA sanity : group_size = {group_size}")
for h in range(n_q):
    kv = h // group_size
    print(f"  Q-head {h:>2} ↔ KV-head {kv}")

print("\n[test] ✓ Tous les tests passent. Slicing OK pour Qwen2.5-0.5B + GQA.")
