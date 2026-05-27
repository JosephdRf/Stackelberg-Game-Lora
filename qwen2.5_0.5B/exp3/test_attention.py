"""
Smoke test : valide la reconstruction des cartes d'attention Qwen2
(get_attention_maps) contre la sortie native eager (output_attentions=True).

À lancer (CPU suffit, modèle ~1GB) :
  python qwen2.5_0.5B/exp3/test_attention.py
"""
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.dirname(_HERE)
sys.path.insert(0, _MODEL)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from gradient_mask import HiddenStateCapture
from stackelberg_losses import get_attention_maps, get_attention_outputs

DL = 19           # design layer testé
N_Q, N_KV, DH = 14, 2, 64

print("[test] Chargement Qwen2.5-0.5B (eager) + LoRA r=32 ...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32,
    attn_implementation="eager", trust_remote_code=True,
)
model = get_peft_model(base, LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
))
model.eval()

# Résolution des modules (comme dans train_exp3._resolve_attn_ctx)
rotary_emb = next(m for n, m in model.named_modules()
                  if n.endswith("model.rotary_emb") or n == "rotary_emb")
q_mod = next(m for n, m in model.named_modules() if n.endswith(f"layers.{DL}.self_attn.q_proj"))
k_mod = next(m for n, m in model.named_modules() if n.endswith(f"layers.{DL}.self_attn.k_proj"))
v_mod = next(m for n, m in model.named_modules() if n.endswith(f"layers.{DL}.self_attn.v_proj"))
ln_mod = next(m for n, m in model.named_modules() if n.endswith(f"layers.{DL}.input_layernorm"))

cap = HiddenStateCapture()
cap.register(model, DL - 1)  # hidden = entrée du layer DL

# Forward avec attentions natives
torch.manual_seed(0)
input_ids = torch.randint(0, 1000, (1, 32))
with torch.no_grad():
    out = model(input_ids=input_ids, output_attentions=True)
hidden = cap.get()

# Reconstruction maison
with torch.no_grad():
    A = get_attention_maps(hidden, q_mod, k_mod, v_mod,
                           n_heads=N_Q, d_head=DH, rotary_emb=rotary_emb,
                           n_kv_heads=N_KV, input_layernorm=ln_mod)
    A2, Z = get_attention_outputs(hidden, q_mod, k_mod, v_mod,
                                  n_heads=N_Q, d_head=DH, rotary_emb=rotary_emb,
                                  n_kv_heads=N_KV, input_layernorm=ln_mod)

print(f"[test] A shape          = {tuple(A.shape)}  (attendu (1, {N_Q}, 32, 32))")
print(f"[test] Z shape          = {tuple(Z.shape)}  (attendu (1, 32, {N_Q}, {DH}))")
assert A.shape == (1, N_Q, 32, 32)
assert Z.shape == (1, 32, N_Q, DH)
assert torch.allclose(A, A2), "A et A2 diffèrent"

# softmax → lignes somment à 1
row_sums = A.sum(dim=-1)
assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), "lignes A != 1"
print("[test] ✓ softmax (lignes = 1)")

# masque causal : A[..., i, j] == 0 pour j > i
causal_ok = (A[0, 0].triu(diagonal=1).abs().max().item() < 1e-6)
assert causal_ok, "masque causal violé"
print("[test] ✓ masque causal")

# Comparaison avec attentions natives Qwen
native = out.attentions[DL]  # (1, N_Q, 32, 32)
print(f"[test] native attn shape= {tuple(native.shape)}")
diff = (A - native.float()).abs()
print(f"[test] max|A - native|  = {diff.max().item():.2e}   mean = {diff.mean().item():.2e}")
if diff.max().item() < 1e-3:
    print("[test] ✓ reconstruction cohérente avec l'attention native (tol 1e-3)")
else:
    print("[test] ⚠ écart > 1e-3 — vérifier RoPE / repeat_kv / bias")

cap.remove()
print("\n[test] Terminé.")
