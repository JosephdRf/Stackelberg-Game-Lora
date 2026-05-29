"""
Leader→Follower gating au design layer (Pythia-160M / GPT-NeoX).

Idée
----
La sortie d'attention est o = Σ_h W_O^(h) z_h (somme des contributions par tête,
poids fixe 1), où W_O = attention.dense. On rend les contributions des têtes
FOLLOWER pondérées par un signal calculé depuis les têtes LEADER :

    s = concat({z_l}_{l∈leader})   ∈ R^{|L|·d_head}
    g = 2·σ(MLP(s))                ∈ (0,2)^{|F|}   (par token)
    o = Σ_{l∈L} W_O^(l) z_l  +  Σ_{i∈F} g_i · W_O^(i) z_i

Implémentation : forward_pre_hook sur attention.dense. L'entrée de dense est
exactement concat(z_h) ∈ (B,L,n_heads·d_head). Architecture GPT-NeoX :
  model.gpt_neox.layers[dl].attention.dense

Init : dernière couche du MLP à zéro → g ≡ 1 → couche identique au modèle
pré-entraîné à l'init (entraînement stable).

Le gate n'est PAS un adaptateur PEFT : sauvegardé/rechargé séparément
(save_gate / load_gate) et ré-enregistré à l'éval.
"""

import os
import torch
import torch.nn as nn


class LeaderFollowerGate(nn.Module):
    """MLP concat(z_leader) → hidden → |F| gates ∈ (0,2) ; hook sur attention.dense."""

    def __init__(self, leader_heads, follower_heads, d_head: int,
                 n_heads: int, hidden: int = 128):
        super().__init__()
        self.leader_heads = list(leader_heads)
        self.follower_heads = list(follower_heads)
        self.d_head = d_head
        self.n_heads = n_heads
        self.hidden = hidden

        self.mlp = nn.Sequential(
            nn.Linear(len(self.leader_heads) * d_head, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(self.follower_heads)),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)   # logit 0 → 2·σ(0)=1 → gates ≡ 1

        self._handle = None
        self.last_g = None   # dernier g calculé (detaché), pour le logging des stats
        self.register_buffer("_lh", torch.tensor(self.leader_heads, dtype=torch.long), persistent=False)
        self.register_buffer("_fh", torch.tensor(self.follower_heads, dtype=torch.long), persistent=False)

    def _pre_hook(self, module, args):
        x = args[0]                              # (B, L, H·d_head)
        B, L, D = x.shape
        xh = x.view(B, L, self.n_heads, self.d_head)
        lh = self._lh.to(x.device)
        fh = self._fh.to(x.device)
        s = xh.index_select(2, lh).reshape(B, L, -1)
        g = 2.0 * torch.sigmoid(self.mlp(s))     # (B, L, |F|)
        self.last_g = g.detach()                 # pour le logging
        gate = x.new_ones(B, L, self.n_heads, 1)
        gate = gate.index_copy(2, fh, g.unsqueeze(-1))
        xh = xh * gate
        return (xh.reshape(B, L, D),) + tuple(args[1:])

    def register(self, model, design_layer: int):
        dense = _find_dense(model, design_layer)
        self._handle = dense.register_forward_pre_hook(self._pre_hook)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def config(self):
        return {
            "leader_heads": self.leader_heads,
            "follower_heads": self.follower_heads,
            "d_head": self.d_head,
            "n_heads": self.n_heads,
            "hidden": self.hidden,
        }


def gate_stats(gate):
    """
    Stats du dernier g calculé (gate.last_g, shape (B, L, |F|)) pour le logging.

    Returns dict :
      mean_dev   : mean|g-1|         — activité du gating (0 = inerte = identité)
      token_std  : std de g sur les tokens, moyenné — conditionnement contextuel
      saturation : frac(g>1.9 ou g<0.1) — proportion de gates aux bornes
      per_head   : (|F|,) tensor      — gate moyen par tête follower
    """
    g = gate.last_g
    if g is None:
        return None
    mean_dev = (g - 1.0).abs().mean().item()
    token_std = g.std(dim=1).mean().item()
    saturation = ((g > 1.9) | (g < 0.1)).float().mean().item()
    per_head = g.mean(dim=(0, 1))
    return {"mean_dev": mean_dev, "token_std": token_std,
            "saturation": saturation, "per_head": per_head}


def gate_grad_norm(gate):
    """Norme L2 du gradient du MLP de gating (confirme qu'il apprend)."""
    gs = [p.grad.norm() for p in gate.parameters() if p.grad is not None]
    if not gs:
        return 0.0
    return torch.norm(torch.stack(gs)).item()


def _find_dense(model, design_layer: int):
    """Localise attention.dense du design layer (PEFT-wrappé ou nu)."""
    target = f"layers.{design_layer}.attention.dense"
    for name, mod in model.named_modules():
        if name.endswith(target):
            return mod
    raise RuntimeError(f"attention.dense introuvable pour le layer {design_layer}")


def save_gate(gate: LeaderFollowerGate, out_dir: str, design_layer: int):
    os.makedirs(out_dir, exist_ok=True)
    payload = {"state_dict": gate.state_dict(),
               "config": gate.config(),
               "design_layer": design_layer}
    torch.save(payload, os.path.join(out_dir, "gate.pt"))


def load_gate(model, ckpt_dir: str, device):
    """Si ckpt_dir/gate.pt existe : instancie, charge, enregistre le hook, renvoie le gate. Sinon None."""
    path = os.path.join(ckpt_dir, "gate.pt")
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location=device)
    cfg = payload["config"]
    gate = LeaderFollowerGate(
        leader_heads=cfg["leader_heads"], follower_heads=cfg["follower_heads"],
        d_head=cfg["d_head"], n_heads=cfg["n_heads"], hidden=cfg["hidden"],
    )
    gate.load_state_dict(payload["state_dict"])
    gate = gate.to(device)
    gate.eval()
    gate.register(model, payload["design_layer"])
    return gate
