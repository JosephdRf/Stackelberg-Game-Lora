"""
GAME-LoRA losses — adapté pour Pythia-160M (GPT-NeoX)
"Multi-Head Attention is a Multi-Player Game" (Chakrabarti & Balachundar, 2026)

Différences vs qwen2.5_0.5B/game_lora/game_losses.py :
  - Module d'attention : model.base_model.model.gpt_neox.layers[l].attention
  - Projection output  : attention.dense  (vs self_attn.o_proj chez Qwen)
  - LM head            : embed_out        (vs lm_head chez Qwen)
  - num_heads / head_dim extraits depuis attn_module.config

Contient :
  - HeadInteractionMatrix  : calcul de G (Def 2.3) = ω_ij · ρ_ij
  - LogDetBarrierLoss      : L_LDB = -log det(G + εI)         (Eq 28)
  - AdaptiveBarlowTwinsLoss: L_ABT avec weighting adaptatif   (Eq 29)
  - NashMTL                : arbitrage multi-objectif          (Navon et al., 2022)
  - GAMELossScheduler      : schedule 3 phases (warmup/constant/cooldown)
  - HeadOutputCapture      : hook sur attention.dense (pré-W_O)
  - get_output_projection_weights : W_O par tête (LoRA-aware)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Head Interaction Matrix  (Definition 2.3)
# ---------------------------------------------------------------------------


class HeadInteractionMatrix:
    """
    Calcule G ∈ R^{H×H} avec G_ij = ω_ij · ρ_ij
      - ω_ij : cosine similarity des output projections W_O^(i), W_O^(j)   (Def 2.1)
      - ρ_ij : cosine similarity des gradients backpropagés g_i, g_j        (Def 2.2)

    Γ(G) = ||G - I||_F  (interaction strength)
    """

    @staticmethod
    def compute_weight_coupling(W_O: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W_O: (H, d, d_h) — output projection weights per head
        Returns:
            omega: (H, H) — weight coupling matrix
        """
        W_flat = W_O.reshape(W_O.shape[0], -1).float()
        norms = W_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        W_normed = W_flat / norms
        omega = W_normed @ W_normed.T  # (H, H)
        return omega

    @staticmethod
    def compute_gradient_coupling(
        model, logits: torch.Tensor, labels: torch.Tensor,
        W_O: torch.Tensor, head_dim: int
    ) -> torch.Tensor:
        """
        Calcule ρ_ij = cosine(g_i, g_j) où g_i = (W_O^(i))^T η
        η = ∇_ℓ L_CE est le gradient par rapport aux logits.

        Args:
            model: le modèle PEFT (Pythia-160M)
            logits: (B, T, V)
            labels: (B, T)
            W_O: (H, d, d_h)
            head_dim: d_h
        Returns:
            rho: (H, H)
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        eta = torch.autograd.grad(ce_loss, logits, retain_graph=True)[0]  # (B, T, V)
        eta_mean = eta.mean(dim=(0, 1))  # (V,)

        # Cherche la LM head — Pythia utilise embed_out (GPT-NeoX)
        lm_head_weight = None
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and any(k in name for k in ('embed_out', 'lm_head')):
                lm_head_weight = module.weight  # (V, d)
                break
        if lm_head_weight is None:
            raise RuntimeError("LM head (embed_out / lm_head) introuvable dans le modèle")

        # Projeter η de l'espace vocabulaire vers l'espace caché : (V,) @ (V, d) = (d,)
        eta_mean = eta_mean.float() @ lm_head_weight.float()  # (d,)

        # g_i = (W_O^(i))^T η ∈ R^{d_h},  W_O: (H, d, d_h)
        g = torch.einsum('hdi,d->hi', W_O.float(), eta_mean.float())  # (H, d_h)
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        g_normed = g / norms
        rho = g_normed @ g_normed.T  # (H, H)
        return rho

    @staticmethod
    def compute_G(omega: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """G_ij = ω_ij · ρ_ij  (Eq 6)"""
        return omega * rho

    @staticmethod
    def interaction_strength(G: torch.Tensor) -> torch.Tensor:
        """Γ(G) = ||G - I||_F  (Definition 2.3)"""
        H = G.shape[0]
        I = torch.eye(H, device=G.device, dtype=G.dtype)
        return (G - I).norm(p='fro')


# ---------------------------------------------------------------------------
# Log-Determinant Barrier Loss (Eq 28)
# ---------------------------------------------------------------------------


class LogDetBarrierLoss(nn.Module):
    """
    L_LDB = -log det(G + εI)

    Force G à avoir des valeurs propres éloignées de 0 → full rank → têtes diversifiées.
    """

    def __init__(self, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        H = G.shape[0]
        G_reg = G.float() + self.epsilon * torch.eye(H, device=G.device, dtype=torch.float32)
        eigenvalues = torch.linalg.eigvalsh(G_reg)
        eigenvalues = eigenvalues.clamp(min=self.epsilon)
        loss = -eigenvalues.log().sum()
        return loss


# ---------------------------------------------------------------------------
# Adaptive Barlow Twins Loss (Eq 29)
# ---------------------------------------------------------------------------


class AdaptiveBarlowTwinsLoss(nn.Module):
    """
    L_ABT = E_{i<j} [ w_ij ||Ĉ_ij - I||²_F ]

    Ĉ_ij = (1/N) Õ_i^T Õ_j   (cross-correlation of z-scored head outputs)
    w_ij = α + (1-α)·softplus(-β(G_ij - τ))   (adaptive weighting)
    """

    def __init__(
        self,
        alpha: float = 0.929,
        beta: float = 15.99,
        tau: float = 0.0,
        subtract_identity: bool = True,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.subtract_identity = subtract_identity

    def forward(
        self,
        head_outputs: torch.Tensor,
        G: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            head_outputs: (B, T, H, d_h)
            G: (H, H) — head interaction matrix (pour le weighting adaptatif)
        Returns:
            loss: scalar
        """
        B, T, H, d_h = head_outputs.shape
        N = B * T

        outputs = head_outputs.reshape(N, H, d_h)

        mu = outputs.mean(dim=0, keepdim=True)      # (1, H, d_h)
        sigma = outputs.std(dim=0, keepdim=True)     # (1, H, d_h)
        O_tilde = (outputs - mu) / (sigma + 1e-8)    # (N, H, d_h)

        loss = torch.tensor(0.0, device=head_outputs.device, dtype=head_outputs.dtype)
        n_pairs = 0

        for i in range(H):
            for j in range(i + 1, H):
                C_ij = (O_tilde[:, i, :].T @ O_tilde[:, j, :]) / N  # (d_h, d_h)

                if self.subtract_identity:
                    I_dh = torch.eye(d_h, device=C_ij.device, dtype=C_ij.dtype)
                    diff = C_ij - I_dh
                else:
                    diff = C_ij

                pair_loss = (diff ** 2).sum()

                if G is not None:
                    g_ij = G[i, j]
                    w_ij = self.alpha + (1 - self.alpha) * F.softplus(
                        -self.beta * (g_ij - self.tau)
                    )
                else:
                    w_ij = 1.0

                loss = loss + w_ij * pair_loss
                n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss


# ---------------------------------------------------------------------------
# EMA Loss Normalization (Appendix A)
# ---------------------------------------------------------------------------


class EMALossNormalizer:
    """
    L_norm = L · (target / ema)   — stabilise l'entraînement (Appendix A, Eq 34).
    """

    def __init__(self, target: float = 20.0, alpha: float = 0.1):
        self.target = target
        self.alpha = alpha
        self.ema = target  # ema_0 = target

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.ema = self.alpha * loss.item() + (1 - self.alpha) * self.ema
        if self.ema > 0:
            return loss * (self.target / self.ema)
        return loss


# ---------------------------------------------------------------------------
# Nash-MTL (Navon et al., 2022) — arbitrage multi-objectif
# ---------------------------------------------------------------------------


class NashMTL:
    """
    Nash Multi-Task Learning : arbitrage des gradients entre L_CE, L_LDB, L_ABT.
    """

    def __init__(self, n_tasks: int = 3, max_iter: int = 20, lr: float = 0.1):
        self.n_tasks = n_tasks
        self.max_iter = max_iter
        self.lr = lr

    def get_weights(self, grads: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            grads: liste de n_tasks vecteurs 1D detachés (un gradient par loss)
        Returns:
            weights: (n_tasks,) — poids optimaux (solution Nash bargaining)
        """
        n = len(grads)
        device = grads[0].device

        G_mat = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(i, n):
                dot = (grads[i] * grads[j]).sum()
                G_mat[i, j] = dot
                G_mat[j, i] = dot

        log_w = torch.zeros(n, device=device)
        for _ in range(self.max_iter):
            w = F.softmax(log_w, dim=0)
            utilities = (G_mat @ w).clamp(min=1e-12)
            inv_u = 1.0 / utilities
            grad_w = G_mat.T @ inv_u
            grad_log_w = w * (grad_w - (w * grad_w).sum())
            log_w = log_w + self.lr * grad_log_w

        return F.softmax(log_w, dim=0)


# ---------------------------------------------------------------------------
# GAME Loss Scheduler — schedule 3 phases (Appendix A)
# ---------------------------------------------------------------------------


class GAMELossScheduler:
    """
    Schedule des λ en 3 phases :
      1. Linear warmup : 0 → 2%
      2. Constant      : 2% → 95%  à λ_ABT=0.179, λ_LDB=0.352
      3. Cooldown      : 95% → 100% avec λ → 0
    """

    def __init__(
        self,
        total_steps: int,
        lambda_abt: float = 0.179,
        lambda_ldb: float = 0.352,
        warmup_frac: float = 0.02,
        cooldown_start_frac: float = 0.95,
    ):
        self.total_steps = total_steps
        self.lambda_abt = lambda_abt
        self.lambda_ldb = lambda_ldb
        self.warmup_end = int(warmup_frac * total_steps)
        self.cooldown_start = int(cooldown_start_frac * total_steps)

    def get_lambdas(self, step: int) -> Tuple[float, float]:
        if step < self.warmup_end:
            frac = step / max(self.warmup_end, 1)
            return self.lambda_abt * frac, self.lambda_ldb * frac
        elif step < self.cooldown_start:
            return self.lambda_abt, self.lambda_ldb
        else:
            remaining = self.total_steps - self.cooldown_start
            if remaining <= 0:
                return 0.0, 0.0
            frac = max(1.0 - (step - self.cooldown_start) / remaining, 0.0)
            return self.lambda_abt * frac, self.lambda_ldb * frac


# ---------------------------------------------------------------------------
# Hook pour capturer les head outputs avant attention.dense  (Pythia/GPT-NeoX)
# ---------------------------------------------------------------------------


class HeadOutputCapture:
    """
    Forward pre-hook sur attention.dense pour capturer les sorties par tête
    AVANT la projection W_O.

    Architecture GPT-NeoX (Pythia) :
      model.base_model.model.gpt_neox.layers[l].attention.dense
      Input shape : (B, T, num_heads * head_size)

    Usage :
        capture = HeadOutputCapture()
        capture.register(model, design_layer=9)
        out = model(input_ids)
        head_outputs = capture.get()  # (B, T, H, d_h)
        capture.remove()
    """

    def __init__(self):
        self.head_outputs: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, model, design_layer: int = 9):
        """
        Enregistre le hook sur attention.dense de la couche spécifiée.
        Compatible Pythia / GPT-NeoX avec PEFT.
        """
        attn_module = model.base_model.model.gpt_neox.layers[design_layer].attention
        cfg = attn_module.config
        self._num_heads = cfg.num_attention_heads
        self._head_dim = cfg.hidden_size // cfg.num_attention_heads
        # Pré-hook sur dense : input[0] = (B, T, H*d_h) avant mixage
        self._handle = attn_module.dense.register_forward_pre_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, input):
        x = input[0]  # (B, T, H*d_h)
        B, T, _ = x.shape
        self.head_outputs = x.view(B, T, self._num_heads, self._head_dim)

    def get(self) -> Optional[torch.Tensor]:
        return self.head_outputs

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def get_output_projection_weights(model, design_layer: int = 9) -> torch.Tensor:
    """
    Extrait les poids W_O effectifs (base + LoRA delta) de la couche spécifiée,
    reshapés par tête. Différentiable par rapport aux paramètres LoRA.

    Architecture Pythia :
      model.base_model.model.gpt_neox.layers[l].attention.dense
      W_O shape : (hidden_size, hidden_size) = (768, 768)

    Returns:
        W_O: (H, d, d_h) — poids de projection output par tête
    """
    attn = model.base_model.model.gpt_neox.layers[design_layer].attention
    dense = attn.dense  # module LoRA wrappé

    W_O_full = dense.base_layer.weight + dense.get_delta_weight("default")
    # W_O_full : (hidden_size, hidden_size) = (768, 768)

    cfg = attn.config
    num_heads = cfg.num_attention_heads   # 12
    head_dim = cfg.hidden_size // num_heads  # 64
    d = W_O_full.shape[0]  # 768

    # Reshape : (d, H, d_h) → permute → (H, d, d_h)
    W_O_heads = W_O_full.view(d, num_heads, head_dim)  # (768, 12, 64)
    W_O_heads = W_O_heads.permute(1, 0, 2)              # (12, 768, 64)

    return W_O_heads
