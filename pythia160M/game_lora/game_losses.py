"""
GAME-LoRA losses — adapté pour Pythia-160M (GPT-NeoX)
"Multi-Head Attention is a Multi-Player Game" (Chakrabarti & Balachundar, 2026)

Contient :
  - LogDetBarrierLoss      : L_LDB = -log det(G + εI)         (Eq 28)
  - AdaptiveBarlowTwinsLoss: L_ABT avec weighting adaptatif   (Eq 29)
  - NashMTL                : arbitrage multi-objectif          (Navon et al., 2022)
  - GAMELossScheduler      : schedule 3 phases (warmup/constant/cooldown)

HeadInteractionMatrix, HeadOutputCapture et get_output_projection_weights
sont dans train_utils (communs à toutes les runs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from train_utils import HeadInteractionMatrix, HeadOutputCapture, get_output_projection_weights


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
        # G shape : (H, H)
        assert G.shape[0] == G.shape[1], "G doit être carré"
        
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
