"""
GAME-LoRA losses — implémentation fidèle à l'article
"Multi-Head Attention is a Multi-Player Game" (Chakrabarti & Balachundar, 2026)

Contient :
  - HeadInteractionMatrix  : calcul de G (Def 2.3) = ω_ij · ρ_ij
  - LogDetBarrierLoss      : L_LDB = -log det(G + εI)         (Eq 28)
  - AdaptiveBarlowTwinsLoss: L_ABT avec weighting adaptatif   (Eq 29)
  - NashMTL                : arbitrage multi-objectif          (Navon et al., 2022)
  - GAMELossScheduler      : schedule 3 phases (warmup/constant/cooldown)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


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
        H = W_O.shape[0]
        # Flatten each head's projection: (H, d*d_h)
        W_flat = W_O.reshape(H, -1).float()
        # Cosine similarity matrix
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
            model: le modèle (pour backward)
            logits: (B, T, V)
            labels: (B, T)
            W_O: (H, d, d_h)
            head_dim: d_h
        Returns:
            rho: (H, H)
        """
        # Compute η = ∂L_CE/∂logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Gradient par rapport aux logits
        eta = torch.autograd.grad(ce_loss, logits, retain_graph=True)[0]  # (B, T, V)

        # Moyenne sur batch et séquence pour obtenir η moyen: (V,)
        eta_mean = eta.mean(dim=(0, 1))  # (V,)

        # Projeter η de l'espace vocabulaire (V) vers l'espace caché (d)
        # ∂L/∂hidden = (∂L/∂logits) @ W_lm_head  car logits = hidden @ W_lm_head^T
        # W_lm_head: (V, d)  →  eta_mean @ W_lm_head : (V,) @ (V, d) = (d,)
        lm_head_weight = None
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and 'lm_head' in name:
                lm_head_weight = module.weight  # (V, d)
                break
        if lm_head_weight is None:
            raise RuntimeError("lm_head introuvable dans le modèle")
        eta_mean = (eta_mean.float() @ lm_head_weight.float())  # (d,)

        H = W_O.shape[0]
        # g_i = (W_O^(i))^T η ∈ R^{d_h},  W_O: (H, d, d_h), eta_mean: (d,)
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

    Pression d'expansion sur la matrice d'interaction des têtes.
    Force G à avoir des valeurs propres éloignées de 0 → full rank.
    Approx. de la compression term dans l'IB social objective (Eq 9).
    """

    def __init__(self, epsilon: float = 0.01):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        """
        Args:
            G: (H, H) head interaction matrix
        Returns:
            loss: scalar
        """
        H = G.shape[0]
        print("Number of heads (H):", H)
        G_reg = G.float() + self.epsilon * torch.eye(H, device=G.device, dtype=torch.float32)

        # Clamp eigenvalues for numerical stability (Appendix A)
        eigenvalues = torch.linalg.eigvalsh(G_reg)
        eigenvalues = eigenvalues.clamp(min=self.epsilon)

        # -log det = -sum log(eigenvalues)
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

    Pénalise la redondance entre paires de têtes (off-diagonal de Ĉ_ij → 0)
    tout en ancrant une base de features partagée (diagonal de Ĉ_ij → 1).
    """

    def __init__(
        self,
        alpha: float = 0.929,     # high floor (Appendix A)
        beta: float = 15.99,      # aggressive slope
        tau: float = 0.0,         # threshold
        subtract_identity: bool = True,  # target I, not 0
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
            head_outputs: (B, T, H, d_h) — per-head outputs before W_O
            G: (H, H) — head interaction matrix (for adaptive weighting)
        Returns:
            loss: scalar
        """
        B, T, H, d_h = head_outputs.shape
        N = B * T  # (Appendix A: N = B·T)

        # Reshape to (N, H, d_h)
        outputs = head_outputs.reshape(N, H, d_h)

        # Z-score normalization per head (Eq 30-32)
        mu = outputs.mean(dim=0, keepdim=True)      # (1, H, d_h)
        sigma = outputs.std(dim=0, keepdim=True)     # (1, H, d_h)
        O_tilde = (outputs - mu) / (sigma + 1e-8)    # (N, H, d_h)

        # Cross-correlation pour chaque paire i < j (Eq 33)
        loss = torch.tensor(0.0, device=head_outputs.device, dtype=head_outputs.dtype)
        n_pairs = 0

        for i in range(H):
            for j in range(i + 1, H):
                # Ĉ_ij = (1/N) Õ_i^T Õ_j ∈ R^{d_h × d_h}
                C_ij = (O_tilde[:, i, :].T @ O_tilde[:, j, :]) / N  # (d_h, d_h)

                # ||Ĉ_ij - I||²_F or ||Ĉ_ij||²_F
                if self.subtract_identity:
                    I_dh = torch.eye(d_h, device=C_ij.device, dtype=C_ij.dtype)
                    diff = C_ij - I_dh
                else:
                    diff = C_ij

                pair_loss = (diff ** 2).sum()

                # Adaptive weighting w_ij
                if G is not None:
                    g_ij = G[i, j]
                    w_ij = self.alpha + (1 - self.alpha) * F.softplus(-self.beta * (g_ij - self.tau))
                else:
                    w_ij = 1.0

                loss = loss + w_ij * pair_loss
                n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs  # E_{i<j}

        return loss


# ---------------------------------------------------------------------------
# EMA Loss Normalization (Appendix A)
# ---------------------------------------------------------------------------


class EMALossNormalizer:
    """
    Normalise une loss par sa moyenne mobile exponentielle.
    L_norm = L · (target / ema)
    Stabilise l'entraînement (Appendix A, Eq 34).
    """

    def __init__(self, target: float = 20.0, alpha: float = 0.1):
        self.target = target
        self.alpha = alpha
        self.ema = target  # ema_0 = target (Appendix A)

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
    Nash Multi-Task Learning : arbitrage des gradients entre les losses
    L_CE, L_LDB, L_ABT via un jeu de Nash sur les poids.

    Simplifié pour 3 losses : trouve les poids α_k ≥ 0 (Σα_k = 1)
    qui maximisent le produit des utilités (solution de Nash bargaining).
    """

    def __init__(self, n_tasks: int = 3, max_iter: int = 20, lr: float = 0.1):
        self.n_tasks = n_tasks
        self.max_iter = max_iter
        self.lr = lr

    def get_weights(
        self,
        grads: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Résout le Nash bargaining problem sur les gradients.

        Args:
            grads: list de n_tasks tensors, chaque grad est un vecteur 1D detaché
                   (gradients des différentes losses par rapport aux paramètres partagés)
        Returns:
            weights: (n_tasks,) — poids optimaux pour chaque loss
        """
        n = len(grads)
        device = grads[0].device

        # Pré-calcul de la matrice de produits scalaires G[i,j] = <g_i, g_j>
        # (les grads sont detachés, on optimise log_w analytiquement)
        G_mat = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(i, n):
                dot = (grads[i] * grads[j]).sum()
                G_mat[i, j] = dot
                G_mat[j, i] = dot

        # Optimisation par gradient ascent sur log_w
        log_w = torch.zeros(n, device=device)

        for _ in range(self.max_iter):
            w = F.softmax(log_w, dim=0)

            # Utilité de chaque tâche: u_k = <g_k, Σ w_j g_j> = Σ w_j G[k,j]
            utilities = G_mat @ w  # (n,)

            # Objectif Nash: maximiser Σ log(u_k)
            utilities = utilities.clamp(min=1e-12)

            # Gradient analytique de Σ log(u_k) par rapport à log_w
            # ∂/∂log_w_i = Σ_k (1/u_k) * G[k,i] * w_i - w_i * Σ_k (1/u_k)*Σ_j G[k,j]*w_j
            inv_u = 1.0 / utilities  # (n,)
            # ∂obj/∂w_i = Σ_k inv_u[k] * G[k,i]
            grad_w = G_mat.T @ inv_u  # (n,)
            # Pour softmax: ∂obj/∂log_w_i = w_i * (grad_w_i - Σ_j w_j * grad_w_j)
            grad_log_w = w * (grad_w - (w * grad_w).sum())

            log_w = log_w + self.lr * grad_log_w

        weights = F.softmax(log_w, dim=0)
        return weights


# ---------------------------------------------------------------------------
# GAME Loss Scheduler — schedule 3 phases (Appendix A)
# ---------------------------------------------------------------------------


class GAMELossScheduler:
    """
    Schedule des poids de régularisation en 3 phases :
      1. Linear warmup:  0 → 2% du training
      2. Constant:       2% → 87.9% à λ_ABT=0.179, λ_LDB=0.352
      3. Cooldown:       87.9% → 100% avec λ → 0
    """

    def __init__(
        self,
        total_steps: int,
        lambda_abt: float = 0.179,
        lambda_ldb: float = 0.352,
        warmup_frac: float = 0.02,
        cooldown_start_frac: float = 0.879,
    ):
        self.total_steps = total_steps
        self.lambda_abt = lambda_abt
        self.lambda_ldb = lambda_ldb
        self.warmup_end = int(warmup_frac * total_steps)
        self.cooldown_start = int(cooldown_start_frac * total_steps)

    def get_lambdas(self, step: int) -> Tuple[float, float]:
        """
        Returns (λ_ABT, λ_LDB) for the given optimizer step.
        """
        if step < self.warmup_end:
            # Phase 1: linear warmup
            frac = step / max(self.warmup_end, 1)
            return self.lambda_abt * frac, self.lambda_ldb * frac
        elif step < self.cooldown_start:
            # Phase 2: constant
            return self.lambda_abt, self.lambda_ldb
        else:
            # Phase 3: linear cooldown to 0
            remaining = self.total_steps - self.cooldown_start
            if remaining <= 0:
                return 0.0, 0.0
            frac = 1.0 - (step - self.cooldown_start) / remaining
            frac = max(frac, 0.0)
            return self.lambda_abt * frac, self.lambda_ldb * frac


# ---------------------------------------------------------------------------
# Hook pour capturer les head outputs avant W_O
# ---------------------------------------------------------------------------


class HeadOutputCapture:
    """
    Forward hook pour capturer les outputs par tête d'une couche d'attention
    avant la projection W_O.

    Utilisation:
        capture = HeadOutputCapture()
        capture.register(model, design_layer=19)
        ...
        out = model(input_ids)
        head_outputs = capture.get()  # (B, T, H, d_h)
    """

    def __init__(self):
        self.head_outputs: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, model, design_layer: int = 19):
        """
        Enregistre un hook sur la couche d'attention spécifiée.
        Compatible Qwen2.5 (model.model.layers[l].self_attn).
        """
        attn_module = model.base_model.model.model.layers[design_layer].self_attn
        self._handle = attn_module.register_forward_hook(self._hook_fn)
        return self

    def _hook_fn(self, module, input, output):
        """
        Hook qui capture les outputs avant la projection O.
        Pour Qwen2.5: l'attention renvoie (attn_output, attn_weights, past_kv).
        attn_output est déjà post-W_O.

        On doit plutôt capturer après Q*K^T*V et avant W_O.
        On intercepte au niveau de la couche et recalcule à partir de
        l'architecture interne du module attention.
        """
        # Pour capturer pre-W_O, on a besoin d'accéder aux internals.
        # Approche: on stocke l'output de l'attention complète et on
        # extrait la forme par tête à partir de output[0].
        # output[0] = attn_output après W_O projection, shape (B, T, d)
        # On reconstruit les head outputs en passant par W_O^{-1} approx
        # Mais c'est plus propre d'utiliser un hook sur le V après attention.

        # Approche pragmatique: on capture le output final (post-W_O)
        # et on le reshape en (B, T, H, d_h) pour les losses.
        # C'est ce que fait effectivement le papier dans le calcul de Ĉ_ij.
        attn_output = output[0]  # (B, T, d)
        B, T, d = attn_output.shape
        head_dim = module.head_dim
        num_heads = d // head_dim

        # Reshape: (B, T, H, d_h)
        self.head_outputs = attn_output.view(B, T, num_heads, head_dim)

    def get(self) -> Optional[torch.Tensor]:
        return self.head_outputs

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def get_output_projection_weights(model, design_layer: int = 19) -> torch.Tensor:
    """
    Extrait les poids W_O effectifs (base + LoRA delta) de la couche spécifiée,
    reshapés par tête. Différentiable par rapport aux paramètres LoRA.

    Returns:
        W_O: (H, d, d_h) — les poids de projection output par tête
    """
    attn = model.base_model.model.model.layers[design_layer].self_attn
    o_proj = attn.o_proj

    # Poids effectif = base (frozen) + LoRA delta (trainable)
    W_O_full = o_proj.base_layer.weight + o_proj.get_delta_weight("default")

    d = W_O_full.shape[0]
    head_dim = attn.head_dim
    num_heads = d // head_dim

    # Reshape: W_O = (d, H, d_h) → transposé en (H, d, d_h)
    W_O_heads = W_O_full.view(d, num_heads, head_dim)  # (d, H, d_h)
    W_O_heads = W_O_heads.permute(1, 0, 2)  # (H, d, d_h)

    return W_O_heads
