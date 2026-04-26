"""
Module commun pour le LoRA fine-tuning de Pythia-160M sur WikiText-103.
Partagé entre le baseline LoRA et les expériences GAME-LoRA / Stackelberg.

Contient :
  - TrainConfig
  - WikiTextDataset (map-style, packing)
  - build_model_and_tokenizer (LoRA via PEFT)
  - setup_training (optimizer, scheduler, train/val dataloaders)
  - evaluate (loss CE + perplexité sur val)
  - seed_everything
  - log_config / add_common_args
  - HeadInteractionMatrix, HeadOutputCapture, get_output_projection_weights
  - log_head_matrices (omega, rho, G images + Γ(G) scalaire)
"""

import os
import math
import random
import logging
import itertools

_DATASETS_CACHE = os.path.join(
    os.environ.get("SCRATCH", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "datasets"
)
from dataclasses import dataclass, field
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------


def seed_everything(seed: int):
    """Couvre Python, NumPy, PyTorch CPU et CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Modèle
    model_name: str = "EleutherAI/pythia-160m"

    # Données — WikiText-103 (OOD léger pour Pythia)
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"
    seq_len: int = 1024
    total_tokens: int = 100_000_000  # ~1 epoch sur WikiText-103 (~103M tokens)

    # LoRA — même config que la version Qwen (Appendix A)
    lora_rank: int = 16
    lora_alpha: int = 32           # scaling = alpha/rank = 2
    lora_dropout: float = 0.1
    # Pythia/GPT-NeoX : QKV fusionné + projection de sortie
    lora_target_modules: list = field(
        default_factory=lambda: ["query_key_value", "dense"]
    )

    # Optimisation — LoRA fine-tuning
    lr: float = 3e-4               # LR standard LoRA (vs 3e-5 full FT)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    batch_size_per_gpu: int = 4
    grad_accum: int = 4           # batch effectif = 16
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)    # beta2=0.95 classique pour les LM

    # Évaluation
    eval_every: int = 100         # en steps optimizer
    eval_max_batches: int = 50

    # Logging / sauvegarde
    output_dir: str = "./checkpoints/baseline"
    log_every: int = 20
    save_every: int = 500
    wandb_project: Optional[str] = "Stackelberg-Pythia160M"
    wandb_group: Optional[str] = None
    run_name: Optional[str] = None
    seed: int = 42
    dry_run: bool = False
    random_init: bool = False
    num_workers: int = 8

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.grad_accum

    @property
    def total_steps(self) -> int:
        tokens_per_step = self.seq_len * self.effective_batch_size
        return math.ceil(self.total_tokens / tokens_per_step)

    @property
    def warmup_steps(self) -> int:
        return max(1, math.ceil(self.warmup_ratio * self.total_steps))


# ---------------------------------------------------------------------------
# Dataset : WikiText-103 map-style, packing fixe
# ---------------------------------------------------------------------------


class WikiTextDataset(Dataset):
    """
    Tokenise tout le split en une passe, concatène, et découpe en blocs
    de seq_len+1 tokens (packing). Map-style donc compatible num_workers > 0.
    """

    def __init__(self, tokenizer, seq_len: int, split: str,
                 dataset_name: str, dataset_config: str,
                 max_tokens: Optional[int] = None):
        super().__init__()
        from datasets import load_dataset

        ds = load_dataset(dataset_name, dataset_config, split=split, cache_dir=_DATASETS_CACHE)
        eos = tokenizer.eos_token_id
        all_ids: List[int] = []
        BATCH = 1000
        texts = [ex["text"] for ex in ds if ex["text"].strip()]
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            enc = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
            for ids in enc:
                all_ids.extend(ids)
                all_ids.append(eos)

        if max_tokens is not None and len(all_ids) > max_tokens:
            all_ids = all_ids[:max_tokens]

        block = seq_len + 1
        n_blocks = len(all_ids) // block
        all_ids = all_ids[: n_blocks * block]
        self.data = torch.tensor(all_ids, dtype=torch.long).view(n_blocks, block)
        self.seq_len = seq_len
        logger.info(
            f"  Dataset '{split}' : {len(all_ids):,} tokens → {n_blocks} blocs "
            f"de {seq_len+1} tokens"
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        block = self.data[idx]
        return {"input_ids": block.clone(), "labels": block.clone()}


# ---------------------------------------------------------------------------
# Construction du modèle (LoRA via PEFT)
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(cfg: TrainConfig):
    logger.info(f"Chargement de {cfg.model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.random_init:
        logger.info("Initialisation ALÉATOIRE des poids (pas de préentraînement)")
        config = AutoConfig.from_pretrained(cfg.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Setup de l'entraînement (optimizer, scheduler, dataloaders)
# ---------------------------------------------------------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_training(cfg: TrainConfig, model, tokenizer):
    """
    Returns: (train_loader, val_loader, optimizer, scheduler, total_steps)
    """
    max_train_tokens = (
        100 * cfg.seq_len * cfg.effective_batch_size if cfg.dry_run else None
    )

    train_ds = WikiTextDataset(
        tokenizer, cfg.seq_len, split="train",
        dataset_name=cfg.dataset_name, dataset_config=cfg.dataset_config,
        max_tokens=max_train_tokens,
    )
    val_ds = WikiTextDataset(
        tokenizer, cfg.seq_len, split="validation",
        dataset_name=cfg.dataset_name, dataset_config=cfg.dataset_config,
    )

    g = make_generator(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=g,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=(cfg.num_workers > 0),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=cfg.betas,
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    total_steps = 100 if cfg.dry_run else cfg.total_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    return train_loader, val_loader, optimizer, scheduler, total_steps


# ---------------------------------------------------------------------------
# Évaluation périodique (val loss + perplexité)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches: Optional[int] = None,
             autocast_dtype=torch.bfloat16):
    """
    Renvoie (val_loss, val_ppl) moyennés sur max_batches batches.
    """
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    use_autocast = device.type in ("cuda", "cpu")

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=autocast_dtype,
            enabled=use_autocast,
        ):
            out = model(input_ids=input_ids, labels=labels)

        total_loss += out.loss.item()
        n_batches  += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    if was_training:
        model.train()

    val_loss = total_loss / max(1, n_batches)
    val_ppl  = math.exp(min(val_loss, 20))
    return val_loss, val_ppl


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_config(cfg: TrainConfig):
    logger.info("=== Configuration (Pythia-160M, FULL fine-tuning) ===")
    logger.info(f"  Modèle           : {cfg.model_name}")
    logger.info(f"  Dataset          : {cfg.dataset_name} / {cfg.dataset_config}")
    logger.info(f"  Seq len          : {cfg.seq_len}")
    logger.info(f"  Tokens totaux    : {cfg.total_tokens:,}")
    logger.info(f"  Batch effectif   : {cfg.effective_batch_size}")
    logger.info(f"  Steps totaux     : {cfg.total_steps}")
    logger.info(f"  Warmup steps     : {cfg.warmup_steps}")
    logger.info(f"  LR               : {cfg.lr}")
    logger.info(f"  Weight decay     : {cfg.weight_decay}")
    logger.info(f"  Betas            : {cfg.betas}")
    logger.info(f"  Eval every       : {cfg.eval_every} steps "
                f"({cfg.eval_max_batches} batches)")
    logger.info(f"  Num workers      : {cfg.num_workers}")
    logger.info(f"  Seed             : {cfg.seed}")
    logger.info(f"  Dry run          : {cfg.dry_run}")


def add_common_args(parser):
    parser.add_argument("--model_name", default="EleutherAI/pythia-160m")
    parser.add_argument("--dataset_name", default="Salesforce/wikitext")
    parser.add_argument("--dataset_config", default="wikitext-103-raw-v1")
    parser.add_argument("--total_tokens", type=int, default=100_000_000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--wandb_project", default="Stackelberg-Pythia160M")
    parser.add_argument("--wandb_group",   default=None,
                        help="Groupe W&B (ex: 'baseline', 'game_lora', 'exp1')")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_max_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--random_init", action="store_true",
                        help="Initialiser les poids aléatoirement (pas de préentraînement)")
    parser.add_argument("--head_log_layer", type=int, default=9,
                        help="Couche d'attention observée pour les matrices omega/rho/G "
                             "(layer 9 ≈ 79%% du réseau pour Pythia-160M à 12 couches)")
    return parser


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
    def _eta_stats(logits: torch.Tensor, labels: torch.Tensor):
        """
        Accumule les statistiques du gradient analytique ∇_ℓ L_CE pour un batch.

        Retourne (probs_sum, label_counts, n_valid) — additifs entre batches,
        ce qui permet de construire un eta_mean stable sur N batches avant de
        calculer rho une seule fois.
        """
        shift_logits = logits[..., :-1, :].contiguous().detach().float()
        shift_labels = labels[..., 1:].contiguous()
        mask = (shift_labels != -100)
        V = shift_logits.shape[-1]

        probs_sum = torch.zeros(V, device=shift_logits.device)
        for b in range(shift_logits.shape[0]):
            p = torch.softmax(shift_logits[b], dim=-1)
            probs_sum += (p * mask[b].unsqueeze(-1)).sum(0)
            del p

        valid_labels = shift_labels[mask].clamp(min=0)
        label_counts = torch.bincount(valid_labels, minlength=V).float()
        n_valid = mask.sum().item()
        return probs_sum, label_counts, n_valid

    @staticmethod
    def _rho_from_eta_stats(model, probs_sum, label_counts, n_valid, W_O):
        """
        Calcule rho depuis des stats accumulées sur plusieurs batches.
        Sépare la projection W_O de l'accumulation, pour stabiliser rho.
        """
        eta_mean = (probs_sum - label_counts) / max(n_valid, 1)  # (V,)

        lm_head_weight = None
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and any(k in name for k in ('embed_out', 'lm_head')):
                lm_head_weight = module.weight
                break
        if lm_head_weight is None:
            raise RuntimeError("LM head (embed_out / lm_head) introuvable dans le modèle")

        eta_mean = eta_mean.float() @ lm_head_weight.float()  # (d,)
        g = torch.einsum('hdi,d->hi', W_O.float(), eta_mean.float())  # (H, d_h)
        norms = g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        g_normed = g / norms
        return g_normed @ g_normed.T  # (H, H)

    @staticmethod
    def compute_gradient_coupling(
        model, logits: torch.Tensor, labels: torch.Tensor,
        W_O: torch.Tensor, head_dim: int
    ) -> torch.Tensor:
        """
        Calcule ρ_ij = cosine(g_i, g_j) sur un seul batch.
        Préférer log_head_matrices (N batches accumulés) pour un rho stable.
        """
        ps, lc, nv = HeadInteractionMatrix._eta_stats(logits, labels)
        return HeadInteractionMatrix._rho_from_eta_stats(model, ps, lc, nv, W_O)

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
# Hook pour capturer les head outputs avant attention.dense  (Pythia/GPT-NeoX)
# ---------------------------------------------------------------------------


class HeadOutputCapture:
    """
    Forward pre-hook sur attention.dense pour capturer les sorties par tête
    AVANT la projection W_O.

    Architecture GPT-NeoX (Pythia) :
      model.gpt_neox.layers[l].attention.dense
      Input shape : (B, T, num_heads * head_size)
    """

    def __init__(self):
        self.head_outputs: Optional[torch.Tensor] = None
        self._handle = None

    def register(self, model, design_layer: int = 9):
        attn_module = model.gpt_neox.layers[design_layer].attention
        cfg = attn_module.config
        self._num_heads = cfg.num_attention_heads
        self._head_dim = cfg.hidden_size // cfg.num_attention_heads
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
    Extrait les poids W_O effectifs de la couche spécifiée, reshapés par tête.

    Architecture Pythia :
      model.gpt_neox.layers[l].attention.dense  (768, 768)

    Returns:
        W_O: (H, d, d_h)
    """
    attn = model.gpt_neox.layers[design_layer].attention
    dense = attn.dense

    W_O_full = dense.weight.detach()  # (hidden_size, hidden_size) — poids de base

    # Avec LoRA, le poids effectif = W_base + scaling · lora_B @ lora_A
    if hasattr(dense, "lora_A") and "default" in dense.lora_A:
        scaling = dense.scaling["default"]
        lora_A  = dense.lora_A["default"].weight.detach()  # (r, in_features)
        lora_B  = dense.lora_B["default"].weight.detach()  # (out_features, r)
        W_O_full = W_O_full + scaling * (lora_B @ lora_A)

    cfg = attn.config
    num_heads = cfg.num_attention_heads   # 12
    head_dim = cfg.hidden_size // num_heads  # 64
    d = W_O_full.shape[0]  # 768

    W_O_heads = W_O_full.view(d, num_heads, head_dim)  # (768, 12, 64)
    W_O_heads = W_O_heads.permute(1, 0, 2)              # (12, 768, 64)

    return W_O_heads


# ---------------------------------------------------------------------------
# Visualisation et logging des matrices d'interaction de têtes
# ---------------------------------------------------------------------------


def _matrices_figure(omega: torch.Tensor, rho: torch.Tensor, G: torch.Tensor):
    """Crée une figure 1×3 : omega, rho, G comme heatmaps."""
    mats = [
        (omega.cpu().float().numpy(), "ω  (weight coupling)"),
        (rho.cpu().float().numpy(),   "ρ  (gradient coupling)"),
        (G.cpu().float().numpy(),     "G = ω · ρ"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    # rho est un produit scalaire normalisé → échelle fixe [-1, 1]
    # omega et G ont des valeurs bien plus petites : échelle adaptée à chaque matrice
    fixed_scale = [False, True, False]
    for ax, (mat, title), fixed in zip(axes, mats, fixed_scale):
        mat = np.nan_to_num(mat, nan=0.0)
        if fixed:
            vmin, vmax = -1.0, 1.0
        else:
            off_diag = mat[~np.eye(mat.shape[0], dtype=bool)]
            vmax = float(np.abs(off_diag).max()) or 1e-8
            vmin = -vmax
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel("head j")
        ax.set_ylabel("head i")
    fig.tight_layout()
    return fig


def log_head_matrices(
    model,
    device: torch.device,
    design_layer: int,
    step: int,
    val_loader,
    wandb_mod=None,
    n_batches: int = 20,
) -> float:
    """
    Calcule omega, rho, G pour `design_layer`, logue une image wandb + Γ(G).

    rho est estimé en accumulant les stats du gradient (probs_sum, label_counts)
    sur n_batches batches de validation avant de calculer eta_mean une seule fois.
    Cela stabilise rho (~160k tokens avec les defaults) vs un seul batch (4k tokens).

    Args:
        n_batches  : nombre de batches val accumulés pour rho (défaut 20)
        wandb_mod  : module wandb importé, ou None pour désactiver le logging

    Returns:
        gamma : Γ(G) = ||G − I||_F  (float)
    """
    was_training = model.training
    model.eval()

    use_autocast = device.type in ("cuda", "cpu")
    autocast_ctx = torch.autocast(
        device_type=device.type if device.type != "mps" else "cpu",
        dtype=torch.bfloat16,
        enabled=use_autocast,
    )

    with torch.no_grad():
        W_O   = get_output_projection_weights(model, design_layer).to(device)
        omega = HeadInteractionMatrix.compute_weight_coupling(W_O)

        probs_sum_acc    = None
        label_counts_acc = None
        n_valid_acc      = 0

        for batch in itertools.islice(iter(val_loader), n_batches):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            with autocast_ctx:
                out = model(input_ids=input_ids, labels=labels)
            ps, lc, nv = HeadInteractionMatrix._eta_stats(out.logits, labels)
            if probs_sum_acc is None:
                probs_sum_acc    = ps
                label_counts_acc = lc
            else:
                probs_sum_acc    = probs_sum_acc + ps
                label_counts_acc = label_counts_acc + lc
            n_valid_acc += nv

        rho   = HeadInteractionMatrix._rho_from_eta_stats(
            model, probs_sum_acc, label_counts_acc, n_valid_acc, W_O
        )
        G     = HeadInteractionMatrix.compute_G(omega, rho)
        gamma = HeadInteractionMatrix.interaction_strength(G).item()

    if was_training:
        model.train()

    if wandb_mod is not None:
        fig = _matrices_figure(omega, rho, G)
        wandb_mod.log({
            "head/matrices": wandb_mod.Image(fig),
            "head/gamma_G":  gamma,
        }, step=step)
        plt.close(fig)

    logger.info(f"  [head matrices] layer={design_layer}  Γ(G)={gamma:.4f}  "
                f"(rho sur {n_batches} batches)")
    return gamma
