"""
Module commun pour le FULL fine-tuning de Pythia-160M sur WikiText-103.
Partagé entre le baseline et les expériences GAME-LoRA / Stackelberg.

Changements par rapport à la version précédente :
  - LoRA → FULL fine-tuning (tous les poids entraînables)
  - The Pile → WikiText-103 (OOD léger vs pré-entraînement Pythia)
  - Streaming → dataset map-style (train + validation held-out)
  - Eval périodique sur le split validation (loss CE + perplexité)
  - Seeds complets (torch CPU + CUDA + numpy + random + generator dataloader)
  - num_workers > 0 possible (plus de streaming)
  - Poids en fp32 + autocast bf16 (stable pour full FT sur 160M)

Contient :
  - TrainConfig
  - WikiTextDataset (map-style, packing)
  - build_model_and_tokenizer (full FT)
  - setup_training (optimizer, scheduler, train/val dataloaders)
  - evaluate (loss CE + perplexité sur val)
  - seed_everything
  - log_config / add_common_args
"""

import os
import math
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

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
    # Optionnel : pour reproductibilité exacte (ralentit ~10-20%)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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

    # Optimisation — FULL fine-tuning
    # Pour du full FT d'un LM pré-entraîné : LR beaucoup plus faible qu'en LoRA.
    # 2e-5 à 5e-5 est la plage standard. On reste conservateur à 3e-5.
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    batch_size_per_gpu: int = 8
    grad_accum: int = 2           # batch effectif = 16
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)    # beta2=0.95 classique pour les LM

    # Évaluation
    eval_every: int = 100         # en steps optimizer
    eval_max_batches: int = 50    # limite l'eval pour rester rapide (~50 × 16 × 1024 ≈ 0.8M tokens)

    # Logging / sauvegarde
    output_dir: str = "./checkpoints/baseline"
    log_every: int = 20
    save_every: int = 500
    wandb_project: Optional[str] = None
    run_name: str = "baseline_fullft_pythia"
    seed: int = 42
    dry_run: bool = False
    random_init: bool = False
    num_workers: int = 2

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

        ds = load_dataset(dataset_name, dataset_config, split=split)
        # On concatène tous les textes non vides, sans ré-introduire de BOS/EOS
        # explicite (WikiText est un corpus continu ; le tokenizer GPT-NeoX n'a
        # pas de BOS par défaut). On ajoute un EOS entre les articles.
        eos = tokenizer.eos_token_id
        all_ids: List[int] = []
        for ex in ds:
            t = ex["text"]
            if not t.strip():
                continue
            ids = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
            all_ids.extend(ids)
            all_ids.append(eos)

        if max_tokens is not None and len(all_ids) > max_tokens:
            all_ids = all_ids[:max_tokens]

        # Nombre de blocs complets de seq_len+1
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
# Construction du modèle (FULL fine-tuning)
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
        # Poids en fp32 pour un full FT stable. Autocast bf16 dans la boucle.
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params entraînables : {n_trainable:,} / {n_total:,} "
                f"({100*n_trainable/n_total:.1f}%)")

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
    # Cap des tokens train pour dry_run
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
    La loss est une CE moyenne par token (standard HF).
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
    val_ppl  = math.exp(min(val_loss, 20))  # clip pour éviter les overflows
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
    parser.add_argument("--batch_size_per_gpu", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_max_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--random_init", action="store_true",
                        help="Initialiser les poids aléatoirement (pas de préentraînement)")
    return parser
