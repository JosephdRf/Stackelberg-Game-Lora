"""
Module commun pour l'entraînement LoRA de Qwen2.5-0.5B sur The Pile.
Partagé entre le baseline (train_baseline.py) et GAME-LoRA (train_game_lora.py).

Contient :
  - TrainConfig : configuration d'entraînement
  - PileStreamDataset : dataset streaming sur The Pile
  - build_model_and_tokenizer : construction du modèle + LoRA
  - setup_training : création optimizer, scheduler, dataloader
  - log_config : affichage de la configuration
"""

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — reproduction exacte de l'article
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Modèle
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # LoRA — Appendix A, Training configuration
    lora_rank: int = 16
    lora_alpha: int = 32  # ratio alpha/rank = 2 → scaling = 1.0
    lora_dropout: float = 0.1
    # "LoRA targets: Q, K, V, O projections (all layers)"
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Données
    dataset_name: str = "EleutherAI/pile"
    dataset_config: str = "all"  # pile complet en streaming
    seq_len: int = 1024
    total_tokens: int = 20_000_000  # 20M tokens

    # Optimisation — Appendix A
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02  # "2% warmup"
    batch_size_per_gpu: int = 8  # à ajuster selon VRAM
    grad_accum: int = 2  # batch effectif = batch_size_per_gpu * grad_accum = 16

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size_per_gpu * self.grad_accum

    @property
    def total_steps(self) -> int:
        tokens_per_step = self.seq_len * self.effective_batch_size
        return math.ceil(self.total_tokens / tokens_per_step)

    @property
    def warmup_steps(self) -> int:
        return math.ceil(self.warmup_ratio * self.total_steps)

    # Logging / sauvegarde
    output_dir: str = "./checkpoints/baseline"
    log_every: int = 50
    save_every: int = 2000
    wandb_project: Optional[str] = None
    run_name: str = "baseline_lora"
    seed: int = 42
    dry_run: bool = False  # 100 steps pour tester


# ---------------------------------------------------------------------------
# Dataset : The Pile en streaming, tokenisé à la volée
# ---------------------------------------------------------------------------


class PileStreamDataset(IterableDataset):
    """
    Streaming dataset sur The Pile.
    Concatène les textes et les découpe en blocs de seq_len tokens
    (stratégie 'pack' classique, identique à ce que fait l'article).
    """

    def __init__(self, tokenizer, seq_len: int, total_tokens: int, seed: int = 42):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.total_tokens = total_tokens
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
        ).shuffle(seed=self.seed, buffer_size=50_000)

        buffer = []
        tokens_yielded = 0

        for example in ds:
            text = example.get("text", "")
            if not text.strip():
                continue

            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            # Ajoute EOS entre les documents
            ids = ids + [self.tokenizer.eos_token_id]
            buffer.extend(ids)

            # Émet des blocs de seq_len+1 tokens (input / label décalé de 1)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len + 1 :]

                block = torch.tensor(chunk, dtype=torch.long)  # 1025 tokens
                yield {"input_ids": block, "labels": block.clone()}

                tokens_yielded += self.seq_len
                if tokens_yielded >= self.total_tokens:
                    return


# ---------------------------------------------------------------------------
# Construction du modèle avec LoRA
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(cfg: TrainConfig):
    logger.info(f"Chargement de {cfg.model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16,  # bf16 pour Qwen2.5
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
# Setup de l'entraînement (optimizer, scheduler, dataloader)
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
    Crée les composants d'entraînement communs.
    Returns: (dataloader, optimizer, scheduler, total_steps)
    """
    max_tokens = 100 * cfg.seq_len * cfg.effective_batch_size if cfg.dry_run else cfg.total_tokens
    dataset = PileStreamDataset(tokenizer, cfg.seq_len, max_tokens, cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=0,
        pin_memory=(torch.cuda.is_available()),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=cfg.weight_decay,
    )

    total_steps = 100 if cfg.dry_run else cfg.total_steps

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    return dataloader, optimizer, scheduler, total_steps


def log_config(cfg: TrainConfig):
    logger.info("=== Configuration ===")
    logger.info(f"  Modèle        : {cfg.model_name}")
    logger.info(f"  LoRA rank     : {cfg.lora_rank}  alpha={cfg.lora_alpha}")
    logger.info(f"  Tokens totaux : {cfg.total_tokens:,}")
    logger.info(f"  Batch effectif: {cfg.effective_batch_size}")
    logger.info(f"  Steps totaux  : {cfg.total_steps}")
    logger.info(f"  Warmup steps  : {cfg.warmup_steps}")
    logger.info(f"  LR            : {cfg.lr}")
    logger.info(f"  Weight decay  : {cfg.weight_decay}")
    logger.info(f"  Dry run       : {cfg.dry_run}")


def add_common_args(parser):
    """Ajoute les arguments communs au parser."""
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--total_tokens", type=int, default=20_000_000)
    parser.add_argument("--batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=200)
    return parser
