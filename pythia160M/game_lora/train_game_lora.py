"""
GAME-LoRA training — Pythia-160M (GPT-NeoX)
"Multi-Head Attention is a Multi-Player Game" (Chakrabarti & Balachundar, 2026)

L = L_CE + λ_LDB · L_LDB + λ_ABT · L_ABT   (Eq 27)
avec gradients arbitrés via Nash-MTL.

Adapté depuis qwen2.5_0.5B/game_lora/train_game_lora.py :
  - design_layer par défaut = 9  (≈ 79% de 12 couches, analogue du layer 19/24 de Qwen)
  - Imports depuis ../train.py   (Pythia TrainConfig)
  - game_losses.py local         (hooks/accès W_O adaptés à GPT-NeoX)

Usage :
    python train_game_lora.py
    python train_game_lora.py --design_layer 9 --dry_run
    python train_game_lora.py --wandb_project my_project --run_name game_lora_pythia_v1
    python train_game_lora.py --no_nash_mtl   # combinaison linéaire simple (Eq 27)
"""

import os
import sys
import argparse
import logging
import time

_HERE  = os.path.dirname(os.path.abspath(__file__))   # pythia160M/game_lora/
_MODEL = os.path.dirname(_HERE)                        # pythia160M/
sys.path.insert(0, _MODEL)   # pour importer pythia160M/train.py
sys.path.insert(0, _HERE)    # pour importer game_losses.py local

import torch
import numpy as np
from tqdm import tqdm

from train import (
    TrainConfig,
    build_model_and_tokenizer,
    setup_training,
    get_device,
    log_config,
    add_common_args,
)
from game_losses import (
    HeadInteractionMatrix,
    LogDetBarrierLoss,
    AdaptiveBarlowTwinsLoss,
    EMALossNormalizer,
    NashMTL,
    GAMELossScheduler,
    HeadOutputCapture,
    get_output_projection_weights,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boucle d'entraînement GAME-LoRA
# ---------------------------------------------------------------------------


def train_game_lora(cfg: TrainConfig, design_layer: int = 9, use_nash_mtl: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    logger.info(f"Device : {device}")

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    model, tokenizer = build_model_and_tokenizer(cfg)
    model = model.to(device)

    dataloader, optimizer, scheduler, total_steps = setup_training(cfg, model, tokenizer)
    logger.info(f"Total steps : {total_steps}  |  Warmup : {cfg.warmup_steps}")
    logger.info(f"Design layer : {design_layer}  |  Nash-MTL : {use_nash_mtl}")

    _ema_loss  = None
    _ema_alpha = 0.05

    # --- Composants GAME-LoRA ---
    ldb_loss_fn   = LogDetBarrierLoss(epsilon=0.01).to(device)
    abt_loss_fn   = AdaptiveBarlowTwinsLoss(
        alpha=0.929, beta=15.99, tau=0.0, subtract_identity=True
    ).to(device)
    ema_normalizer  = EMALossNormalizer(target=20.0, alpha=0.1)
    loss_scheduler  = GAMELossScheduler(total_steps=total_steps)
    nash_mtl        = NashMTL(n_tasks=3) if use_nash_mtl else None

    # Hook sur attention.dense de la design layer (GPT-NeoX)
    head_capture = HeadOutputCapture()
    head_capture.register(model, design_layer=design_layer)

    # Boucle principale
    model.train()
    global_step = 0
    accum_loss  = 0.0
    accum_ce    = 0.0
    accum_ldb   = 0.0
    accum_abt   = 0.0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(total=total_steps, desc="GAME-LoRA Training (Pythia-160M)", unit="step")

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        opt_step = (global_step + 1) // cfg.grad_accum
        lambda_abt, lambda_ldb = loss_scheduler.get_lambdas(opt_step)

        with torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=torch.bfloat16,
            enabled=(device.type in ("cuda", "cpu")),
        ):
            out     = model(input_ids=input_ids, labels=labels)
            ce_loss = out.loss

            head_outputs = head_capture.get()  # (B, T, H, d_h)
            game_loss = torch.tensor(0.0, device=device)
            ldb_val   = torch.tensor(0.0, device=device)
            abt_val   = torch.tensor(0.0, device=device)

            if head_outputs is not None and (lambda_abt > 0 or lambda_ldb > 0):
                W_O   = get_output_projection_weights(model, design_layer).to(device)
                omega = HeadInteractionMatrix.compute_weight_coupling(W_O)
                rho   = HeadInteractionMatrix.compute_gradient_coupling(
                    model, out.logits, labels, W_O,
                    head_dim=head_outputs.shape[-1],
                )
                G = HeadInteractionMatrix.compute_G(omega, rho)

                if lambda_ldb > 0:
                    ldb_val = ldb_loss_fn(G)

                if lambda_abt > 0:
                    abt_raw = abt_loss_fn(head_outputs, G)
                    abt_val = ema_normalizer.normalize(abt_raw)

        if use_nash_mtl and (lambda_abt > 0 or lambda_ldb > 0) and head_outputs is not None:
            params = [p for p in model.parameters() if p.requires_grad]

            grad_ce = torch.autograd.grad(
                ce_loss, params, retain_graph=True, allow_unused=True
            )
            grad_ce_flat = torch.cat([
                g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                for g, p in zip(grad_ce, params)
            ])

            if lambda_ldb > 0:
                grad_ldb = torch.autograd.grad(
                    ldb_val, params, retain_graph=True, allow_unused=True
                )
                grad_ldb_flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                    for g, p in zip(grad_ldb, params)
                ])
            else:
                grad_ldb_flat = torch.zeros_like(grad_ce_flat)

            if lambda_abt > 0:
                grad_abt = torch.autograd.grad(
                    abt_val, params, retain_graph=True, allow_unused=True
                )
                grad_abt_flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                    for g, p in zip(grad_abt, params)
                ])
            else:
                grad_abt_flat = torch.zeros_like(grad_ce_flat)

            with torch.no_grad():
                weights = nash_mtl.get_weights([
                    grad_ce_flat.detach(),
                    (lambda_ldb * grad_ldb_flat).detach(),
                    (lambda_abt * grad_abt_flat).detach(),
                ])

            total_loss = (
                weights[0] * ce_loss
                + weights[1] * lambda_ldb * ldb_val
                + weights[2] * lambda_abt * abt_val
            ) / cfg.grad_accum
        else:
            # Combinaison linéaire simple (Eq 27)
            total_loss = (
                ce_loss + lambda_ldb * ldb_val + lambda_abt * abt_val
            ) / cfg.grad_accum

        total_loss.backward()
        accum_loss += total_loss.item()
        accum_ce   += ce_loss.item() / cfg.grad_accum
        accum_ldb  += ldb_val.item() / cfg.grad_accum if lambda_ldb > 0 else 0.0
        accum_abt  += abt_val.item() / cfg.grad_accum if lambda_abt > 0 else 0.0

        # Gradient accumulation
        if (global_step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step       = (global_step + 1) // cfg.grad_accum
            step_time      = time.perf_counter() - _step_start
            tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
            _step_start    = time.perf_counter()

            _ema_loss = (
                accum_loss if _ema_loss is None
                else _ema_alpha * accum_loss + (1 - _ema_alpha) * _ema_loss
            )

            pbar.update(1)
            pbar.set_postfix(
                loss=f"{accum_loss:.4f}",
                ema=f"{_ema_loss:.4f}",
                ce=f"{accum_ce:.4f}",
                ldb=f"{accum_ldb:.4f}",
                abt=f"{accum_abt:.4f}",
                tok_s=f"{tokens_per_sec:,}",
            )

            if opt_step % cfg.log_every == 0 or opt_step == 1:
                lr_now = scheduler.get_last_lr()[0]
                tqdm.write(
                    f"Step {opt_step:>6d}/{total_steps}"
                    f"  loss={accum_loss:.4f}"
                    f"  ema={_ema_loss:.4f}"
                    f"  CE={accum_ce:.4f}"
                    f"  LDB={accum_ldb:.4f}"
                    f"  ABT={accum_abt:.4f}"
                    f"  λ_ABT={lambda_abt:.4f}  λ_LDB={lambda_ldb:.4f}"
                    f"  lr={lr_now:.2e}"
                    f"  tok/s={tokens_per_sec:,}"
                )
                if use_wandb:
                    log_dict = {
                        "train/loss":      accum_loss,
                        "train/ce_loss":   accum_ce,
                        "train/ldb_loss":  accum_ldb,
                        "train/abt_loss":  accum_abt,
                        "train/lambda_abt": lambda_abt,
                        "train/lambda_ldb": lambda_ldb,
                        "train/lr":        lr_now,
                        "train/step":      opt_step,
                        "train/tokens":    opt_step * cfg.seq_len * cfg.effective_batch_size,
                        "train/loss_ema":  _ema_loss,
                    }
                    if head_outputs is not None:
                        try:
                            W_O_log = get_output_projection_weights(model, design_layer).to(device)
                            omega_log = HeadInteractionMatrix.compute_weight_coupling(W_O_log)
                            gamma = HeadInteractionMatrix.interaction_strength(omega_log)
                            log_dict["train/gamma_G"] = gamma.item()
                        except Exception:
                            pass
                    wandb.log(log_dict, step=opt_step)

            accum_loss = 0.0
            accum_ce   = 0.0
            accum_ldb  = 0.0
            accum_abt  = 0.0

            if opt_step >= total_steps:
                break

        global_step += 1

    pbar.close()
    head_capture.remove()

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="GAME-LoRA training — Pythia-160M")
    parser = add_common_args(parser)
    parser.add_argument("--run_name",   default="game_lora_pythia")
    parser.add_argument(
        "--design_layer", type=int, default=9,
        help="Couche d'attention pour les losses GAME. "
             "Pythia-160M a 12 couches ; layer 9 ≈ 79%% du réseau (analogue du layer 19/24 de Qwen).",
    )
    parser.add_argument(
        "--no_nash_mtl", action="store_true",
        help="Désactiver Nash-MTL, utiliser combinaison linéaire simple (Eq 27)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name         = args.model_name,
        dataset_name       = args.dataset_name,
        total_tokens       = args.total_tokens,
        batch_size_per_gpu = args.batch_size_per_gpu,
        grad_accum         = args.grad_accum,
        lr                 = args.lr,
        wandb_project      = args.wandb_project,
        run_name           = args.run_name,
        seed               = args.seed,
        dry_run            = args.dry_run,
        log_every          = args.log_every,
    )

    log_config(cfg)
    logger.info(f"  Design layer  : {args.design_layer}")
    logger.info(f"  Nash-MTL      : {not args.no_nash_mtl}")
    train_game_lora(cfg, design_layer=args.design_layer, use_nash_mtl=not args.no_nash_mtl)
