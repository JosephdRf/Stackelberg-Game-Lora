"""
GAME-LoRA training — fine-tuning avec régularisation game-théorique
"Multi-Head Attention is a Multi-Player Game" (Chakrabarti & Balachundar, 2026)

L = L_CE + λ_LDB · L_LDB + λ_ABT · L_ABT   (Eq 27)
avec gradients arbitrés via Nash-MTL.

Usage :
    python train_game_lora.py
    python train_game_lora.py --design_layer 19 --dry_run
    python train_game_lora.py --wandb_project my_project --run_name game_lora_v1
"""

import os
import sys
import json
import argparse
import logging
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root and current dir to path for imports
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root (for train.py)
sys.path.insert(0, _HERE)                   # current dir (for game_losses.py)

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


def train_game_lora(cfg: TrainConfig, design_layer: int = 19, use_nash_mtl: bool = True):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    logger.info(f"Device : {device}")

    # Wandb (optionnel)
    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    # Modèle
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = model.to(device)

    # Setup commun
    dataloader, optimizer, scheduler, total_steps = setup_training(cfg, model, tokenizer)
    logger.info(f"Total steps : {total_steps}  |  Warmup : {cfg.warmup_steps}")
    logger.info(f"Design layer : {design_layer}  |  Nash-MTL : {use_nash_mtl}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    _exp_dir = os.path.dirname(os.path.abspath(cfg.output_dir))
    logs_dir  = os.path.join(_exp_dir, "logs")
    plots_dir = os.path.join(_exp_dir, "plots")
    os.makedirs(logs_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    loss_history: dict = {"step": [], "loss": [], "ce_loss": [], "ldb_loss": [], "abt_loss": []}

    # --- GAME-LoRA specific components ---
    ldb_loss_fn = LogDetBarrierLoss(epsilon=0.01).to(device)
    abt_loss_fn = AdaptiveBarlowTwinsLoss(
        alpha=0.929, beta=15.99, tau=0.0, subtract_identity=True
    ).to(device)
    ema_normalizer = EMALossNormalizer(target=20.0, alpha=0.1)
    loss_scheduler = GAMELossScheduler(total_steps=total_steps)
    nash_mtl = NashMTL(n_tasks=3) if use_nash_mtl else None

    # Hook pour capturer les head outputs
    head_capture = HeadOutputCapture()
    head_capture.register(model, design_layer=design_layer)

    # Boucle principale
    model.train()
    global_step = 0
    accum_loss = 0.0
    accum_ce = 0.0
    accum_ldb = 0.0
    accum_abt = 0.0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(total=total_steps, desc="GAME-LoRA Training", unit="step")

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        opt_step = (global_step + 1) // cfg.grad_accum
        lambda_abt, lambda_ldb = loss_scheduler.get_lambdas(opt_step)

        with torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=torch.bfloat16,
            enabled=(device.type in ("cuda", "cpu")),
        ):
            # Forward pass — le hook capture les head outputs
            out = model(input_ids=input_ids, labels=labels)
            ce_loss = out.loss

            # Récupérer head outputs et calculer les losses GAME
            head_outputs = head_capture.get()  # (B, T, H, d_h)
            game_loss = torch.tensor(0.0, device=device)
            ldb_val = torch.tensor(0.0, device=device)
            abt_val = torch.tensor(0.0, device=device)

            if head_outputs is not None and (lambda_abt > 0 or lambda_ldb > 0):
                # Interaction matrix G = ω ⊙ ρ (Def 2.3, Eq 6)
                W_O = get_output_projection_weights(model, design_layer).to(device)
                omega = HeadInteractionMatrix.compute_weight_coupling(W_O)
                rho = HeadInteractionMatrix.compute_gradient_coupling(
                    model, out.logits, labels, W_O,
                    head_dim=head_outputs.shape[-1]
                )
                G = HeadInteractionMatrix.compute_G(omega, rho)

                # L_LDB (Eq 28)
                if lambda_ldb > 0:
                    ldb_val = ldb_loss_fn(G)

                # L_ABT (Eq 29)
                if lambda_abt > 0:
                    abt_raw = abt_loss_fn(head_outputs, G)
                    abt_val = ema_normalizer.normalize(abt_raw)

            if use_nash_mtl and (lambda_abt > 0 or lambda_ldb > 0) and head_outputs is not None:
                # Nash-MTL: trouver les poids optimaux pour les 3 losses
                # On calcule les gradients de chaque loss séparément
                params = [p for p in model.parameters() if p.requires_grad]

                # Gradient de CE
                grad_ce = torch.autograd.grad(
                    ce_loss, params, retain_graph=True, allow_unused=True
                )
                grad_ce_flat = torch.cat([
                    g.flatten() if g is not None else torch.zeros(p.numel(), device=device)
                    for g, p in zip(grad_ce, params)
                ])

                # Gradient de LDB
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

                # Gradient de ABT
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

                # Nash bargaining
                with torch.no_grad():
                    weights = nash_mtl.get_weights([
                        grad_ce_flat.detach(),
                        (lambda_ldb * grad_ldb_flat).detach(),
                        (lambda_abt * grad_abt_flat).detach(),
                    ])

                # Loss combinée avec poids Nash
                total_loss = (
                    weights[0] * ce_loss
                    + weights[1] * lambda_ldb * ldb_val
                    + weights[2] * lambda_abt * abt_val
                ) / cfg.grad_accum
            else:
                # Sans Nash-MTL: combinaison linéaire simple (Eq 27)
                total_loss = (
                    ce_loss + lambda_ldb * ldb_val + lambda_abt * abt_val
                ) / cfg.grad_accum

        total_loss.backward()
        accum_loss += total_loss.item()
        accum_ce += (ce_loss.item() / cfg.grad_accum)
        accum_ldb += (ldb_val.item() / cfg.grad_accum) if lambda_ldb > 0 else 0
        accum_abt += (abt_val.item() / cfg.grad_accum) if lambda_abt > 0 else 0

        # Gradient accumulation
        if (global_step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step = (global_step + 1) // cfg.grad_accum
            step_time = time.perf_counter() - _step_start
            tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)

            _step_start = time.perf_counter()
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{accum_loss:.4f}",
                ce=f"{accum_ce:.4f}",
                ldb=f"{accum_ldb:.4f}",
                abt=f"{accum_abt:.4f}",
                tok_s=f"{tokens_per_sec:,}",
            )

            if opt_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                tqdm.write(
                    f"Step {opt_step:>6d}/{total_steps}"
                    f"  loss={accum_loss:.4f}"
                    f"  CE={accum_ce:.4f}"
                    f"  LDB={accum_ldb:.4f}"
                    f"  ABT={accum_abt:.4f}"
                    f"  λ_ABT={lambda_abt:.4f}  λ_LDB={lambda_ldb:.4f}"
                    f"  lr={lr_now:.2e}"
                    f"  tok/s={tokens_per_sec:,}"
                )
                if use_wandb:
                    log_dict = {
                        "train/loss": accum_loss,
                        "train/ce_loss": accum_ce,
                        "train/ldb_loss": accum_ldb,
                        "train/abt_loss": accum_abt,
                        "train/lambda_abt": lambda_abt,
                        "train/lambda_ldb": lambda_ldb,
                        "train/lr": lr_now,
                        "train/step": opt_step,
                        "train/tokens": opt_step * cfg.seq_len * cfg.effective_batch_size,
                    }
                    # Log Γ(G) si disponible
                    if head_outputs is not None:
                        try:
                            W_O = get_output_projection_weights(model, design_layer).to(device)
                            omega = HeadInteractionMatrix.compute_weight_coupling(W_O)
                            gamma = HeadInteractionMatrix.interaction_strength(omega)
                            log_dict["train/gamma_G"] = gamma.item()
                        except Exception:
                            pass
                    wandb.log(log_dict, step=opt_step)
                loss_history["step"].append(opt_step)
                loss_history["loss"].append(accum_loss)
                loss_history["ce_loss"].append(accum_ce)
                loss_history["ldb_loss"].append(accum_ldb)
                loss_history["abt_loss"].append(accum_abt)
                with open(os.path.join(logs_dir, "loss.json"), "w") as _f:
                    json.dump(loss_history, _f, indent=2)

            accum_loss = 0.0
            accum_ce = 0.0
            accum_ldb = 0.0
            accum_abt = 0.0

            # Sauvegarde checkpoint
            if opt_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{opt_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                logger.info(f"Checkpoint sauvegardé → {ckpt_path}")

            if opt_step >= total_steps:
                break

        global_step += 1

    pbar.close()
    head_capture.remove()

    # Sauvegarde finale
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modèle final sauvegardé → {final_path}")

    # Plot des losses
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("loss", "ce_loss", "ldb_loss", "abt_loss"):
        ax.plot(loss_history["step"], loss_history[key], label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses — GAME-LoRA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Plot sauvegardé → {os.path.join(plots_dir, 'loss.png')}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="GAME-LoRA training")
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    parser.add_argument("--run_name", default="game_lora")
    parser.add_argument("--design_layer", type=int, default=19,
                        help="Couche d'attention pour les losses GAME (Appendix A: layer 19)")
    parser.add_argument("--no_nash_mtl", action="store_true",
                        help="Désactiver Nash-MTL, utiliser combinaison linéaire simple")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        total_tokens=args.total_tokens,
        batch_size_per_gpu=args.batch_size_per_gpu,
        grad_accum=args.grad_accum,
        lr=args.lr,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        seed=args.seed,
        dry_run=args.dry_run,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    log_config(cfg)
    logger.info(f"  Design layer  : {args.design_layer}")
    logger.info(f"  Nash-MTL      : {not args.no_nash_mtl}")
    train_game_lora(cfg, design_layer=args.design_layer, use_nash_mtl=not args.no_nash_mtl)
