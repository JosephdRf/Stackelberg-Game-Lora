"""
Baseline : LoRA fine-tuning de Pythia-160M sur WikiText-103 (CE seul, sans losses GAME).

Mêmes hyperparamètres LoRA que GAME-LoRA (r=16, alpha=32, dropout=0.1,
target_modules=[query_key_value, dense]) — seule la loss diffère.

Usage :
    python pythia160M/baseline/train_baseline.py
    python pythia160M/baseline/train_baseline.py --dry_run
    python pythia160M/baseline/train_baseline.py --wandb_project my_proj --run_name pythia_lora_baseline_v1
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

# Repo root
_HERE    = os.path.dirname(os.path.abspath(__file__))          # pythia160M/baseline/
_MODEL   = os.path.dirname(_HERE)                              # pythia160M/
_ROOT    = os.path.dirname(_MODEL)                             # repo root
sys.path.insert(0, _MODEL)

import torch
from tqdm import tqdm

from train_utils import (
    TrainConfig,
    build_model_and_tokenizer,
    setup_training,
    evaluate,
    seed_everything,
    get_device,
    log_config,
    add_common_args,
    log_head_matrices,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Boucle d'entraînement
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig, head_log_layer: int = 9):
    seed_everything(cfg.seed)

    device = get_device()
    logger.info(f"Device : {device}")

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    model, tokenizer = build_model_and_tokenizer(cfg)
    model = model.to(device)

    train_loader, val_loader, optimizer, scheduler, total_steps = setup_training(
        cfg, model, tokenizer
    )
    logger.info(f"Total steps : {total_steps}  |  Warmup : {cfg.warmup_steps}")

    # Répertoires
    os.makedirs(cfg.output_dir, exist_ok=True)
    _exp_dir  = os.path.dirname(os.path.abspath(cfg.output_dir))
    logs_dir  = os.path.join(_exp_dir, "logs")
    plots_dir = os.path.join(_exp_dir, "plots")
    os.makedirs(logs_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Historique
    history = {
        "train": {"step": [], "loss": [], "loss_ema": []},
        "val":   {"step": [], "loss": [], "ppl": []},
    }
    _ema_loss  = None
    _ema_alpha = 0.05

    # Eval initiale (step 0) pour avoir le point de référence avant entraînement
    logger.info("Eval initiale (modèle pré-entraîné, avant fine-tuning) ...")
    v_loss, v_ppl = evaluate(model, val_loader, device,
                             max_batches=cfg.eval_max_batches)
    history["val"]["step"].append(0)
    history["val"]["loss"].append(v_loss)
    history["val"]["ppl"].append(v_ppl)
    logger.info(f"  [step 0] val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
    if use_wandb:
        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl, "train/step": 0}, step=0)

    model.train()
    opt_step    = 0
    accum_loss  = 0.0
    micro_count = 0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(total=total_steps, desc="Training (Pythia-160M LoRA baseline)", unit="step")

    _autocast_dtype = torch.bfloat16
    _use_autocast   = device.type in ("cuda", "cpu")  # pas en MPS

    done = False
    # Boucle sur epochs (au cas où le dataset est consommé avant total_steps)
    while not done:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type if device.type != "mps" else "cpu",
                dtype=_autocast_dtype,
                enabled=_use_autocast,
            ):
                out  = model(input_ids=input_ids, labels=labels)
                loss = out.loss / cfg.grad_accum

            loss.backward()
            accum_loss += loss.item()
            micro_count += 1

            if micro_count % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=cfg.grad_clip,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                opt_step += 1
                avg_loss = accum_loss
                accum_loss = 0.0

                step_time      = time.perf_counter() - _step_start
                tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
                _step_start    = time.perf_counter()

                pbar.update(1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", tok_s=f"{tokens_per_sec:,}")

                _ema_loss = avg_loss if _ema_loss is None \
                    else _ema_alpha * avg_loss + (1 - _ema_alpha) * _ema_loss

                # Log train
                if opt_step % cfg.log_every == 0 or opt_step == 1:
                    lr_now = scheduler.get_last_lr()[0]
                    logger.info(
                        f"[train] step {opt_step:>6d}/{total_steps}"
                        f"  loss={avg_loss:.4f}  ema={_ema_loss:.4f}"
                        f"  lr={lr_now:.2e}  tok/s={tokens_per_sec:,}"
                    )
                    if use_wandb:
                        wandb.log({
                            "train/loss":     avg_loss,
                            "train/loss_ema": _ema_loss,
                            "train/lr":       lr_now,
                            "train/step":     opt_step,
                            "train/tokens":   opt_step * cfg.seq_len * cfg.effective_batch_size,
                        }, step=opt_step)
                    history["train"]["step"].append(opt_step)
                    history["train"]["loss"].append(avg_loss)
                    history["train"]["loss_ema"].append(_ema_loss)

                # Eval périodique
                if opt_step % cfg.eval_every == 0:
                    v_loss, v_ppl = evaluate(
                        model, val_loader, device,
                        max_batches=cfg.eval_max_batches,
                        autocast_dtype=_autocast_dtype,
                    )
                    logger.info(f"[val]   step {opt_step:>6d}  "
                               f"val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
                    log_head_matrices(
                        model, device, head_log_layer, opt_step, val_loader,
                        wandb_mod=wandb if use_wandb else None,
                    )
                    if use_wandb:
                        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl},
                                  step=opt_step)
                    history["val"]["step"].append(opt_step)
                    history["val"]["loss"].append(v_loss)
                    history["val"]["ppl"].append(v_ppl)

                # Dump JSON après chaque log/eval
                if (opt_step % cfg.log_every == 0
                        or opt_step % cfg.eval_every == 0
                        or opt_step == 1):
                    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
                        json.dump(history, _f, indent=2)

                # Checkpoint
                if opt_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{opt_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    logger.info(f"Checkpoint → {ckpt_path}")

                if opt_step >= total_steps:
                    done = True
                    break

        if not done:
            logger.info("Fin d'epoch — on recommence un passage sur le dataset.")

    pbar.close()

    # Eval finale
    logger.info("Eval finale ...")
    v_loss, v_ppl = evaluate(model, val_loader, device,
                             max_batches=cfg.eval_max_batches)
    history["val"]["step"].append(opt_step)
    history["val"]["loss"].append(v_loss)
    history["val"]["ppl"].append(v_ppl)
    logger.info(f"  [final] val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
    if use_wandb:
        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=opt_step)

    # Sauvegarde finale
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modèle final sauvegardé → {final_path}")

    # JSON final
    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
        json.dump(history, _f, indent=2)

    # Plot train + val
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(history["train"]["step"], history["train"]["loss"],
             alpha=0.25, color="darkorange", label="train loss (raw)")
    ax1.plot(history["train"]["step"], history["train"]["loss_ema"],
             color="darkorange", label="train loss (EMA)")
    ax1.plot(history["val"]["step"], history["val"]["loss"],
             color="steelblue", marker="o", markersize=4, label="val loss")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training — Pythia-160M LoRA baseline on WikiText-103")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)

    # Plot perplexité val
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["val"]["step"], history["val"]["ppl"],
            color="steelblue", marker="o", markersize=4)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity — WikiText-103")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_ppl.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots → {plots_dir}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline LoRA fine-tuning — Pythia-160M sur WikiText-103 (CE seul)"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--run_name",   default="baseline_fullft_pythia")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = TrainConfig(
        model_name         = args.model_name,
        dataset_name       = args.dataset_name,
        dataset_config     = args.dataset_config,
        total_tokens       = args.total_tokens,
        batch_size_per_gpu = args.batch_size_per_gpu,
        grad_accum         = args.grad_accum,
        lr                 = args.lr,
        output_dir         = args.output_dir,
        wandb_project      = args.wandb_project,
        run_name           = args.run_name,
        seed               = args.seed,
        dry_run            = args.dry_run,
        log_every          = args.log_every,
        eval_every         = args.eval_every,
        eval_max_batches   = args.eval_max_batches,
        save_every         = args.save_every,
        num_workers        = args.num_workers,
        random_init        = args.random_init,
    )

    log_config(cfg)
    train(cfg, head_log_layer=args.head_log_layer)
