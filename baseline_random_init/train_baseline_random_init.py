"""
Baseline avec poids aléatoires : LoRA fine-tuning de Qwen2.5-0.5B (non préentrainé)
sur The Pile. Même hyperparamètres que le baseline préentrainé, seuls les poids
sont initialisés aléatoirement (via AutoConfig + AutoModelForCausalLM.from_config).

Usage :
    python train_baseline_random_init.py
    python train_baseline_random_init.py --dry_run
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

logger = logging.getLogger(__name__)


def train(cfg: TrainConfig):
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

    os.makedirs(cfg.output_dir, exist_ok=True)
    _exp_dir = os.path.dirname(os.path.abspath(cfg.output_dir))
    logs_dir  = os.path.join(_exp_dir, "logs")
    plots_dir = os.path.join(_exp_dir, "plots")
    os.makedirs(logs_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    loss_history: dict = {"step": [], "loss": [], "loss_ema": []}
    _ema_loss = None
    _ema_alpha = 0.05

    model.train()
    global_step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(total=total_steps, desc="Training (random init)", unit="step")

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=torch.bfloat16,
            enabled=(device.type in ("cuda", "cpu")),
        ):
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / cfg.grad_accum

        loss.backward()
        accum_loss += loss.item()

        if (global_step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            opt_step = (global_step + 1) // cfg.grad_accum
            avg_loss = accum_loss
            accum_loss = 0.0

            step_time = time.perf_counter() - _step_start
            tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
            _step_start = time.perf_counter()

            pbar.update(1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", tok_s=f"{tokens_per_sec:,}")

            _ema_loss = avg_loss if _ema_loss is None else _ema_alpha * avg_loss + (1 - _ema_alpha) * _ema_loss

            if opt_step % cfg.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                tqdm.write(
                    f"Step {opt_step:>6d}/{total_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  ema={_ema_loss:.4f}"
                    f"  lr={lr_now:.2e}"
                    f"  tok/s={tokens_per_sec:,}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/loss_ema": _ema_loss,
                            "train/lr": lr_now,
                            "train/step": opt_step,
                        },
                        step=opt_step,
                    )
                loss_history["step"].append(opt_step)
                loss_history["loss"].append(avg_loss)
                loss_history["loss_ema"].append(_ema_loss)
                with open(os.path.join(logs_dir, "loss.json"), "w") as _f:
                    json.dump(loss_history, _f, indent=2)

            if opt_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{opt_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                logger.info(f"Checkpoint sauvegardé → {ckpt_path}")

            if opt_step >= total_steps:
                break

        global_step += 1

    pbar.close()

    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modèle final sauvegardé → {final_path}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history["step"], loss_history["loss"], alpha=0.3, color="tomato", label="loss (raw)")
    ax.plot(loss_history["step"], loss_history["loss_ema"], color="tomato", label="loss (EMA α=0.05)")
    ax.set_xlabel("step")
    ax.set_ylabel("Loss CE")
    ax.set_title("Training loss — baseline (random init)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Plot sauvegardé → {os.path.join(plots_dir, 'loss.png')}")

    if use_wandb:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline LoRA — poids aléatoires")
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    parser.add_argument("--run_name", default="baseline_random_init")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
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
        random_init=True,  # poids aléatoires
    )

    log_config(cfg)
    train(cfg)
