"""
Experiment 1 — Stackelberg Attention Diversity LoRA
===================================================

Principle
---------
Multi-head attention heads are modeled as players in a Stackelberg game.
Head 0 is the "leader": it optimises purely for cross-entropy.
Heads 1..H-1 are "followers": they optimise for cross-entropy plus a
diversity penalty that discourages redundant attention patterns.

The bilevel (K=1 Stackelberg) update per optimizer step:

  Phase 1 — Accumulate follower gradients
      Forward → CE + diversity loss → backward (accumulated over micro-batches)

  Phase 2 — Leader looks ahead
      θ_F' = θ_F − η_sim · g_F          (simulated follower step, vanilla SGD)
      Forward with θ_F' → leader loss = CE(θ_L, θ_F')
      θ_L  ← Adam(θ_L, ∇_{θ_L} CE)     (leader update)

  Phase 3 — Follower update
      Restore θ_F, apply saved accumulated gradients via Adam

Parameter split (LoRA):
  Leader   = o_proj LoRA     (head output mixing)
  Follower = q/k/v_proj LoRA (attention computation)

Loss structure:
  Leader:   L_1 = L_CE
  Follower: L_i = L_CE + λ_lead·sim(A_i, A_0) + λ_peer·Σ_{j≠i,j≠0} sim(A_i, A_j)
  where sim = normalised Frobenius inner product of attention matrices.

Usage:
    python train_exp1.py --dry_run
    python train_exp1.py --design_layer 19 --lambda_lead 0.1 --lambda_peer 0.01
    python train_exp1.py --wandb_project my_project --run_name stackelberg_v1
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
sys.path.insert(0, _HERE)                   # current dir (for stackelberg_losses.py)

import torch
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

from train import (
    TrainConfig,
    PileStreamDataset,
    get_device,
    log_config,
    add_common_args,
)
from stackelberg_losses import (
    compute_diversity_loss,
    split_leader_follower_params,
    AttentionWeightCapture,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model builder (eager attention required for weight extraction)
# ---------------------------------------------------------------------------


def build_model_eager(cfg: TrainConfig):
    """Build model + LoRA with ``attn_implementation='eager'``."""
    logger.info(f"Chargement de {cfg.model_name} (eager attention) ...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=torch.bfloat16,
        attn_implementation="eager",
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
# Stackelberg training loop
# ---------------------------------------------------------------------------


def train_stackelberg(
    cfg: TrainConfig,
    design_layer: int = 19,
    lr_leader: float = 1e-4,
    lr_follower: float = 3e-4,
    lr_sim: float = 1e-3,
    lambda_lead: float = 0.1,
    lambda_peer: float = 0.01,
    leader_idx: int = 0,
):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = get_device()
    logger.info(f"Device: {device}")

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    # ── Model with eager attention ──
    model, tokenizer = build_model_eager(cfg)
    model = model.to(device)

    # ── Dataset & dataloader (reuse from train.py) ──
    max_tokens = 100 * cfg.seq_len * cfg.grad_accum if cfg.dry_run else cfg.total_tokens
    dataset = PileStreamDataset(tokenizer, cfg.seq_len, max_tokens, cfg.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    total_steps = 100 if cfg.dry_run else cfg.total_steps

    # ── Split parameters: leader (o_proj) vs followers (q/k/v_proj) ──
    leader_params, follower_params = split_leader_follower_params(model)
    logger.info(
        f"Leader params (o_proj LoRA): {sum(p.numel() for p in leader_params):,}"
    )
    logger.info(
        f"Follower params (q/k/v LoRA): {sum(p.numel() for p in follower_params):,}"
    )

    # ── Two separate Adam optimizers ──
    leader_optimizer = torch.optim.AdamW(
        leader_params,
        lr=lr_leader,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    follower_optimizer = torch.optim.AdamW(
        follower_params,
        lr=lr_follower,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )

    leader_scheduler = get_cosine_schedule_with_warmup(
        leader_optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )
    follower_scheduler = get_cosine_schedule_with_warmup(
        follower_optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Attention hook on design layer ──
    attn_capture = AttentionWeightCapture()
    attn_capture.register(model, layer_idx=design_layer)

    os.makedirs(cfg.output_dir, exist_ok=True)
    _exp_dir = os.path.dirname(os.path.abspath(cfg.output_dir))
    logs_dir  = os.path.join(_exp_dir, "logs")
    plots_dir = os.path.join(_exp_dir, "plots")
    os.makedirs(logs_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    loss_history: dict = {"step": [], "ce_loss": [], "ce_loss_ema": [], "div_loss": [], "leader_ce": []}
    _ema_ce = None
    _ema_alpha = 0.05

    # ── Training state ──
    model.train()
    global_step = 0
    accum_ce = 0.0
    accum_div = 0.0
    accum_leader_ce = 0.0
    leader_optimizer.zero_grad()
    follower_optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(total=total_steps, desc="Stackelberg Training", unit="step")

    last_batch = None  # keep last micro-batch for the bilevel leader forward

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        last_batch = (input_ids, labels)

        # ==============================================================
        # Phase 1: Forward — accumulate follower loss gradients
        # L_F = L_CE + L_diversity (scaled by 1/grad_accum)
        # ==============================================================
        with torch.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=torch.bfloat16,
            enabled=(device.type in ("cuda", "cpu")),
        ):
            out = model(input_ids=input_ids, labels=labels)
            ce_loss = out.loss

            # Attention weights captured by hook: (B, H, L, L)
            attn_w = attn_capture.get()

            if attn_w is not None:
                div_loss = compute_diversity_loss(
                    attn_w,
                    leader_idx=leader_idx,
                    lambda_lead=lambda_lead,
                    lambda_peer=lambda_peer,
                )
            else:
                div_loss = torch.tensor(0.0, device=device)

            follower_loss = (ce_loss + div_loss) / cfg.grad_accum

        # Backward — accumulates grads on ALL params
        follower_loss.backward()

        accum_ce += ce_loss.item() / cfg.grad_accum
        accum_div += div_loss.item() / cfg.grad_accum

        # ==============================================================
        # Optimizer step every grad_accum micro-batches
        # ==============================================================
        if (global_step + 1) % cfg.grad_accum == 0:
            opt_step = (global_step + 1) // cfg.grad_accum

            # ==========================================================
            # Phase 2: Stackelberg bilevel — leader looks ahead
            # ==========================================================

            # Step 2: Save follower accumulated gradients
            saved_follower_grads = [
                p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                for p in follower_params
            ]

            # Step 3: Simulated follower step (vanilla SGD, NO Adam/momentum)
            #   θ_F' = θ_F − η_sim · g_F
            saved_follower_data = [p.data.clone() for p in follower_params]
            with torch.no_grad():
                for p, g in zip(follower_params, saved_follower_grads):
                    p.data.sub_(lr_sim * g)

            # Step 4-5: Forward with adapted followers → leader loss = CE
            leader_optimizer.zero_grad()

            inp, lab = last_batch
            with torch.autocast(
                device_type=device.type if device.type != "mps" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type in ("cuda", "cpu")),
            ):
                out_leader = model(input_ids=inp, labels=lab)
                leader_ce = out_leader.loss
            # Discard captured attention weights from this forward
            attn_capture.get()

            # Step 6: Update leader via Adam
            leader_ce.backward()
            torch.nn.utils.clip_grad_norm_(leader_params, max_norm=1.0)
            leader_optimizer.step()
            leader_scheduler.step()

            accum_leader_ce = leader_ce.item()

            # ==========================================================
            # Phase 3: Restore followers & update with saved gradients
            # ==========================================================

            # Restore follower params to pre-simulated state
            with torch.no_grad():
                for p, saved in zip(follower_params, saved_follower_data):
                    p.data.copy_(saved)

            # Apply saved follower gradients
            follower_optimizer.zero_grad()
            for p, g in zip(follower_params, saved_follower_grads):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(follower_params, max_norm=1.0)
            follower_optimizer.step()
            follower_scheduler.step()

            # Clean up
            leader_optimizer.zero_grad()
            follower_optimizer.zero_grad()

            # EMA update (every optimizer step)
            _ema_ce = accum_ce if _ema_ce is None else _ema_alpha * accum_ce + (1 - _ema_alpha) * _ema_ce

            # ==========================================================
            # Logging
            # ==========================================================
            step_time = time.perf_counter() - _step_start
            tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
            _step_start = time.perf_counter()

            pbar.update(1)
            pbar.set_postfix(
                ce=f"{accum_ce:.4f}",
                ema=f"{_ema_ce:.4f}",
                div=f"{accum_div:.4f}",
                l_ce=f"{accum_leader_ce:.4f}",
                tok_s=f"{tokens_per_sec:,}",
            )

            if opt_step % cfg.log_every == 0:
                lr_l = leader_scheduler.get_last_lr()[0]
                lr_f = follower_scheduler.get_last_lr()[0]
                tqdm.write(
                    f"Step {opt_step:>6d}/{total_steps}"
                    f"  CE={accum_ce:.4f}"
                    f"  ema={_ema_ce:.4f}"
                    f"  div={accum_div:.4f}"
                    f"  leader_CE={accum_leader_ce:.4f}"
                    f"  lr_L={lr_l:.2e}  lr_F={lr_f:.2e}"
                    f"  tok/s={tokens_per_sec:,}"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/ce_loss": accum_ce,
                            "train/div_loss": accum_div,
                            "train/leader_ce": accum_leader_ce,
                            "train/lr_leader": lr_l,
                            "train/lr_follower": lr_f,
                            "train/step": opt_step,
                            "train/tokens": opt_step
                            * cfg.seq_len
                            * cfg.effective_batch_size,
                        },
                        step=opt_step,
                    )
                loss_history["step"].append(opt_step)
                loss_history["ce_loss"].append(accum_ce)
                loss_history["ce_loss_ema"].append(_ema_ce)
                loss_history["div_loss"].append(accum_div)
                loss_history["leader_ce"].append(accum_leader_ce)
                with open(os.path.join(logs_dir, "loss.json"), "w") as _f:
                    json.dump(loss_history, _f, indent=2)

            accum_ce = 0.0
            accum_div = 0.0
            accum_leader_ce = 0.0

            # Checkpoint
            if opt_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{opt_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                logger.info(f"Checkpoint sauvegardé → {ckpt_path}")

            if opt_step >= total_steps:
                break

        global_step += 1

    pbar.close()
    attn_capture.remove()

    # Final save
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modèle final sauvegardé → {final_path}")

    # Plot des losses
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history["step"], loss_history["ce_loss"], alpha=0.3, color="steelblue", label="ce_loss (raw)")
    ax.plot(loss_history["step"], loss_history["ce_loss_ema"], color="steelblue", label="ce_loss (EMA α=0.05)")
    for key in ("div_loss", "leader_ce"):
        ax.plot(loss_history["step"], loss_history[key], label=key, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training losses — exp1 (Stackelberg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Plot sauvegardé → {os.path.join(plots_dir, 'loss.png')}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stackelberg Attention Diversity Training"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    parser.add_argument("--run_name", default="stackelberg_lora")
    parser.add_argument(
        "--design_layer",
        type=int,
        default=19,
        help="Attention layer to hook for diversity loss",
    )
    parser.add_argument("--lr_leader", type=float, default=1e-4)
    parser.add_argument("--lr_follower", type=float, default=3e-4)
    parser.add_argument(
        "--lr_sim",
        type=float,
        default=1e-3,
        help="LR for simulated follower step (vanilla SGD, no momentum)",
    )
    parser.add_argument("--lambda_lead", type=float, default=0.1)
    parser.add_argument("--lambda_peer", type=float, default=0.01)
    parser.add_argument("--leader_idx", type=int, default=0)
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
    )

    log_config(cfg)
    logger.info(f"  Design layer  : {args.design_layer}")
    logger.info(f"  LR leader     : {args.lr_leader}")
    logger.info(f"  LR follower   : {args.lr_follower}")
    logger.info(f"  LR sim step   : {args.lr_sim}")
    logger.info(f"  λ_lead        : {args.lambda_lead}")
    logger.info(f"  λ_peer        : {args.lambda_peer}")
    logger.info(f"  Leader head   : {args.leader_idx}")

    train_stackelberg(
        cfg,
        design_layer=args.design_layer,
        lr_leader=args.lr_leader,
        lr_follower=args.lr_follower,
        lr_sim=args.lr_sim,
        lambda_lead=args.lambda_lead,
        lambda_peer=args.lambda_peer,
        leader_idx=args.leader_idx,
    )
