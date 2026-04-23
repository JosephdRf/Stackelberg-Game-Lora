"""
Experiment 1 — Stackelberg Attention Diversity LoRA (Pythia-160M)
==================================================================

Principle
---------
Multi-head attention heads are modeled as players in a Stackelberg game.
The leader (dense / output projection LoRA) optimises purely for cross-entropy.
The followers (query_key_value LoRA) optimise for cross-entropy plus a
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
  Leader   = dense LoRA           (head output mixing, output projection)
  Follower = query_key_value LoRA (attention computation, fused QKV)

Loss structure (current):
  Leader:   L_1 = L_CE
  Follower: L_i = L_CE                              (λ_lead = λ_peer = 0 for now)

  → Set --lambda_lead / --lambda_peer > 0 to activate the diversity penalty:
  Follower: L_i = L_CE + λ_lead·sim(A_i, A_0) + λ_peer·Σ_{j≠i,j≠0} sim(A_i, A_j)

Usage:
    python pythia160M/exp1/train_exp1.py --dry_run
    python pythia160M/exp1/train_exp1.py --design_layer 9
    python pythia160M/exp1/train_exp1.py --wandb_project my_project --run_name stackelberg_v1
    python pythia160M/exp1/train_exp1.py --lambda_lead 0.1 --lambda_peer 0.01
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

_HERE  = os.path.dirname(os.path.abspath(__file__))   # pythia160M/exp1/
_MODEL = os.path.dirname(_HERE)                        # pythia160M/
sys.path.insert(0, _MODEL)   # pour importer train_utils.py
sys.path.insert(0, _HERE)    # pour importer stackelberg_losses.py

import torch
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader

from train_utils import (
    TrainConfig,
    WikiTextDataset,
    make_generator,
    get_device,
    log_config,
    add_common_args,
    seed_everything,
    evaluate,
    log_head_matrices,
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
    """Build Pythia-160M + LoRA with attn_implementation='eager'."""
    logger.info(f"Chargement de {cfg.model_name} (eager attention) ...")

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
    design_layer: int = 9,
    lr_leader: float = 1e-4,
    lr_follower: float = 3e-4,
    lr_sim: float = 1e-3,
    lambda_lead: float = 0.0,
    lambda_peer: float = 0.0,
    leader_idx: int = 0,
):
    seed_everything(cfg.seed)

    device = get_device()
    logger.info(f"Device : {device}")

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))

    # ── Model with eager attention ──
    model, tokenizer = build_model_eager(cfg)
    model = model.to(device)

    # ── Datasets & dataloaders ──
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
    total_steps = 100 if cfg.dry_run else cfg.total_steps

    # ── Split parameters: leader (dense) vs followers (query_key_value) ──
    leader_params, follower_params = split_leader_follower_params(model)
    logger.info(f"Leader params (dense LoRA)          : {sum(p.numel() for p in leader_params):,}")
    logger.info(f"Follower params (query_key_value LoRA): {sum(p.numel() for p in follower_params):,}")
    logger.info(f"Design layer : {design_layer}  |  Leader head idx : {leader_idx}")
    logger.info(f"λ_lead={lambda_lead}  λ_peer={lambda_peer}"
                + ("  (CE only — diversity inactive)" if lambda_lead == 0 and lambda_peer == 0 else ""))

    # ── Two separate Adam optimizers ──
    leader_optimizer = torch.optim.AdamW(
        leader_params,
        lr=lr_leader,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    follower_optimizer = torch.optim.AdamW(
        follower_params,
        lr=lr_follower,
        betas=cfg.betas,
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

    # ── Attention hook on design layer (only needed when diversity is active) ──
    need_attn = lambda_lead > 0 or lambda_peer > 0
    attn_capture = AttentionWeightCapture()
    if need_attn:
        attn_capture.register(model, layer_idx=design_layer)

    # ── Directories & history ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    _exp_dir  = os.path.dirname(os.path.abspath(cfg.output_dir))
    logs_dir  = os.path.join(_exp_dir, "logs")
    plots_dir = os.path.join(_exp_dir, "plots")
    os.makedirs(logs_dir,  exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    history = {
        "train": {"step": [], "ce": [], "ce_ema": [], "div": [], "leader_ce": []},
        "val":   {"step": [], "loss": [], "ppl": []},
    }
    _ema_ce    = None
    _ema_alpha = 0.05

    # ── Eval initiale ──
    logger.info("Eval initiale ...")
    v_loss, v_ppl = evaluate(model, val_loader, device, max_batches=cfg.eval_max_batches)
    history["val"]["step"].append(0)
    history["val"]["loss"].append(v_loss)
    history["val"]["ppl"].append(v_ppl)
    logger.info(f"  [step 0] val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
    if use_wandb:
        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=0)

    # ── Training state ──
    model.train()
    global_step    = 0
    accum_ce       = 0.0
    accum_div      = 0.0
    accum_leader_ce = 0.0
    leader_optimizer.zero_grad()
    follower_optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(total=total_steps, desc="Stackelberg Training (Pythia-160M)", unit="step")

    last_batch = None

    done = False
    while not done:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
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
                out = model(
                    input_ids=input_ids,
                    labels=labels,
                    output_attentions=need_attn,
                )
                ce_loss = out.loss

                if need_attn:
                    attn_w = attn_capture.get()
                    div_loss = (
                        compute_diversity_loss(
                            attn_w,
                            leader_idx=leader_idx,
                            lambda_lead=lambda_lead,
                            lambda_peer=lambda_peer,
                        )
                        if attn_w is not None
                        else torch.tensor(0.0, device=device)
                    )
                else:
                    div_loss = torch.tensor(0.0, device=device)

                follower_loss = (ce_loss + div_loss) / cfg.grad_accum

            follower_loss.backward()
            accum_ce  += ce_loss.item() / cfg.grad_accum
            accum_div += div_loss.item() / cfg.grad_accum

            # ==============================================================
            # Optimizer step every grad_accum micro-batches
            # ==============================================================
            if (global_step + 1) % cfg.grad_accum == 0:
                opt_step = (global_step + 1) // cfg.grad_accum

                # ==========================================================
                # Phase 2: Stackelberg bilevel — leader looks ahead
                # ==========================================================

                # Save follower accumulated gradients
                saved_follower_grads = [
                    p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                    for p in follower_params
                ]

                # Simulated follower step (vanilla SGD, no Adam/momentum)
                #   θ_F' = θ_F − η_sim · g_F
                saved_follower_data = [p.data.clone() for p in follower_params]
                with torch.no_grad():
                    for p, g in zip(follower_params, saved_follower_grads):
                        p.data.sub_(lr_sim * g)

                # Forward with adapted followers → leader loss = CE
                leader_optimizer.zero_grad()
                inp, lab = last_batch
                with torch.autocast(
                    device_type=device.type if device.type != "mps" else "cpu",
                    dtype=torch.bfloat16,
                    enabled=(device.type in ("cuda", "cpu")),
                ):
                    out_leader = model(input_ids=inp, labels=lab)
                    leader_ce  = out_leader.loss
                if need_attn:
                    attn_capture.get()  # discard

                leader_ce.backward()
                torch.nn.utils.clip_grad_norm_(leader_params, max_norm=cfg.grad_clip)
                leader_optimizer.step()
                leader_scheduler.step()
                accum_leader_ce = leader_ce.item()

                # ==========================================================
                # Phase 3: Restore followers & apply saved gradients
                # ==========================================================
                with torch.no_grad():
                    for p, saved in zip(follower_params, saved_follower_data):
                        p.data.copy_(saved)

                follower_optimizer.zero_grad()
                for p, g in zip(follower_params, saved_follower_grads):
                    p.grad = g
                torch.nn.utils.clip_grad_norm_(follower_params, max_norm=cfg.grad_clip)
                follower_optimizer.step()
                follower_scheduler.step()

                leader_optimizer.zero_grad()
                follower_optimizer.zero_grad()

                # EMA
                _ema_ce = (
                    accum_ce if _ema_ce is None
                    else _ema_alpha * accum_ce + (1 - _ema_alpha) * _ema_ce
                )

                step_time      = time.perf_counter() - _step_start
                tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
                _step_start    = time.perf_counter()

                pbar.update(1)
                pbar.set_postfix(
                    ce=f"{accum_ce:.4f}",
                    ema=f"{_ema_ce:.4f}",
                    div=f"{accum_div:.4f}",
                    l_ce=f"{accum_leader_ce:.4f}",
                    tok_s=f"{tokens_per_sec:,}",
                )

                # ── Log train ──
                if opt_step % cfg.log_every == 0 or opt_step == 1:
                    lr_l = leader_scheduler.get_last_lr()[0]
                    lr_f = follower_scheduler.get_last_lr()[0]
                    logger.info(
                        f"[train] step {opt_step:>6d}/{total_steps}"
                        f"  CE={accum_ce:.4f}  ema={_ema_ce:.4f}"
                        f"  div={accum_div:.4f}  leader_CE={accum_leader_ce:.4f}"
                        f"  lr_L={lr_l:.2e}  lr_F={lr_f:.2e}"
                        f"  tok/s={tokens_per_sec:,}"
                    )
                    if use_wandb:
                        wandb.log({
                            "train/ce_loss":    accum_ce,
                            "train/ce_ema":     _ema_ce,
                            "train/div_loss":   accum_div,
                            "train/leader_ce":  accum_leader_ce,
                            "train/lr_leader":  lr_l,
                            "train/lr_follower": lr_f,
                            "train/tokens":     opt_step * cfg.seq_len * cfg.effective_batch_size,
                        }, step=opt_step)
                    history["train"]["step"].append(opt_step)
                    history["train"]["ce"].append(accum_ce)
                    history["train"]["ce_ema"].append(_ema_ce)
                    history["train"]["div"].append(accum_div)
                    history["train"]["leader_ce"].append(accum_leader_ce)

                # ── Eval périodique ──
                if opt_step % cfg.eval_every == 0:
                    v_loss, v_ppl = evaluate(
                        model, val_loader, device,
                        max_batches=cfg.eval_max_batches,
                        autocast_dtype=torch.bfloat16,
                    )
                    logger.info(f"[val]   step {opt_step:>6d}  val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
                    log_head_matrices(
                        model, device, design_layer, opt_step, val_loader,
                        wandb_mod=wandb if use_wandb else None,
                    )
                    if use_wandb:
                        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=opt_step)
                    history["val"]["step"].append(opt_step)
                    history["val"]["loss"].append(v_loss)
                    history["val"]["ppl"].append(v_ppl)

                # ── JSON ──
                if (opt_step % cfg.log_every == 0
                        or opt_step % cfg.eval_every == 0
                        or opt_step == 1):
                    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
                        json.dump(history, _f, indent=2)

                accum_ce        = 0.0
                accum_div       = 0.0
                accum_leader_ce = 0.0

                # ── Checkpoint ──
                if opt_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{opt_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    logger.info(f"Checkpoint → {ckpt_path}")

                if opt_step >= total_steps:
                    done = True
                    break

            global_step += 1

        if not done:
            logger.info("Fin d'epoch — on recommence un passage sur le dataset.")

    pbar.close()
    attn_capture.remove()

    # ── Eval finale ──
    logger.info("Eval finale ...")
    v_loss, v_ppl = evaluate(model, val_loader, device, max_batches=cfg.eval_max_batches)
    history["val"]["step"].append(opt_step)
    history["val"]["loss"].append(v_loss)
    history["val"]["ppl"].append(v_ppl)
    logger.info(f"  [final] val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
    if use_wandb:
        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=opt_step)

    # ── Sauvegarde finale ──
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modèle final sauvegardé → {final_path}")

    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
        json.dump(history, _f, indent=2)

    # ── Plots ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train"]["step"], history["train"]["ce"],
            alpha=0.25, color="darkorange", label="CE (raw)")
    ax.plot(history["train"]["step"], history["train"]["ce_ema"],
            color="darkorange", label="CE (EMA)")
    ax.plot(history["train"]["step"], history["train"]["leader_ce"],
            color="purple", alpha=0.6, label="leader CE (lookahead)")
    if any(v > 0 for v in history["train"]["div"]):
        ax.plot(history["train"]["step"], history["train"]["div"],
                color="green", alpha=0.6, label="diversity loss")
    ax.plot(history["val"]["step"], history["val"]["loss"],
            color="steelblue", marker="o", markersize=4, label="val loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Loss")
    ax.set_title("Training — Pythia-160M Stackelberg exp1")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["val"]["step"], history["val"]["ppl"],
            color="steelblue", marker="o", markersize=4)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity — WikiText-103 (Stackelberg exp1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_ppl.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots → {plots_dir}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stackelberg Attention Diversity Training — Pythia-160M exp1"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--run_name",   default="stackelberg_exp1_pythia")
    parser.add_argument(
        "--design_layer", type=int, default=9,
        help="Attention layer for diversity loss and head matrix logging "
             "(Pythia-160M has 12 layers; layer 9 ≈ 79%%)",
    )
    parser.add_argument("--lr_leader",   type=float, default=1e-4)
    parser.add_argument("--lr_follower", type=float, default=3e-4)
    parser.add_argument(
        "--lr_sim", type=float, default=1e-3,
        help="LR for simulated follower step (vanilla SGD, no momentum)",
    )
    parser.add_argument(
        "--lambda_lead", type=float, default=0.0,
        help="Penalty weight for leader-follower similarity (0 = CE only)",
    )
    parser.add_argument(
        "--lambda_peer", type=float, default=0.0,
        help="Penalty weight for peer-follower similarity (0 = CE only)",
    )
    parser.add_argument("--leader_idx", type=int, default=0,
                        help="Index of the leader head")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
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
    logger.info(f"  Design layer  : {args.design_layer}")
    logger.info(f"  LR leader     : {args.lr_leader}")
    logger.info(f"  LR follower   : {args.lr_follower}")
    logger.info(f"  LR sim step   : {args.lr_sim}")
    logger.info(f"  λ_lead        : {args.lambda_lead}")
    logger.info(f"  λ_peer        : {args.lambda_peer}")
    logger.info(f"  Leader head   : {args.leader_idx}")

    train_stackelberg(
        cfg,
        design_layer = args.design_layer,
        lr_leader    = args.lr_leader,
        lr_follower  = args.lr_follower,
        lr_sim       = args.lr_sim,
        lambda_lead  = args.lambda_lead,
        lambda_peer  = args.lambda_peer,
        leader_idx   = args.leader_idx,
    )
