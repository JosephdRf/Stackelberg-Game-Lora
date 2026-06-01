"""
Experiment 1 — Stackelberg LoRA (Qwen2.5-0.5B)
================================================

Port direct de pythia160M/exp1/train_exp1.py adapté à Qwen2 (GQA).

Principle
---------
One attention layer (design_layer, default 19) is treated as a Stackelberg game.
Leader heads (default = 7 premiers Q-heads = un groupe GQA complet) committent
en premier en anticipant la meilleure réponse des followers ; les followers
(les 7 autres Q-heads + KV head non-leader) best-respond. Tous minimisent
la cross-entropy.

Bilevel (K=1 Stackelberg) update par optimizer step :

  Phase 1 — Accumulate follower gradients
      Forward → L_F = L_CE / grad_accum → backward (micro-batches)
      mask_follower_grad() zéro les tranches leader dans p.grad
      Save g_follower = {id(p): p.grad.clone()}

  Phase 2 — Leader lookahead
      θ_F' = θ_F − η_sim · clip(g_F)
      Forward at θ_F' → L_leader = L_CE → backward sur tous les micro-batches accumulés
      mask_leader_grad() : garde uniquement les tranches θ_L, zéro tout le reste
      Save g_leader = {id(p): p.grad.clone()}

  Phase 3 — Restore, assemble, step
      Restore θ_F ← saved values
      assemble_gradients(g_follower, g_leader) :
        θ_L ∪ θ_F (q_lora_B, k_lora_B, v_lora_B, o_lora_A) : g_F + g_L
        θ_S (q_lora_A, k_lora_A, v_lora_A, o_lora_B, other) : g_F
      clip_grad_norm → optimizer.step()

Parameter split (design layer LoRA, Qwen2 GQA) :
  θ_L (tranches leader) :
    q_proj.lora_B   : rows [0:448]            (Q-heads 0..6)
    k_proj.lora_B   : rows [0:64]             (KV-head 0)
    v_proj.lora_B   : rows [0:64]             (KV-head 0)
    o_proj.lora_A   : cols [:, 0:448]         (input cols Q-heads 0..6)
  θ_F (tranches follower, même tenseur) :
    q_proj.lora_B   : rows [448:896]
    k_proj.lora_B   : rows [64:128]
    v_proj.lora_B   : rows [64:128]
    o_proj.lora_A   : cols [:, 448:896]
  θ_S (shared, follower-only update) :
    q_proj.lora_A, k_proj.lora_A, v_proj.lora_A
    o_proj.lora_B
    autres layers (other)

Usage :
    python qwen2.5_0.5B/exp1/train_exp1.py --dry_run
    python qwen2.5_0.5B/exp1/train_exp1.py --design_layer 19
    python qwen2.5_0.5B/exp1/train_exp1.py --wandb_project my_project --run_name stackelberg_qwen_v1
"""

import os
import sys
import glob
import json
import argparse
import logging
import time
import dataclasses
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))   # qwen2.5_0.5B/exp1/
_MODEL = os.path.dirname(_HERE)                       # qwen2.5_0.5B/
sys.path.insert(0, _MODEL)  # train_utils.py, gradient_mask.py, stackelberg_losses.py

import torch
import numpy as np
from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup
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
    build_model_and_tokenizer,
    QWEN_HIDDEN_SIZE,
    QWEN_N_Q_HEADS,
    QWEN_N_KV_HEADS,
    QWEN_D_HEAD,
)
from gradient_mask import (
    collect_lora_params,
    mask_follower_grad,
    mask_leader_grad,
    assemble_gradients,
)
from stackelberg_losses import ldb_loss, head_interaction_matrix

logger = logging.getLogger(__name__)


def _parse_int_list(s: str) -> list:
    """Parse "0,1,2,3,4,5,6" → [0, 1, 2, 3, 4, 5, 6]."""
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Stackelberg training loop
# ---------------------------------------------------------------------------


def train_stackelberg(
    cfg: TrainConfig,
    design_layer: int = 19,
    lr_leader: float = 1e-4,
    lr_follower: float = 3e-4,
    lr_sim: float = 1e-3,
    leader_q_heads: list = None,
    leader_kv_heads: list = None,
    lambda_ldb: float = 0.0,
    keep_wandb_open: bool = False,
):
    if leader_q_heads is None:
        leader_q_heads = list(range(7))
    if leader_kv_heads is None:
        leader_kv_heads = [0]

    seed_everything(cfg.seed)

    device = get_device()
    logger.info(f"Device : {device}")

    use_wandb = cfg.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.run_name,
            group=cfg.wandb_group,
            config={
                **vars(cfg),
                "lr_leader": lr_leader,
                "lr_follower": lr_follower,
                "lr_sim": lr_sim,
                "design_layer": design_layer,
                "leader_q_heads": leader_q_heads,
                "leader_kv_heads": leader_kv_heads,
                "lambda_ldb": lambda_ldb,
            },
        )

    # ── Model (float32, SDPA) ──
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = model.to(device)
    if device.type == "cuda":
        logger.info(
            f"VRAM après chargement modèle : {torch.cuda.memory_allocated() / 1e9:.2f} GB "
            f"/ {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total"
        )

    # ── Datasets & dataloaders ──
    max_train_tokens = (
        100 * cfg.seq_len * cfg.effective_batch_size if cfg.dry_run else None
    )
    train_ds = WikiTextDataset(
        tokenizer,
        cfg.seq_len,
        split="train",
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        max_tokens=max_train_tokens,
    )
    val_ds = WikiTextDataset(
        tokenizer,
        cfg.seq_len,
        split="validation",
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
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

    # ── Collect params with leader/follower assembly ──
    all_params, grad_assembly = collect_lora_params(
        model,
        design_layers=[design_layer],
        leader_q_heads=leader_q_heads,
        leader_kv_heads=leader_kv_heads,
        n_q_heads=QWEN_N_Q_HEADS,
        n_kv_heads=QWEN_N_KV_HEADS,
        d_head=QWEN_D_HEAD,
        hidden_size=QWEN_HIDDEN_SIZE,
    )

    # θ_L ∪ θ_F : tranches décomposables
    _decomposable_kinds = ("q_lora_B", "k_lora_B", "v_lora_B", "o_lora_A")
    n_design_params = sum(
        p.numel()
        for r in grad_assembly.roles
        if r.kind in _decomposable_kinds
        for p in [r.param]
    )
    # θ_S : shared (lora_A de q/k/v + lora_B de o) + autres layers
    _shared_kinds = ("q_lora_A", "k_lora_A", "v_lora_A", "o_lora_B", "other")
    n_shared_params = sum(
        p.numel()
        for r in grad_assembly.roles
        if r.kind in _shared_kinds
        for p in [r.param]
    )
    logger.info(f"Total trainable params : {sum(p.numel() for p in all_params):,}")
    logger.info(
        f"Batch size / GPU       : {cfg.batch_size_per_gpu}  |  Grad accum : {cfg.grad_accum}"
        f"  |  Effective batch : {cfg.effective_batch_size}"
    )
    logger.info(
        f"Design layer           : {design_layer}  "
        f"|  Leader Q-heads : {leader_q_heads}  |  Leader KV-heads : {leader_kv_heads}"
    )
    logger.info(
        f"Design params (θ_L∪θ_F) : {n_design_params:,}  |  Shared+other (θ_S) : {n_shared_params:,}"
    )

    # leader-only paramètres = ceux dont les tranches leader sont les SEULES updates leader.
    # Pour exp1 Qwen, le seul param entièrement "leader-LR" est o_proj.lora_A (cols leader).
    # Comme on partage un tenseur entre θ_L et θ_F, on accepte la simplification : tout le
    # tenseur o_proj.lora_A utilise lr_leader, le reste lr_follower.
    leader_param_ids = {
        id(r.param) for r in grad_assembly.roles if r.kind == "o_lora_A"
    }
    param_groups = [
        {
            "params": [p for p in all_params if id(p) not in leader_param_ids],
            "lr": lr_follower,
            "name": "follower_and_shared",
        },
        {
            "params": [p for p in all_params if id(p) in leader_param_ids],
            "lr": lr_leader,
            "name": "leader",
        },
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    # ── LDB config ──
    _d_head_ldb = QWEN_D_HEAD
    logger.info(f"λ_ldb={lambda_ldb}")

    # ── Directories & history ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    logs_dir = os.path.join(cfg.output_dir, "logs")
    plots_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    history = {
        "train": {"step": [], "ce": [], "ce_ema": [], "leader_ce": [], "ldb": []},
        "val": {"step": [], "loss": [], "ppl": []},
    }
    _ema_ce = None
    _ema_alpha = 0.05

    # ── Eval initiale ──
    logger.info("Eval initiale ...")
    v_loss, v_ppl = evaluate(
        model, val_loader, device, max_batches=cfg.eval_max_batches
    )
    history["val"]["step"].append(0)
    history["val"]["loss"].append(v_loss)
    history["val"]["ppl"].append(v_ppl)
    logger.info(f"  [step 0] val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
    if use_wandb:
        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=0)

    # ── Training state ──
    model.train()
    global_step = 0
    accum_ce = 0.0
    accum_leader_ce = 0.0
    accum_ldb = 0.0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(
        total=total_steps, desc="Stackelberg Training (Qwen2.5-0.5B)", unit="step",
        disable=not sys.stderr.isatty(),
    )

    accum_inputs: list = []
    accum_labels: list = []

    done = False
    while not done:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            accum_inputs.append(input_ids)
            accum_labels.append(labels)

            # ==================================================================
            # Phase 1: Follower forward — accumulate follower gradients
            # ==================================================================
            with torch.autocast(
                device_type=device.type if device.type != "mps" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type in ("cuda", "cpu")),
            ):
                out = model(input_ids=input_ids, labels=labels)
                ce_loss = out.loss

            follower_loss = ce_loss / cfg.grad_accum

            if lambda_ldb > 0:
                G = head_interaction_matrix(
                    model, out.logits, labels, design_layer, _d_head_ldb,
                )
                ldb_raw = lambda_ldb * ldb_loss(G)
                follower_loss = follower_loss + ldb_raw / cfg.grad_accum
                accum_ldb += ldb_raw.item()

            follower_loss.backward()
            accum_ce += ce_loss.item() / cfg.grad_accum

            if global_step == 0 and device.type == "cuda":
                logger.info(
                    f"VRAM pic step 0 (batch={cfg.batch_size_per_gpu}, seq={cfg.seq_len}) : "
                    f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                    f"/ {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB total"
                )

            # ==================================================================
            # Optimizer step every grad_accum micro-batches
            # ==================================================================
            if (global_step + 1) % cfg.grad_accum == 0:
                opt_step = (global_step + 1) // cfg.grad_accum

                with torch.no_grad():
                    mask_follower_grad(grad_assembly)

                g_follower = {
                    id(r.param): r.param.grad.clone()
                    if r.param.grad is not None
                    else torch.zeros_like(r.param)
                    for r in grad_assembly.roles
                }

                # ==============================================================
                # Phase 2: Leader lookahead
                # ==============================================================

                sim_gnorm = torch.norm(
                    torch.stack(
                        [g.norm() for g in g_follower.values() if g.numel() > 0]
                    )
                )
                sim_clip = min(1.0, cfg.grad_clip / (sim_gnorm.item() + 1e-8))

                design_roles = [
                    r
                    for r in grad_assembly.roles
                    if r.kind in _decomposable_kinds
                ]
                saved_data = {id(r.param): r.param.data.clone() for r in design_roles}
                with torch.no_grad():
                    for r in design_roles:
                        gf = g_follower[id(r.param)]
                        r.param.data.sub_(lr_sim * gf * sim_clip)

                # Leader forward over all accumulated micro-batches
                optimizer.zero_grad()
                leader_ce_accum = torch.tensor(0.0, device=device)
                for inp, lab in zip(accum_inputs, accum_labels):
                    with torch.autocast(
                        device_type=device.type if device.type != "mps" else "cpu",
                        dtype=torch.bfloat16,
                        enabled=(device.type in ("cuda", "cpu")),
                    ):
                        out_leader = model(input_ids=inp, labels=lab)
                        leader_ce_mb = out_leader.loss / cfg.grad_accum
                    if lambda_ldb > 0:
                        G_l = head_interaction_matrix(
                            model, out_leader.logits, lab, design_layer, _d_head_ldb,
                        )
                        ldb_raw_l = lambda_ldb * ldb_loss(G_l)
                        leader_ce_mb = leader_ce_mb + ldb_raw_l / cfg.grad_accum
                    leader_ce_mb.backward()
                    leader_ce_accum = leader_ce_accum + leader_ce_mb.detach()

                accum_leader_ce = leader_ce_accum.item()

                with torch.no_grad():
                    mask_leader_grad(grad_assembly)

                g_leader = {
                    id(r.param): r.param.grad.clone()
                    if r.param.grad is not None
                    else torch.zeros_like(r.param)
                    for r in grad_assembly.roles
                }

                # ==============================================================
                # Phase 3: Restore, assemble, step
                # ==============================================================
                with torch.no_grad():
                    for r in design_roles:
                        r.param.data.copy_(saved_data[id(r.param)])

                with torch.no_grad():
                    assemble_gradients(grad_assembly, g_follower, g_leader)

                torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA
                _ema_ce = (
                    accum_ce
                    if _ema_ce is None
                    else _ema_alpha * accum_ce + (1 - _ema_alpha) * _ema_ce
                )

                step_time = time.perf_counter() - _step_start
                tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
                _step_start = time.perf_counter()

                pbar.update(1)
                pbar.set_postfix(
                    ce=f"{accum_ce:.4f}",
                    ema=f"{_ema_ce:.4f}",
                    l_ce=f"{accum_leader_ce:.4f}",
                    ldb=f"{accum_ldb:.4f}",
                    tok_s=f"{tokens_per_sec:,}",
                )

                # ── Log train ──
                if opt_step % cfg.log_every == 0 or opt_step == 1:
                    lr_f = scheduler.get_last_lr()[0]
                    lr_l = (
                        scheduler.get_last_lr()[1]
                        if len(scheduler.get_last_lr()) > 1
                        else lr_f
                    )
                    logger.info(
                        f"[train] step {opt_step:>6d}/{total_steps}"
                        f"  CE={accum_ce:.4f}  ema={_ema_ce:.4f}"
                        f"  leader_CE={accum_leader_ce:.4f}"
                        f"  ldb={accum_ldb:.4f}"
                        f"  lr_L={lr_l:.2e}  lr_F={lr_f:.2e}"
                        f"  tok/s={tokens_per_sec:,}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/ce_loss": accum_ce,
                                "train/ce_ema": _ema_ce,
                                "train/leader_ce": accum_leader_ce,
                                "train/ldb_loss": accum_ldb,
                                "train/lr_leader": lr_l,
                                "train/lr_follower": lr_f,
                                "train/tokens": opt_step
                                * cfg.seq_len
                                * cfg.effective_batch_size,
                            },
                            step=opt_step,
                        )
                    history["train"]["step"].append(opt_step)
                    history["train"]["ce"].append(accum_ce)
                    history["train"]["ce_ema"].append(_ema_ce)
                    history["train"]["leader_ce"].append(accum_leader_ce)
                    history["train"]["ldb"].append(accum_ldb)

                # ── Eval périodique ──
                if opt_step % cfg.eval_every == 0:
                    v_loss, v_ppl = evaluate(
                        model,
                        val_loader,
                        device,
                        max_batches=cfg.eval_max_batches,
                        autocast_dtype=torch.bfloat16,
                    )
                    logger.info(
                        f"[val]   step {opt_step:>6d}  val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}"
                    )
                    log_head_matrices(
                        model,
                        device,
                        design_layer,
                        opt_step,
                        val_loader,
                        wandb_mod=wandb if use_wandb else None,
                        log_image=(opt_step % (cfg.eval_every * 5) == 0),
                    )
                    if use_wandb:
                        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=opt_step)
                    history["val"]["step"].append(opt_step)
                    history["val"]["loss"].append(v_loss)
                    history["val"]["ppl"].append(v_ppl)

                if (
                    opt_step % cfg.log_every == 0
                    or opt_step % cfg.eval_every == 0
                    or opt_step == 1
                ):
                    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
                        json.dump(history, _f, indent=2)

                accum_ce = 0.0
                accum_leader_ce = 0.0
                accum_ldb = 0.0
                accum_inputs = []
                accum_labels = []

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

    # ── Eval finale ──
    logger.info("Eval finale ...")
    v_loss, v_ppl = evaluate(
        model, val_loader, device, max_batches=cfg.eval_max_batches
    )
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
    ax.plot(
        history["train"]["step"],
        history["train"]["ce"],
        alpha=0.25,
        color="darkorange",
        label="CE (raw)",
    )
    ax.plot(
        history["train"]["step"],
        history["train"]["ce_ema"],
        color="darkorange",
        label="CE (EMA)",
    )
    ax.plot(
        history["train"]["step"],
        history["train"]["leader_ce"],
        color="purple",
        alpha=0.6,
        label="leader CE (lookahead)",
    )
    ax.plot(
        history["val"]["step"],
        history["val"]["loss"],
        color="steelblue",
        marker="o",
        markersize=4,
        label="val loss",
    )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Loss")
    ax.set_title("Training — Qwen2.5-0.5B Stackelberg exp1")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        history["val"]["step"],
        history["val"]["ppl"],
        color="steelblue",
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("Validation perplexity — WikiText-103 (Stackelberg exp1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "val_ppl.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots → {plots_dir}")

    if use_wandb and not keep_wandb_open:
        with open(os.path.join(cfg.output_dir, "wandb_run_id.txt"), "w") as _f:
            _f.write(wandb.run.id)
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stackelberg LoRA Training — Qwen2.5-0.5B exp1"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--run_name", default="stackelberg_exp1_qwen")
    parser.add_argument(
        "--design_layer",
        type=int,
        default=19,
        help="Attention layer pour le jeu Stackelberg (Qwen-0.5B a 24 layers ; "
             "layer 19 ≈ 79%% du réseau, équivalent au layer 9/12 de Pythia)",
    )
    parser.add_argument("--lr_leader", type=float, default=1e-4)
    parser.add_argument("--lr_follower", type=float, default=3e-4)
    parser.add_argument(
        "--lr_sim",
        type=float,
        default=1e-3,
        help="LR pour le simulated follower step (vanilla SGD, no momentum)",
    )
    parser.add_argument(
        "--leader_q_heads",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Liste séparée par des virgules d'indices Q-heads leader "
             "(default = 7 premiers heads = 1 groupe GQA complet)",
    )
    parser.add_argument(
        "--leader_kv_heads",
        type=str,
        default="0",
        help="Liste des indices KV-heads leader (default = '0', "
             "la KV head correspondant aux Q-heads leader)",
    )
    parser.add_argument(
        "--nb_runs", type=int, default=3,
        help="Nombre d'entraînements consécutifs (seeds seed, seed+1, …).",
    )
    parser.add_argument(
        "--run_eval", action="store_true", default=True,
        help="Lancer l'évaluation après l'entraînement et logger les métriques.",
    )
    parser.add_argument(
        "--lambda_ldb", type=float, default=0.0,
        help="Weight for the log-determinant barrier loss (0 = disabled).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    leader_q = _parse_int_list(args.leader_q_heads)
    leader_kv = _parse_int_list(args.leader_kv_heads)

    cfg = TrainConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        total_tokens=args.total_tokens,
        batch_size_per_gpu=args.batch_size_per_gpu,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        run_name=args.run_name,
        seed=args.seed,
        dry_run=args.dry_run,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches,
        save_every=args.save_every,
        num_workers=args.num_workers,
        random_init=args.random_init,
    )

    log_config(cfg)
    logger.info(f"  Design layer    : {args.design_layer}")
    logger.info(f"  Leader Q-heads  : {leader_q}")
    logger.info(f"  Leader KV-heads : {leader_kv}")
    logger.info(f"  LR leader       : {args.lr_leader}")
    logger.info(f"  LR follower     : {args.lr_follower}")
    logger.info(f"  LR sim step     : {args.lr_sim}")
    logger.info(f"  Nb runs         : {args.nb_runs}")
    logger.info(f"  λ_ldb           : {args.lambda_ldb}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        logger.info(f"Anciens checkpoints supprimés : {args.output_dir}")

    _run_durations = []
    _train_wall_start = time.perf_counter()
    for i in range(args.nb_runs):
        cfg_i = dataclasses.replace(
            cfg,
            output_dir=os.path.join(args.output_dir, f"run_{i}"),
            run_name=args.run_name,
            seed=args.seed + i,
            wandb_project=cfg.wandb_project if i == 0 else None,
        )
        logger.info(
            f"\n{'='*60}\n"
            f"Run {i+1}/{args.nb_runs}  seed={cfg_i.seed}  output={cfg_i.output_dir}\n"
            f"{'='*60}"
        )
        keep_open = args.run_eval and i == 0 and cfg.wandb_project is not None
        _t0 = time.perf_counter()
        train_stackelberg(
            cfg_i,
            design_layer=args.design_layer,
            lr_leader=args.lr_leader,
            lr_follower=args.lr_follower,
            lr_sim=args.lr_sim,
            leader_q_heads=leader_q,
            leader_kv_heads=leader_kv,
            lambda_ldb=args.lambda_ldb,
            keep_wandb_open=keep_open,
        )
        _run_durations.append(time.perf_counter() - _t0)
        logger.info(f"  Run {i} duration : {_run_durations[-1]/60:.1f} min")

    if args.run_eval:
        from eval import run_eval, load_model, METRIC_ORDER

        run_dirs = sorted(glob.glob(os.path.join(args.output_dir, "run_*/final")))
        if not run_dirs:
            run_dirs = [os.path.join(args.output_dir, "final")]

        _total_train_s = time.perf_counter() - _train_wall_start
        logger.info(f"  Total training : {_total_train_s/60:.1f} min  "
                    f"(mean/run={_total_train_s/len(_run_durations)/60:.1f} min)")
        if cfg.wandb_project is not None:
            import wandb
            wandb.run.summary["train/total_duration_s"] = round(_total_train_s)
            wandb.run.summary["train/mean_run_duration_s"] = round(_total_train_s / len(_run_durations))
            for _i, _d in enumerate(_run_durations):
                wandb.run.summary[f"train/run_{_i}_duration_s"] = round(_d)

        logger.info(f"\n{'='*60}\nÉvaluation sur {len(run_dirs)} checkpoint(s)\n{'='*60}")
        all_results = []
        for run_dir in run_dirs:
            logger.info(f"  eval: {run_dir}")
            model, tokenizer, device = load_model(run_dir)
            r = run_eval(model, tokenizer, device)
            all_results.append(r)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_keys = list(all_results[0].keys())
        results = {k: round(float(np.mean([r[k] for r in all_results])), 4) for k in all_keys}
        results_std = {k: round(float(np.std([r[k] for r in all_results])), 4) for k in all_keys}

        logger.info("=== Résultats eval (moyenne ± std) ===")
        for k in METRIC_ORDER:
            if k in results:
                logger.info(f"  {k:<20} = {results[k]:.4f} ± {results_std[k]:.4f}")

        if cfg.wandb_project is not None:
            log_dict = {}
            for k in results:
                wandb.run.summary[f"eval/{k}"] = results[k]
                wandb.run.summary[f"eval/{k}_std"] = results_std[k]
                log_dict[f"eval/{k}"] = results[k]
            wandb.log(log_dict)
            run0_dir = os.path.join(args.output_dir, "run_0")
            with open(os.path.join(run0_dir, "wandb_run_id.txt"), "w") as _f:
                _f.write(wandb.run.id)
            wandb.finish()
