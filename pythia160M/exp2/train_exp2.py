"""
Experiment 2 — Stackelberg LoRA + Diversity Loss (Pythia-160M)
==============================================================

Principle
---------
One attention layer (design_layer, default 9) is treated as a Stackelberg game.
Leader head (h0) commits first by anticipating the followers' best response;
followers (h1–11) best-respond. Both minimise cross-entropy; followers also
penalise inter-head similarity via L_div when λ > 0.

The bilevel (K=1 Stackelberg) update per optimizer step:

  Phase 1 — Accumulate follower gradients
      Forward → L_F = L_CE [+ λ·L_div] / grad_accum → backward (micro-batches)
      mask_follower_grad() zeros the leader slices in p.grad
      Save g_follower = {id(p): p.grad.clone()}

  Phase 2 — Leader lookahead
      θ_F' = θ_F − η_sim · clip(g_F)   (seulement θ_F : qkv_lora_B + dense_lora_A)
      Forward at θ_F' → L_leader = L_CE → backward over all accumulated micro-batches
      mask_leader_grad() : garde uniquement les tranches θ_L, zero tout le reste (y compris θ_S)
      Save g_leader = {id(p): p.grad.clone()}

  Phase 3 — Restore, assemble, step
      Restore θ_F ← saved values (qkv_lora_B + dense_lora_A)
      assemble_gradients(g_follower, g_leader) → writes final p.grad:
        θ_L ∪ θ_F (qkv_lora_B, dense_lora_A) : g_F + g_L  (tranches disjointes)
        θ_S (qkv_lora_A, dense_lora_B, other) : g_F        (g_L = 0 après mask)
      clip_grad_norm → optimizer.step() — single AdamW step, single t counter.

Parameter split (design layer LoRA):
  θ_L ∪ θ_F — décomposables (même tenseur, tranches disjointes) :
    qkv_lora_B  : lignes lq/lk/lv → θ_L ;  reste → θ_F  (lr_follower)
    dense_lora_A : colonnes lo    → θ_L ;  reste → θ_F  (lr_leader)
  θ_S — partagés, mis à jour uniquement par g_F :
    qkv_lora_A   (lr_follower)
    dense_lora_B (lr_follower)
  θ_S — autres layers : follower only  (lr_follower)

  Simplification lr : θ_L et θ_F partagent le même tenseur → même lr.
    Tranches θ_L de qkv_lora_B → lr_follower (au lieu de lr_leader).
    Tranches θ_F de dense_lora_A → lr_leader  (au lieu de lr_follower).

Usage:
    python pythia160M/exp2/train_exp2.py --dry_run
    python pythia160M/exp2/train_exp2.py --design_layer 9
    python pythia160M/exp2/train_exp2.py --wandb_project my_project --run_name stackelberg_v2
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

_HERE = os.path.dirname(os.path.abspath(__file__))  # pythia160M/exp2/
_MODEL = os.path.dirname(_HERE)  # pythia160M/
sys.path.insert(0, _HERE)   # stackelberg_losses.py local (exp2)
sys.path.insert(1, _MODEL)  # train_utils.py, gradient_mask.py

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
)
from gradient_mask import (
    collect_lora_params,
    mask_follower_grad,
    mask_leader_grad,
    assemble_gradients,
    HiddenStateCapture,
)
from stackelberg_losses import (
    get_attention_maps, get_attention_outputs,
    follower_diversity_loss, follower_diversity_loss_sq, follower_diversity_loss_hadamard,
    follower_erank_loss, follower_output_diversity_loss, follower_diversity_loss_cka,
    entropy_heads,
    leader_confidence_loss, leader_confidence_loss_smooth, minus_entropy_head,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Eval helpers — attention heatmaps (no grad)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_attn0_heatmap(
    model, fixed_ids, qkv_module, d_head, rotary_emb, rotary_ndims,
    input_layernorm, capture, leader_idx, device,
):
    """A_leader heatmap on a fixed validation batch (first example only)."""
    model.eval()
    model(input_ids=fixed_ids)
    hidden = capture.get()
    A = get_attention_maps(
        hidden, qkv_module, n_heads=12, d_head=d_head,
        rotary_emb=rotary_emb, rotary_ndims=rotary_ndims,
        input_layernorm=input_layernorm,
    )
    model.train()
    return A[0, leader_idx].cpu().float()  # (L, L) — first batch element


@torch.no_grad()
def _compute_val_head_metrics(
    model, val_loader, qkv_module, d_head, rotary_emb, rotary_ndims,
    input_layernorm, capture, leader_idx, device, n_batches=20,
):
    """
    Un seul passage sur val_loader — retourne :
      S          : (12, 12)  matrice de similarité cosinus S^A
      conf_max   : float     mean max_l' A_leader[b,l,l']
      conf_l2    : float     mean ||A_leader[b,l,:]||^2
      h_entropy  : float     entropie de la tête leader
    """
    model.eval()
    S_accum = torch.zeros(12, 12)
    conf_max_sum = conf_l2_sum = entropy_sum = 0.0
    count = 0
    for batch in val_loader:
        if count >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids)
        hidden = capture.get()
        if hidden is None:
            continue
        A = get_attention_maps(
            hidden, qkv_module, n_heads=12, d_head=d_head,
            rotary_emb=rotary_emb, rotary_ndims=rotary_ndims,
            input_layernorm=input_layernorm,
        )
        B, H, L, _ = A.shape
        A_flat = A.view(B, H, L * L)
        A_norm = A_flat / A_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S_accum += torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0).cpu().float()
        A_leader = A[:, leader_idx, :, :]  # (B, L, L)
        conf_max_sum += A_leader.max(dim=-1).values.mean().item()
        conf_l2_sum += (A_leader ** 2).sum(dim=-1).mean().item()
        entropy_sum += entropy_heads(A)[leader_idx].item()
        count += 1
    model.train()
    n = max(1, count)
    return S_accum / n, conf_max_sum / n, conf_l2_sum / n, entropy_sum / n


# ---------------------------------------------------------------------------
# Stackelberg training loop
# ---------------------------------------------------------------------------


_CONF_LOSS_FN = {
    "max":    leader_confidence_loss,
    "smooth": leader_confidence_loss_smooth,
    "entropy": minus_entropy_head,
}

_DIV_LOSS_FN = {
    "cos":      follower_diversity_loss,
    "cos_sq":   follower_diversity_loss_sq,
    "hadamard": follower_diversity_loss_hadamard,
}


def train_stackelberg(
    cfg: TrainConfig,
    design_layers: list = None,
    lr_leader: float = 1e-4,
    lr_follower: float = 3e-4,
    lr_sim: float = 1e-3,
    lambda_lead: float = 0.0,
    lambda_peer: float = 0.0,
    lambda_conf: float = 0.0,
    lambda_rank: float = 0.0,
    leader_idx: int = 0,
    conf_loss_type: str = "max",
    div_loss_type: str = "cos",
    keep_wandb_open: bool = False,
):
    if design_layers is None:
        design_layers = [9]
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
            settings=wandb.Settings(init_timeout=300, _service_wait=120),
            config={
                **vars(cfg),
                "lr_leader": lr_leader,
                "lr_follower": lr_follower,
                "lr_sim": lr_sim,
                "lambda_lead": lambda_lead,
                "lambda_peer": lambda_peer,
                "lambda_conf": lambda_conf,
                "lambda_rank": lambda_rank,
                "conf_loss_type": conf_loss_type,
                "div_loss_type": div_loss_type,
                "design_layers": design_layers,
                "leader_idx": leader_idx,
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

    # ── Single optimizer over all trainable params ──
    all_params, grad_assembly = collect_lora_params(
        model,
        design_layers=design_layers,
        d_model=768,
        n_heads=12,
        leader_idx=leader_idx,
    )

    # θ_L ∪ θ_F : tranches décomposables (qkv_lora_B + dense_lora_A), même tenseur
    n_design_params = sum(
        p.numel()
        for r in grad_assembly.roles
        if r.kind in ("qkv_lora_B", "dense_lora_A")
        for p in [r.param]
    )
    # θ_S : partagés (qkv_lora_A + dense_lora_B) + autres layers
    n_shared_params = sum(
        p.numel()
        for r in grad_assembly.roles
        if r.kind in ("qkv_lora_A", "dense_lora_B", "other")
        for p in [r.param]
    )
    logger.info(f"Total trainable params : {sum(p.numel() for p in all_params):,}")
    logger.info(
        f"Batch size / GPU       : {cfg.batch_size_per_gpu}  |  Grad accum : {cfg.grad_accum}"
        f"  |  Effective batch : {cfg.effective_batch_size}"
    )
    logger.info(
        f"Design layers          : {design_layers}  |  Leader head idx : {leader_idx}"
    )
    logger.info(
        f"Design params (θ_L∪θ_F) : {n_design_params:,}  |  Shared+other (θ_S) : {n_shared_params:,}"
    )
    # lr_follower is the reference lr; leader uses lr_leader.
    # Since there is a single optimizer we use lr_follower as base and apply
    # a per-param-group lr_leader for the leader-only parameters.
    # We split into two param groups for lr scheduling but share Adam state.
    leader_param_ids = {
        id(r.param) for r in grad_assembly.roles if r.kind == "dense_lora_A"
    }
    # dense_lora_A (θ_L ∩ dense) → lr_leader ; everything else (θ_F, θ_S) → lr_follower.
    # Simplification acceptée : tranches θ_F de dense_lora_A utilisent lr_leader
    # et tranches θ_L de qkv_lora_B utilisent lr_follower (même tenseur, même lr).
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

    # ── Diversity / confidence hook ──
    need_div = lambda_lead > 0 or lambda_peer > 0 or lambda_rank > 0
    need_hook = need_div or lambda_conf > 0
    d_head = 768 // 12  # 64
    if need_hook:
        gpt_neox_module = next(
            mod for name, mod in model.named_modules()
            if name == "gpt_neox" or name.endswith(".gpt_neox")
        )
        rotary_emb = gpt_neox_module.rotary_emb
        _layer_ctx = {}
        for dl in design_layers:
            _qkv_mod = next(
                mod for name, mod in model.named_modules()
                if name.endswith(f"gpt_neox.layers.{dl}.attention.query_key_value")
            )
            _attn_mod = next(
                mod for name, mod in model.named_modules()
                if name.endswith(f"gpt_neox.layers.{dl}.attention")
            )
            _lay_mod = next(
                mod for name, mod in model.named_modules()
                if name.endswith(f"gpt_neox.layers.{dl}")
            )
            _cap = HiddenStateCapture()
            _cap.register(model, dl - 1)
            _layer_ctx[dl] = {
                "qkv_module":      _qkv_mod,
                "input_layernorm": _lay_mod.input_layernorm,
                "rotary_ndims":    _attn_mod.rotary_ndims,
                "capture":         _cap,
            }
        _conf_loss_fn = _CONF_LOSS_FN[conf_loss_type]
        _div_loss_fn = _DIV_LOSS_FN.get(div_loss_type, None)
        logger.info(
            f"λ_lead={lambda_lead}  λ_peer={lambda_peer}  λ_conf={lambda_conf}  λ_rank={lambda_rank}  "
            f"conf_loss_type={conf_loss_type}  div_loss_type={div_loss_type}  "
            f"— hooks on layers {[dl - 1 for dl in design_layers]}"
        )
    else:
        logger.info("λ_lead=0  λ_peer=0  λ_conf=0  — CE only (no hook, identical to baseline)")

    # ── Directories & history ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    logs_dir = os.path.join(cfg.output_dir, "logs")
    plots_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Batch fixe pour la heatmap A_0 (même input à chaque eval)
    _fixed_batch = next(iter(val_loader))
    fixed_ids = _fixed_batch["input_ids"][:1].to(device)

    history = {
        "train": {"step": [], "ce": [], "ce_ema": [], "div": [], "conf": []},
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
    accum_div = 0.0
    accum_conf = 0.0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(
        total=total_steps, desc="Stackelberg Training (Pythia-160M)", unit="step",
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
            # L_F = L_CE  [+ L_div if λ > 0]  scaled by 1/grad_accum
            # ==================================================================
            with torch.autocast(
                device_type=device.type if device.type != "mps" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type in ("cuda", "cpu")),
            ):
                out = model(input_ids=input_ids, labels=labels)
                ce_loss = out.loss

            if need_div:
                _div_losses = []
                for dl in design_layers:
                    ctx = _layer_ctx[dl]
                    hidden = ctx["capture"].get()
                    if div_loss_type in ("erank", "output_cos", "cka", "cos_output_cos"):
                        A, Z = get_attention_outputs(
                            hidden, ctx["qkv_module"], n_heads=12, d_head=d_head,
                            rotary_emb=rotary_emb, rotary_ndims=ctx["rotary_ndims"],
                            input_layernorm=ctx["input_layernorm"],
                        )
                        if div_loss_type == "erank":
                            _div_losses.append(follower_erank_loss(
                                Z, n_heads=12, leader_idx=leader_idx, lambda_rank=lambda_rank,
                            ))
                        elif div_loss_type == "cka":
                            _div_losses.append(follower_diversity_loss_cka(
                                Z, n_heads=12, leader_idx=leader_idx,
                                lambda_lead=lambda_lead, lambda_peer=lambda_peer,
                            ))
                        elif div_loss_type == "cos_output_cos":
                            _div_losses.append(
                                follower_diversity_loss(
                                    A, n_heads=12, leader_idx=leader_idx,
                                    lambda_lead=lambda_lead, lambda_peer=lambda_peer,
                                )
                                + follower_output_diversity_loss(
                                    Z, n_heads=12, leader_idx=leader_idx,
                                    lambda_lead=lambda_lead, lambda_peer=lambda_peer,
                                )
                            )
                        else:
                            _div_losses.append(follower_output_diversity_loss(
                                Z, n_heads=12, leader_idx=leader_idx,
                                lambda_lead=lambda_lead, lambda_peer=lambda_peer,
                            ))
                    else:
                        A = get_attention_maps(
                            hidden, ctx["qkv_module"], n_heads=12, d_head=d_head,
                            rotary_emb=rotary_emb, rotary_ndims=ctx["rotary_ndims"],
                            input_layernorm=ctx["input_layernorm"],
                        )
                        _div_losses.append(_div_loss_fn(
                            A, n_heads=12, leader_idx=leader_idx,
                            lambda_lead=lambda_lead, lambda_peer=lambda_peer,
                        ))
                div_loss = torch.stack(_div_losses).mean()
                follower_loss = (ce_loss + div_loss) / cfg.grad_accum
                accum_div += div_loss.item()
            else:
                follower_loss = ce_loss / cfg.grad_accum

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

                # Mask leader slice out of follower gradients
                with torch.no_grad():
                    mask_follower_grad(grad_assembly)

                # Save masked follower gradients
                g_follower = {
                    id(r.param): r.param.grad.clone()
                    if r.param.grad is not None
                    else torch.zeros_like(r.param)
                    for r in grad_assembly.roles
                }

                # ==============================================================
                # Phase 2: Leader lookahead
                # θ_F' = θ_F − η_sim · ĝ_F  (clipped vanilla SGD)
                # ==============================================================

                # Clip before simulated step to match Phase 3 clipping
                sim_gnorm = torch.norm(
                    torch.stack(
                        [g.norm() for g in g_follower.values() if g.numel() > 0]
                    )
                )
                sim_clip = min(1.0, cfg.grad_clip / (sim_gnorm.item() + 1e-8))

                design_roles = [
                    r
                    for r in grad_assembly.roles
                    if r.kind in ("qkv_lora_B", "dense_lora_A")
                ]
                saved_data = {id(r.param): r.param.data.clone() for r in design_roles}
                with torch.no_grad():
                    for r in design_roles:
                        gf = g_follower[id(r.param)]
                        r.param.data.sub_(lr_sim * gf * sim_clip)

                # Leader forward over all accumulated micro-batches
                optimizer.zero_grad()
                for inp, lab in zip(accum_inputs, accum_labels):
                    with torch.autocast(
                        device_type=device.type if device.type != "mps" else "cpu",
                        dtype=torch.bfloat16,
                        enabled=(device.type in ("cuda", "cpu")),
                    ):
                        out_leader = model(input_ids=inp, labels=lab)
                        leader_ce_mb = out_leader.loss / cfg.grad_accum
                    if lambda_conf > 0:
                        _conf_losses = []
                        for dl in design_layers:
                            ctx = _layer_ctx[dl]
                            hidden_leader = ctx["capture"].get()
                            A_leader = get_attention_maps(
                                hidden_leader, ctx["qkv_module"], 12, d_head,
                                rotary_emb, ctx["rotary_ndims"], ctx["input_layernorm"],
                            )
                            _conf_losses.append(_conf_loss_fn(A_leader, leader_idx))
                        conf_raw = lambda_conf * torch.stack(_conf_losses).mean()
                        leader_ce_mb = leader_ce_mb + conf_raw / cfg.grad_accum
                        accum_conf += conf_raw.detach().item()
                    leader_ce_mb.backward()

                # Mask: keep only leader slices, zero "other" params
                with torch.no_grad():
                    mask_leader_grad(grad_assembly)

                # Save masked leader gradients
                g_leader = {
                    id(r.param): r.param.grad.clone()
                    if r.param.grad is not None
                    else torch.zeros_like(r.param)
                    for r in grad_assembly.roles
                }

                # ==============================================================
                # Phase 3: Restore followers, assemble final gradient, single step
                # ==============================================================
                with torch.no_grad():
                    for r in design_roles:
                        r.param.data.copy_(saved_data[id(r.param)])

                # Write assembled gradient into p.grad
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
                    div=f"{accum_div:.4f}",
                    conf=f"{accum_conf:.4f}",
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
                        f"  div={accum_div:.4f}  conf={accum_conf:.4f}"
                        f"  lr_L={lr_l:.2e}  lr_F={lr_f:.2e}"
                        f"  tok/s={tokens_per_sec:,}"
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "train/ce_loss": accum_ce,
                                "train/ce_ema": _ema_ce,
                                "train/div_loss": accum_div,
                                "train/conf_loss": accum_conf,
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
                    history["train"]["div"].append(accum_div)
                    history["train"]["conf"].append(accum_conf)

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
                    # log_head_matrices désactivé
                    if use_wandb:
                        log_dict = {"val/loss": v_loss, "val/ppl": v_ppl}
                        if need_hook:
                            from matplotlib.colors import LogNorm
                            _conf_max_vals, _conf_l2_vals, _entropy_vals = [], [], []
                            S_sum = torch.zeros(12, 12)
                            for dl in design_layers:
                                ctx = _layer_ctx[dl]
                                A0 = _compute_attn0_heatmap(
                                    model, fixed_ids, ctx["qkv_module"], d_head,
                                    rotary_emb, ctx["rotary_ndims"], ctx["input_layernorm"],
                                    ctx["capture"], leader_idx, device,
                                )
                                fig, ax = plt.subplots(figsize=(7, 6))
                                A0_np = A0.numpy()
                                _vmax = float(np.percentile(A0_np, 99.5))
                                _vmin = max(float(A0_np.min()), _vmax * 1e-4)
                                im = ax.imshow(
                                    A0_np.clip(_vmin, None),
                                    cmap="inferno", aspect="auto",
                                    norm=LogNorm(vmin=_vmin, vmax=_vmax),
                                )
                                plt.colorbar(im, ax=ax, label="attention weight (log)")
                                ax.set_title(f"A_leader (head {leader_idx}, layer {dl}, step {opt_step})")
                                log_dict[f"eval/A0_heatmap_layer{dl}"] = wandb.Image(fig)
                                plt.close(fig)

                                S, conf_max, conf_l2, h_entropy = _compute_val_head_metrics(
                                    model, val_loader, ctx["qkv_module"], d_head,
                                    rotary_emb, ctx["rotary_ndims"], ctx["input_layernorm"],
                                    ctx["capture"], leader_idx, device,
                                )
                                S_sum += S
                                _conf_max_vals.append(conf_max)
                                _conf_l2_vals.append(conf_l2)
                                _entropy_vals.append(h_entropy)

                            S = S_sum / len(design_layers)
                            fig, ax = plt.subplots(figsize=(7, 6))
                            S_np = S.numpy()
                            _off = S_np[~np.eye(S_np.shape[0], dtype=bool)]
                            _vext = max(abs(float(np.percentile(_off, 1))),
                                        abs(float(np.percentile(_off, 99))), 0.05)
                            im = ax.imshow(S_np, cmap="RdBu_r", vmin=-_vext, vmax=_vext, aspect="auto")
                            plt.colorbar(im, ax=ax, label="cosine similarity (diag saturated)")
                            ax.set_title(f"S^A cosine similarity (step {opt_step})")
                            ax.set_xlabel("head j")
                            ax.set_ylabel("head i")
                            log_dict["eval/SA_heatmap"] = wandb.Image(fig)
                            plt.close(fig)
                            log_dict["leader/conf_max"] = float(np.mean(_conf_max_vals))
                            log_dict["leader/conf_l2"] = float(np.mean(_conf_l2_vals))
                            log_dict["leader/entropy"] = float(np.mean(_entropy_vals))
                        wandb.log(log_dict, step=opt_step)
                    history["val"]["step"].append(opt_step)
                    history["val"]["loss"].append(v_loss)
                    history["val"]["ppl"].append(v_ppl)

                # ── JSON ──
                if (
                    opt_step % cfg.log_every == 0
                    or opt_step % cfg.eval_every == 0
                    or opt_step == 1
                ):
                    os.makedirs(logs_dir, exist_ok=True)
                    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
                        json.dump(history, _f, indent=2)

                accum_ce = 0.0
                accum_div = 0.0
                accum_conf = 0.0
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

    if need_hook:
        for dl in design_layers:
            _layer_ctx[dl]["capture"].remove()

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
    if any(v > 0 for v in history["train"]["div"]):
        ax.plot(
            history["train"]["step"],
            history["train"]["div"],
            color="green",
            alpha=0.6,
            label="diversity loss",
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
    ax.set_title("Training — Pythia-160M Stackelberg exp2")
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
    ax.set_title("Validation perplexity — WikiText-103 (Stackelberg exp2)")
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
        description="Stackelberg Attention Diversity Training — Pythia-160M exp2"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--run_name", default="stackelberg_exp2_pythia")
    parser.add_argument(
        "--design_layer",
        nargs="+",
        type=int,
        default=[9],
        help="Design layer(s) for the Stackelberg game. Ex: --design_layer 9  or  --design_layer 6 7 8 9",
    )
    parser.add_argument("--lr_leader", type=float, default=1e-4)
    parser.add_argument("--lr_follower", type=float, default=3e-4)
    parser.add_argument(
        "--lr_sim",
        type=float,
        default=1e-3,
        help="LR for simulated follower step (vanilla SGD, no momentum)",
    )
    parser.add_argument(
        "--lambda_lead",
        type=float,
        default=0.0,
        help="Penalty weight for leader-follower similarity (0 = CE only)",
    )
    parser.add_argument(
        "--lambda_peer",
        type=float,
        default=0.0,
        help="Penalty weight for peer-follower similarity (0 = CE only)",
    )
    parser.add_argument(
        "--lambda_conf",
        type=float,
        default=0.0,
        help="Penalty weight for leader confidence loss (0 = désactivé)",
    )
    parser.add_argument(
        "--lambda_rank",
        type=float,
        default=0.0,
        help="Coefficient pour la loss de rang effectif des followers (utilisé si --div_loss_type erank)",
    )
    parser.add_argument(
        "--leader_idx", type=int, default=0, help="Index of the leader head"
    )
    parser.add_argument(
        "--conf_loss_type", choices=["max", "smooth", "entropy"], default="max",
        help="Confidence loss variant: max=leader_confidence_loss, smooth=leader_confidence_loss_smooth, entropy=minus_entropy_head",
    )
    parser.add_argument(
        "--div_loss_type", choices=["cos", "cos_sq", "hadamard", "erank", "output_cos", "cka", "cos_output_cos"], default="cos",
        help=(
            "Diversity loss variant: "
            "cos=cosine similarity sur A_i (Exp2_1–4), "
            "cos_sq=squared cosine similarity sur A_i (Exp2_5), "
            "hadamard=|A_i ⊙ A_j| dot product sur A_i (Exp2_6), "
            "erank=-effective rank des sorties followers (Exp2_7, utilise --lambda_rank), "
            "output_cos=cosine similarity sur Z_i=A_i@V_i (Exp2_8, utilise --lambda_lead/peer), "
            "cka=CKA linéaire sur Z_i=A_i@V_i (utilise --lambda_lead/peer, évite artefacts simplexe)"
        ),
    )
    parser.add_argument(
        "--nb_runs", type=int, default=5,
        help="Nombre d'entraînements consécutifs (seeds seed, seed+1, …). Chaque run sauvegardé dans output_dir/run_i/",
    )
    parser.add_argument(
        "--run_eval", action="store_true", default=True,
        help="Lancer l'évaluation après l'entraînement et logger les métriques dans le même run wandb.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        total_tokens=args.total_tokens,
        batch_size_per_gpu=args.batch_size_per_gpu,
        grad_accum=args.grad_accum,
        lr=args.lr,
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
    logger.info(f"  Design layers : {args.design_layer}")
    logger.info(f"  LR leader     : {args.lr_leader}")
    logger.info(f"  LR follower   : {args.lr_follower}")
    logger.info(f"  LR sim step   : {args.lr_sim}")
    logger.info(f"  λ_lead        : {args.lambda_lead}")
    logger.info(f"  λ_peer        : {args.lambda_peer}")
    logger.info(f"  λ_conf        : {args.lambda_conf}")
    logger.info(f"  λ_rank        : {args.lambda_rank}")
    logger.info(f"  Leader head   : {args.leader_idx}")
    logger.info(f"  Nb runs       : {args.nb_runs}")
    logger.info(f"  Conf loss     : {args.conf_loss_type}")
    logger.info(f"  Div loss      : {args.div_loss_type}")

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
            design_layers=args.design_layer,
            lr_leader=args.lr_leader,
            lr_follower=args.lr_follower,
            lr_sim=args.lr_sim,
            lambda_lead=args.lambda_lead,
            lambda_peer=args.lambda_peer,
            lambda_conf=args.lambda_conf,
            lambda_rank=args.lambda_rank,
            leader_idx=args.leader_idx,
            conf_loss_type=args.conf_loss_type,
            div_loss_type=args.div_loss_type,
            keep_wandb_open=keep_open,
        )
        _run_durations.append(time.perf_counter() - _t0)
        logger.info(f"  Run {i} duration : {_run_durations[-1]/60:.1f} min")

    if args.run_eval and cfg.wandb_project is not None:
        import wandb
        from eval import run_eval, load_model, METRIC_ORDER

        run_dirs = sorted(glob.glob(os.path.join(args.output_dir, "run_*/final")))
        if not run_dirs:
            run_dirs = [os.path.join(args.output_dir, "final")]

        _total_train_s = time.perf_counter() - _train_wall_start
        wandb.run.summary["train/total_duration_s"] = round(_total_train_s)
        wandb.run.summary["train/mean_run_duration_s"] = round(_total_train_s / len(_run_durations))
        for _i, _d in enumerate(_run_durations):
            wandb.run.summary[f"train/run_{_i}_duration_s"] = round(_d)
        logger.info(f"  Total training : {_total_train_s/60:.1f} min  "
                    f"(mean/run={_total_train_s/len(_run_durations)/60:.1f} min)")

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
