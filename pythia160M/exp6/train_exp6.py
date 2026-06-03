"""
Experiment 6 — Stackelberg LoRA + Leader→Follower Gating (Pythia-160M)
=======================================================================

Base : exp1 (bilevel Stackelberg 3-phases, CE only). On AJOUTE un couplage
STRUCTUREL leader→followers au design layer via un MLP de gating.

Architecture (design layer)
---------------------------
Sortie d'attention standard : o = Σ_h W_O^(h) z_h  (somme, poids fixe 1),
W_O = attention.dense. On pondère les contributions FOLLOWER par des gates
calculés depuis le LEADER :

    s = concat({z_l}_{l∈leader})  ∈ R^{|L|·d_head}
    g = 2·σ(MLP(s))               ∈ (0,2)^{|F|}   (par token)
    o = Σ_{l∈L} W_O^(l) z_l  +  Σ_{i∈F} g_i · W_O^(i) z_i

cf. pythia160M/gate.py. Hook forward_pre sur attention.dense → agit APRÈS
l'attention (sur concat(z_h)).

Rôle Stackelberg : le MLP de gating est un paramètre LEADER (θ_L) :
  masqué pour le follower (Phase 1), gardé pour le leader (Phase 2),
  anticipé dans le lookahead, lr_leader.
Init : gates ≡ 1 → couche identique au modèle pré-entraîné au départ.

Usage:
    python pythia160M/exp6/train_exp6.py --dry_run
    python pythia160M/exp6/train_exp6.py --design_layer 9 --leader_idx 0 --gate_hidden 128
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

_HERE = os.path.dirname(os.path.abspath(__file__))   # pythia160M/exp6/
_MODEL = os.path.dirname(_HERE)                       # pythia160M/
sys.path.insert(0, _MODEL)

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
    add_gate_roles,
    gate_param_ids,
    HiddenStateCapture,
)
from stackelberg_losses import (
    get_attention_maps, get_attention_outputs,
    follower_diversity_loss, follower_diversity_loss_sq, follower_diversity_loss_hadamard,
    follower_erank_loss, follower_output_diversity_loss, follower_diversity_loss_cka,
    entropy_heads,
    leader_confidence_loss, leader_confidence_loss_smooth, minus_entropy_head,
    ldb_loss, head_interaction_matrix,
)
from gate import LeaderFollowerGate, save_gate, gate_stats, gate_grad_norm

logger = logging.getLogger(__name__)

_NH = 12        # têtes Pythia-160M
_DM = 768
_DH = _DM // _NH  # 64


_CONF_LOSS_FN = {
    "max":     leader_confidence_loss,
    "smooth":  leader_confidence_loss_smooth,
    "entropy": minus_entropy_head,
}

_DIV_LOSS_FN = {
    "cos":      follower_diversity_loss,
    "cos_sq":   follower_diversity_loss_sq,
    "hadamard": follower_diversity_loss_hadamard,
}


def _parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Reconstruction d'attention (pour les losses exp3) — GPT-NeoX qkv fusionné
# ---------------------------------------------------------------------------


def _resolve_attn_ctx(model, design_layers):
    """Renvoie (rotary_emb, {dl: {qkv_module, input_layernorm, rotary_ndims, capture}})."""
    gpt_neox = next(
        mod for name, mod in model.named_modules()
        if name == "gpt_neox" or name.endswith(".gpt_neox")
    )
    rotary_emb = gpt_neox.rotary_emb
    ctx = {}
    for dl in design_layers:
        qkv = next(m for n, m in model.named_modules()
                   if n.endswith(f"gpt_neox.layers.{dl}.attention.query_key_value"))
        attn = next(m for n, m in model.named_modules()
                    if n.endswith(f"gpt_neox.layers.{dl}.attention"))
        lay = next(m for n, m in model.named_modules()
                   if n.endswith(f"gpt_neox.layers.{dl}"))
        cap = HiddenStateCapture()
        cap.register(model, dl - 1)
        ctx[dl] = {"qkv_module": qkv, "input_layernorm": lay.input_layernorm,
                   "rotary_ndims": attn.rotary_ndims, "capture": cap}
    return rotary_emb, ctx


def _A(hidden, ctx, rotary_emb):
    return get_attention_maps(
        hidden, ctx["qkv_module"], n_heads=_NH, d_head=_DH,
        rotary_emb=rotary_emb, rotary_ndims=ctx["rotary_ndims"],
        input_layernorm=ctx["input_layernorm"],
    )


def _AZ(hidden, ctx, rotary_emb):
    return get_attention_outputs(
        hidden, ctx["qkv_module"], n_heads=_NH, d_head=_DH,
        rotary_emb=rotary_emb, rotary_ndims=ctx["rotary_ndims"],
        input_layernorm=ctx["input_layernorm"],
    )


@torch.no_grad()
def _compute_leader_heatmaps(model, fixed_ids, ctx, rotary_emb, leader_indices):
    model.eval()
    model(input_ids=fixed_ids)
    hidden = ctx["capture"].get()
    A = _A(hidden, ctx, rotary_emb)
    model.train()
    return {k: A[0, k].cpu().float() for k in leader_indices}


@torch.no_grad()
def _compute_val_head_metrics(model, val_loader, ctx, rotary_emb,
                              leader_indices, device, n_batches=20):
    model.eval()
    S_accum = torch.zeros(_NH, _NH)
    conf_max_sum = conf_l2_sum = entropy_sum = 0.0
    count = 0
    li_t = torch.tensor(leader_indices)
    for batch in val_loader:
        if count >= n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        model(input_ids=input_ids)
        hidden = ctx["capture"].get()
        if hidden is None:
            continue
        A = _A(hidden, ctx, rotary_emb)
        B, H, L, _ = A.shape
        A_flat = A.view(B, H, L * L)
        A_norm = A_flat / A_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        S_accum += torch.bmm(A_norm, A_norm.transpose(1, 2)).mean(0).cpu().float()
        A_leaders = A[:, li_t, :, :]
        conf_max_sum += A_leaders.max(dim=-1).values.mean().item()
        conf_l2_sum += (A_leaders ** 2).sum(dim=-1).mean().item()
        entropy_sum += entropy_heads(A)[li_t].mean().item()
        count += 1
    model.train()
    n = max(1, count)
    return S_accum / n, conf_max_sum / n, conf_l2_sum / n, entropy_sum / n


# ---------------------------------------------------------------------------
# Stackelberg + gating training loop
# ---------------------------------------------------------------------------


def train_stackelberg(
    cfg: TrainConfig,
    design_layer: int = 9,
    lr_leader: float = 1e-4,
    lr_follower: float = 3e-4,
    lr_sim: float = 1e-3,
    lr_gate: float = None,
    lambda_lead: float = 0.0,
    lambda_peer: float = 0.0,
    lambda_conf: float = 0.0,
    lambda_rank: float = 0.0,
    lambda_ldb: float = 0.0,
    conf_loss_type: str = "max",
    div_loss_type: str = "cos",
    leader_indices: list = None,
    gate_hidden: int = 128,
    keep_wandb_open: bool = False,
):
    if leader_indices is None:
        leader_indices = [0]
    if lr_gate is None:
        lr_gate = lr_leader   # rétro-compat : gate au lr_leader si non spécifié
    follower_indices = [h for h in range(_NH) if h not in set(leader_indices)]

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
                "lr_gate": lr_gate,
                "lr_sim": lr_sim,
                "lambda_lead": lambda_lead,
                "lambda_peer": lambda_peer,
                "lambda_conf": lambda_conf,
                "lambda_rank": lambda_rank,
                "lambda_ldb": lambda_ldb,
                "conf_loss_type": conf_loss_type,
                "div_loss_type": div_loss_type,
                "design_layer": design_layer,
                "leader_indices": leader_indices,
                "follower_indices": follower_indices,
                "gate_hidden": gate_hidden,
                "exp": "exp6_gating",
            },
        )

    # ── Model ──
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = model.to(device)

    # ── Gate leader→followers, sur attention.dense du design layer ──
    gate = LeaderFollowerGate(
        leader_heads=leader_indices, follower_heads=follower_indices,
        d_head=_DH, n_heads=_NH, hidden=gate_hidden,
    ).to(device)
    gate.register(model, design_layer)
    logger.info(
        f"Gate MLP : {sum(p.numel() for p in gate.parameters()):,} params  "
        f"(in={len(leader_indices)*_DH} → {gate_hidden} → {len(follower_indices)})  "
        f"init gates ≡ 1"
    )

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
        train_ds, batch_size=cfg.batch_size_per_gpu, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=True, generator=g, persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size_per_gpu, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=False, persistent_workers=(cfg.num_workers > 0),
    )
    total_steps = 100 if cfg.dry_run else cfg.total_steps

    # ── Collect params (LoRA leader/follower) + gate (leader entier) ──
    all_params, grad_assembly = collect_lora_params(
        model, design_layers=[design_layer], d_model=_DM, n_heads=_NH,
        leader_indices=leader_indices,
    )
    all_params = add_gate_roles(grad_assembly, all_params, gate)
    _gate_ids = gate_param_ids(grad_assembly)

    n_design = sum(p.numel() for r in grad_assembly.roles
                   if r.kind in ("qkv_lora_B", "dense_lora_A") for p in [r.param])
    n_gate = sum(p.numel() for r in grad_assembly.roles
                 if r.kind == "gate_leader" for p in [r.param])
    logger.info(f"Total trainable params : {sum(p.numel() for p in all_params):,}")
    logger.info(
        f"Design layer           : {design_layer}  |  Leader heads : {leader_indices}"
        f"  |  Follower heads : {follower_indices}"
    )
    logger.info(f"Design LoRA (θ_L∪θ_F) : {n_design:,}  |  Gate (θ_L) : {n_gate:,}")

    # 3 groupes lr : follower/shared (lr_follower), leader-LoRA dense_lora_A (lr_leader),
    # gate MLP (lr_gate, séparé pour pouvoir l'accélérer indépendamment).
    leader_lora_ids = {
        id(r.param) for r in grad_assembly.roles if r.kind == "dense_lora_A"
    }
    param_groups = [
        {"params": [p for p in all_params
                    if id(p) not in leader_lora_ids and id(p) not in _gate_ids],
         "lr": lr_follower, "name": "follower_and_shared"},
        {"params": [p for p in all_params if id(p) in leader_lora_ids],
         "lr": lr_leader, "name": "leader"},
        {"params": [p for p in all_params if id(p) in _gate_ids],
         "lr": lr_gate, "name": "gate"},
    ]
    logger.info(f"lr : follower={lr_follower}  leader={lr_leader}  gate={lr_gate}")
    optimizer = torch.optim.AdamW(param_groups, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps, num_training_steps=total_steps,
    )

    # ── Losses exp3 (diversity / confidence / LDB) — hooks d'attention ──
    # Si tous les λ=0 (cas run6_1) : need_hook=False → forward identique exp6 "CE+gate".
    need_div = lambda_lead > 0 or lambda_peer > 0 or lambda_rank > 0
    need_hook = need_div or lambda_conf > 0
    rotary_emb = None
    _layer_ctx = {}
    if need_hook:
        rotary_emb, _layer_ctx = _resolve_attn_ctx(model, [design_layer])
        _conf_loss_fn = _CONF_LOSS_FN[conf_loss_type]
        _div_loss_fn = _DIV_LOSS_FN.get(div_loss_type, None)
        logger.info(
            f"λ_lead={lambda_lead} λ_peer={lambda_peer} λ_conf={lambda_conf} "
            f"λ_rank={lambda_rank} λ_ldb={lambda_ldb}  conf={conf_loss_type} div={div_loss_type}"
        )
    else:
        logger.info("λ=0 partout — CE + gate uniquement, pas de hook d'attention")

    # ── Directories & history ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    logs_dir = os.path.join(cfg.output_dir, "logs")
    plots_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    _fixed_batch = next(iter(val_loader))
    fixed_ids = _fixed_batch["input_ids"][:1].to(device)

    history = {
        "train": {"step": [], "ce": [], "ce_ema": [], "leader_ce": [],
                  "div": [], "conf": [], "ldb": []},
        "val": {"step": [], "loss": [], "ppl": []},
    }
    _ema_ce = None
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
    gate.train()
    global_step = 0
    accum_ce = 0.0
    accum_leader_ce = 0.0
    accum_div = 0.0
    accum_conf = 0.0
    accum_ldb = 0.0
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(
        total=total_steps, desc="Stackelberg+Gate (Pythia-160M exp6)", unit="step",
        disable=not sys.stderr.isatty(),
    )

    accum_inputs, accum_labels = [], []

    done = False
    while not done:
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            accum_inputs.append(input_ids)
            accum_labels.append(labels)

            # === Phase 1 : Follower forward ===
            with torch.autocast(
                device_type=device.type if device.type != "mps" else "cpu",
                dtype=torch.bfloat16, enabled=(device.type in ("cuda", "cpu")),
            ):
                out = model(input_ids=input_ids, labels=labels)
                ce_loss = out.loss

            # diversity loss sur les followers (exp3)
            if need_div:
                ctx = _layer_ctx[design_layer]
                hidden = ctx["capture"].get()
                if div_loss_type in ("erank", "output_cos", "cka"):
                    _A_unused, Z = _AZ(hidden, ctx, rotary_emb)
                    if div_loss_type == "erank":
                        div_loss = follower_erank_loss(
                            Z, n_heads=_NH, leader_indices=leader_indices, lambda_rank=lambda_rank)
                    elif div_loss_type == "cka":
                        div_loss = follower_diversity_loss_cka(
                            Z, n_heads=_NH, leader_indices=leader_indices,
                            lambda_lead=lambda_lead, lambda_peer=lambda_peer)
                    else:
                        div_loss = follower_output_diversity_loss(
                            Z, n_heads=_NH, leader_indices=leader_indices,
                            lambda_lead=lambda_lead, lambda_peer=lambda_peer)
                else:
                    A = _A(hidden, ctx, rotary_emb)
                    div_loss = _div_loss_fn(
                        A, n_heads=_NH, leader_indices=leader_indices,
                        lambda_lead=lambda_lead, lambda_peer=lambda_peer)
                follower_loss = (ce_loss + div_loss) / cfg.grad_accum
                accum_div += div_loss.item()
            else:
                follower_loss = ce_loss / cfg.grad_accum

            if lambda_ldb > 0:
                ldb_raw = lambda_ldb * ldb_loss(
                    head_interaction_matrix(model, out.logits, labels, design_layer, _DH))
                follower_loss = follower_loss + ldb_raw / cfg.grad_accum
                accum_ldb += ldb_raw.item()

            follower_loss.backward()
            accum_ce += ce_loss.item() / cfg.grad_accum

            if (global_step + 1) % cfg.grad_accum == 0:
                opt_step = (global_step + 1) // cfg.grad_accum

                with torch.no_grad():
                    mask_follower_grad(grad_assembly)
                g_follower = {
                    id(r.param): r.param.grad.clone() if r.param.grad is not None
                    else torch.zeros_like(r.param)
                    for r in grad_assembly.roles
                }

                # === Phase 2 : Leader lookahead ===
                sim_gnorm = torch.norm(
                    torch.stack([gg.norm() for gg in g_follower.values() if gg.numel() > 0])
                )
                sim_clip = min(1.0, cfg.grad_clip / (sim_gnorm.item() + 1e-8))

                design_roles = [r for r in grad_assembly.roles
                                if r.kind in ("qkv_lora_B", "dense_lora_A")]
                saved_data = {id(r.param): r.param.data.clone() for r in design_roles}
                with torch.no_grad():
                    for r in design_roles:
                        r.param.data.sub_(lr_sim * g_follower[id(r.param)] * sim_clip)

                optimizer.zero_grad()
                leader_ce_accum = torch.tensor(0.0, device=device)
                for inp, lab in zip(accum_inputs, accum_labels):
                    with torch.autocast(
                        device_type=device.type if device.type != "mps" else "cpu",
                        dtype=torch.bfloat16, enabled=(device.type in ("cuda", "cpu")),
                    ):
                        out_leader = model(input_ids=inp, labels=lab)
                        leader_ce_mb = out_leader.loss / cfg.grad_accum
                    # confidence loss sur les têtes leader (exp3)
                    if lambda_conf > 0:
                        ctx = _layer_ctx[design_layer]
                        hidden_leader = ctx["capture"].get()
                        A_leader = _A(hidden_leader, ctx, rotary_emb)
                        conf_raw = lambda_conf * _conf_loss_fn(A_leader, leader_indices)
                        leader_ce_mb = leader_ce_mb + conf_raw / cfg.grad_accum
                        accum_conf += conf_raw.detach().item()
                    if lambda_ldb > 0:
                        ldb_raw_l = lambda_ldb * ldb_loss(
                            head_interaction_matrix(model, out_leader.logits, lab, design_layer, _DH))
                        leader_ce_mb = leader_ce_mb + ldb_raw_l / cfg.grad_accum
                    leader_ce_mb.backward()
                    leader_ce_accum = leader_ce_accum + leader_ce_mb.detach()
                accum_leader_ce = leader_ce_accum.item()

                with torch.no_grad():
                    mask_leader_grad(grad_assembly)
                g_leader = {
                    id(r.param): r.param.grad.clone() if r.param.grad is not None
                    else torch.zeros_like(r.param)
                    for r in grad_assembly.roles
                }

                # === Phase 3 : Restore, assemble, step ===
                with torch.no_grad():
                    for r in design_roles:
                        r.param.data.copy_(saved_data[id(r.param)])
                    assemble_gradients(grad_assembly, g_follower, g_leader)

                # norme du gradient du gate (avant clip) — confirme qu'il apprend
                _gate_gnorm = gate_grad_norm(gate)

                torch.nn.utils.clip_grad_norm_(all_params, max_norm=cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                _ema_ce = accum_ce if _ema_ce is None \
                    else _ema_alpha * accum_ce + (1 - _ema_alpha) * _ema_ce

                step_time = time.perf_counter() - _step_start
                tokens_per_sec = int(cfg.seq_len * cfg.effective_batch_size / step_time)
                _step_start = time.perf_counter()

                pbar.update(1)
                pbar.set_postfix(ce=f"{accum_ce:.4f}", ema=f"{_ema_ce:.4f}",
                                 l_ce=f"{accum_leader_ce:.4f}", tok_s=f"{tokens_per_sec:,}")

                if opt_step % cfg.log_every == 0 or opt_step == 1:
                    lr_f = scheduler.get_last_lr()[0]
                    lr_l = scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else lr_f
                    logger.info(
                        f"[train] step {opt_step:>6d}/{total_steps}  CE={accum_ce:.4f}  "
                        f"ema={_ema_ce:.4f}  leader_CE={accum_leader_ce:.4f}  "
                        f"div={accum_div:.4f}  conf={accum_conf:.4f}  ldb={accum_ldb:.4f}  "
                        f"lr_L={lr_l:.2e}  lr_F={lr_f:.2e}  tok/s={tokens_per_sec:,}"
                    )
                    # ── Stats du gate (signaux exp6) ──
                    _gs = gate_stats(gate)
                    if _gs is not None:
                        logger.info(
                            f"        gate: mean|g-1|={_gs['mean_dev']:.4f}  "
                            f"token_std={_gs['token_std']:.4f}  sat={_gs['saturation']:.3f}  "
                            f"grad_norm={_gate_gnorm:.2e}"
                        )
                    if use_wandb:
                        log_dict = {
                            "train/ce_loss": accum_ce, "train/ce_ema": _ema_ce,
                            "train/leader_ce": accum_leader_ce,
                            "train/div_loss": accum_div, "train/conf_loss": accum_conf,
                            "train/ldb_loss": accum_ldb,
                            "train/lr_leader": lr_l, "train/lr_follower": lr_f,
                            "train/tokens": opt_step * cfg.seq_len * cfg.effective_batch_size,
                        }
                        if _gs is not None:
                            log_dict["gate/mean_dev"] = _gs["mean_dev"]
                            log_dict["gate/token_std"] = _gs["token_std"]
                            log_dict["gate/saturation"] = _gs["saturation"]
                            log_dict["gate/grad_norm"] = _gate_gnorm
                            for _h, _v in zip(follower_indices, _gs["per_head"].tolist()):
                                log_dict[f"gate/head_{_h}"] = _v
                        wandb.log(log_dict, step=opt_step)
                    history["train"]["step"].append(opt_step)
                    history["train"]["ce"].append(accum_ce)
                    history["train"]["ce_ema"].append(_ema_ce)
                    history["train"]["leader_ce"].append(accum_leader_ce)
                    history["train"]["div"].append(accum_div)
                    history["train"]["conf"].append(accum_conf)
                    history["train"]["ldb"].append(accum_ldb)

                if opt_step % cfg.eval_every == 0:
                    v_loss, v_ppl = evaluate(model, val_loader, device,
                                             max_batches=cfg.eval_max_batches,
                                             autocast_dtype=torch.bfloat16)
                    logger.info(f"[val]   step {opt_step:>6d}  val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
                    log_head_matrices(model, device, design_layer, opt_step, val_loader,
                                      wandb_mod=wandb if use_wandb else None,
                                      log_image=(opt_step % (cfg.eval_every * 5) == 0))
                    if use_wandb:
                        ev_dict = {"val/loss": v_loss, "val/ppl": v_ppl}
                        if need_hook:
                            from matplotlib.colors import LogNorm
                            _log_img = (opt_step % (cfg.eval_every * 5) == 0)
                            ctx = _layer_ctx[design_layer]
                            if _log_img:
                                heatmaps = _compute_leader_heatmaps(
                                    model, fixed_ids, ctx, rotary_emb, leader_indices)
                                for rank, (k, A0) in enumerate(heatmaps.items()):
                                    fig, ax = plt.subplots(figsize=(7, 6))
                                    A0_np = A0.numpy()
                                    _vmax = float(np.percentile(A0_np, 99.5))
                                    _vmin = max(max(float(A0_np.min()), _vmax * 1e-4), 1e-9)
                                    if _vmax <= _vmin:
                                        _vmax = _vmin * 100
                                    im = ax.imshow(A0_np.clip(_vmin, None), cmap="inferno",
                                                   aspect="auto", norm=LogNorm(vmin=_vmin, vmax=_vmax))
                                    plt.colorbar(im, ax=ax, label="attention weight (log)")
                                    ax.set_title(f"A_leader_{rank} (head {k}, step {opt_step})")
                                    ev_dict[f"eval/A_leader_{rank}_heatmap"] = wandb.Image(fig)
                                    plt.close(fig)
                            S, conf_max, conf_l2, h_entropy = _compute_val_head_metrics(
                                model, val_loader, ctx, rotary_emb,
                                leader_indices, device, n_batches=cfg.eval_max_batches)
                            if _log_img:
                                fig, ax = plt.subplots(figsize=(7, 6))
                                S_np = S.numpy()
                                _off = S_np[~np.eye(S_np.shape[0], dtype=bool)]
                                _vext = max(abs(float(np.percentile(_off, 1))),
                                            abs(float(np.percentile(_off, 99))), 0.05)
                                im = ax.imshow(S_np, cmap="RdBu_r", vmin=-_vext, vmax=_vext, aspect="auto")
                                plt.colorbar(im, ax=ax, label="cosine similarity")
                                ax.set_title(f"S^A cosine similarity (step {opt_step})")
                                ev_dict["eval/SA_heatmap"] = wandb.Image(fig)
                                plt.close(fig)
                            ev_dict["leader/conf_max"] = conf_max
                            ev_dict["leader/conf_l2"] = conf_l2
                            ev_dict["leader/entropy"] = h_entropy
                        wandb.log(ev_dict, step=opt_step)
                    history["val"]["step"].append(opt_step)
                    history["val"]["loss"].append(v_loss)
                    history["val"]["ppl"].append(v_ppl)

                if (opt_step % cfg.log_every == 0 or opt_step % cfg.eval_every == 0 or opt_step == 1):
                    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
                        json.dump(history, _f, indent=2)

                accum_ce = 0.0
                accum_leader_ce = 0.0
                accum_div = 0.0
                accum_conf = 0.0
                accum_ldb = 0.0
                accum_inputs, accum_labels = [], []

                if opt_step % cfg.save_every == 0:
                    ckpt_path = os.path.join(cfg.output_dir, f"step_{opt_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    save_gate(gate, ckpt_path, design_layer)
                    logger.info(f"Checkpoint → {ckpt_path}")

                if opt_step >= total_steps:
                    done = True
                    break
            global_step += 1

        if not done:
            logger.info("Fin d'epoch — nouveau passage sur le dataset.")

    pbar.close()

    # ── Eval finale ──
    logger.info("Eval finale ...")
    v_loss, v_ppl = evaluate(model, val_loader, device, max_batches=cfg.eval_max_batches)
    history["val"]["step"].append(opt_step)
    history["val"]["loss"].append(v_loss)
    history["val"]["ppl"].append(v_ppl)
    logger.info(f"  [final] val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
    if use_wandb:
        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=opt_step)

    # ── Sauvegarde finale (adapter PEFT + gate.pt) ──
    final_path = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    save_gate(gate, final_path, design_layer)
    logger.info(f"Modèle final + gate sauvegardés → {final_path}")

    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
        json.dump(history, _f, indent=2)

    # ── Plots ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["train"]["step"], history["train"]["ce"], alpha=0.25,
            color="darkorange", label="CE (raw)")
    ax.plot(history["train"]["step"], history["train"]["ce_ema"],
            color="darkorange", label="CE (EMA)")
    ax.plot(history["train"]["step"], history["train"]["leader_ce"],
            color="purple", alpha=0.6, label="leader CE (lookahead)")
    ax.plot(history["val"]["step"], history["val"]["loss"], color="steelblue",
            marker="o", markersize=4, label="val loss")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("Loss")
    ax.set_title("Training — Pythia-160M exp6 (Stackelberg + gating)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots → {plots_dir}")

    if use_wandb and not keep_wandb_open:
        with open(os.path.join(cfg.output_dir, "wandb_run_id.txt"), "w") as _f:
            _f.write(wandb.run.id)
        wandb.finish()

    gate.remove()
    if need_hook:
        for dl in _layer_ctx:
            _layer_ctx[dl]["capture"].remove()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stackelberg + Leader→Follower Gating — Pythia-160M exp6"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--run_name", default="stackelberg_exp6_pythia")
    parser.add_argument("--design_layer", type=int, default=9,
                        help="Design layer (Pythia-160M : 0-11)")
    parser.add_argument("--lr_leader", type=float, default=1e-4)
    parser.add_argument("--lr_follower", type=float, default=3e-4)
    parser.add_argument("--lr_gate", type=float, default=None,
                        help="LR dédié au MLP de gating (défaut = lr_leader). "
                             "Augmenter (ex. 1e-2) pour un effet plus marqué du gate.")
    parser.add_argument("--lr_sim", type=float, default=1e-3,
                        help="LR du simulated follower step (vanilla SGD)")
    parser.add_argument("--leader_idx", nargs="+", type=int, default=[0],
                        help="Indices des têtes leader (les autres sont followers). Ex: --leader_idx 0  ou  0 1 2")
    parser.add_argument("--gate_hidden", type=int, default=128,
                        help="Dim cachée du MLP de gating")
    # ── Losses exp3 (défaut 0 = CE + gate uniquement) ──
    parser.add_argument("--lambda_lead", type=float, default=0.0)
    parser.add_argument("--lambda_peer", type=float, default=0.0)
    parser.add_argument("--lambda_conf", type=float, default=0.0)
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--lambda_ldb", type=float, default=0.0)
    parser.add_argument("--conf_loss_type", choices=["max", "smooth", "entropy"], default="max")
    parser.add_argument("--div_loss_type",
                        choices=["cos", "cos_sq", "hadamard", "erank", "output_cos", "cka"],
                        default="cos")
    parser.add_argument("--nb_runs", type=int, default=5,
                        help="Nombre d'entraînements consécutifs (seeds seed, seed+1, …).")
    parser.add_argument("--run_eval", action="store_true", default=True,
                        help="Évaluation après training, métriques dans le même run wandb.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        model_name=args.model_name, dataset_name=args.dataset_name,
        dataset_config=args.dataset_config, total_tokens=args.total_tokens,
        batch_size_per_gpu=args.batch_size_per_gpu, grad_accum=args.grad_accum,
        lr=args.lr, output_dir=args.output_dir, wandb_project=args.wandb_project,
        wandb_group=args.wandb_group, run_name=args.run_name, seed=args.seed,
        dry_run=args.dry_run, log_every=args.log_every, eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches, save_every=args.save_every,
        num_workers=args.num_workers, random_init=args.random_init,
    )

    log_config(cfg)
    logger.info(f"  Design layer    : {args.design_layer}")
    logger.info(f"  Leader heads    : {args.leader_idx}")
    logger.info(f"  Gate hidden     : {args.gate_hidden}")
    logger.info(f"  LR leader/foll/sim : {args.lr_leader}/{args.lr_follower}/{args.lr_sim}")
    logger.info(f"  LR gate         : {args.lr_gate if args.lr_gate is not None else f'{args.lr_leader} (=lr_leader)'}")
    logger.info(f"  Nb runs         : {args.nb_runs}")

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
        logger.info(f"Anciens checkpoints supprimés : {args.output_dir}")

    _run_durations = []
    _train_wall_start = time.perf_counter()
    for i in range(args.nb_runs):
        cfg_i = dataclasses.replace(
            cfg, output_dir=os.path.join(args.output_dir, f"run_{i}"),
            run_name=args.run_name, seed=args.seed + i,
            wandb_project=cfg.wandb_project if i == 0 else None,
        )
        logger.info(f"\n{'='*60}\nRun {i+1}/{args.nb_runs}  seed={cfg_i.seed}  "
                    f"output={cfg_i.output_dir}\n{'='*60}")
        keep_open = args.run_eval and i == 0 and cfg.wandb_project is not None
        _t0 = time.perf_counter()
        train_stackelberg(
            cfg_i, design_layer=args.design_layer, lr_leader=args.lr_leader,
            lr_follower=args.lr_follower, lr_sim=args.lr_sim, lr_gate=args.lr_gate,
            lambda_lead=args.lambda_lead, lambda_peer=args.lambda_peer,
            lambda_conf=args.lambda_conf, lambda_rank=args.lambda_rank,
            lambda_ldb=args.lambda_ldb, conf_loss_type=args.conf_loss_type,
            div_loss_type=args.div_loss_type,
            leader_indices=args.leader_idx, gate_hidden=args.gate_hidden,
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
        logger.info(f"  Total training : {_total_train_s/60:.1f} min")

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
