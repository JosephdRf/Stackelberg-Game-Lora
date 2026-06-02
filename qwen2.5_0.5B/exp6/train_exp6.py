"""
Experiment 6 — Stackelberg LoRA + Leader→Follower Gating (Qwen2.5-0.5B)
========================================================================

Base : exp1 (bilevel Stackelberg 3-phases, SDPA, CE only). On AJOUTE un
couplage STRUCTUREL leader→followers au design layer via un MLP de gating.

Architecture (design layer)
---------------------------
Sortie d'attention standard : o = Σ_h W_O^(h) z_h  (somme, poids fixe 1).
On pondère les contributions FOLLOWER par des gates calculés depuis le LEADER :

    s = concat({z_l}_{l∈leader})  ∈ R^{|L|·d_head}
    g = 2·σ(MLP(s))               ∈ (0,2)^{|F|}   (par token)
    o = Σ_{l∈L} W_O^(l) z_l  +  Σ_{i∈F} g_i · W_O^(i) z_i

cf. qwen2.5_0.5B/gate.py. Le gate s'enregistre en forward_pre_hook sur o_proj
→ agit APRÈS l'attention (sur concat(z_h)), donc compatible **SDPA** (pas
d'eager, pas de reconstruction de cartes d'attention).

Rôle Stackelberg : le MLP de gating est un paramètre LEADER (θ_L) :
  - masqué pour le follower (Phase 1)
  - gardé pour le leader (Phase 2), anticipé dans le lookahead
  - lr_leader
Init : gates ≡ 1 → couche identique au modèle pré-entraîné au départ.

Usage:
    python qwen2.5_0.5B/exp6/train_exp6.py --dry_run
    python qwen2.5_0.5B/exp6/train_exp6.py --design_layer 19 --gate_hidden 128
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

_HERE = os.path.dirname(os.path.abspath(__file__))   # qwen2.5_0.5B/exp6/
_MODEL = os.path.dirname(_HERE)                       # qwen2.5_0.5B/
sys.path.insert(0, _MODEL)  # train_utils, gradient_mask, gate, eval

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
    add_gate_roles,
    gate_param_ids,
)
from gate import LeaderFollowerGate, save_gate, gate_stats, gate_grad_norm

logger = logging.getLogger(__name__)

_NQ = QWEN_N_Q_HEADS      # 14
_NKV = QWEN_N_KV_HEADS    # 2
_DH = QWEN_D_HEAD         # 64
_GROUP = _NQ // _NKV      # 7


def _parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Stackelberg + gating training loop
# ---------------------------------------------------------------------------


def train_stackelberg(
    cfg: TrainConfig,
    design_layer: int = 19,
    lr_leader: float = 1e-4,
    lr_follower: float = 3e-4,
    lr_sim: float = 1e-3,
    lr_gate: float = None,
    leader_q_heads: list = None,
    leader_kv_heads: list = None,
    gate_hidden: int = 128,
    keep_wandb_open: bool = False,
):
    if leader_q_heads is None:
        leader_q_heads = list(range(7))
    if leader_kv_heads is None:
        leader_kv_heads = [0]
    if lr_gate is None:
        lr_gate = lr_leader   # rétro-compat : gate au lr_leader si non spécifié
    follower_q_heads = [h for h in range(_NQ) if h not in set(leader_q_heads)]

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
                "lr_gate": lr_gate,
                "design_layer": design_layer,
                "leader_q_heads": leader_q_heads,
                "leader_kv_heads": leader_kv_heads,
                "follower_q_heads": follower_q_heads,
                "gate_hidden": gate_hidden,
                "exp": "exp6_gating",
            },
        )

    # ── Model (float32, SDPA — PAS d'eager) ──
    model, tokenizer = build_model_and_tokenizer(cfg)
    model = model.to(device)

    # ── Gate leader→followers, enregistré sur o_proj du design layer ──
    gate = LeaderFollowerGate(
        leader_heads=leader_q_heads, follower_heads=follower_q_heads,
        d_head=_DH, n_heads=_NQ, hidden=gate_hidden,
    ).to(device)
    gate.register(model, design_layer)
    logger.info(
        f"Gate MLP : {sum(p.numel() for p in gate.parameters()):,} params  "
        f"(in={len(leader_q_heads)*_DH} → {gate_hidden} → {len(follower_q_heads)})  "
        f"init gates ≡ 1"
    )

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
        model,
        design_layers=[design_layer],
        leader_q_heads=leader_q_heads,
        leader_kv_heads=leader_kv_heads,
        n_q_heads=_NQ, n_kv_heads=_NKV, d_head=_DH, hidden_size=QWEN_HIDDEN_SIZE,
    )
    all_params = add_gate_roles(grad_assembly, all_params, gate)
    _gate_ids = gate_param_ids(grad_assembly)

    _decomposable_kinds = ("q_lora_B", "k_lora_B", "v_lora_B", "o_lora_A")
    n_design = sum(p.numel() for r in grad_assembly.roles
                   if r.kind in _decomposable_kinds for p in [r.param])
    n_gate = sum(p.numel() for r in grad_assembly.roles
                 if r.kind == "gate_leader" for p in [r.param])
    logger.info(f"Total trainable params : {sum(p.numel() for p in all_params):,}")
    logger.info(
        f"Design layer           : {design_layer}  |  Leader Q-heads : {leader_q_heads}"
        f"  |  Follower Q-heads : {follower_q_heads}"
    )
    logger.info(f"Design LoRA (θ_L∪θ_F) : {n_design:,}  |  Gate (θ_L) : {n_gate:,}")

    # 3 groupes lr : follower/shared (lr_follower), leader-LoRA o_lora_A (lr_leader),
    # gate MLP (lr_gate, séparé pour pouvoir l'accélérer indépendamment).
    leader_lora_ids = {
        id(r.param) for r in grad_assembly.roles if r.kind == "o_lora_A"
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

    # ── Directories & history ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    logs_dir = os.path.join(cfg.output_dir, "logs")
    plots_dir = os.path.join(cfg.output_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    history = {
        "train": {"step": [], "ce": [], "ce_ema": [], "leader_ce": []},
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
    optimizer.zero_grad()

    _step_start = time.perf_counter()
    pbar = tqdm(
        total=total_steps, desc="Stackelberg+Gate (Qwen2.5-0.5B exp6)", unit="step",
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

            follower_loss = ce_loss / cfg.grad_accum
            follower_loss.backward()
            accum_ce += ce_loss.item() / cfg.grad_accum

            if global_step == 0 and device.type == "cuda":
                logger.info(
                    f"VRAM pic step 0 : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
                    f"/ {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
                )

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
                                if r.kind in _decomposable_kinds]
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
                            "train/lr_leader": lr_l, "train/lr_follower": lr_f,
                            "train/tokens": opt_step * cfg.seq_len * cfg.effective_batch_size,
                        }
                        if _gs is not None:
                            log_dict["gate/mean_dev"] = _gs["mean_dev"]
                            log_dict["gate/token_std"] = _gs["token_std"]
                            log_dict["gate/saturation"] = _gs["saturation"]
                            log_dict["gate/grad_norm"] = _gate_gnorm
                            for _h, _v in zip(follower_q_heads, _gs["per_head"].tolist()):
                                log_dict[f"gate/head_{_h}"] = _v
                        wandb.log(log_dict, step=opt_step)
                    history["train"]["step"].append(opt_step)
                    history["train"]["ce"].append(accum_ce)
                    history["train"]["ce_ema"].append(_ema_ce)
                    history["train"]["leader_ce"].append(accum_leader_ce)

                if opt_step % cfg.eval_every == 0:
                    v_loss, v_ppl = evaluate(model, val_loader, device,
                                             max_batches=cfg.eval_max_batches,
                                             autocast_dtype=torch.bfloat16)
                    logger.info(f"[val]   step {opt_step:>6d}  val_loss={v_loss:.4f}  val_ppl={v_ppl:.3f}")
                    log_head_matrices(model, device, design_layer, opt_step, val_loader,
                                      wandb_mod=wandb if use_wandb else None,
                                      log_image=(opt_step % (cfg.eval_every * 5) == 0))
                    if use_wandb:
                        wandb.log({"val/loss": v_loss, "val/ppl": v_ppl}, step=opt_step)
                    history["val"]["step"].append(opt_step)
                    history["val"]["loss"].append(v_loss)
                    history["val"]["ppl"].append(v_ppl)

                if (opt_step % cfg.log_every == 0 or opt_step % cfg.eval_every == 0 or opt_step == 1):
                    with open(os.path.join(logs_dir, "history.json"), "w") as _f:
                        json.dump(history, _f, indent=2)

                accum_ce = 0.0
                accum_leader_ce = 0.0
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
    ax.set_title("Training — Qwen2.5-0.5B exp6 (Stackelberg + gating)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "loss.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots → {plots_dir}")

    if use_wandb and not keep_wandb_open:
        with open(os.path.join(cfg.output_dir, "wandb_run_id.txt"), "w") as _f:
            _f.write(wandb.run.id)
        wandb.finish()

    gate.remove()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stackelberg + Leader→Follower Gating — Qwen2.5-0.5B exp6"
    )
    parser = add_common_args(parser)
    parser.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--run_name", default="stackelberg_exp6_qwen")
    parser.add_argument("--design_layer", type=int, default=19,
                        help="Design layer (Qwen-0.5B : 0-23)")
    parser.add_argument("--lr_leader", type=float, default=1e-4)
    parser.add_argument("--lr_follower", type=float, default=3e-4)
    parser.add_argument("--lr_gate", type=float, default=None,
                        help="LR dédié au MLP de gating (défaut = lr_leader). "
                             "Augmenter (ex. 1e-2) pour un effet plus marqué du gate.")
    parser.add_argument("--lr_sim", type=float, default=1e-3,
                        help="LR du simulated follower step (vanilla SGD)")
    parser.add_argument("--leader_q_heads", type=str, default="0,1,2,3,4,5,6",
                        help="Q-heads leader (défaut groupe GQA complet [0..6])")
    parser.add_argument("--leader_kv_heads", type=str, default="0",
                        help="KV-heads leader (défaut '0')")
    parser.add_argument("--gate_hidden", type=int, default=128,
                        help="Dim cachée du MLP de gating")
    parser.add_argument("--nb_runs", type=int, default=3,
                        help="Nombre d'entraînements consécutifs (seeds seed, seed+1, …).")
    parser.add_argument("--run_eval", action="store_true", default=True,
                        help="Évaluation après training, métriques dans le même run wandb.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    leader_q = _parse_int_list(args.leader_q_heads)
    leader_kv = _parse_int_list(args.leader_kv_heads)

    cfg = TrainConfig(
        model_name=args.model_name, dataset_name=args.dataset_name,
        dataset_config=args.dataset_config, total_tokens=args.total_tokens,
        batch_size_per_gpu=args.batch_size_per_gpu, grad_accum=args.grad_accum,
        lr=args.lr, output_dir=args.output_dir, wandb_project=args.wandb_project,
        wandb_group=args.wandb_group, run_name=args.run_name, seed=args.seed,
        dry_run=args.dry_run, log_every=args.log_every, eval_every=args.eval_every,
        eval_max_batches=args.eval_max_batches, save_every=args.save_every,
        num_workers=args.num_workers, random_init=args.random_init,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank,
    )

    log_config(cfg)
    logger.info(f"  Design layer    : {args.design_layer}")
    logger.info(f"  Leader Q-heads  : {leader_q}")
    logger.info(f"  Leader KV-heads : {leader_kv}")
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
            leader_q_heads=leader_q, leader_kv_heads=leader_kv,
            gate_hidden=args.gate_hidden, keep_wandb_open=keep_open,
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
