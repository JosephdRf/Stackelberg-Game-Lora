# Qwen2.5-0.5B — Stackelberg Game LoRA

Pipeline Stackelberg porté sur `Qwen/Qwen2.5-0.5B` (GQA : 24 layers, 14 Q-heads /
2 KV-heads, 7 Q par groupe). LoRA `r=32`, `alpha=64`, targets `q/k/v/o_proj`, fp32,
WikiText-103. `design_layer=19`, leader = groupe GQA complet `[0..6]`.
Métriques sur WandB (projet `Stackelberg-Qwen0.5B`).

Voir le [README global](../README.md) pour la vue d'ensemble du repo et la liste des
expériences.

## Baseline (LoRA + CE seul)

```bash
python qwen2.5_0.5B/baseline/train_baseline.py --dry_run
python qwen2.5_0.5B/baseline/train_baseline.py \
    --output_dir checkpoints/baseline_qwen --run_name Baseline_seed_42 --run_eval
```

## Exp1 — Stackelberg bilevel (CE)

```bash
python qwen2.5_0.5B/exp1/train_exp1.py --dry_run
python qwen2.5_0.5B/exp1/train_exp1.py \
    --output_dir checkpoints/exp1_qwen --run_name Exp1_seed_42 \
    --design_layer 19 --leader_q_heads "0,1,2,3,4,5,6" --leader_kv_heads "0" \
    --run_eval
```

## Exp3 — bilevel + losses diversity / confiance / LDB

```bash
python qwen2.5_0.5B/exp3/train_exp3.py \
    --output_dir checkpoints/exp3_qwen --run_name Exp3_seed_42 \
    --design_layer 19 --leader_idx 0 1 2 3 4 5 6 \
    --div_loss_type cos --lambda_lead 1e-2 --lambda_peer 1e-2 \
    --conf_loss_type entropy --lambda_conf 0.05 --lambda_ldb 1.0 \
    --run_eval

# Validation de la reconstruction d'attention (A vs SDPA) :
python qwen2.5_0.5B/exp3/test_attention.py
```

## Exp6 — bilevel + gating leader→follower

Un MLP transforme le signal du leader en poids `g ∈ (0,2)` sur les sorties des
followers (init zéro → identité, donc neutre au départ). `--lr_gate` contrôle le LR
dédié du gate. Toutes les losses d'exp3 sont disponibles (toutes à 0 par défaut →
forward identique à exp1 + gate).

```bash
# Gate seul (CE + gating)
python qwen2.5_0.5B/exp6/train_exp6.py \
    --output_dir checkpoints/exp6_qwen --run_name Exp6_seed_42 \
    --design_layer 19 --leader_q_heads "0,1,2,3,4,5,6" --leader_kv_heads "0" \
    --gate_hidden 128 --lr_gate 1e-2 --run_eval

# Gate + losses exp3
python qwen2.5_0.5B/exp6/train_exp6.py \
    --output_dir checkpoints/exp6_qwen --run_name Exp6_full \
    --design_layer 19 --leader_q_heads "0,1,2,3,4,5,6" --leader_kv_heads "0" \
    --gate_hidden 128 --lr_gate 3e-3 \
    --div_loss_type cos --lambda_lead 1e-2 --lambda_peer 1e-2 \
    --conf_loss_type entropy --lambda_conf 0.05 --lambda_ldb 5.0 --run_eval
```

> **Mode KV-follower** : passer `--leader_q_heads "0" --leader_kv_heads ""` met le
> leader sur la seule tête Q 0 (params Q/O exclusifs), la KV-head 0 — partagée par
> les têtes 0-6 du groupe GQA — restant follower. Maximise le nombre de têtes gated
> (13 au lieu de 7).

## Évaluation (7 benchmarks)

`eval.py` reprend le run WandB du checkpoint. Les 7 benchmarks sont identiques à
ceux de Pythia (comparaison directe) : WikiText103, PTB, LAMBADA, HellaSwag, PIQA,
ARC-Easy, MemoTrap.

```bash
# Modèle de base (référence)
python qwen2.5_0.5B/eval.py --model_path Qwen/Qwen2.5-0.5B \
    --wandb_project Stackelberg-Qwen0.5B --wandb_run_name Eval_base

# Checkpoint LoRA fine-tuné
python qwen2.5_0.5B/eval.py \
    --model_path checkpoints/exp1_qwen/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --wandb_project Stackelberg-Qwen0.5B --wandb_run_name Eval_exp1
```
