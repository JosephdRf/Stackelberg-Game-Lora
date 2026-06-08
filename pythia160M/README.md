# Pythia-160M — Stackelberg Game LoRA

LoRA fine-tuning de Pythia-160M sur WikiText-103.
Les métriques sont stockées sur WandB (projet `Stackelberg-Pythia160M`), visibles dans le tableau des runs.

**Config LoRA commune** (baseline et GAME-LoRA) : `r=16`, `alpha=32`, `dropout=0.1`, `target_modules=[query_key_value, dense]`, `lr=3e-4`.

Voir le [README global](../README.md) pour la vue d'ensemble du repo et la liste des expériences.

## 1. Eval du modèle de base (référence)

```bash
python pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --wandb_project Stackelberg-Pythia160M --wandb_group Base --wandb_run_name Eval_seed_42
```

## 2. Baseline LoRA (CE seul)

```bash
python pythia160M/baseline/train_baseline.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/baseline \
    --wandb_project Stackelberg-Pythia160M --wandb_group Baseline --run_name Train_seed_42

python pythia160M/eval.py \
    --model_path /Data/joseph.de-roffignac/checkpoints/baseline/final \
    --wandb_project Stackelberg-Pythia160M --wandb_group Baseline --wandb_run_name Eval_baseline_seed_42
```

## 3. GAME-LoRA

```bash
python pythia160M/game_lora/train_game_lora.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/game_lora \
    --wandb_project Stackelberg-Pythia160M --wandb_group Game_Lora --run_name Train_seed_42

python pythia160M/eval.py \
    --model_path /Data/joseph.de-roffignac/checkpoints/game_lora/final \
    --wandb_project Stackelberg-Pythia160M --wandb_group Game_Lora --wandb_run_name Eval_game_lora
```


## 4. Ablation studies (GAME-LoRA)

Les flags `--no_ldb` et `--no_abt` permettent de désactiver chaque terme de la loss GAME.

| Flags | Loss effective | Remarque |
|---|---|---|
| *(aucun)* | CE + λ_LDB · L_LDB + λ_ABT · L_ABT | GAME-LoRA complet |
| `--no_abt` | CE + λ_LDB · L_LDB | LDB seul |
| `--no_ldb` | CE + λ_ABT · L_ABT | ABT seul |
| `--no_ldb --no_abt` | CE seul | LoRA seul (**≠ baseline** : LoRA uniquement, pas full fine-tuning) |

```bash
# Ablation : LDB seul
python pythia160M/game_lora/train_game_lora.py --no_abt \
    --output_dir /Data/joseph.de-roffignac/checkpoints/ablation_ldb \
    --wandb_project Stackelberg-Pythia160M --wandb_group Ablation --run_name ldb_only_seed_42

# Ablation : ABT seul
python pythia160M/game_lora/train_game_lora.py --no_ldb \
    --output_dir /Data/joseph.de-roffignac/checkpoints/ablation_abt \
    --wandb_project Stackelberg-Pythia160M --wandb_group Ablation --run_name abt_only_seed_42

# LoRA seul (CE uniquement — sans les losses GAME)
python pythia160M/game_lora/train_game_lora.py --no_ldb --no_abt \
    --output_dir /Data/joseph.de-roffignac/checkpoints/ablation_lora_only \
    --wandb_project Stackelberg-Pythia160M --wandb_group Ablation --run_name lora_only_seed_42
```


## 5. Stackelberg exp1 — bilevel optimization

Schéma de Stackelberg : leader = première tête d'attention (`query_key_value` LoRA, head 0), followers = têtes d'attention restantes.
Par défaut `λ_lead=0.1` et `λ_peer=0.01`. Passer `--lambda_lead 0 --lambda_peer 0` pour CE pure (valider la boucle bilevel sans diversité).

```bash
python pythia160M/exp1/train_exp1.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/exp1 \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --run_name Train_stackelberg_exp1_n

python pythia160M/eval.py \
    --model_path /Data/joseph.de-roffignac/checkpoints/exp1/final \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --wandb_run_name Eval_exp1_n
```


## 6. Stackelberg exp3 — bilevel + losses diversity / confiance / LDB

exp1 enrichi : diversity loss sur les followers (`cos/cos_sq/hadamard/erank/output_cos/cka`),
confidence loss sur les leaders (`max/smooth/entropy`), barrière log-det (LDB) ; supporte
multi-leaders et multi-design-layers.

```bash
python pythia160M/exp3/train_exp3.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/exp3 \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp3 --run_name Exp3_seed_42 \
    --design_layer 9 --leader_idx 0 \
    --div_loss_type cos --lambda_lead 1e-2 --lambda_peer 1e-2 \
    --conf_loss_type entropy --lambda_conf 0.05 --lambda_ldb 1.0 --run_eval
```


## 7. Stackelberg exp6 — bilevel + gating leader→follower

Un MLP transforme le signal du leader en poids `g ∈ (0,2)` appliqués aux sorties des
têtes followers (init zéro → `g≡1`, identité au départ). `--lr_gate` = LR dédié du gate.
Toutes les losses d'exp3 sont disponibles (toutes à 0 par défaut → exp1 + gate pur).

```bash
# Gate seul (CE + gating)
python pythia160M/exp6/train_exp6.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/exp6 \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp6 --run_name Exp6_seed_42 \
    --design_layer 9 --leader_idx 0 --gate_hidden 128 --lr_gate 1e-2 --run_eval

# Gate + losses exp3
python pythia160M/exp6/train_exp6.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/exp6 \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp6 --run_name Exp6_full \
    --design_layer 9 --leader_idx 0 --gate_hidden 128 --lr_gate 1e-2 \
    --div_loss_type cos --lambda_lead 1e-2 --lambda_peer 1e-2 \
    --conf_loss_type entropy --lambda_conf 0.05 --lambda_ldb 5.0 --run_eval
```
