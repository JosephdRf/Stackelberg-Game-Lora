# Pythia-160M — Stackelberg Game LoRA

LoRA fine-tuning de Pythia-160M sur WikiText-103.
Les métriques sont stockées sur WandB (projet `Stackelberg-Pythia160M`), visibles dans le tableau des runs.

**Config LoRA commune** (baseline et GAME-LoRA) : `r=16`, `alpha=32`, `dropout=0.1`, `target_modules=[query_key_value, dense]`, `lr=3e-4`.

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

## 4. Stackelberg exp1 — bilevel optimization

Schéma de Stackelberg : leader = `dense` LoRA (projection de sortie), followers = `query_key_value` LoRA.
Par défaut `λ_lead=0` et `λ_peer=0` (CE pure, pour valider la boucle bilevel avant d'activer la diversité).

```bash
python pythia160M/exp1/train_exp1.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/exp1 \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --run_name stackelberg_exp1_seed_42

python pythia160M/eval.py \
    --model_path /Data/joseph.de-roffignac/checkpoints/exp1/final \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --wandb_run_name Eval_exp1_seed_42
```

Activer la diversité :
```bash
python pythia160M/exp1/train_exp1.py --lambda_lead 0.1 --lambda_peer 0.01 \
    --output_dir /Data/joseph.de-roffignac/checkpoints/exp1_div \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --run_name stackelberg_div_seed_42
```

## 5. Ablation studies (GAME-LoRA)

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
