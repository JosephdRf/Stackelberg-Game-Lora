# Pythia-160M — Stackelberg Game LoRA

Full fine-tuning de Pythia-160M sur WikiText-103.
Les métriques sont stockées sur WandB (projet `Stackelberg-Pythia160M`), visibles dans le tableau des runs.

## 1. Eval du modèle de base (référence)

```bash
python pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --wandb_project Stackelberg-Pythia160M --wandb_group Base --wandb_run_name Eval_seed_42
```

## 2. Baseline full FT

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
