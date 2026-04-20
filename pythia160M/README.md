# Pythia-160M — Stackelberg Game LoRA

Full fine-tuning de Pythia-160M sur WikiText-103.
Les métriques sont stockées sur WandB (projet `Stackelberg`), visibles dans le tableau des runs.

## 1. Eval du modèle de base (référence)

```bash
python pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --wandb_project Stackelberg --wandb_group baseline --wandb_run_name eval_base
```

## 2. Baseline full FT

```bash
python pythia160M/baseline/train_baseline.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/baseline \
    --wandb_project Stackelberg --wandb_group baseline --run_name seed_42

python pythia160M/eval.py \
    --model_path /Data/joseph.de-roffignac/checkpoints/baseline/final \
    --wandb_project Stackelberg --wandb_group baseline --wandb_run_name eval_baseline
```

## 3. GAME-LoRA

```bash
python pythia160M/game_lora/train_game_lora.py \
    --output_dir /Data/joseph.de-roffignac/checkpoints/game_lora \
    --wandb_project Stackelberg --wandb_group Game_Lora --run_name seed_42

python pythia160M/eval.py \
    --model_path /Data/joseph.de-roffignac/checkpoints/game_lora/final \
    --wandb_project Stackelberg --wandb_group Game_Lora --wandb_run_name eval_game_lora
```
