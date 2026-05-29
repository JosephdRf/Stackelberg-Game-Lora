# Qwen2.5-0.5B — Stackelberg Game LoRA

## Entraînement baseline

```bash
python qwen2.5_0.5B/baseline/train_baseline.py --dry_run
python qwen2.5_0.5B/baseline/train_baseline.py
```

## Entraînement GAME-LoRA

```bash
python qwen2.5_0.5B/game_lora/train_game_lora.py --dry_run
python qwen2.5_0.5B/game_lora/train_game_lora.py
python qwen2.5_0.5B/game_lora/train_game_lora.py --no_nash_mtl
```

## Entraînement Stackelberg (Exp1)

```bash
python qwen2.5_0.5B/exp1/train_exp1.py --dry_run
python qwen2.5_0.5B/exp1/train_exp1.py
```

## Évaluation

```bash
# Modèle de base
python qwen2.5_0.5B/eval.py --model_path Qwen/Qwen2.5-0.5B --csv_column base

# Baseline fine-tuné
python qwen2.5_0.5B/eval.py \
    --model_path qwen2.5_0.5B/baseline/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column baseline

# GAME-LoRA
python qwen2.5_0.5B/eval.py \
    --model_path qwen2.5_0.5B/game_lora/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column game-lora

# Stackelberg
python qwen2.5_0.5B/eval.py \
    --model_path qwen2.5_0.5B/exp1/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column exp1
```
