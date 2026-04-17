# Pythia-160M — Stackelberg Game LoRA

Full fine-tuning de Pythia-160M sur WikiText-103, avec comparaison baseline vs GAME-LoRA / Stackelberg.

## Usage

### 1. Eval du modèle de base (référence)

```bash
python pythia160M/eval.py --model_path EleutherAI/pythia-160m --csv_column base
```

### 2. Baseline full FT

```bash
python pythia160M/baseline/train_baseline.py --wandb_project stackelberg
python pythia160M/eval.py --model_path pythia160M/baseline/checkpoints/final --csv_column baseline_fullft
```
