#!/bin/bash
#SBATCH --job-name=game-lora-baseline
#SBATCH --account=def-omar12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

# Charger les modules requis
module load python/3.10
module load cuda/11.8
module load intel/2023.2.1
module load StdEnv/2023


# Activer le virtualenv
source "$SLURM_SUBMIT_DIR/venv/bin/activate"

# Se placer dans le répertoire du projet
cd "$SLURM_SUBMIT_DIR"

# Configurer WandB pour le mode hors ligne
export WANDB_MODE=offline

# Configurer Hugging Face pour le mode hors ligne
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


# Lancer l'entraînement

python pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --wandb_project Stackelberg-Pythia160M --wandb_group Base --wandb_run_name Eval_seed_42 \
    "$@"
