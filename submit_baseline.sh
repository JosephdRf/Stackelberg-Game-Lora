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

# Lancer l'entraînement
# python qwen2.5_0.5B/baseline/train_baseline.py \
#     --batch_size_per_gpu 4 \
#     --grad_accum 4 \
#     --lr 3e-4 \
#     "$@"
