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
module load gcc arrow/21.0.0

# Activer le virtualenv
source "$SLURM_SUBMIT_DIR/venv/bin/activate"

# Se placer dans le répertoire du projet
cd "$SLURM_SUBMIT_DIR"

# Lancer l'entraînement
python train_baseline.py \
    --batch_size_per_gpu 4 \
    --grad_accum 4 \
    --output_dir ./checkpoints/baseline \
    "$@"
