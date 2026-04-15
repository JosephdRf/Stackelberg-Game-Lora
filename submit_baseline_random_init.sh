#!/bin/bash
#SBATCH --job-name=game-lora-random-init
#SBATCH --account=def-omar12
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/baseline_random_init_%j.out
#SBATCH --error=logs/baseline_random_init_%j.err

module load gcc arrow/21.0.0

source "$SLURM_SUBMIT_DIR/venv/bin/activate"

cd "$SLURM_SUBMIT_DIR"

python baseline_random_init/train_baseline_random_init.py \
    --batch_size_per_gpu 4 \
    --grad_accum 4 \
    "$@"
