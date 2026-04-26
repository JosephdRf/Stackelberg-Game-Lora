#!/bin/bash
#SBATCH --job-name=game-lora-baseline
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err


# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/11.8
module load intel/2023.2.1
module load arrow/21.0.0


# Aller au projet
cd "$SLURM_SUBMIT_DIR"

# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Runs and evals
python -u pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --wandb_project Stackelberg-Pythia160M \
    --wandb_group Base \
    --wandb_run_name Eval_testCC \
    "$@"