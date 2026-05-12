#!/bin/bash
#SBATCH --job-name=joseph-stackelberg-job
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load intel/2023.2.1
module load arrow/21.0.0

# Aller au projet
cd "$SLURM_SUBMIT_DIR"

# Purge des anciens logs (garder les 10 derniers)
ls -t logs/*.out 2>/dev/null | tail -n +11 | xargs -r rm --
ls -t logs/*.err 2>/dev/null | tail -n +11 | xargs -r rm --


# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run et evals
CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp2/$RUN_NAME

python pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --wandb_run_name Eval_base_recent \
    --wandb_group Base
