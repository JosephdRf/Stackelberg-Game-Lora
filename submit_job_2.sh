#!/bin/bash
#SBATCH --job-name=joseph-stackelberg-job
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
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
ls -t logs/baseline_*.out 2>/dev/null | tail -n +11 | xargs -r rm --
ls -t logs/baseline_*.err 2>/dev/null | tail -n +11 | xargs -r rm --


# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run et evals
RUN_NAME_TRAIN=Train_stackelberg_exp1_5
RUN_NAME_EVAL=Eval_exp1_5
CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp1/$RUN_NAME_TRAIN

python pythia160M/exp1/train_exp1.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --run_name $RUN_NAME_TRAIN \
    --lr_sim 3e-4 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lambda_lead 0 \
    --lambda_peer 0

python pythia160M/eval.py \
    --model_path $CKPT_DIR/final \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --wandb_run_name $RUN_NAME_EVAL
