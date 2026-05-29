#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --job-name=Exp6

# exp6 Pythia-160M : Stackelberg bilevel (CE) + gating leader→followers au
# design layer (hook sur attention.dense). Le MLP de gating est un paramètre
# leader (lr_leader). Init gates ≡ 1.
# lr alignés sur les runs pythia exp2 (3e-5/3e-5/1e-5) pour comparabilité.

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load intel/2023.2.1
module load arrow/21.0.0

cd "$SLURM_SUBMIT_DIR"

ls -t logs/*.out 2>/dev/null | tail -n +11 | xargs -r rm --
ls -t logs/*.err 2>/dev/null | tail -n +11 | xargs -r rm --

source $SLURM_SUBMIT_DIR/.venv/bin/activate

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_NAME=Exp6
CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp6/$RUN_NAME

python pythia160M/exp6/train_exp6.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp6 --run_name $RUN_NAME \
    --design_layer 9 \
    --leader_idx 0 \
    --gate_hidden 128 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lr_sim 1e-5 \
    --nb_runs 5 \
    --run_eval
