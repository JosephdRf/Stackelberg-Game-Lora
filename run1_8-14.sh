#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=8-14
#SBATCH --output=/dev/null
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com

# lr_sim par run (index 0 = run 8, ..., index 6 = run 14)
LR_SIMS=(0 3e-5 1e-5 1e-4 3e-6 5e-6 2e-5)
LR_SIM=${LR_SIMS[$((SLURM_ARRAY_TASK_ID - 8))]}

RUN_NAME=Exp1_${SLURM_ARRAY_TASK_ID}
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load intel/2023.2.1
module load arrow/21.0.0

# Aller au projet
cd "$SLURM_SUBMIT_DIR"

# Purge des anciens logs (garder les 20 derniers .err)
ls -t logs/*.err 2>/dev/null | tail -n +21 | xargs -r rm --

# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp1/$RUN_NAME

python pythia160M/exp1/train_exp1.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp1 --run_name $RUN_NAME \
    --leader_idx 0 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lr_sim $LR_SIM \
