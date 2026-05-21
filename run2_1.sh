#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --array=0-8

x=$SLURM_ARRAY_TASK_ID

LAMBDA_LEADS=(0.0 1e-4 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0)
LAMBDA_PEERS=(0.0 1e-4 3e-3 1e-2 3e-2 1e-1 3e-1 1.0 3.0)

RUN_NAME=Exp2_1_${x}
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp2/$RUN_NAME

python pythia160M/exp3/train_exp3.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp2_sliced --run_name $RUN_NAME \
    --leader_idx 0 \
    --lr_sim 1e-5 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --div_loss_type cos \
    --lambda_conf 0.0 \
    --lambda_lead ${LAMBDA_LEADS[$x]} \
    --lambda_peer ${LAMBDA_PEERS[$x]} \
