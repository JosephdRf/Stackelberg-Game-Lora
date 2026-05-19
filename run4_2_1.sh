#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=9:00:00
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

# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RUN_NAME=Exp4_2_1
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME
CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp4/$RUN_NAME

python pythia160M/exp2/train_exp2.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp4 --run_name $RUN_NAME \
    --design_layer 6 7 8 9 \
    --leader_idx 0 \
    --lr_sim 1e-5 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lambda_conf 1.0 \
    --conf_loss_type max \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --div_loss_type cos \
