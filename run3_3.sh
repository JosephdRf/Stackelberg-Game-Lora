#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --array=0-3

RANKS=(16 32 64 128)
RANK=${RANKS[$SLURM_ARRAY_TASK_ID]}

RUN_NAME=Exp3_3_${RANK}
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME

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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp3/$RUN_NAME

python pythia160M/exp3/train_exp3.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp3 --run_name $RUN_NAME \
    --leader_idx 0 \
    --lr_sim 1e-5 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lambda_lead 0.01 \
    --lambda_peer 0.01 \
    --lambda_conf 0.0 \
    --div_loss_type cos \
    --lora_rank $RANK \
 