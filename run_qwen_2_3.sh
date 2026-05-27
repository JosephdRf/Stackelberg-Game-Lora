#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --job-name=Exp2_3

# Exp2_3 (Qwen) — confidence loss "L2-smooth" sur les têtes leader.
#   L_leader   = CE + λ_conf·(-1/BL Σ_{b,l} Σ_l' A_leader[l,l']²)     λ_conf = 1.0
#   L_follower = CE + λ_lead·cos(A_i,A_leaders) + λ_peer·cos(A_i,A_j)   (λ=1e-2)
LR_LEADER=3e-4
LR_FOLLOWER=3e-4
LR_SIM=1e-4

RUN_NAME=Exp2_3

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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp2_qwen/$RUN_NAME

python qwen2.5_0.5B/exp3/train_exp3.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B --wandb_group Exp2 --run_name $RUN_NAME \
    --design_layer 19 \
    --leader_idx 0 1 2 3 4 5 6 \
    --lr_sim $LR_SIM \
    --lr_leader $LR_LEADER \
    --lr_follower $LR_FOLLOWER \
    --conf_loss_type smooth \
    --lambda_conf 1.0 \
    --div_loss_type cos \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --nb_runs 5 \
    --run_eval
