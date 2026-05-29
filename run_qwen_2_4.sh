#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --array=0-5

# Exp2_4 (Qwen) — confidence loss "entropy" (maximise l'entropie des têtes leader),
# sweep λ_conf. Miroir de pythia run2_4.
#   L_leader   = CE + λ_conf·(1/BL Σ_{b,l} Σ_l' A_leader[l,l']·log A_leader[l,l'])
#   L_follower = CE + λ_lead·cos(A_i,A_leaders) + λ_peer·cos(A_i,A_j)   (λ=1e-2)
# index 0 → Exp2_4_1, ..., index 5 → Exp2_4_6
x=$SLURM_ARRAY_TASK_ID

LAMBDA_CONFS=(0.001 0.01 0.05 0.2 1.0 10.0)
LAM_CONF=${LAMBDA_CONFS[$x]}

LR_LEADER=3e-4
LR_FOLLOWER=3e-4
LR_SIM=1e-4

RUN_NAME=Exp2_4_$((x+1))
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load intel/2023.2.1
module load arrow/21.0.0

cd "$SLURM_SUBMIT_DIR"

ls -t logs/*.err 2>/dev/null | tail -n +21 | xargs -r rm --

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
    --conf_loss_type entropy \
    --lambda_conf $LAM_CONF \
    --div_loss_type cos \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --nb_runs 3 \
    --run_eval
