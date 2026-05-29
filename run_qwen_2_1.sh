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
#SBATCH --array=0-8

# Exp2_1 (Qwen) — sweep diversity loss (followers), leader = CE seul.
#   L_follower = CE + λ_lead·(1/F)Σ_i cos(A_i,A_leaders) + λ_peer·(1/F(F-1))Σ_{i≠j} cos(A_i,A_j)
#   L_leader   = CE
# Miroir de pythia run2_1 (même grille λ). Forme MEAN (pair-normalisée) → λ
# transférable malgré 14 têtes (vs 12 Pythia).
x=$SLURM_ARRAY_TASK_ID

LAMBDAS=(0 1e-4 3e-3 1e-2 3e-2 1e-1 3e-1 1e0 3e0)
LAM=${LAMBDAS[$x]}

# lr ×10 vs pythia Exp2 (échelle LoRA d'exp1 : 3e-4/3e-4/1e-4).
LR_LEADER=3e-4
LR_FOLLOWER=3e-4
LR_SIM=1e-4

RUN_NAME=Exp2_1_${x}
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load intel/2023.2.1
module load arrow/21.0.0

cd "$SLURM_SUBMIT_DIR"

# Purge des anciens logs (garder les 20 derniers .err)
ls -t logs/*.err 2>/dev/null | tail -n +21 | xargs -r rm --

# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
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
    --div_loss_type cos \
    --lambda_conf 0.0 \
    --lambda_lead $LAM \
    --lambda_peer $LAM \
    --nb_runs 3 \
    --run_eval
