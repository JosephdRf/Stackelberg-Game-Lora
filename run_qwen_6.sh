#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --job-name=Exp6_qwen

# exp6 Qwen2.5-0.5B : Stackelberg bilevel (CE) + gating leader→followers au
# design layer. SDPA (pas d'eager), fp32, batch 4 × grad_accum 4 + grad ckpt.
# Le MLP de gating est un paramètre leader (lr_leader). Init gates ≡ 1.

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

RUN_NAME=Exp6_qwen
CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp6_qwen/$RUN_NAME

python qwen2.5_0.5B/exp6/train_exp6.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B --wandb_group Exp6 --run_name $RUN_NAME \
    --design_layer 19 \
    --leader_q_heads "0,1,2,3,4,5,6" \
    --leader_kv_heads "0" \
    --gate_hidden 128 \
    --lr_leader 3e-4 \
    --lr_follower 3e-4 \
    --lr_sim 1e-4 \
    --nb_runs 3 \
    --run_eval
