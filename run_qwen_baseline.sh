#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --array=0-3

# Ablation baseline Qwen2.5-0.5B (miroir de pythia baseline_ablation) :
#   index 0 : Baseline_ref           — fp32 + SDPA            (référence)
#   index 1 : Baseline_eager         — fp32 + eager attention
#   index 2 : Baseline_bf16          — bf16 + SDPA
#   index 3 : Baseline_eager_bf16    — bf16 + eager attention
# nb_runs=1 par variante (comme pythia : un seul `final/` par config).

x=$SLURM_ARRAY_TASK_ID

NAMES=(Baseline_ref Baseline_eager Baseline_bf16 Baseline_eager_bf16)
FLAGS=(""           "--attention_eager"  "--bfloat16"  "--attention_eager --bfloat16")

RUN_NAME=${NAMES[$x]}
EXTRA_FLAGS=${FLAGS[$x]}

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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/baseline_ablation_qwen/$RUN_NAME

python qwen2.5_0.5B/baseline/train_baseline.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B \
    --wandb_group Ablation \
    --run_name $RUN_NAME \
    --nb_runs 1 \
    --run_eval \
    $EXTRA_FLAGS
