#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --job-name=Baseline_qwen

# Estimation ressources :
#   - 5 runs × ~50-60 min (Qwen-0.5B ≈ 3.1× params Pythia, baseline 1 fwd/bwd) ≈ ~5h
#   - eval 7 benchmarks × 5 checkpoints en fp32 (HellaSwag = goulot) ≈ ~2-2.5h
#   → walltime 10h (marge confortable)
#   - mem 24G : modèle fp32 sur GPU (~2GB) + dataset WikiText-103 packé en RAM
#   - 10 CPUs pour num_workers=8 (dataloader)

# Modules
module load StdEnv/2023
module load python/3.10
module load cuda/12.2
module load intel/2023.2.1
module load arrow/21.0.0

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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/qwen_baseline

# float32 (pas de --bfloat16) + SDPA (pas de --attention_eager)
python qwen2.5_0.5B/baseline/train_baseline.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B \
    --wandb_group Baseline \
    --run_name Baseline_ref \
    --nb_runs 5 \
    --run_eval
