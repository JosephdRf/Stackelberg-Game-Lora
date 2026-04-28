#!/bin/bash
# Soumettre les 4 ablations : sbatch --export=RUN=1 submit_job_baseline.sh
# RUN=1 : baseline ref       (float32, SDPA)
# RUN=2 : + attention eager  (float32, eager)
# RUN=3 : + bfloat16         (bfloat16, SDPA)
# RUN=4 : + eager + bfloat16 (bfloat16, eager) → conditions modèle de exp1

#SBATCH --job-name=baseline-ablation
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
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

# Purge des anciens logs (garder les 10 derniers)
ls -t logs/baseline_*.out 2>/dev/null | tail -n +11 | xargs -r rm --
ls -t logs/baseline_*.err 2>/dev/null | tail -n +11 | xargs -r rm --

# Virtualenv
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Offline mode
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Sélection de la config selon RUN
case "${RUN:-1}" in
    1)
        RUN_NAME=Baseline_ref
        EXTRA_FLAGS=""
        ;;
    2)
        RUN_NAME=Baseline_eager
        EXTRA_FLAGS="--attention_eager"
        ;;
    3)
        RUN_NAME=Baseline_bf16
        EXTRA_FLAGS="--bfloat16"
        ;;
    4)
        RUN_NAME=Baseline_eager_bf16
        EXTRA_FLAGS="--attention_eager --bfloat16"
        ;;
    *)
        echo "RUN invalide : ${RUN}. Utiliser RUN=1,2,3 ou 4."
        exit 1
        ;;
esac

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/baseline_ablation/$RUN_NAME

python pythia160M/baseline/train_baseline.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Ablation --run_name $RUN_NAME \
    $EXTRA_FLAGS

python pythia160M/eval.py \
    --model_path $CKPT_DIR/final \
    --wandb_project Stackelberg-Pythia160M --wandb_group Ablation --wandb_run_name Eval_$RUN_NAME
