#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com

# Ré-évaluation des 4 baselines d'ablation Pythia-160M avec TOUTES les métriques
# (les 7 benchmarks sont actifs par défaut dans pythia160M/eval.py).
# Chaque tâche évalue un checkpoint ; aucun wandb_run_id.txt → nouveaux runs Eval_*.
MODELS=(Baseline_ref Baseline_eager Baseline_bf16 Baseline_eager_bf16)
M=${MODELS[$SLURM_ARRAY_TASK_ID]}

RUN_NAME=Eval_${M}
scontrol update JobId=$SLURM_JOB_ID JobName=$RUN_NAME

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

CKPT=$SLURM_SUBMIT_DIR/checkpoints/baseline_ablation/$M/final

python pythia160M/eval.py \
    --model_path $CKPT \
    --wandb_project Stackelberg-Pythia160M \
    --wandb_group Ablation \
    --wandb_run_name $RUN_NAME \
    --seed 42
