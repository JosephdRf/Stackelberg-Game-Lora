#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=8-14
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com

# Recherche du lr_sim optimal pour Qwen2.5-0.5B exp1 (Stackelberg).
# lr_leader = lr_follower = 3e-4 (fixés).
#
# Grille lr_sim : ×10 vs Pythia (run1_8-14.sh) pour préserver les mêmes ratios
# lr_sim/lr_follower, puisque lr_follower passe de 3e-5 → 3e-4.
#   Pythia : (0  3e-5 1e-5 1e-4 3e-6 5e-6 2e-5)   (lr_follower=3e-5)
#   Qwen   : (0  3e-4 1e-4 1e-3 3e-5 5e-5 2e-4)   (lr_follower=3e-4)
# index 0 = run 8, ..., index 6 = run 14

LR_SIMS=(0 3e-4 1e-4 1e-3 3e-5 5e-5 2e-4)
LR_SIM=${LR_SIMS[$((SLURM_ARRAY_TASK_ID - 8))]}

RUN_NAME=Exp1_${SLURM_ARRAY_TASK_ID}
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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp1_qwen/$RUN_NAME

# nb_runs=1 : sweep → 1 run par config (comparer les lr_sim). Une fois le meilleur
# lr_sim trouvé, relancer avec nb_runs=5 pour la variance.
# float32 + SDPA (pas de --bfloat16, pas de --attention_eager).
python qwen2.5_0.5B/exp1/train_exp1.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B --wandb_group Exp1 --run_name $RUN_NAME \
    --design_layer 19 \
    --leader_q_heads "0,1,2,3,4,5,6" \
    --leader_kv_heads "0" \
    --lr_leader 3e-4 \
    --lr_follower 3e-4 \
    --lr_sim $LR_SIM \
    --nb_runs 1
