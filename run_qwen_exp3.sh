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
#SBATCH --job-name=Exp3_qwen

# exp3 Qwen2.5-0.5B (Stackelberg + diversity + confidence + LDB).
# Adapté de run2_45.sh (qui lance pythia exp3), config losses identique.
#
# Ressources : exp3 utilise eager attention + reconstruction des cartes
# d'attention à chaque step → nettement plus lent qu'exp1.
#   ~2.5-3h/run × 5 runs + eval (7 benchmarks fp32 × 5 ckpt) ≈ ~15-18h
#   → walltime 24h (marge). Si trop long : réduire --nb_runs ou splitter.

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

RUN_NAME=Exp3_qwen
CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp3_qwen/$RUN_NAME

# fp32 (pas de --bfloat16) ; exp3 force eager attention en interne.
# Leaders = groupe GQA complet [0..6] (KV head 0). lr_leader=lr_follower=3e-4.
python qwen2.5_0.5B/exp3/train_exp3.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B --wandb_group Exp3 --run_name $RUN_NAME \
    --design_layer 19 \
    --leader_idx 0 1 2 3 4 5 6 \
    --lr_leader 3e-4 \
    --lr_follower 3e-4 \
    --lr_sim 1e-3 \
    --conf_loss_type entropy \
    --lambda_conf 0.05 \
    --div_loss_type cos \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --lambda_ldb 0.0 \
    --nb_runs 5 \
    --run_eval
