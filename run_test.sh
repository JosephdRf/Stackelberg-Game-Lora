#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=logs/%A.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com

# Exemple de script de lancement (exp3 Pythia-160M) — sert de gabarit.
# Stackelberg bilevel + losses diversity (cos) + confiance (entropy) + LDB.
RUN_NAME=Exp3_test
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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp3/$RUN_NAME

python pythia160M/exp3/train_exp3.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp3 --run_name $RUN_NAME \
    --design_layer 9 \
    --leader_idx 0 \
    --lr_leader 1e-4 \
    --lr_follower 3e-4 \
    --lr_sim 1e-3 \
    --div_loss_type cos \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --conf_loss_type entropy \
    --lambda_conf 0.05 \
    --lambda_ldb 1.0 \
    --nb_runs 5 \
    --run_eval
