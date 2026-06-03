#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --array=0-2

# exp6 Pythia-160M : gate + losses run2_45 (div cos + conf entropy + LDB),
# sweep lambda_ldb (2/5/10). lr_gate FIXE.
# index → lambda_ldb :
LAMBDA_LDBS=(2.0 5.0 10.0)
LDB=${LAMBDA_LDBS[$SLURM_ARRAY_TASK_ID]}

LR_GATE=1e-2     # FIXE — à ajuster avec le meilleur lr_gate de run6_1

RUN_NAME=Exp6_2_ldb_${LDB}
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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp6/$RUN_NAME

# nb_runs=1 pour le sweep. Relancer le meilleur en nb_runs=5.
python pythia160M/exp6/train_exp6.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp6 --run_name $RUN_NAME \
    --design_layer 9 \
    --leader_idx 0 \
    --gate_hidden 128 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lr_sim 1e-5 \
    --lr_gate $LR_GATE \
    --div_loss_type cos \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --conf_loss_type entropy \
    --lambda_conf 0.05 \
    --lambda_ldb $LDB \
    --nb_runs 5 \
    --run_eval
