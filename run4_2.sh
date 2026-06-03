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

# Exp4_2_x (x=3,4,5) — Pythia-160M : multi-design-layer Stackelberg.
# Même hyperparamètres que Exp4_2_1/2_2 ; on fait varier le choix des couches.
#
# index 0 → Exp4_2_3 : [3,6,9,11]   étalé (vs [6,7,8,9] consécutif de 4_2_1)
# index 1 → Exp4_2_4 : [2,5,8,11]   espacement maximal (un layer tous les 3 blocs)
# index 2 → Exp4_2_5 : [8,9,10,11]  zone haute dense (étend [8,9,10] de 4_2_2 au dernier layer)

NAMES=(   "Exp4_2_3"   "Exp4_2_4"   "Exp4_2_5"  )
LAYERS=( "3 6 9 11"   "2 5 8 11"   "8 9 10 11"  )

RUN_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
DL=${LAYERS[$SLURM_ARRAY_TASK_ID]}

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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp4/$RUN_NAME

python pythia160M/exp3/train_exp3.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Pythia160M --wandb_group Exp4 --run_name $RUN_NAME \
    --design_layer $DL \
    --leader_idx 0 \
    --lr_sim 1e-5 \
    --lr_leader 3e-5 \
    --lr_follower 3e-5 \
    --lambda_conf 1.0 \
    --conf_loss_type max \
    --lambda_lead 1e-2 \
    --lambda_peer 1e-2 \
    --div_loss_type cos \
    --nb_runs 5 \
    --run_eval
