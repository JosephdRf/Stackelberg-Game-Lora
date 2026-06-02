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
#SBATCH --array=0-4

# exp6 Pythia-160M : sweep du lr dédié au MLP de gating (lr_gate).
# Avec lr_gate=lr_leader (3e-5) le gate bouge à peine → on teste des lr plus
# grands pour un effet plus marqué. Reste (lr_leader/follower/sim) inchangé.
# index → lr_gate :
LR_GATES=(3e-5 3e-4 1e-3 3e-3 1e-2)
LR_GATE=${LR_GATES[$SLURM_ARRAY_TASK_ID]}

RUN_NAME=Exp6_lrgate_${LR_GATE}
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

# nb_runs=1 pour le sweep (comparer les lr_gate). Relancer le meilleur en nb_runs=3.
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
    --nb_runs 1 \
    --run_eval
