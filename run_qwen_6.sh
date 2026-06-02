#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=logs/%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --array=0-4

# exp6 Qwen2.5-0.5B : sweep du lr dédié au MLP de gating (lr_gate).
# Stackelberg bilevel (CE) + gating leader→followers, SDPA, fp32, grad ckpt.
# lr_leader/follower = 3e-4 ; on teste lr_gate de 3e-4 (= leader) à 3e-2.
# index → lr_gate :
LR_GATES=(3e-4 1e-3 3e-3 1e-2 3e-2)
LR_GATE=${LR_GATES[$SLURM_ARRAY_TASK_ID]}

RUN_NAME=Exp6_qwen_lrgate_${LR_GATE}
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

CKPT_DIR=$SLURM_SUBMIT_DIR/checkpoints/exp6_qwen/$RUN_NAME

# nb_runs=1 pour le sweep (comparer les lr_gate). Relancer le meilleur en nb_runs=3.
python qwen2.5_0.5B/exp6/train_exp6.py \
    --output_dir $CKPT_DIR \
    --wandb_project Stackelberg-Qwen0.5B --wandb_group Exp6 --run_name $RUN_NAME \
    --design_layer 19 \
    --leader_q_heads "0,1,2,3,4,5,6" \
    --leader_kv_heads "0" \
    --gate_hidden 128 \
    --lr_leader 3e-4 \
    --lr_follower 3e-4 \
    --lr_sim 1e-4 \
    --lr_gate $LR_GATE \
    --nb_runs 1 \
    --run_eval
