#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --job-name=Eval_Exp2_2

# Ré-évaluation Exp2_2 (Pythia-160M) sur les 5 checkpoints (run_0..run_4/final),
# avec TOUTES les métriques (7 benchmarks actifs par défaut dans pythia160M/eval.py).
#
# eval.py rouvre AUTOMATIQUEMENT la run wandb d'origine via
#   checkpoints/exp2/Exp2_2/run_0/wandb_run_id.txt  →  ID = 4jb9hond
# (= la dernière run nommée Exp2_2, 2026-05-21). Les eval/* existantes seront
# écrasées par les nouvelles valeurs, les nouvelles clés (PIQA/ARC/MemoTrap/PTB)
# ajoutées. L'historique training n'est pas touché.

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

# Offline mode (nœud de calcul sans Internet) → sync ensuite depuis le login node
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --model_path = dossier parent : eval.py détecte run_*/final, évalue les 5,
# moyenne ± std, et reprend la run via run_0/wandb_run_id.txt.
python pythia160M/eval.py \
    --model_path $SLURM_SUBMIT_DIR/checkpoints/exp2/Exp2_2 \
    --wandb_project Stackelberg-Pythia160M \
    --wandb_group Exp2 \
    --wandb_run_name Exp2_2 \
    --seed 42

# Après le job, depuis le LOGIN NODE (Internet), pousser les nouvelles métriques :
#   wandb sync wandb/offline-run-*-4jb9hond
echo ""
echo ">>> Pour pousser sur wandb depuis le login node :"
echo ">>>   wandb sync \$(ls -dt wandb/offline-run-*-4jb9hond | head -1)"
