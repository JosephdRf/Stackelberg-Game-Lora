#!/bin/bash
#SBATCH --account=def-omar12
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/dev/null
#SBATCH --error=logs/%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joseph.deroffignac@gmail.com
#SBATCH --job-name=eval_missing

# Usage : bash eval_missing.sh <run_id1> [run_id2 ...]
#   → soumet un job SLURM séparé (1h, A100) par run ID.
#   Chaque job évalue tous les sous-runs (run_*/final) du checkpoint,
#   reprend le run wandb existant (resume="allow") et uploade via wandb sync.

# Si on n'est pas dans un job SLURM, on est le lanceur.
if [ -z "$SLURM_JOB_ID" ]; then
    SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
    CSV="$SUBMIT_DIR/run_ids_evaluation.csv"

    if [ $# -gt 0 ]; then
        # IDs passés explicitement en argument
        IDS=("$@")
    else
        # Lecture du CSV : garde les lignes dont le Name commence par Exp
        if [ ! -f "$CSV" ]; then
            echo "ERREUR : $CSV introuvable et aucun run ID fourni en argument."
            exit 1
        fi
        mapfile -t IDS < <(python3 - "$CSV" <<'EOF'
import csv, sys
with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        if row["Name"].startswith("Exp"):
            print(row["ID"])
EOF
        )
        echo "CSV : ${#IDS[@]} runs Exp* trouvés dans $CSV"
    fi

    if [ ${#IDS[@]} -eq 0 ]; then
        echo "Aucun run ID à traiter."
        exit 1
    fi

    for RUN_ID in "${IDS[@]}"; do
        sbatch "$0" "$RUN_ID"
    done
    exit 0
fi

# À partir d'ici : on est dans un job SLURM, $1 est le seul run ID à traiter.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
WANDB_DIR="$SUBMIT_DIR/wandb"

module load StdEnv/2023 python/3.10 cuda/12.2 intel/2023.2.1 arrow/21.0.0 2>/dev/null || true

cd "$SUBMIT_DIR"
source "$SUBMIT_DIR/.venv/bin/activate"

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------------------

for RUN_ID in "$@"; do
    echo ""
    echo "========================================"
    echo "Run ID : $RUN_ID"
    echo "========================================"

    # -- Trouve le répertoire offline du run original --
    OFFLINE_ORIG=$(ls -d "$WANDB_DIR/offline-run-"*"-${RUN_ID}" 2>/dev/null | head -1)
    if [ -z "$OFFLINE_ORIG" ]; then
        echo "ERREUR : aucun répertoire offline-run-*-${RUN_ID} trouvé dans $WANDB_DIR"
        continue
    fi
    CONFIG="$OFFLINE_ORIG/files/config.yaml"
    echo "  Offline dir : $OFFLINE_ORIG"

    # -- Extrait output_dir (parent du run_0) et wandb_project depuis config.yaml --
    read OUTPUT_DIR WANDB_PROJECT < <(python3 - "$CONFIG" <<'EOF'
import sys, re, yaml
with open(sys.argv[1]) as f:
    c = yaml.safe_load(f)
d = c.get("output_dir", {}).get("value", "")
d = re.sub(r"/run_\d+$", "", d)      # retire le suffixe /run_N si présent
p = c.get("wandb_project", {}).get("value", "Stackelberg")
print(d, p)
EOF
    )

    if [ -z "$OUTPUT_DIR" ]; then
        echo "ERREUR : impossible d'extraire output_dir depuis $CONFIG"
        continue
    fi
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERREUR : répertoire checkpoint introuvable : $OUTPUT_DIR"
        continue
    fi
    echo "  Checkpoint  : $OUTPUT_DIR"
    echo "  Project     : $WANDB_PROJECT"

    # -- Ancienne valeur WikiText103_BPB (sanity check) --
    OLD_BPB=$(python3 -c "
import json
try:
    with open('$OFFLINE_ORIG/files/wandb-summary.json') as f:
        print(json.load(f).get('eval/WikiText103_BPB', ''))
except Exception:
    print('')
")
    [ -n "$OLD_BPB" ] && echo "  WikiText103_BPB (ancien) : $OLD_BPB"

    # -- Snapshot des répertoires offline existants (pour trouver le nouveau) --
    BEFORE=$(ls -d "$WANDB_DIR/offline-run-"* 2>/dev/null | sort)

    # -- Lance eval.py (résume le run via wandb_run_id.txt dans le checkpoint) --
    python pythia160M/eval.py \
        --model_path   "$OUTPUT_DIR" \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "eval_${RUN_ID}"

    EVAL_STATUS=$?
    if [ $EVAL_STATUS -ne 0 ]; then
        echo "ERREUR : eval.py a échoué (code $EVAL_STATUS) pour $RUN_ID"
        continue
    fi

    # -- Identifie le(s) nouveau(x) répertoire(s) offline créés par eval --
    AFTER=$(ls -d "$WANDB_DIR/offline-run-"* 2>/dev/null | sort)
    NEW_DIRS=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER"))

    if [ -z "$NEW_DIRS" ]; then
        echo "AVERTISSEMENT : aucun nouveau répertoire offline créé pour $RUN_ID"
        continue
    fi

    # -- Sanity check : WikiText103_BPB doit être stable --
    BPB_OK=1
    for NEW_DIR in $NEW_DIRS; do
        echo "  Nouveau répertoire offline : $NEW_DIR"
        if [ -n "$OLD_BPB" ]; then
            python3 -c "
import json, sys
try:
    with open('$NEW_DIR/files/wandb-summary.json') as f:
        new = json.load(f).get('eval/WikiText103_BPB')
except Exception:
    new = None
if new is None:
    print('  [?] WikiText103_BPB absent du nouveau summary — vérification impossible')
    sys.exit(0)
old  = $OLD_BPB
diff = abs(old - new)
pct  = diff / old * 100
if diff < 0.002:
    print(f'  [OK] WikiText103_BPB : ancien={old:.4f}  nouveau={new:.4f}  |diff|={diff:.4f} ({pct:.2f}%)')
    sys.exit(0)
else:
    print(f'  [ERREUR] WikiText103_BPB diverge : ancien={old:.4f}  nouveau={new:.4f}  |diff|={diff:.4f} ({pct:.2f}%)')
    print(f'           Le modèle chargé ne correspond pas — ne pas synchroniser ce run.')
    sys.exit(1)
"
            if [ $? -ne 0 ]; then
                BPB_OK=0
            fi
        fi
    done

    if [ $BPB_OK -eq 0 ]; then
        echo "  ABORT — run $RUN_ID ignoré (BPB incohérent, répertoire offline conservé mais non synchronisé)"
        continue
    fi
    echo "  OK — eval terminée pour $RUN_ID (sync à faire depuis le login node)"
done

echo ""
echo "Terminé."
