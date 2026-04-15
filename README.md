# Stackelberg Game LoRA

Fine-tuning LoRA avec régularisations game-théoriques sur les attention heads.
Deux modèles disponibles en parallèle :

| Modèle | Dossier | Params | Têtes | Couches | dtype |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B** | `qwen2.5_0.5B/` | 500M | 14 | 24 | bfloat16 |
| **Pythia-160M** | `pythia160M/` | 162M | 12 | 12 | bfloat16 |

Les deux modèles partagent le même protocole : 20M tokens sur The Pile,
LoRA rank=16, batch effectif=16, cosine schedule, 2% warmup.


## Expériences

### 1. Baseline
LoRA fine-tuning standard : `loss = L_CE`

### 2. GAME-LoRA (`game_lora/` — Qwen uniquement pour l'instant)
Reproduction de "Multi-Head Attention is a Multi-Player Game" (Chakrabarti & Balachundar, 2026) :
- `L = L_CE + λ_LDB · L_LDB + λ_ABT · L_ABT` (Eq 27)
- Gradients arbitrés via Nash-MTL (Navon et al., 2022)

### 3. Stackelberg Attention Diversity (`exp1/` — Qwen uniquement pour l'instant)
- Head 0 = leader, Heads 1..N-1 = followers
- Bilevel K=1 update avec diversity penalty


## Installation (Compute Canada / Alliance)

```bash
module load gcc arrow/21.0.0
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Toujours faire `module load gcc arrow/21.0.0` **avant** `source venv/bin/activate`.

Optionnel — pré-télécharger The Pile sur `$SCRATCH` (évite les téléchargements répétés sur nœud de calcul) :
```bash
export HF_HOME=$SCRATCH/.cache/huggingface
python -c "from datasets import load_dataset; load_dataset('monology/pile-uncopyrighted', split='train')"
```
Puis ajouter `export HF_HOME=$SCRATCH/.cache/huggingface` dans les scripts SLURM.


---

## Qwen2.5-0.5B

### Entraînement baseline

```bash
# Test rapide (100 steps)
python qwen2.5_0.5B/baseline/train_baseline.py --dry_run

# Run complet
python qwen2.5_0.5B/baseline/train_baseline.py

# Avec hyperparamètres custom
python qwen2.5_0.5B/baseline/train_baseline.py \
    --batch_size_per_gpu 4 --grad_accum 4 --lr 3e-4
```

### Entraînement GAME-LoRA

```bash
python qwen2.5_0.5B/game_lora/train_game_lora.py --dry_run
python qwen2.5_0.5B/game_lora/train_game_lora.py
python qwen2.5_0.5B/game_lora/train_game_lora.py --no_nash_mtl
```

### Entraînement Stackelberg (Exp1)

```bash
python qwen2.5_0.5B/exp1/train_exp1.py --dry_run
python qwen2.5_0.5B/exp1/train_exp1.py
python qwen2.5_0.5B/exp1/train_exp1.py \
    --lr_leader 1e-4 --lr_follower 3e-4 --lambda_lead 0.1 --lambda_peer 0.01
```

### Évaluation Qwen2.5-0.5B

```bash
# Modèle de base (sans fine-tuning)
python qwen2.5_0.5B/eval.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --csv_column base

# Baseline fine-tuné
python qwen2.5_0.5B/eval.py \
    --model_path qwen2.5_0.5B/baseline/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column baseline

# GAME-LoRA
python qwen2.5_0.5B/eval.py \
    --model_path qwen2.5_0.5B/game_lora/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column game-lora

# Stackelberg
python qwen2.5_0.5B/eval.py \
    --model_path qwen2.5_0.5B/exp1/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column exp1
```

### Seeds multiples (Qwen)
```bash
for seed in 42 43 44; do
    python qwen2.5_0.5B/baseline/train_baseline.py --seed $seed
    python qwen2.5_0.5B/game_lora/train_game_lora.py --seed $seed
    python qwen2.5_0.5B/exp1/train_exp1.py --seed $seed
done
```


---

## Pythia-160M

Architecture GPT-NeoX : projections QKV fusionnées (`query_key_value` + `dense`),
12 couches, 12 têtes, context 2048, float16.

### Entraînement baseline

```bash
# Test rapide (100 steps)
python pythia160M/baseline/train_baseline.py --dry_run

# Run complet (defaults : batch=8, grad_accum=2, batch_eff=16, lr=3e-4)
python pythia160M/baseline/train_baseline.py

# Avec hyperparamètres custom
python pythia160M/baseline/train_baseline.py \
    --batch_size_per_gpu 8 --grad_accum 2 --lr 3e-4
```

### Évaluation Pythia-160M

```bash
# Modèle de base (sans fine-tuning)
python pythia160M/eval.py \
    --model_path EleutherAI/pythia-160m \
    --csv_column base

# Baseline fine-tuné
python pythia160M/eval.py \
    --model_path pythia160M/baseline/checkpoints/final \
    --base_model EleutherAI/pythia-160m \
    --csv_column baseline
```

### Seeds multiples (Pythia)
```bash
for seed in 42 43 44; do
    python pythia160M/baseline/train_baseline.py --seed $seed
done
```


---

## Résultats attendus — GAME-LoRA sur Qwen2.5-0.5B (Table 1, papier)

| Benchmark    | Baseline | GAME-LoRA | Δ Halluc. |
|--------------|----------|-----------|-----------|
| HE-Dial      | 0.458    | 0.491     | +7.2%     |
| HE-QA        | 0.376    | 0.445     | +18.4%    |
| HE-Summ      | 0.438    | 0.500     | +14.2%    |
| MemoTrap     | 0.642    | 0.650     |           |
| TFQA-MC1     | 0.252    | 0.263     |           |
| TFQA-MC2     | 0.401    | 0.412     |           |
| MMLU         | 0.477    | 0.469     |           |
| WikiText BPB | 0.784    | 0.786     |           |
| Winogrande   | 0.573    | 0.565     |           |
| **Halluc. Δ** | –       | **+8.0%** |           |
| **Knowl. Δ**  | –       | **-0.1%** |           |

*Pas de valeurs de référence papier pour Pythia-160M (expériences nouvelles).*


---

## Structure du repo

```
.
├── requirements.txt
├── README.md
│
├── qwen2.5_0.5B/
│   ├── train.py                  # config, dataset, modèle, optimizer (Qwen)
│   ├── eval.py                   # évaluation complète (Qwen)
│   ├── results.csv               # résultats centralisés Qwen
│   │
│   ├── baseline/
│   │   ├── train_baseline.py
│   │   ├── checkpoints/          # step_N/ et final/
│   │   ├── logs/loss.json
│   │   └── plots/loss.png
│   │
│   ├── game_lora/
│   │   ├── train_game_lora.py
│   │   ├── game_losses.py
│   │   └── checkpoints/
│   │
│   └── exp1/
│       ├── train_exp1.py
│       ├── stackelberg_losses.py
│       └── checkpoints/
│
└── pythia160M/
    ├── train.py                  # config, dataset, modèle, optimizer (Pythia)
    ├── eval.py                   # évaluation (HE-Dial, HE-QA, HE-Summ, WikiText BPB)
    ├── results.csv               # résultats centralisés Pythia
    │
    └── baseline/
        ├── train_baseline.py
        ├── checkpoints/          # step_N/ et final/
        ├── logs/loss.json
        └── plots/loss.png
```


---

## Notes importantes

### Différences Qwen vs Pythia

| | Qwen2.5-0.5B | Pythia-160M |
|---|---|---|
| LoRA targets | `q_proj, k_proj, v_proj, o_proj` | `query_key_value, dense` |
| dtype | bfloat16 | bfloat16 |
| Design layer (GAME-LoRA) | 19 (sur 24) | 9 (sur 12) |
| Tokenizer vocab | 151 936 | 50 304 |
| Batch size GPU recommandé | 4–8 | 8–16 |

### The Pile — accès
`monology/pile-uncopyrighted` (utilisé par défaut) : The Pile sans textes sous copyright,
compatible streaming HuggingFace. Pré-télécharger sur `$SCRATCH` recommandé pour Narval.

### Reproductibilité
L'article rapporte des résultats moyennés sur 3 seeds (42, 43, 44).
Pour forcer le déterminisme GPU :
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python <script>.py ...
```
