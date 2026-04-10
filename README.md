# Stackelberg Game LoRA

Trois expériences de fine-tuning LoRA de **Qwen2.5-0.5B** sur The Pile (20M tokens),
explorant des régularisations game-théoriques sur les attention heads.

## Expériences

### 1. Baseline (`baseline/`)
LoRA fine-tuning standard avec `loss = L_CE`.

### 2. GAME-LoRA (`game_lora/`)
Reproduction de "Multi-Head Attention is a Multi-Player Game"
(Chakrabarti & Balachundar, 2026) :
- `L = L_CE + λ_LDB · L_LDB + λ_ABT · L_ABT` (Eq 27)
- `L_LDB` : log-determinant barrier sur la matrice d'interaction G (Eq 28)
- `L_ABT` : Barlow Twins adaptatif entre paires de têtes (Eq 29)
- Gradients arbitrés via Nash-MTL (Navon et al., 2022)
- Schedule 3 phases : warmup 0–2%, constant 2–87.9%, cooldown 87.9–100%

### 3. Stackelberg Attention Diversity (`exp1/`)
Modélisation des attention heads comme un jeu de Stackelberg :
- Head 0 = leader (optimise L_CE via o_proj LoRA)
- Heads 1..13 = followers (optimise L_CE + diversity penalty via q/k/v LoRA)
- Bilevel K=1 update : simulated follower step → leader lookahead → follower update
- Similarité = Frobenius inner product normalisé des matrices d'attention

Paramètres communs :
- LoRA rank=16, alpha=32, dropout=0.1, cibles Q/K/V/O toutes couches
- AdamW, cosinus schedule, 2% warmup
- Batch effectif=16, séquences 1024 tokens
- Design layer: 19 (Qwen2.5-0.5B, 14 heads, 24 layers)


## Résultats attendus — GAME-LoRA (Table 1, papier)

| Benchmark    | Baseline | GAME-LoRA |
|--------------|----------|-----------|
| HE-Dial      | 0.458    | 0.491     |
| HE-QA        | 0.376    | 0.445     |
| HE-Summ      | 0.438    | 0.500     |
| MemoTrap     | 0.642    | 0.650     |
| TFQA-MC1     | 0.252    | 0.263     |
| TFQA-MC2     | 0.401    | 0.412     |
| MMLU         | 0.477    | 0.469     |
| NQ           | 0.066    | 0.067     |
| PopQA        | 0.111    | 0.112     |
| WikiText BPB | 0.784    | 0.786     |
| Winogrande   | 0.573    | 0.565     |
| **Halluc. Δ** | –       | **+8.0%** |
| **Knowl. Δ**  | –       | **-0.1%** |


## Installation (Compute Canada / Alliance)

Sur les clusters de l'Alliance de recherche numérique du Canada (Narval, Béluga, Cedar…),
`pyarrow` et plusieurs dépendances sont fournies via modules Lmod.

```bash
# 1. Charger les modules système requis
module load gcc arrow/21.0.0

# 2. Créer le virtualenv (une seule fois)
python -m venv venv

# 3. Activer le venv (toujours APRÈS module load)
source venv/bin/activate

# 4. Installer les dépendances Python
pip install -r requirements.txt
```

> **Important :** à chaque nouvelle session, toujours faire `module load gcc arrow/21.0.0`
> **avant** `source venv/bin/activate`.


## Entraînement

Toutes les commandes se lancent depuis la **racine du repo**.

### Baseline

```bash
# Test rapide (100 steps)
python baseline/train_baseline.py --dry_run

# Run complet
python baseline/train_baseline.py
```

### GAME-LoRA

```bash
# Test rapide (100 steps)
python game_lora/train_game_lora.py --dry_run

# Run complet
python game_lora/train_game_lora.py

# Sans Nash-MTL
python game_lora/train_game_lora.py --no_nash_mtl
```

### Stackelberg (Exp1)

```bash
# Test rapide (100 steps)
python exp1/train_exp1.py --dry_run

# Run complet
python exp1/train_exp1.py

# Avec hyperparamètres custom
python exp1/train_exp1.py --lr_leader 1e-4 --lr_follower 3e-4 --lambda_lead 0.1 --lambda_peer 0.01
```

### Seeds multiples (3 seeds)
```bash
for seed in 42 43 44; do
    python baseline/train_baseline.py --seed $seed
    python game_lora/train_game_lora.py --seed $seed
    python exp1/train_exp1.py --seed $seed
done
```


## Évaluation

Les résultats sont centralisés dans `results.csv` à la racine du repo.

```bash
# Évaluer le baseline → colonne "baseline" dans results.csv
python eval.py \
    --model_path baseline/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column baseline

# Évaluer GAME-LoRA → colonne "game-lora"
python eval.py \
    --model_path game_lora/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column game-lora

# Évaluer Stackelberg → colonne "exp1"
python eval.py \
    --model_path exp1/checkpoints/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --csv_column exp1

# Évaluer le modèle de base (sans fine-tuning)
python eval.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --csv_column base
```


## Structure du repo

```
.
├── train.py                    # code commun (config, dataset, modèle, optimizer)
├── eval.py                     # évaluation (benchmarks hallucination + knowledge)
├── results.csv                 # résultats centralisés (toutes expériences)
├── requirements.txt
│
├── baseline/                   # Exp: Baseline LoRA
│   ├── train_baseline.py
│   └── checkpoints/            # (généré) step_N/ et final/
│
├── game_lora/                  # Exp: GAME-LoRA
│   ├── train_game_lora.py
│   ├── game_losses.py
│   └── checkpoints/            # (généré) step_N/ et final/
│
└── exp1/                       # Exp: Stackelberg Attention Diversity
    ├── train_exp1.py
    ├── stackelberg_losses.py
    └── checkpoints/            # (généré) step_N/ et final/
```

### Architecture du code

- **`eval.py`** : évaluation sur tous les benchmarks (HaluEval, TruthfulQA, MemoTrap, MMLU, NQ, PopQA, WikiText, WinoGrande)
- **`train.py`** : partagé entre les 3 expériences — `TrainConfig`, `PileStreamDataset`,
  `build_model_and_tokenizer`, `setup_training`
- **`baseline/train_baseline.py`** : boucle simple `loss = L_CE`
- **`game_lora/train_game_lora.py`** : 3 losses + Nash-MTL, hooks sur layer 19
- **`game_lora/game_losses.py`** : `HeadInteractionMatrix`, `LogDetBarrierLoss`,
  `AdaptiveBarlowTwinsLoss`, `NashMTL`, `GAMELossScheduler`
- **`exp1/train_exp1.py`** : bilevel Stackelberg K=1, 2 Adam optimizers (leader/follower)
- **`exp1/stackelberg_losses.py`** : `compute_diversity_loss`, `split_leader_follower_params`,
  `AttentionWeightCapture`


## Notes importantes

### Vérification des steps
L'article indique 19 531 steps. Vérification :
```
total_tokens = 20_000_000
seq_len      = 1024
batch_eff    = 16
tokens/step  = 1024 * 16 = 16 384
steps        = ceil(20_000_000 / 16_384) = 1221 → NON

# L'article compte en steps d'optimiseur avec gradient accumulation :
# tokens/opt_step = 1024 * 16 = 16384
# opt_steps = 20M / 16384 ≈ 1221

# Sauf si "steps" = forward passes (sans gradient accum) :
# avec grad_accum=16 : forward_passes = 1221 * 16 = 19531 ✓
```
Le script compte les forward passes comme l'article.

### The Pile — accès
The Pile complet nécessite une acceptation des conditions d'utilisation sur HuggingFace.
Connexion requise :
```bash
huggingface-cli login
```
Ou utiliser un miroir :
```python
# Remplacer dans PileStreamDataset.__iter__ :
ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
```

### Reproductibilité exacte
L'article rapporte des résultats moyennés sur 3 seeds.
Les légères variations (~0.5-1%) entre seeds sont normales.
L'écart avec l'article peut venir de :
1. La version exacte de The Pile utilisée (ordre des données)
2. La version de transformers/peft
3. Les opérations non-déterministes sur GPU (attention flash, etc.)

Pour forcer le déterminisme :
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_baseline.py ...
```
et ajouter `torch.use_deterministic_algorithms(True)` dans le script.
