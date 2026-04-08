# Reproduction GAME-LoRA

Reproduction de l'article "Multi-Head Attention is a Multi-Player Game"
(Chakrabarti & Balachundar, 2026) — baseline LoRA + GAME-LoRA.

## Ce qu'on reproduit

**Baseline** : LoRA fine-tuning de Qwen2.5-0.5B sur The Pile (20M tokens).

**GAME-LoRA** : même setup + régularisation game-théorique (Eq 27) :
- `L = L_CE + λ_LDB · L_LDB + λ_ABT · L_ABT`
- `L_LDB` : log-determinant barrier sur la matrice d'interaction G (Eq 28)
- `L_ABT` : Barlow Twins adaptatif entre paires de têtes (Eq 29)
- Gradients arbitrés via Nash-MTL (Navon et al., 2022)
- Schedule 3 phases : warmup 0–2%, constant 2–87.9%, cooldown 87.9–100%

Paramètres communs (Appendix A) :
- LoRA rank=16, alpha=32, dropout=0.1, cibles Q/K/V/O toutes couches
- AdamW lr=3e-4, weight_decay=0.1, cosinus schedule, 2% warmup
- Batch effectif=16, séquences 1024 tokens, 19 531 forward passes


## Résultats attendus (Table 1)

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


## Prérequis matériel

| GPU       | VRAM  | batch/gpu | grad_accum | temps estimé (baseline) | temps estimé (GAME) |
|-----------|-------|-----------|------------|-------------------------|---------------------|
| A100 80GB | 80GB  | 4         | 4          | ~45 min                 | ~50 min             |
| A100 40GB | 40GB  | 2         | 8          | ~55 min                 | ~65 min             |
| RTX 4090  | 24GB  | 1         | 16         | ~2h                     | ~2h15               |
| RTX 3090  | 24GB  | 1         | 16         | ~3h                     | ~3h30               |
| V100 16GB | 16GB  | 1         | 16         | ~4h                     | ~4h30               |

GAME-LoRA ajoute ~5% de surcoût (Appendix A) dû au calcul des losses de régularisation sur la couche design (layer 19).


## Installation

```bash
pip install -r requirements.txt
```


## Entraînement

### Baseline

```bash
# Test rapide (100 steps)
python train_baseline.py --dry_run

# Run complet
python train_baseline.py \
    --batch_size_per_gpu 1 \
    --grad_accum 16 \
    --output_dir ./checkpoints/baseline
```

### GAME-LoRA

```bash
# Test rapide (100 steps)
python train_game_lora.py --dry_run

# Run complet
python train_game_lora.py \
    --batch_size_per_gpu 1 \
    --grad_accum 16 \
    --design_layer 19 \
    --output_dir ./checkpoints/game_lora

# Sans Nash-MTL (combinaison linéaire simple)
python train_game_lora.py --no_nash_mtl --output_dir ./checkpoints/game_lora_no_nash
```

### Seeds multiples (3 seeds comme dans l'article)
```bash
for seed in 42 43 44; do
    python train_baseline.py --seed $seed --output_dir ./checkpoints/baseline_s${seed}
    python train_game_lora.py --seed $seed --output_dir ./checkpoints/game_lora_s${seed}
done
```


## Évaluation

Le même script `eval_baseline.py` sert pour évaluer les deux méthodes.

```bash
# Évaluer le baseline
python eval_baseline.py \
    --model_path ./checkpoints/baseline/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --output_json results_baseline.json

# Évaluer GAME-LoRA
python eval_baseline.py \
    --model_path ./checkpoints/game_lora/final \
    --base_model Qwen/Qwen2.5-0.5B \
    --output_json results_game_lora.json

# Évaluer le modèle de base (sans fine-tuning)
python eval_baseline.py \
    --model_path Qwen/Qwen2.5-0.5B \
    --output_json results_base.json
```


## Structure des fichiers

```
.
├── requirements.txt        # dépendances
├── train.py                # code commun (config, dataset, modèle, optimizer)
├── train_baseline.py       # entraînement baseline (CE only)
├── train_game_lora.py      # entraînement GAME-LoRA (CE + LDB + ABT + Nash-MTL)
├── game_losses.py          # losses GAME-LoRA (interaction matrix, LDB, ABT, Nash-MTL, scheduler)
├── eval_baseline.py        # évaluation (benchmarks hallucination + knowledge)
└── checkpoints/
    ├── baseline/
    │   └── final/
    └── game_lora/
        └── final/
```

### Architecture du code

- **`train.py`** : partagé entre baseline et GAME-LoRA — `TrainConfig`, `PileStreamDataset`,
  `build_model_and_tokenizer`, `setup_training` (optimizer, scheduler, dataloader)
- **`train_baseline.py`** : boucle d'entraînement simple avec `loss = L_CE`
- **`train_game_lora.py`** : boucle avec les 3 losses, hooks sur la couche design, Nash-MTL
- **`game_losses.py`** : implémentation fidèle au papier des composants GAME :
  - `HeadInteractionMatrix` : G = ω⊙ρ (Def 2.3), Γ(G) = ||G-I||_F
  - `LogDetBarrierLoss` : -log det(G+εI) (Eq 28)
  - `AdaptiveBarlowTwinsLoss` : E_{i<j}[w_ij ||Ĉ_ij - I||²_F] (Eq 29)
  - `NashMTL` : Nash bargaining pour arbitrer les gradients
  - `GAMELossScheduler` : schedule 3 phases (warmup/constant/cooldown)
  - `HeadOutputCapture` : hook pour capturer les sorties par tête


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
