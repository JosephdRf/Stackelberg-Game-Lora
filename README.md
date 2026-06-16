# Stackelberg-Multi-Head-Attention

LoRA fine-tuning d'un LLM sous forme de **jeu de Stackelberg** entre têtes d'attention :
une (ou plusieurs) tête *leader* est optimisée en anticipant la réponse des têtes
*followers* (optimisation bilevel à 3 phases), avec des pénalités de diversité /
confiance / barrière log-det sur les têtes, et un mécanisme de *gating* leader→follower.

Le repo décline le même pipeline sur deux modèles :

| Dossier | Modèle | Attention | `design_layer` | Leader |
|---|---|---|---|---|
| [`pythia160M/`](pythia160M/) | `EleutherAI/pythia-160m` | MHA, 12 layers / 12 têtes | 9 | tête 0 |
| [`qwen2.5_0.5B/`](qwen2.5_0.5B/) | `Qwen/Qwen2.5-0.5B` | GQA, 24 layers / 14 Q + 2 KV | 19 | groupe GQA [0..6] |

Données : **WikiText-103** (~100M tokens). Métriques sur **WandB**
(projets `Stackelberg-Pythia160M` et `Stackelberg-Qwen0.5B`). Voir le README de
chaque dossier pour les commandes détaillées.

## Structure

```
.
├── pythia160M/                 # pipeline Pythia-160M (référence)
│   ├── train_utils.py          # TrainConfig, dataset WikiText, build modèle+LoRA
│   ├── gradient_mask.py        # slicing leader/follower des grads LoRA + HiddenStateCapture
│   ├── stackelberg_losses.py   # reconstruction attention + losses diversity/confiance/LDB
│   ├── gate.py                 # gating leader→follower (MLP, hook sur la projection de sortie)
│   ├── eval.py                 # 7 benchmarks (voir plus bas)
│   ├── baseline/               # LoRA + CE seul (référence)
│   ├── game_lora/              # GAME-LoRA (losses LDB + ABT)
│   ├── exp1/                   # Stackelberg bilevel (CE)
│   ├── exp3/                   # exp1 + losses diversity/confiance/LDB (+ multi-leader/layer)
│   ├── exp5/                   # variante exp
│   └── exp6/                   # exp1 + gating leader→follower (+ losses exp3 optionnelles)
│
└── qwen2.5_0.5B/               # même pipeline porté sur Qwen2.5-0.5B (GQA, rank 32)
    ├── train_utils.py
    ├── gradient_mask.py        # slicing GQA (q/k/v/o séparés)
    ├── stackelberg_losses.py   # reconstruction attention Qwen (RoPE full + repeat_kv)
    ├── gate.py
    ├── eval.py
    ├── test_slicing.py         # smoke test du slicing GQA
    ├── baseline/
    ├── exp1/
    ├── exp3/                   # + test_attention.py (validation reconstruction A vs SDPA)
    └── exp6/
```

> Non versionnés / volumineux : `checkpoints/`, `wandb/`, `datasets/`, `logs/`, `.venv/`.

## Expériences

| Exp | Idée |
|---|---|
| **baseline** | LoRA + cross-entropy seul. Point de comparaison. |
| **game_lora** | Losses GAME (LDB + ABT). Flags d'ablation `--no_ldb` / `--no_abt` (Pythia). |
| **exp1** | Jeu de Stackelberg bilevel à 3 phases (follower forward+mask → leader lookahead+mask → restore+assemble+step), CE pure. |
| **exp3** | exp1 enrichi : diversity loss sur followers (`cos/cos_sq/hadamard/erank/output_cos/cka`), confidence loss sur leaders (`max/smooth/entropy`), barrière log-det (LDB) ; multi-leaders / multi-layers. |
| **exp6** | exp1 + **gating leader→follower** : un MLP transforme le signal du leader en poids `g ∈ (0,2)` appliqués aux sorties des têtes followers (init zéro → identité). Peut combiner toutes les losses d'exp3. |

## Eval (7 benchmarks, identiques Pythia ↔ Qwen)

`WikiText103` · `PTB` · `LAMBADA` · `HellaSwag` · `PIQA` · `ARC-Easy` · `MemoTrap`
(MemoTrap lu en offline-first depuis `datasets/memotrap/`). `eval.py` reprend le run
WandB du checkpoint et calcule perplexité / BPB / accuracy selon le benchmark.

## Environnement (Compute Canada / Narval)

```bash
module load StdEnv/2023 python/3.10 cuda/12.2 intel/2023.2.1 arrow/21.0.0
source .venv/bin/activate
export WANDB_MODE=offline HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Les nœuds de calcul sont hors-ligne : WandB tourne en mode `offline` puis se
synchronise depuis le nœud de login (`wandb sync wandb/offline-run-*`). Les datasets
HuggingFace sont mis en cache dans `datasets/`.
