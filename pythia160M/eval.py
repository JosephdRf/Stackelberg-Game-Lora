"""
Évaluation Pythia-160M — benchmarks adaptés au FT sur WikiText-103.

DEUX catégories de métriques :

  (A) Métrique qui DOIT s'améliorer après FT sur WikiText-103
      (sanity check : le modèle finetuné doit battre le modèle de base) :
        - WikiText-103 BPB/PPL (in-domain test split)

  (B) Métriques pour discriminer différentes méthodes de FT
      (baseline vs Stackelberg, variantes de λ, etc.) :
        - WikiText-2 BPB         : OOD léger (généralisation LM)
        - LAMBADA   (acc, ppl)   : complétion long-contexte, très sensible
        - HellaSwag (acc_norm)   : sens commun, standard
        - PIQA      (acc)        : raisonnement physique
        - ARC-Easy  (acc_norm)   : QA facile
        - MemoTrap  (acc)        : résistance à la mémorisation — pertinent
                                   pour un mécanisme visant la diversité
                                   des têtes (Stackelberg)

Benchmarks retirés vs version précédente :
  HE-Summ, TFQA, PopQA  → orientés "hallucination", peu discriminants
                           pour comparer des variantes de FT sur WikiText
  WinoGrande            → ~52% ≈ random à 160M, variance > signal
  MMLU, ARC-Challenge   → ~26% ≈ random à 160M

Usage :
    # Modèle de base
    python pythia160M/eval.py --model_path EleutherAI/pythia-160m --csv_column base

    # Modèle full-finetuné (moyenne sur 3 seeds)
    python pythia160M/eval.py \\
        --model_path pythia160M/baseline/checkpoints/final \\
        --csv_column baseline_fullft

    # Seeds personnalisés
    python pythia160M/eval.py \\
        --model_path pythia160M/baseline/checkpoints/final \\
        --csv_column baseline_fullft --seeds 42 43 44
"""

import os
import re
import math
import string
import json
import argparse
import logging
from typing import Optional

_DATASETS_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset"
)

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

EVAL_PARAMS = {
    "n_samples": 2000,   # plus large qu'avant car benchmarks moins coûteux
    "seed":      42,
}

# Benchmarks actifs — commenter pour sauter
BENCHMARKS_TO_EVALUATE = [
    "WikiText103_PPL",   # (A) métrique cible : doit s'améliorer après FT
    "WikiText2_BPB",     # (B) OOD léger
    "LAMBADA",           # (B) complétion, sensible
    "HellaSwag",         # (B) sens commun
    "PIQA",              # (B) raisonnement physique
    "ARC-Easy",          # (B) QA facile
    "MemoTrap",          # (B) diversité / anti-mémorisation
]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------


def load_model(model_path: str):
    logger.info(f"Chargement du modèle depuis {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Utilitaires log-vraisemblance
# ---------------------------------------------------------------------------


def conditional_log_likelihood(
    model, tokenizer, device,
    context: str, completion: str,
    max_length: int = 2048,
    length_normalize: bool = False,
) -> float:
    ctx_ids  = tokenizer(context,              add_special_tokens=True,
                         truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(context + completion, add_special_tokens=True,
                         truncation=True, max_length=max_length)["input_ids"]
    n_ctx = len(ctx_ids)
    if len(full_ids) <= n_ctx:
        return float("-inf")
    n_completion = len(full_ids) - n_ctx
    input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
    labels    = input_ids.clone()
    labels[0, :n_ctx] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    mean_nll = out.loss.item()
    if length_normalize:
        return -mean_nll
    return -mean_nll * n_completion


# ---------------------------------------------------------------------------
# Perplexité / BPB sur WikiText (sliding window)
# ---------------------------------------------------------------------------


def _eval_wikitext_generic(model, tokenizer, device,
                           config_name: str, seq_len: int = 512,
                           stride: int = 256):
    """
    Calcule NLL moyen (nats/token), perplexité et BPB sur WikiText (test split).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", config_name, split="test", streaming=True)
    except Exception as e:
        logger.warning(f"WikiText ({config_name}) non disponible : {e}")
        return None

    full_text  = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    encodings  = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids  = encodings["input_ids"]
    num_bytes  = len(full_text.encode("utf-8"))
    n_tok      = input_ids.shape[1]
    bytes_per_token = num_bytes / n_tok

    nlls     = []
    prev_end = 0
    for begin in tqdm(range(0, n_tok - 1, stride),
                      desc=f"WikiText ({config_name})", leave=False):
        end        = min(begin + seq_len, n_tok)
        target_len = end - prev_end
        chunk      = input_ids[:, begin:end].to(device)
        labels     = chunk.clone()
        labels[:, :-target_len] = -100
        with torch.no_grad():
            out = model(chunk, labels=labels)
        nlls.append(out.loss.item() * target_len)
        prev_end = end
        if end == n_tok:
            break

    avg_nll = sum(nlls) / (n_tok - 1)
    ppl     = math.exp(min(avg_nll, 20))
    bpb     = avg_nll / math.log(2) / bytes_per_token
    return {"nll": avg_nll, "ppl": ppl, "bpb": bpb}


def eval_wikitext103_ppl(model, tokenizer, device):
    """Métrique cible du FT : doit s'améliorer après fine-tuning sur WikiText-103."""
    return _eval_wikitext_generic(model, tokenizer, device, "wikitext-103-raw-v1")


def eval_wikitext2_bpb(model, tokenizer, device):
    """OOD léger (plus petit, plus rapide)."""
    return _eval_wikitext_generic(model, tokenizer, device, "wikitext-2-raw-v1")


# ---------------------------------------------------------------------------
# LAMBADA — complétion du dernier mot
# ---------------------------------------------------------------------------


def eval_lambada(model, tokenizer, device, n, seed):
    """
    LAMBADA : accuracy = le modèle prédit-il le dernier mot (greedy argmax) ?
    Calcule aussi la perplexité conditionnelle sur le dernier mot.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test",
                          cache_dir=_DATASETS_CACHE)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"LAMBADA non disponible : {e}")
        return None

    correct = 0
    nll_sum = 0.0
    n_tok_sum = 0

    for ex in tqdm(ds, desc="LAMBADA", leave=False):
        text = ex["text"].strip()
        # Découpage : tout sauf le dernier mot = contexte, dernier mot = cible
        last_space = text.rfind(" ")
        if last_space == -1:
            continue
        context = text[:last_space]
        target  = text[last_space:]  # commence par un espace, ex: " word"

        ctx_ids    = tokenizer(context,         add_special_tokens=True)["input_ids"]
        full_ids   = tokenizer(context + target, add_special_tokens=True)["input_ids"]
        n_ctx      = len(ctx_ids)
        target_ids = full_ids[n_ctx:]
        if len(target_ids) == 0:
            continue

        input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits[0]  # (T, V)

        # Accuracy : on prédit token par token en greedy sur la cible
        # La position qui prédit target_ids[i] est n_ctx - 1 + i
        pred_ids = logits[n_ctx - 1 : n_ctx - 1 + len(target_ids)].argmax(dim=-1)
        gold_ids = torch.tensor(target_ids, device=device)
        if torch.equal(pred_ids, gold_ids):
            correct += 1

        # NLL du target
        log_probs = F.log_softmax(
            logits[n_ctx - 1 : n_ctx - 1 + len(target_ids)].float(), dim=-1
        )
        nll = -log_probs.gather(1, gold_ids.unsqueeze(1)).sum().item()
        nll_sum   += nll
        n_tok_sum += len(target_ids)

    acc = correct / len(ds)
    ppl = math.exp(min(nll_sum / max(1, n_tok_sum), 20))
    return {"acc": acc, "ppl": ppl}


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------


def eval_hellaswag(model, tokenizer, device, n, seed):
    """
    HellaSwag : parmi 4 completions, choisir la bonne.
    Score = acc_norm (log-likelihood normalisée par longueur, standard).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation",
                          cache_dir=_DATASETS_CACHE)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"HellaSwag non disponible : {e}")
        return None

    def preprocess(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    correct = 0
    for ex in tqdm(ds, desc="HellaSwag", leave=False):
        ctx = preprocess(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
        endings = [preprocess(e) for e in ex["endings"]]
        gold    = int(ex["label"])

        scores = [
            conditional_log_likelihood(model, tokenizer, device,
                                       ctx + " ", e, length_normalize=True)
            for e in endings
        ]
        if int(np.argmax(scores)) == gold:
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------


def eval_piqa(model, tokenizer, device, n, seed):
    """PIQA : choix binaire entre 2 solutions à un problème physique."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True,
                          cache_dir=_DATASETS_CACHE)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"PIQA non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="PIQA", leave=False):
        ctx  = "Question: " + ex["goal"] + "\nAnswer:"
        sols = [ex["sol1"], ex["sol2"]]
        gold = int(ex["label"])
        scores = [
            conditional_log_likelihood(model, tokenizer, device,
                                       ctx + " ", s, length_normalize=True)
            for s in sols
        ]
        if int(np.argmax(scores)) == gold:
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# ARC-Easy
# ---------------------------------------------------------------------------


def eval_arc_easy(model, tokenizer, device, n, seed):
    """ARC-Easy : QA scientifique élémentaire, 4 choix (parfois 3 ou 5)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test",
                          cache_dir=_DATASETS_CACHE)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"ARC-Easy non disponible : {e}")
        return None

    correct = 0
    total   = 0
    for ex in tqdm(ds, desc="ARC-Easy", leave=False):
        q       = ex["question"]
        choices = ex["choices"]["text"]
        labels  = ex["choices"]["label"]
        gold    = ex["answerKey"]
        if gold not in labels:
            continue

        ctx = "Question: " + q + "\nAnswer:"
        scores = [
            conditional_log_likelihood(model, tokenizer, device,
                                       ctx + " ", c, length_normalize=True)
            for c in choices
        ]
        pred = labels[int(np.argmax(scores))]
        if pred == gold:
            correct += 1
        total += 1
    return correct / max(1, total)


# ---------------------------------------------------------------------------
# MemoTrap (pertinent pour évaluer la diversité des têtes / Stackelberg)
# ---------------------------------------------------------------------------


def eval_memotrap(model, tokenizer, device, n, seed):
    """
    MemoTrap (Inverse Scaling Prize) : le modèle doit suivre une instruction
    qui contredit un pattern mémorisé. Particulièrement pertinent pour évaluer
    un mécanisme qui favorise la diversité des têtes d'attention.
    """
    import ast, csv, io
    import urllib.request

    BASE_URL = "https://raw.githubusercontent.com/liujch1998/memo-trap/master/data/"
    FILES = [
        "1-proverb-ending.csv",
        "2-proverb-translation.csv",
        "3-hate-speech-ending.csv",
        "4-history-of-science-qa.csv",
    ]

    all_examples = []
    for fname in FILES:
        url = BASE_URL + fname
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                all_examples.append({
                    "prompt":       row["prompt"],
                    "classes":      ast.literal_eval(row["classes"]),
                    "answer_index": int(row["answer_index"]),
                })
        except Exception as e:
            logger.warning(f"MemoTrap — impossible de charger {fname} : {e}")

    if not all_examples:
        return None

    rng  = np.random.default_rng(seed)
    idxs = rng.permutation(len(all_examples))[:min(n, len(all_examples))]
    examples = [all_examples[i] for i in idxs]

    correct = 0
    for ex in tqdm(examples, desc="MemoTrap", leave=False):
        scores = [
            conditional_log_likelihood(model, tokenizer, device,
                                       ex["prompt"], c, length_normalize=True)
            for c in ex["classes"]
        ]
        if int(np.argmax(scores)) == ex["answer_index"]:
            correct += 1
    return correct / len(examples)


# ---------------------------------------------------------------------------
# Pipeline d'évaluation
# ---------------------------------------------------------------------------


def run_eval(model, tokenizer, device, n=2000, seed=42):
    results = {}
    logger.info("=== Évaluation Pythia-160M ===")

    skipped = [b for b in [
        "WikiText103_PPL", "WikiText2_BPB", "LAMBADA",
        "HellaSwag", "PIQA", "ARC-Easy", "MemoTrap",
    ] if b not in BENCHMARKS_TO_EVALUATE]
    if skipped:
        logger.info(f"  Ignorés : {skipped}")

    # (A) Métrique cible du FT
    if "WikiText103_PPL" in BENCHMARKS_TO_EVALUATE:
        logger.info("WikiText-103 (in-domain, cible du FT) ...")
        r = eval_wikitext103_ppl(model, tokenizer, device)
        if r is not None:
            results["WikiText103_PPL"] = round(r["ppl"], 4)
            results["WikiText103_BPB"] = round(r["bpb"], 4)
            logger.info(f"  WT103 PPL   = {r['ppl']:.3f}   BPB = {r['bpb']:.4f}")

    # (B) Métriques discriminantes
    if "WikiText2_BPB" in BENCHMARKS_TO_EVALUATE:
        logger.info("WikiText-2 (OOD léger) ...")
        r = eval_wikitext2_bpb(model, tokenizer, device)
        if r is not None:
            results["WikiText2_BPB"] = round(r["bpb"], 4)
            logger.info(f"  WT2   BPB   = {r['bpb']:.4f}")

    if "LAMBADA" in BENCHMARKS_TO_EVALUATE:
        logger.info("LAMBADA ...")
        r = eval_lambada(model, tokenizer, device, n, seed)
        if r is not None:
            results["LAMBADA_acc"] = round(r["acc"], 4)
            results["LAMBADA_ppl"] = round(r["ppl"], 4)
            logger.info(f"  LAMBADA acc = {r['acc']:.4f}   ppl = {r['ppl']:.3f}")

    if "HellaSwag" in BENCHMARKS_TO_EVALUATE:
        logger.info("HellaSwag ...")
        v = eval_hellaswag(model, tokenizer, device, n, seed)
        if v is not None:
            results["HellaSwag"] = round(v, 4)
            logger.info(f"  HellaSwag   = {v:.4f}")

    if "PIQA" in BENCHMARKS_TO_EVALUATE:
        logger.info("PIQA ...")
        v = eval_piqa(model, tokenizer, device, n, seed)
        if v is not None:
            results["PIQA"] = round(v, 4)
            logger.info(f"  PIQA        = {v:.4f}")

    if "ARC-Easy" in BENCHMARKS_TO_EVALUATE:
        logger.info("ARC-Easy ...")
        v = eval_arc_easy(model, tokenizer, device, n, seed)
        if v is not None:
            results["ARC-Easy"] = round(v, 4)
            logger.info(f"  ARC-Easy    = {v:.4f}")

    if "MemoTrap" in BENCHMARKS_TO_EVALUATE:
        logger.info("MemoTrap ...")
        v = eval_memotrap(model, tokenizer, device, n, seed)
        if v is not None:
            results["MemoTrap"] = round(v, 4)
            logger.info(f"  MemoTrap    = {v:.4f}")

    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation Pythia-160M (WikiText-103 FT)")
    parser.add_argument("--model_path", required=True,
                        help="Chemin vers le modèle fine-tuné ou 'EleutherAI/pythia-160m'")
    parser.add_argument("--n_samples", type=int, default=EVAL_PARAMS["n_samples"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Seeds à moyenner (défaut: 42 43 44)")
    parser.add_argument("--wandb_project", default="Stackelberg",
                        help="Projet W&B (défaut: Stackelberg). Passer '' pour désactiver.")
    parser.add_argument("--wandb_run_name", required=True,
                        help="Nom du run W&B")
    parser.add_argument("--wandb_group", default=None,
                        help="Groupe W&B (ex: 'baseline', 'game_lora')")
    args = parser.parse_args()

    import wandb
    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               group=args.wandb_group, job_type="eval",
               config={"model_path": args.model_path, "seeds": args.seeds,
                       "n_samples": args.n_samples})

    model, tokenizer, device = load_model(args.model_path)

    all_results = []
    for seed in args.seeds:
        logger.info(f"\n=== Seed {seed} ===")
        all_results.append(run_eval(model, tokenizer, device, args.n_samples, seed))

    all_keys = {k for r in all_results for k in r}
    agg = {
        k: {
            "mean": round(float(np.mean([r[k] for r in all_results if k in r])), 4),
            "std":  round(float(np.std( [r[k] for r in all_results if k in r])), 4),
        }
        for k in all_keys
    }

    METRIC_ORDER = [
        "WikiText103_PPL", "WikiText103_BPB", "WikiText2_BPB",
        "LAMBADA_acc", "LAMBADA_ppl",
        "HellaSwag", "PIQA", "ARC-Easy", "MemoTrap",
    ]
    ordered_keys = [k for k in METRIC_ORDER if k in agg]

    logger.info(f"\n=== Résultats finaux (moyenne sur seeds {args.seeds}) ===")
    for k in ordered_keys:
        v = agg[k]
        logger.info(f"  {k:<20} = {v['mean']:.4f} ± {v['std']:.4f}")

    for k in ordered_keys:
        wandb.run.summary[k] = agg[k]["mean"]
    wandb.finish()
