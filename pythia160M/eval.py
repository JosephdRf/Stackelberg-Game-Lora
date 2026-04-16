"""
Évaluation Pythia-160M — benchmarks hallucination + perplexité
  HaluEval    : HE-Summ  (signal moyen — format contraignant)
  MemoTrap    : résistance à la mémorisation (signal fort)
  TruthfulQA  : MC2 uniquement (signal moyen)
  PopQA       : open-domain QA sur entités populaires (signal moyen)
  WikiText BPB: perplexité normalisée (signal fort)

Benchmarks retirés (signal quasi-nul sur 160M) :
  HE-Dial, HE-QA  → modèle trop petit pour générer du contenu nuisible discriminant
  MMLU            → ~26% ≈ random (25%)
  NQ              → trop difficile à 160M, variance > signal
  Winogrande      → ~52% ≈ random (50%)

Usage :
    # Modèle de base (sans fine-tuning)
    python pythia160M/eval.py --model_path EleutherAI/pythia-160m

    # Modèle fine-tuné LoRA
    python pythia160M/eval.py \\
        --model_path pythia160M/baseline/checkpoints/final \\
        --base_model EleutherAI/pythia-160m \\
        --csv_column baseline
"""

import os
import re
import math
import string
import json
import argparse
import logging
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

EVAL_PARAMS = {
    "n_samples":   1024,
    "seed":        42,
    "temperature": 0.0,   # greedy decoding
}

# Benchmarks actifs — commenter pour sauter un benchmark
BENCHMARKS_TO_EVALUATE = [
    "HE-Summ",      # signal moyen (meilleur des 3 HE sur petit modèle)
    "MemoTrap",     # signal fort  (résistance à la mémorisation)
    "TFQA",         # signal moyen (MC2 surtout)
    "PopQA",        # signal moyen (entités fréquentes dans pretraining)
    "WikiText_BPB", # signal fort  (perplexité normalisée, métrique principale)
]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------

def load_model(model_path: str, base_model: Optional[str] = None):
    """
    Charge soit :
    - Un modèle LoRA fine-tuné (model_path = dossier peft, base_model requis)
    - Le modèle de base directement (base_model = None)
    """
    logger.info(f"Chargement du modèle depuis {model_path} ...")

    tokenizer_src = model_path if base_model is None else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model is not None:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
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

def log_likelihood(model, tokenizer, device, text: str) -> float:
    """Log-vraisemblance totale d'un texte (WikiText BPB)."""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = ids["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return -out.loss.item() * (input_ids.shape[1] - 1)


def conditional_log_likelihood(
    model, tokenizer, device,
    context: str, completion: str,
    max_length: int = 2048,
    length_normalize: bool = False,
) -> float:
    """
    Log-vraisemblance conditionnelle p(completion | context).

    HuggingFace divise la loss par le nombre de tokens completion,
    donc out.loss est déjà une moyenne NLL/token.

    length_normalize=True  : retourne la moyenne NLL/token
                             (à utiliser quand les completions ont des longueurs
                             très variables — ex: HaluEval)
    length_normalize=False : retourne NLL total = moyenne × n_tokens completion
                             (à utiliser pour MC où les longueurs sont comparables)
    """
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
    # out.loss = mean NLL over completion tokens
    mean_nll = out.loss.item()
    if length_normalize:
        return -mean_nll
    return -mean_nll * n_completion


# ---------------------------------------------------------------------------
# HaluEval — Summarization uniquement (signal moyen, meilleur des 3 HE)
# ---------------------------------------------------------------------------

def eval_halueval_summ(model, tokenizer, device, n, seed):
    """
    HaluEval Summarization : le modèle préfère-t-il le bon résumé au résumé halluciné ?

    Utilise length_normalize=True car les résumés (right vs hallucinated)
    peuvent avoir des longueurs sensiblement différentes.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"HaluEval Summarization non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="HaluEval-Summ", leave=False):
        doc     = ex.get("document", "")
        right_s = ex.get("right_summary", "")
        hall_s  = ex.get("hallucinated_summary", "")

        score_right = conditional_log_likelihood(
            model, tokenizer, device, doc + " ", right_s, length_normalize=True
        )
        score_hall  = conditional_log_likelihood(
            model, tokenizer, device, doc + " ", hall_s, length_normalize=True
        )

        if score_right > score_hall:
            correct += 1

    return correct / len(ds)


# ---------------------------------------------------------------------------
# MemoTrap
# ---------------------------------------------------------------------------

def eval_memotrap(model, tokenizer, device, n, seed):
    """
    MemoTrap : résistance à la complétion automatique de phrases mémorisées.

    Le modèle reçoit une instruction du type "Complete the proverb with a
    different ending:" suivi d'une amorce connue.
    - target_new  : la réponse attendue (non-mémorisée, suit l'instruction)
    - target_true : la complétion mémorisée (le "piège")

    Score = proportion où le modèle préfère target_new à target_true.

    Note : le split disponible sur HF est "train" (pas de split "test" public).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/MemoTrap", split="train")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"MemoTrap non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="MemoTrap", leave=False):
        prompt      = ex.get("prompt", "")
        target_new  = ex.get("target_new", "")
        target_true = ex.get("target_true", "")
        score_new   = conditional_log_likelihood(
            model, tokenizer, device, prompt, target_new, length_normalize=True
        )
        score_true  = conditional_log_likelihood(
            model, tokenizer, device, prompt, target_true, length_normalize=True
        )
        if score_new > score_true:
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# TruthfulQA — MC1 + MC2
# ---------------------------------------------------------------------------

def eval_truthfulqa(model, tokenizer, device, n, seed):
    """
    TruthfulQA MC.
    MC1 : accuracy sur la meilleure réponse unique (argmax).
    MC2 : score probabiliste sur toutes les réponses vraies (somme des probs
          sur les bonnes réponses après softmax). Préférer MC2 pour Pythia-160M.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"TruthfulQA non disponible : {e}")
        return None, None

    mc1_correct = 0
    mc2_correct = 0.0
    for ex in tqdm(ds, desc="TruthfulQA", leave=False):
        q = ex["question"]

        # MC1
        mc = ex["mc1_targets"]
        choices_mc1 = mc["choices"]
        labels_mc1  = mc["labels"]
        scores = [conditional_log_likelihood(model, tokenizer, device, q + " ", c)
                  for c in choices_mc1]
        pred_mc1 = int(np.argmax(scores))
        if labels_mc1[pred_mc1] == 1:
            mc1_correct += 1

        # MC2
        mc2 = ex["mc2_targets"]
        choices_mc2 = mc2["choices"]
        labels_mc2  = mc2["labels"]
        scores2 = [conditional_log_likelihood(model, tokenizer, device, q + " ", c)
                   for c in choices_mc2]
        probs2 = F.softmax(torch.tensor(scores2, dtype=torch.float32), dim=0)
        mask2  = torch.tensor([bool(l) for l in labels_mc2])
        mc2_correct += probs2[mask2].sum().item()

    return mc1_correct / len(ds), mc2_correct / len(ds)


# ---------------------------------------------------------------------------
# PopQA — open-domain QA (entités populaires Wikipedia)
# ---------------------------------------------------------------------------

def generate_greedy(model, tokenizer, device, prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=450).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("\n")[0].strip()


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match_any(prediction: str, gold_answers: list) -> bool:
    pred_norm = normalize_answer(prediction)
    for g in gold_answers:
        g_norm = normalize_answer(g)
        if g_norm and (g_norm == pred_norm or g_norm in pred_norm):
            return True
    return False


def eval_popqa(model, tokenizer, device, n, seed):
    """
    PopQA : questions factuelles sur entités populaires Wikipedia.
    Génération greedy + exact match (ou inclusion) contre les réponses acceptables.
    Plus adapté que NQ pour Pythia-160M (entités très fréquentes dans le pretraining).
    """
    POPQA_FEWSHOT = (
        "Q: What is George Rankin's occupation?\nA: politician\n\n"
        "Q: What is John Mayne's occupation?\nA: journalist\n\n"
        "Q: Who is the spouse of Barack Obama?\nA: Michelle Obama\n\n"
        "Q: What country is Mount Everest located in?\nA: Nepal\n\n"
        "Q: Who directed the movie The Godfather?\nA: Francis Ford Coppola\n\n"
    )
    try:
        from datasets import load_dataset
        ds = load_dataset("akariasai/PopQA", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"PopQA non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="PopQA", leave=False):
        try:
            gold = json.loads(ex.get("possible_answers", "[]"))
        except (ValueError, TypeError):
            gold = [ex.get("possible_answers", "")]
        prompt = POPQA_FEWSHOT + f"Q: {ex['question']}\nA:"
        pred   = generate_greedy(model, tokenizer, device, prompt)
        if exact_match_any(pred, gold):
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# WikiText BPB — métrique principale
# ---------------------------------------------------------------------------

def eval_wikitext_bpb(model, tokenizer, device):
    """
    WikiText BPB (bits per byte) — métrique de perplexité normalisée.

    Formule : BPB = avg_NLL_nats / ln(2) / bytes_per_token
    Plus bas = meilleur. Métrique continue, très sensible aux changements
    de λ, lr_sim, design_layer dans les expériences Stackelberg.

    Utilise wikitext-2-raw-v1 (suffisant pour Pythia-160M, plus rapide que 103).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        logger.warning(f"WikiText non disponible : {e}")
        return None

    full_text  = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    encodings  = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids  = encodings["input_ids"]
    num_bytes  = len(full_text.encode("utf-8"))
    bytes_per_token = num_bytes / input_ids.shape[1]

    stride  = 256
    seq_len = 512
    nlls    = []
    prev_end = 0

    for begin in tqdm(range(0, input_ids.shape[1] - 1, stride),
                      desc="WikiText BPB", leave=False):
        end        = min(begin + seq_len, input_ids.shape[1])
        target_len = end - prev_end
        chunk      = input_ids[:, begin:end].to(device)
        labels     = chunk.clone()
        labels[:, :-target_len] = -100
        with torch.no_grad():
            out = model(chunk, labels=labels)
        nlls.append(out.loss.item() * target_len)
        prev_end = end
        if end == input_ids.shape[1]:
            break

    avg_nll = sum(nlls) / (input_ids.shape[1] - 1)
    return avg_nll / math.log(2) / bytes_per_token


# ---------------------------------------------------------------------------
# Pipeline d'évaluation complète
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, device, n=1024, seed=42):
    results = {}
    logger.info("=== Évaluation en cours (Pythia-160M) ===")

    all_benchmarks = [
        "HE-Summ", "MemoTrap", "TFQA", "PopQA", "WikiText_BPB",
    ]
    skipped = [b for b in all_benchmarks if b not in BENCHMARKS_TO_EVALUATE]
    if skipped:
        logger.info(f"  Benchmarks ignorés : {skipped}")

    if "HE-Summ" in BENCHMARKS_TO_EVALUATE:
        logger.info("HaluEval Summarization ...")
        v = eval_halueval_summ(model, tokenizer, device, n, seed)
        if v is not None:
            results["HE-Summ"] = round(v, 4)
            logger.info(f"  HE-Summ     = {v:.4f}")

    if "MemoTrap" in BENCHMARKS_TO_EVALUATE:
        logger.info("MemoTrap ...")
        v = eval_memotrap(model, tokenizer, device, n, seed)
        if v is not None:
            results["MemoTrap"] = round(v, 4)
            logger.info(f"  MemoTrap    = {v:.4f}")

    if "TFQA" in BENCHMARKS_TO_EVALUATE:
        logger.info("TruthfulQA ...")
        mc1, mc2 = eval_truthfulqa(model, tokenizer, device, n, seed)
        if mc1 is not None:
            results["TFQA-MC1"] = round(mc1, 4)
            results["TFQA-MC2"] = round(mc2, 4)
            logger.info(f"  TFQA-MC1    = {mc1:.4f}")
            logger.info(f"  TFQA-MC2    = {mc2:.4f}")

    if "PopQA" in BENCHMARKS_TO_EVALUATE:
        logger.info("PopQA ...")
        v = eval_popqa(model, tokenizer, device, n, seed)
        if v is not None:
            results["PopQA"] = round(v, 4)
            logger.info(f"  PopQA       = {v:.4f}")

    if "WikiText_BPB" in BENCHMARKS_TO_EVALUATE:
        logger.info("WikiText BPB ...")
        v = eval_wikitext_bpb(model, tokenizer, device)
        if v is not None:
            results["WikiText_BPB"] = round(v, 4)
            logger.info(f"  WikiText BPB= {v:.4f}")

    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation Pythia-160M")
    parser.add_argument("--model_path", required=True,
                        help="Chemin vers le modèle fine-tuné ou 'EleutherAI/pythia-160m'")
    parser.add_argument("--base_model", default=None,
                        help="Base model HF si model_path est un dossier LoRA "
                             "(ex: EleutherAI/pythia-160m)")
    parser.add_argument("--n_samples", type=int, default=EVAL_PARAMS["n_samples"])
    parser.add_argument("--seed",      type=int, default=EVAL_PARAMS["seed"])
    parser.add_argument("--csv",       default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), "results.csv"),
                        help="Fichier CSV de résultats")
    parser.add_argument("--csv_column", default=None,
                        help="Colonne à mettre à jour dans results.csv (ex: 'baseline', 'game-lora')")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path, args.base_model)
    results = run_eval(model, tokenizer, device, args.n_samples, args.seed)

    logger.info("\n=== Résultats finaux (Pythia-160M) ===")
    for k, v in results.items():
        logger.info(f"  {k:<20} = {v:.4f}")

    # Mise à jour results.csv
    if args.csv_column:
        import csv
        csv_path = args.csv
        rows = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames) if reader.fieldnames else ["metric"]
                for row in reader:
                    rows[row["metric"]] = row
        else:
            fieldnames = ["metric"]

        col = args.csv_column
        if col not in fieldnames:
            fieldnames.append(col)

        METRIC_ORDER = [
            "HE-Summ", "MemoTrap",
            "TFQA-MC1", "TFQA-MC2",
            "PopQA", "WikiText_BPB",
        ]
        for m in METRIC_ORDER:
            if m not in rows:
                rows[m] = {"metric": m}
        for metric, value in results.items():
            rows[metric][col] = str(value)

        ordered = [m for m in METRIC_ORDER if m in rows]
        ordered += [m for m in rows if m not in METRIC_ORDER]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in ordered:
                writer.writerow(rows[m])

        logger.info(f"Colonne '{col}' mise à jour dans {csv_path}")