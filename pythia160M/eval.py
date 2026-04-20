"""
Évaluation Pythia-160M — benchmarks adaptés au FT sur WikiText-103.

DEUX catégories de métriques :

  (A) Métrique qui DOIT s'améliorer après FT sur WikiText-103
      (sanity check : le modèle finetuné doit battre le modèle de base) :
        - WikiText-103 BPB/PPL (in-domain test split)

  (B) Métriques pour discriminer différentes méthodes de FT
      (baseline vs Stackelberg, variantes de λ, etc.) :
        - PTB_BPB           : OOD (Penn Treebank, généralisation LM)
        - LAMBADA (acc/ppl) : complétion long-contexte, très sensible
        - HellaSwag         : sens commun, standard
        - PIQA              : raisonnement physique
        - ARC-Easy          : QA facile
        - MemoTrap          : résistance à la mémorisation — pertinent
                              pour un mécanisme visant la diversité
                              des têtes (Stackelberg)

Changements par rapport à la version précédente :
  - WikiText-2 retiré : son test split est identique à celui de WikiText-103
    (mêmes 60 articles) → BPB strictement identique. Remplacé par PTB.
  - LAMBADA : alignement de tokenisation fixé (vérification que ctx_ids
    est bien un préfixe de full_ids, sinon skip).
  - ARC-Easy / HellaSwag / PIQA : format de prompt fixé (espace sur la
    completion et non après le contexte) → compatible avec les merges BPE.
  - Streaming retiré (les fichiers WikiText test font quelques MB).
  - n_samples = set complet par défaut (variance d'eval négligeable).
  - Plusieurs seeds ne changent QUE l'ordre de sampling. Pour mesurer la
    vraie variance d'une méthode, lancer plusieurs FT avec des seeds
    différents et évaluer chacun séparément.

Usage :
    python pythia160M/eval.py --model_path EleutherAI/pythia-160m \\
        --wandb_run_name eval_base --wandb_group base

    python pythia160M/eval.py --model_path pythia160M/baseline/checkpoints/final \\
        --wandb_run_name eval_baseline_seed42 --wandb_group baseline
"""

import os
import re
import math
import argparse
import logging

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
    "n_samples": None,   # None = set complet (recommandé)
    "seed":      42,
}

BENCHMARKS_TO_EVALUATE = [
    "WikiText103_PPL",   # (A) métrique cible
    "PTB_BPB",           # (B) OOD léger
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
# Utilitaires log-vraisemblance (prompts QCM)
# ---------------------------------------------------------------------------


def conditional_log_likelihood(
    model, tokenizer, device,
    context: str, completion: str,
    max_length: int = 2048,
    length_normalize: bool = False,
) -> float:
    """
    Log-vraisemblance de `completion` étant donné `context`.

    IMPORTANT : l'espace séparant contexte et completion doit appartenir à
    `completion` (ex: completion=" answer"). Cela préserve les merges BPE
    du tokenizer GPT-NeoX et aligne correctement les frontières de tokens.
    """
    ctx_ids  = tokenizer(context,              add_special_tokens=True,
                         truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(context + completion, add_special_tokens=True,
                         truncation=True, max_length=max_length)["input_ids"]

    # Vérifier que ctx_ids est bien un préfixe de full_ids. Si la BPE
    # re-fusionne à la frontière, on tombe sur le plus grand préfixe commun.
    if full_ids[:len(ctx_ids)] != ctx_ids:
        k = 0
        while k < min(len(ctx_ids), len(full_ids)) and ctx_ids[k] == full_ids[k]:
            k += 1
        n_ctx = k
    else:
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
# Perplexité / BPB sur corpus LM (sliding window)
# ---------------------------------------------------------------------------


def _eval_lm_sliding(model, tokenizer, device, full_text: str,
                     seq_len: int = 512, stride: int = 256, desc: str = "LM"):
    """NLL moyen (nats/token), perplexité et BPB avec fenêtre glissante."""
    encodings  = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids  = encodings["input_ids"]
    num_bytes  = len(full_text.encode("utf-8"))
    n_tok      = input_ids.shape[1]
    bytes_per_token = num_bytes / n_tok

    nlls     = []
    prev_end = 0
    for begin in tqdm(range(0, n_tok - 1, stride), desc=desc, leave=False):
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
    logger.debug(f"  [{desc}] n_tok={n_tok}  bytes={num_bytes}  "
                 f"bytes/tok={bytes_per_token:.3f}")
    return {"nll": avg_nll, "ppl": ppl, "bpb": bpb}


def eval_wikitext103_ppl(model, tokenizer, device):
    """Métrique cible du FT — doit s'améliorer après FT sur WikiText-103."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="test", cache_dir=_DATASETS_CACHE)
    except Exception as e:
        logger.warning(f"WikiText-103 non disponible : {e}")
        return None

    full_text = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    return _eval_lm_sliding(model, tokenizer, device, full_text,
                            desc="WikiText-103")


def eval_ptb_bpb(model, tokenizer, device):
    """
    Penn Treebank (test split). Corpus Wall Street Journal,
    vraiment OOD par rapport à WikiText-103.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ptb_text_only", split="test",
                          cache_dir=_DATASETS_CACHE, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"PTB non disponible : {e}")
        return None

    full_text = "\n".join([ex["sentence"] for ex in ds if ex["sentence"].strip()])
    return _eval_lm_sliding(model, tokenizer, device, full_text, desc="PTB")


# ---------------------------------------------------------------------------
# LAMBADA — complétion du dernier mot (alignement de tokenisation sécurisé)
# ---------------------------------------------------------------------------


def eval_lambada(model, tokenizer, device, n, seed):
    """
    LAMBADA : accuracy greedy sur le dernier mot + ppl conditionnelle.

    Protocole standard (lm-eval-harness) :
      - context = tout sauf le dernier mot
      - target  = " " + dernier_mot   (espace dans le target, pas avant)
      - skip si la tokenisation jointe n'a pas ctx_ids comme préfixe exact
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test",
                          cache_dir=_DATASETS_CACHE)
    except Exception as e:
        logger.warning(f"LAMBADA non disponible : {e}")
        return None

    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    correct    = 0
    nll_sum    = 0.0
    n_tok_sum  = 0
    n_used     = 0
    n_skipped  = 0

    for ex in tqdm(ds, desc="LAMBADA", leave=False):
        text = ex["text"]
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            n_skipped += 1
            continue
        context = parts[0]
        target  = " " + parts[1]

        ctx_ids  = tokenizer(context,          add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(context + target, add_special_tokens=True)["input_ids"]

        if full_ids[:len(ctx_ids)] != ctx_ids or len(full_ids) <= len(ctx_ids):
            n_skipped += 1
            continue

        n_ctx      = len(ctx_ids)
        target_ids = full_ids[n_ctx:]
        n_tgt      = len(target_ids)

        input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids)
        logits = out.logits[0]  # (T, V)

        # Teacher-forced : argmax par position sur le target
        pred_ids = logits[n_ctx - 1 : n_ctx - 1 + n_tgt].argmax(dim=-1)
        gold_ids = torch.tensor(target_ids, device=device)
        if torch.equal(pred_ids, gold_ids):
            correct += 1

        log_probs = F.log_softmax(
            logits[n_ctx - 1 : n_ctx - 1 + n_tgt].float(), dim=-1
        )
        nll = -log_probs.gather(1, gold_ids.unsqueeze(1)).sum().item()
        nll_sum   += nll
        n_tok_sum += n_tgt
        n_used    += 1

    if n_skipped > 0:
        logger.info(f"  LAMBADA : {n_skipped} exemples skipped (alignement), "
                    f"{n_used} utilisés")

    acc = correct / max(1, n_used)
    ppl = math.exp(min(nll_sum / max(1, n_tok_sum), 20))
    return {"acc": acc, "ppl": ppl}


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------


def eval_hellaswag(model, tokenizer, device, n, seed):
    """
    HellaSwag : parmi 4 completions, choisir la plus plausible.
    acc_norm = argmax de (log p(end | ctx) / n_tokens_end).

    Prompt format (lm-eval-harness style) :
      context    = "{activity_label}: {ctx_a} {ctx_b.capitalize()}"
      completion = " " + ending
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Rowan/hellaswag", split="validation",
                          cache_dir=_DATASETS_CACHE)
    except Exception as e:
        logger.warning(f"HellaSwag non disponible : {e}")
        return None

    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    def preprocess(text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text

    correct = 0
    for ex in tqdm(ds, desc="HellaSwag", leave=False):
        ctx = preprocess(ex["activity_label"] + ": "
                         + ex["ctx_a"] + " "
                         + ex["ctx_b"].capitalize())
        endings = [preprocess(e) for e in ex["endings"]]
        gold    = int(ex["label"])

        scores = [
            conditional_log_likelihood(model, tokenizer, device,
                                       ctx, " " + e, length_normalize=True)
            for e in endings
        ]
        if int(np.argmax(scores)) == gold:
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------


def eval_piqa(model, tokenizer, device, n, seed):
    """
    PIQA : choix binaire entre 2 solutions.
    Prompt : "Question: {goal}\\nAnswer:" + " " + sol
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("ybisk/piqa", split="validation",
                          trust_remote_code=True, cache_dir=_DATASETS_CACHE)
    except Exception as e:
        logger.warning(f"PIQA non disponible : {e}")
        return None

    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    correct = 0
    for ex in tqdm(ds, desc="PIQA", leave=False):
        ctx  = "Question: " + ex["goal"] + "\nAnswer:"
        sols = [ex["sol1"], ex["sol2"]]
        gold = int(ex["label"])
        scores = [
            conditional_log_likelihood(model, tokenizer, device,
                                       ctx, " " + s, length_normalize=True)
            for s in sols
        ]
        if int(np.argmax(scores)) == gold:
            correct += 1
    return correct / len(ds)


# ---------------------------------------------------------------------------
# ARC-Easy
# ---------------------------------------------------------------------------


def eval_arc_easy(model, tokenizer, device, n, seed):
    """
    ARC-Easy : QA scientifique élémentaire, 3-5 choix.
    Prompt : "Question: {q}\\nAnswer:" + " " + choice
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test",
                          cache_dir=_DATASETS_CACHE)
    except Exception as e:
        logger.warning(f"ARC-Easy non disponible : {e}")
        return None

    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

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
                                       ctx, " " + c, length_normalize=True)
            for c in choices
        ]
        pred = labels[int(np.argmax(scores))]
        if pred == gold:
            correct += 1
        total += 1
    return correct / max(1, total)


# ---------------------------------------------------------------------------
# MemoTrap
# ---------------------------------------------------------------------------


def eval_memotrap(model, tokenizer, device, n, seed):
    """
    MemoTrap (Inverse Scaling Prize) : suivre une instruction contre un
    pattern mémorisé. Les prompts contiennent déjà leur propre formatage,
    on ne rajoute PAS d'espace devant les classes.
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

    if n is not None:
        rng  = np.random.default_rng(seed)
        idxs = rng.permutation(len(all_examples))[:min(n, len(all_examples))]
        examples = [all_examples[i] for i in idxs]
    else:
        examples = all_examples

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


def run_eval(model, tokenizer, device, n=None, seed=42):
    results = {}
    logger.info("=== Évaluation Pythia-160M ===")

    skipped = [b for b in [
        "WikiText103_PPL", "PTB_BPB", "LAMBADA",
        "HellaSwag", "PIQA", "ARC-Easy", "MemoTrap",
    ] if b not in BENCHMARKS_TO_EVALUATE]
    if skipped:
        logger.info(f"  Ignorés : {skipped}")

    if "WikiText103_PPL" in BENCHMARKS_TO_EVALUATE:
        logger.info("WikiText-103 (in-domain, cible du FT) ...")
        r = eval_wikitext103_ppl(model, tokenizer, device)
        if r is not None:
            results["WikiText103_PPL"] = round(r["ppl"], 4)
            results["WikiText103_BPB"] = round(r["bpb"], 4)
            logger.info(f"  WT103 PPL   = {r['ppl']:.3f}   BPB = {r['bpb']:.4f}")

    if "PTB_BPB" in BENCHMARKS_TO_EVALUATE:
        logger.info("PTB (OOD) ...")
        r = eval_ptb_bpb(model, tokenizer, device)
        if r is not None:
            results["PTB_BPB"] = round(r["bpb"], 4)
            results["PTB_PPL"] = round(r["ppl"], 4)
            logger.info(f"  PTB   BPB   = {r['bpb']:.4f}   PPL = {r['ppl']:.3f}")

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
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Taille max par benchmark (défaut: set complet). "
                             "Ne pas descendre sous 1000 sans raison.")
    parser.add_argument("--seed", type=int, default=EVAL_PARAMS["seed"],
                        help="Seed pour le sous-échantillonnage (si n_samples < full)")
    parser.add_argument("--wandb_project", default="Stackelberg",
                        help="Projet W&B (passer '' pour désactiver)")
    parser.add_argument("--wandb_run_name", required=True,
                        help="Nom du run W&B")
    parser.add_argument("--wandb_group", default=None,
                        help="Groupe W&B (ex: 'baseline', 'stackelberg_v1')")
    parser.add_argument("--wandb_tags", nargs="*", default=[],
                        help="Tags W&B (ex: 'seed=42 fullft')")
    args = parser.parse_args()

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            job_type="eval",
            config={
                "model_path": args.model_path,
                "seed":       args.seed,
                "n_samples":  args.n_samples,
            },
        )

    model, tokenizer, device = load_model(args.model_path)
    results = run_eval(model, tokenizer, device, args.n_samples, args.seed)

    METRIC_ORDER = [
        "WikiText103_PPL", "WikiText103_BPB",
        "PTB_BPB", "PTB_PPL",
        "LAMBADA_acc", "LAMBADA_ppl",
        "HellaSwag", "PIQA", "ARC-Easy", "MemoTrap",
    ]
    ordered = [k for k in METRIC_ORDER if k in results]
    ordered += [k for k in results if k not in METRIC_ORDER]

    logger.info("\n=== Résultats finaux ===")
    for k in ordered:
        logger.info(f"  {k:<20} = {results[k]}")

    if use_wandb:
        for k in ordered:
            wandb.run.summary[f"eval/{k}"] = results[k]
        wandb.finish()