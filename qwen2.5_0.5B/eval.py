"""
Évaluation Qwen2.5-0.5B — port du eval Pythia-160M (7 benchmarks).

Port de pythia160M/eval.py. Les benchmarks sont identiques pour permettre
une comparaison directe Pythia↔Qwen sur les mêmes tâches.

DEUX catégories :

  (A) Métrique cible (sanity check du FT WikiText-103) :
        - WikiText-103 BPB/PPL (in-domain test split)

  (B) Métriques pour discriminer différentes méthodes de FT :
        - PTB_BPB           : OOD (Penn Treebank, généralisation LM)
        - LAMBADA           : complétion long-contexte
        - HellaSwag         : sens commun
        - PIQA              : raisonnement physique
        - ARC-Easy          : QA facile
        - MemoTrap          : résistance à la mémorisation

Différences vs port Pythia :
  - load_model accepte AutoModelForCausalLM (adapter dirs PEFT autodétectés
    par transformers ; sinon fallback PeftModel.from_pretrained explicite).
  - cache des datasets partagé avec Pythia (datasets/ au repo root).
  - tokenizer Qwen : add_special_tokens=False pour matcher lm-eval-harness ;
    pas de chat template appliqué.

Usage :
    python qwen2.5_0.5B/eval.py --model_path Qwen/Qwen2.5-0.5B \\
        --wandb_run_name eval_qwen_base --wandb_group base

    python qwen2.5_0.5B/eval.py --model_path qwen2.5_0.5B/baseline/checkpoints/run_0/final \\
        --wandb_run_name eval_baseline_seed42 --wandb_group baseline
"""

import os
import re
import glob
import math
import argparse
import logging

_DATASETS_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"
)

import sys
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from tqdm import tqdm as _tqdm
tqdm = partial(_tqdm, file=sys.stderr)
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
# Chargement du modèle (support full checkpoints + PEFT adapter dirs)
# ---------------------------------------------------------------------------


def load_model(model_path: str, base_model: str = None):
    """
    Charge soit :
      - Un modèle HF complet (model_path = HF id ou dir avec config.json + weights)
      - Un dossier PEFT adapter (model_path = dir avec adapter_config.json) ; le
        base model est résolu via adapter_config.base_model_name_or_path ou
        l'argument explicite `base_model`.

    Returns: (model, tokenizer, device).
    """
    logger.info(f"Chargement du modèle depuis {model_path} ...")

    is_local = os.path.isdir(model_path)
    is_peft_dir = is_local and os.path.exists(os.path.join(model_path, "adapter_config.json"))

    # Tokenizer : depuis le model_path si possible, sinon depuis le base model
    if is_peft_dir:
        # Si tokenizer n'est pas dans le dossier adapter, charger depuis le base
        if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
            _tok_src = model_path
        else:
            _tok_src = base_model if base_model is not None else "Qwen/Qwen2.5-0.5B"
    else:
        _tok_src = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        _tok_src, trust_remote_code=True, local_files_only=is_local and _tok_src == model_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # float32 par défaut (cohérence avec eval Pythia : bfloat16 dégrade LAMBADA).
    if is_peft_dir:
        from peft import PeftModel
        # Résolution du base model
        if base_model is None:
            import json
            with open(os.path.join(model_path, "adapter_config.json")) as _f:
                base_model = json.load(_f).get("base_model_name_or_path", "Qwen/Qwen2.5-0.5B")
        logger.info(f"  PEFT adapter détecté → base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=is_local,
        )

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Gate exp6 (opt-in) : si gate.pt présent à côté du checkpoint, le ré-enregistrer.
    # Sans gate.pt (baseline/exp1/exp3), aucun changement.
    if is_local:
        try:
            from gate import load_gate
            gate = load_gate(model, model_path, device)
            if gate is not None:
                logger.info(f"  Gate exp6 chargé et enregistré ({gate._lh.numel()} têtes leader)")
        except Exception as e:
            logger.warning(f"  Gate non chargé ({e}) — éval sans gating")

    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Utilitaires log-vraisemblance (prompts QCM)
# ---------------------------------------------------------------------------


def conditional_log_likelihood(
    model, tokenizer, device,
    context: str, completion: str,
    max_length: int = 2048,
):
    """
    Reproduit exactement la procédure lm-eval-harness (loglikelihood request).

    Returns:
        ll         : somme des log-probs des tokens du target
        is_greedy  : True si argmax de chaque position du target = gold
        n_tokens   : nombre de tokens du target
        n_bytes    : longueur UTF-8 du completion (pour acc_norm byte-normalized)
    """
    ctx_ids  = tokenizer(context,              add_special_tokens=False,
                         truncation=True, max_length=max_length)["input_ids"]
    full_ids = tokenizer(context + completion, add_special_tokens=False,
                         truncation=True, max_length=max_length)["input_ids"]

    n_ctx = len(ctx_ids)
    n_bytes = len(completion.encode("utf-8"))

    if len(full_ids) <= n_ctx:
        return float("-inf"), False, 0, n_bytes
    n_completion = len(full_ids) - n_ctx

    input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0]

    target_logits = logits[n_ctx - 1 : n_ctx - 1 + n_completion]
    log_probs = F.log_softmax(target_logits.float(), dim=-1)
    cont_ids  = torch.tensor(full_ids[n_ctx:], dtype=torch.long, device=device)
    ll = log_probs.gather(1, cont_ids.unsqueeze(1)).sum().item()

    pred_ids  = target_logits.argmax(dim=-1)
    is_greedy = bool(torch.equal(pred_ids, cont_ids))

    return ll, is_greedy, n_completion, n_bytes


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
    """Métrique cible du FT WikiText-103."""
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
    """Penn Treebank (test split). OOD vs WikiText-103."""
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
# LAMBADA
# ---------------------------------------------------------------------------


def eval_lambada(model, tokenizer, device, n, seed):
    """LAMBADA (protocole lm-eval-harness `lambada_openai`)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test",
                          cache_dir=_DATASETS_CACHE)
    except Exception as e:
        logger.warning(f"LAMBADA non disponible : {e}")
        return None

    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    nll_sum   = 0.0
    n_tok_sum = 0
    n_used    = 0

    for ex in tqdm(ds, desc="LAMBADA", leave=False):
        text = ex["text"]
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        context = parts[0]
        target  = " " + parts[1]

        ll, _, n_tgt, _ = conditional_log_likelihood(
            model, tokenizer, device, context, target
        )
        if n_tgt == 0:
            continue

        nll_sum   += -ll
        n_tok_sum += n_tgt
        n_used    += 1

    ppl = math.exp(min(nll_sum / max(1, n_tok_sum), 20))
    return {"ppl": ppl}


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------


def eval_hellaswag(model, tokenizer, device, n, seed):
    """HellaSwag (lm-eval-harness `hellaswag`) : 4 completions, choisir la plus plausible."""
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
    correct_norm = 0
    for ex in tqdm(ds, desc="HellaSwag", leave=False):
        ctx = preprocess(ex["activity_label"] + ": "
                         + ex["ctx_a"] + " "
                         + ex["ctx_b"].capitalize())
        endings = [preprocess(e) for e in ex["endings"]]
        gold    = int(ex["label"])

        lls, lls_norm = [], []
        for e in endings:
            ll, _, _, nb = conditional_log_likelihood(
                model, tokenizer, device, ctx, " " + e
            )
            lls.append(ll)
            lls_norm.append(ll / max(nb, 1))

        if int(np.argmax(lls)) == gold:
            correct += 1
        if int(np.argmax(lls_norm)) == gold:
            correct_norm += 1
    return {"acc": correct / len(ds), "acc_norm": correct_norm / len(ds)}


# ---------------------------------------------------------------------------
# PIQA
# ---------------------------------------------------------------------------


def eval_piqa(model, tokenizer, device, n, seed):
    """PIQA (lm-eval-harness `piqa`) : choix binaire."""
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
    correct_norm = 0
    for ex in tqdm(ds, desc="PIQA", leave=False):
        ctx  = "Question: " + ex["goal"] + "\nAnswer:"
        sols = [ex["sol1"], ex["sol2"]]
        gold = int(ex["label"])

        lls, lls_norm = [], []
        for s in sols:
            ll, _, _, nb = conditional_log_likelihood(
                model, tokenizer, device, ctx, " " + s
            )
            lls.append(ll)
            lls_norm.append(ll / max(nb, 1))

        if int(np.argmax(lls)) == gold:
            correct += 1
        if int(np.argmax(lls_norm)) == gold:
            correct_norm += 1
    return {"acc": correct / len(ds), "acc_norm": correct_norm / len(ds)}


# ---------------------------------------------------------------------------
# ARC-Easy
# ---------------------------------------------------------------------------


def eval_arc_easy(model, tokenizer, device, n, seed):
    """ARC-Easy (lm-eval-harness `arc_easy`) : QA scientifique, 3-5 choix."""
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
    correct_norm = 0
    total   = 0
    for ex in tqdm(ds, desc="ARC-Easy", leave=False):
        q       = ex["question"]
        choices = ex["choices"]["text"]
        labels  = ex["choices"]["label"]
        gold    = ex["answerKey"]
        if gold not in labels:
            continue

        ctx = "Question: " + q + "\nAnswer:"
        lls, lls_norm = [], []
        for c in choices:
            ll, _, _, nb = conditional_log_likelihood(
                model, tokenizer, device, ctx, " " + c
            )
            lls.append(ll)
            lls_norm.append(ll / max(nb, 1))

        if labels[int(np.argmax(lls))] == gold:
            correct += 1
        if labels[int(np.argmax(lls_norm))] == gold:
            correct_norm += 1
        total += 1
    return {"acc": correct / max(1, total),
            "acc_norm": correct_norm / max(1, total)}


# ---------------------------------------------------------------------------
# MemoTrap
# ---------------------------------------------------------------------------


def eval_memotrap(model, tokenizer, device, n, seed):
    """
    MemoTrap (Inverse Scaling Prize). Lit d'abord les CSVs depuis
    `<repo>/datasets/memotrap/` (offline-safe sur les nœuds de calcul) ;
    fallback sur GitHub si absent.
    """
    import ast, csv, io
    import urllib.request

    BASE_URL = "https://raw.githubusercontent.com/liujch1998/memo-trap/master/data/"
    LOCAL_DIR = os.path.join(_DATASETS_CACHE, "memotrap")
    FILES = [
        "1-proverb-ending.csv",
        "2-proverb-translation.csv",
        "3-hate-speech-ending.csv",
        "4-history-of-science-qa.csv",
    ]

    all_examples = []
    for fname in FILES:
        local_path = os.path.join(LOCAL_DIR, fname)
        content = None
        if os.path.exists(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"MemoTrap — lecture locale {fname} échouée : {e}")
        if content is None:
            try:
                with urllib.request.urlopen(BASE_URL + fname, timeout=30) as resp:
                    content = resp.read().decode("utf-8")
            except Exception as e:
                logger.warning(f"MemoTrap — téléchargement {fname} échoué : {e}")
                continue
        try:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                all_examples.append({
                    "prompt":       row["prompt"],
                    "classes":      ast.literal_eval(row["classes"]),
                    "answer_index": int(row["answer_index"]),
                })
        except Exception as e:
            logger.warning(f"MemoTrap — parsing {fname} échoué : {e}")

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
        lls = []
        for c in ex["classes"]:
            ll, _, _, _ = conditional_log_likelihood(
                model, tokenizer, device, ex["prompt"], c
            )
            lls.append(ll)
        if int(np.argmax(lls)) == ex["answer_index"]:
            correct += 1
    return correct / len(examples)


# ---------------------------------------------------------------------------
# Pipeline d'évaluation
# ---------------------------------------------------------------------------

METRIC_ORDER = [
    "WikiText103_PPL", "WikiText103_BPB",
    "LAMBADA_ppl",
    "HellaSwag_acc", "HellaSwag_acc_norm",
    "PTB_BPB", "PTB_PPL",
    "PIQA_acc", "PIQA_acc_norm",
    "ARC-Easy_acc", "ARC-Easy_acc_norm",
    "MemoTrap",
]


def run_eval(model, tokenizer, device, n=None, seed=42):
    results = {}
    logger.info("=== Évaluation Qwen2.5-0.5B ===")

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
            results["LAMBADA_ppl"] = round(r["ppl"], 4)
            logger.info(f"  LAMBADA ppl = {r['ppl']:.3f}")

    if "HellaSwag" in BENCHMARKS_TO_EVALUATE:
        logger.info("HellaSwag ...")
        r = eval_hellaswag(model, tokenizer, device, n, seed)
        if r is not None:
            results["HellaSwag_acc"]      = round(r["acc"], 4)
            results["HellaSwag_acc_norm"] = round(r["acc_norm"], 4)
            logger.info(f"  HellaSwag   acc={r['acc']:.4f}  acc_norm={r['acc_norm']:.4f}")

    if "PIQA" in BENCHMARKS_TO_EVALUATE:
        logger.info("PIQA ...")
        r = eval_piqa(model, tokenizer, device, n, seed)
        if r is not None:
            results["PIQA_acc"]      = round(r["acc"], 4)
            results["PIQA_acc_norm"] = round(r["acc_norm"], 4)
            logger.info(f"  PIQA        acc={r['acc']:.4f}  acc_norm={r['acc_norm']:.4f}")

    if "ARC-Easy" in BENCHMARKS_TO_EVALUATE:
        logger.info("ARC-Easy ...")
        r = eval_arc_easy(model, tokenizer, device, n, seed)
        if r is not None:
            results["ARC-Easy_acc"]      = round(r["acc"], 4)
            results["ARC-Easy_acc_norm"] = round(r["acc_norm"], 4)
            logger.info(f"  ARC-Easy    acc={r['acc']:.4f}  acc_norm={r['acc_norm']:.4f}")

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
    parser = argparse.ArgumentParser(description="Évaluation Qwen2.5-0.5B (WikiText-103 FT)")
    parser.add_argument("--model_path", required=True,
                        help="Chemin vers le modèle fine-tuné ou 'Qwen/Qwen2.5-0.5B'")
    parser.add_argument("--base_model", default=None,
                        help="Si model_path est un dossier PEFT adapter sans le base, "
                             "spécifier le base model ici (sinon résolu depuis adapter_config.json)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Taille max par benchmark (défaut: set complet). "
                             "Ne pas descendre sous 1000 sans raison.")
    parser.add_argument("--seed", type=int, default=EVAL_PARAMS["seed"],
                        help="Seed pour le sous-échantillonnage (si n_samples < full)")
    parser.add_argument("--wandb_project", default="Stackelberg-Qwen0.5B",
                        help="Projet W&B (passer '' pour désactiver)")
    parser.add_argument("--wandb_run_name", required=True,
                        help="Nom du run W&B")
    parser.add_argument("--wandb_group", default=None,
                        help="Groupe W&B (ex: 'baseline', 'stackelberg_v1')")
    parser.add_argument("--wandb_tags", nargs="*", default=[],
                        help="Tags W&B")
    args = parser.parse_args()

    # Auto-detect multi-run directory structure (run_*/final subdirs)
    run_dirs = sorted(glob.glob(os.path.join(args.model_path, "run_*/final")))

    # Try to resume the corresponding training wandb run (run_0 for multi-run, parent dir for single)
    _run_id = None
    if run_dirs:
        _id_file = os.path.join(run_dirs[0].replace("/final", ""), "wandb_run_id.txt")
    else:
        _id_file = os.path.join(os.path.dirname(os.path.abspath(args.model_path)), "wandb_run_id.txt")
    if os.path.exists(_id_file):
        with open(_id_file) as _f:
            _run_id = _f.read().strip()

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        if _run_id:
            logger.info(f"Resuming wandb run {_run_id} (training run)")
            wandb.init(
                project=args.wandb_project,
                id=_run_id,
                resume="allow",
                job_type="eval",
            )
        else:
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

    if run_dirs:
        logger.info(f"Multi-run eval : {len(run_dirs)} runs trouvés dans {args.model_path}")
        all_results = []
        for run_dir in run_dirs:
            logger.info(f"\n--- {run_dir} ---")
            model, tokenizer, device = load_model(run_dir, args.base_model)
            r = run_eval(model, tokenizer, device, args.n_samples, args.seed)
            all_results.append(r)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_keys = list(all_results[0].keys())
        results = {k: round(float(np.mean([r[k] for r in all_results])), 4) for k in all_keys}
        results_std = {k: round(float(np.std([r[k] for r in all_results])), 4) for k in all_keys}

        logger.info("\n=== Résultats (moyenne ± std sur %d runs) ===" % len(run_dirs))
        for k in METRIC_ORDER:
            if k in results:
                logger.info(f"  {k:<20} = {results[k]:.4f} ± {results_std[k]:.4f}")
    else:
        model, tokenizer, device = load_model(args.model_path, args.base_model)
        results = run_eval(model, tokenizer, device, args.n_samples, args.seed)
        results_std = {}

        logger.info("\n=== Résultats finaux ===")
        for k in METRIC_ORDER:
            if k in results:
                logger.info(f"  {k:<20} = {results[k]}")

    ordered = [k for k in METRIC_ORDER if k in results]
    ordered += [k for k in results if k not in METRIC_ORDER]

    if use_wandb:
        for k in ordered:
            wandb.run.summary[f"eval/{k}"] = results[k]
            if k in results_std:
                wandb.run.summary[f"eval/{k}_std"] = results_std[k]
        wandb.finish()
