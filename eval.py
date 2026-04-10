"""
Évaluation — benchmarks identiques à l'article
Table 1 : HaluEval (dial/QA/summ), TruthfulQA MC1/MC2, MemoTrap
           MMLU, NQ, PopQA, WikiText BPB, WinoGrande

Usage :
    python eval.py --model_path ./checkpoints/baseline/final
    python eval.py --model_path Qwen/Qwen2.5-0.5B  # évaluation du modèle de base
"""

import os
import re
import string
import json
import argparse
import logging
from typing import Optional

import torch
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

# Paramètres d'évaluation identiques à l'article (Appendix A)
EVAL_PARAMS = {
    "n_samples": 1024,    # "N=1024 samples per task"
    "seed": 42,           # "seed=42"
    "temperature": 0.0,   # "greedy decoding (temperature=0)"
}


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def load_model(model_path: str, base_model: Optional[str] = None):
    """
    Charge soit :
    - Un modèle LoRA fine-tuné (model_path = dossier peft)
    - Le modèle de base directement
    """
    logger.info(f"Chargement du modèle depuis {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if base_model is None else base_model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model is not None:
        # Charger base + adaptateurs LoRA
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


def log_likelihood(model, tokenizer, device, text: str) -> float:
    """Total log-likelihood of a full text. Used only for WikiText BPB."""
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = ids["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    return -out.loss.item() * (input_ids.shape[1] - 1)


def conditional_log_likelihood(model, tokenizer, device, context: str, completion: str, max_length: int = 2048) -> float:
    """
    Mean log-likelihood of `completion` tokens given `context`.
    Normalised per completion token → length-agnostic comparison.
    This is the correct scoring function for all paired / MC evaluations.
    """
    ctx_ids  = tokenizer(context,             add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(context + completion, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
    n_ctx = len(ctx_ids)
    if len(full_ids) <= n_ctx:
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
    labels    = input_ids.clone()
    labels[0, :n_ctx] = -100  # mask context tokens; only completion contributes to loss
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    return -out.loss.item()  # negative mean NLL over completion tokens (higher = more likely)


def multiple_choice_accuracy(model, tokenizer, device, examples: list) -> float:
    """
    Accuracy par log-vraisemblance sur des QCM.
    Chaque exemple : {"question": str, "choices": [str], "answer": int}
    Protocole standard : on choisit la réponse avec la plus haute p(réponse|question).
    """
    correct = 0
    for ex in tqdm(examples, desc="MC eval", leave=False):
        q = ex["question"]
        scores = []
        for choice in ex["choices"]:
            text = q + " " + choice
            scores.append(log_likelihood(model, tokenizer, device, text))
        pred = int(np.argmax(scores))
        if pred == ex["answer"]:
            correct += 1
    return correct / len(examples)


# ---------------------------------------------------------------------------
# Chargement des benchmarks
# ---------------------------------------------------------------------------

def load_halueval(subset: str, n: int = 1024, seed: int = 42):
    """
    HaluEval — 3 subsets : dialogue, qa, summarization
    Format : classification binaire (hallucination ou non)
    """
    from datasets import load_dataset
    ds = load_dataset("pminervini/HaluEval", subset, split="data")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


def load_truthfulqa(n: int = 1024, seed: int = 42):
    """TruthfulQA — MC1 et MC2"""
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


def load_mmlu(n: int = 1024, seed: int = 42):
    """MMLU — toutes les catégories, split test"""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


def load_wikitext(n: int = 1024, seq_len: int = 512):
    """WikiText-2 BPB (bits per byte)"""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return ds


def load_winogrande(n: int = 1024, seed: int = 42):
    """WinoGrande"""
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# Génération et métriques (NQ / PopQA)
# ---------------------------------------------------------------------------

def generate_greedy(model, tokenizer, device, prompt: str, max_new_tokens: int = 32) -> str:
    """Greedy decode; returns only the generated continuation (first line)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=450).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.split("\n")[0].strip()


def normalize_answer(s: str) -> str:
    """Lower-case, strip articles/punctuation/whitespace (standard EM normalization)."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def exact_match_any(prediction: str, gold_answers: list) -> bool:
    """Return True if normalized prediction matches or contains any gold answer."""
    pred_norm = normalize_answer(prediction)
    for g in gold_answers:
        g_norm = normalize_answer(g)
        if not g_norm:
            continue
        # exact match or gold contained in prediction
        if g_norm == pred_norm or g_norm in pred_norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Évaluations spécifiques
# ---------------------------------------------------------------------------

def eval_halueval_dial(model, tokenizer, device, n, seed):
    """
    HaluEval Dialogue — paired right/hallucinated response comparison.
    Dataset schema (pminervini/HaluEval, 'dialogue' subset):
      knowledge, dialogue_history, right_response, hallucinated_response
    """
    try:
        from datasets import load_dataset
        # 'dialogue' has paired right_response / hallucinated_response fields.
        # 'dialogue_samples' has binary labels only — wrong for this protocol.
        ds = load_dataset("pminervini/HaluEval", "dialogue", split="data")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"HaluEval Dialogue non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="HaluEval-Dial", leave=False):
        knowledge  = ex.get("knowledge", "")
        history    = ex.get("dialogue_history", "")
        right_resp = ex.get("right_response", "")
        hall_resp  = ex.get("hallucinated_response", "")

        # Use natural dialogue format (matches Qwen pre-training distribution better
        # than [Assistant]: which caused below-chance scores)
        context = ""
        if knowledge:
            context += knowledge.strip() + "\n\n"
        context += history.strip() + "\n"
        score_right = conditional_log_likelihood(model, tokenizer, device, context, right_resp)
        score_hall  = conditional_log_likelihood(model, tokenizer, device, context, hall_resp)

        if score_right > score_hall:
            correct += 1

    return correct / len(ds)


def eval_halueval_qa(model, tokenizer, device, n, seed):
    """HaluEval QA — accuracy de détection d'hallucination"""
    try:
        from datasets import load_dataset
        # "qa" subset has paired right_answer / hallucinated_answer fields.
        # "qa_samples" has binary hallucination labels only — wrong format.
        ds = load_dataset("pminervini/HaluEval", "qa", split="data")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"HaluEval QA non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="HaluEval-QA", leave=False):
        q = ex.get("question", "")
        right_ans = ex.get("right_answer", "")
        hall_ans  = ex.get("hallucinated_answer", "")

        score_right = conditional_log_likelihood(model, tokenizer, device, q + " ", right_ans)
        score_hall  = conditional_log_likelihood(model, tokenizer, device, q + " ", hall_ans)

        # Le modèle "détecte" correctement si il préfère la vraie réponse
        if score_right > score_hall:
            correct += 1

    return correct / len(ds)


def eval_halueval_summ(model, tokenizer, device, n, seed):
    """
    HaluEval Summarization — paired right/hallucinated summary comparison.
    Dataset schema (pminervini/HaluEval, 'summarization' subset):
      document, right_summary, hallucinated_summary
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
        doc       = ex.get("document", "")
        right_s   = ex.get("right_summary", "")
        hall_s    = ex.get("hallucinated_summary", "")

        # Use max_length=4096 to avoid truncating long CNN/DM documents
        score_right = conditional_log_likelihood(model, tokenizer, device, doc + "\n\nSummary: ", right_s, max_length=4096)
        score_hall  = conditional_log_likelihood(model, tokenizer, device, doc + "\n\nSummary: ", hall_s, max_length=4096)

        if score_right > score_hall:
            correct += 1

    return correct / len(ds)


def eval_memotrap(model, tokenizer, device, n, seed):
    """
    MemoTrap — fraction where the model follows the instruction over the memorized completion.
    Dataset: pminervini/MemoTrap (may be gated; requires HF login).
    Schema: prompt, target_new (instruction-following), target_true (memorized).
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pminervini/MemoTrap", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"MemoTrap non disponible (dataset peut être privé/gated) : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="MemoTrap", leave=False):
        prompt      = ex.get("prompt", "")
        target_new  = ex.get("target_new", "")   # instruction-following completion
        target_true = ex.get("target_true", "")  # memorized/common completion

        score_new  = conditional_log_likelihood(model, tokenizer, device, prompt, target_new)
        score_true = conditional_log_likelihood(model, tokenizer, device, prompt, target_true)

        if score_new > score_true:
            correct += 1

    return correct / len(ds)


def eval_nq(model, tokenizer, device, n, seed):
    """Natural Questions (open) — exact match after greedy generation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"NQ non disponible : {e}")
        return None

    # 5-shot examples (standard NQ-open few-shot context)
    NQ_FEWSHOT = (
        "Q: when was the last time anyone was on the moon\nA: December 1972\n\n"
        "Q: who wrote he ain't heavy he's my brother lyrics\nA: Bobby Scott\n\n"
        "Q: who is the founder of virgin group\nA: Richard Branson\n\n"
        "Q: who played the lead in grease the movie\nA: John Travolta\n\n"
        "Q: where was the battle of vimy ridge fought\nA: France\n\n"
    )

    correct = 0
    for ex in tqdm(ds, desc="NQ", leave=False):
        question     = ex["question"]
        gold_answers = ex["answer"]  # list[str]

        prompt = NQ_FEWSHOT + f"Q: {question}\nA:"
        pred   = generate_greedy(model, tokenizer, device, prompt, max_new_tokens=32)

        if exact_match_any(pred, gold_answers):
            correct += 1

    return correct / len(ds)


def eval_popqa(model, tokenizer, device, n, seed):
    """PopQA — exact match after greedy generation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("akariasai/PopQA", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    except Exception as e:
        logger.warning(f"PopQA non disponible : {e}")
        return None

    POPQA_FEWSHOT = (
        "Q: What is George Rankin's occupation?\nA: politician\n\n"
        "Q: What is John Mayne's occupation?\nA: journalist\n\n"
        "Q: Who is the spouse of Barack Obama?\nA: Michelle Obama\n\n"
        "Q: What country is Mount Everest located in?\nA: Nepal\n\n"
        "Q: Who directed the movie The Godfather?\nA: Francis Ford Coppola\n\n"
    )

    correct = 0
    for ex in tqdm(ds, desc="PopQA", leave=False):
        question = ex["question"]
        # possible_answers is a JSON-encoded list of strings
        try:
            gold_answers = json.loads(ex.get("possible_answers", "[]"))
        except (ValueError, TypeError):
            gold_answers = [ex.get("possible_answers", "")]

        prompt = POPQA_FEWSHOT + f"Q: {question}\nA:"
        pred   = generate_greedy(model, tokenizer, device, prompt, max_new_tokens=32)

        if exact_match_any(pred, gold_answers):
            correct += 1

    return correct / len(ds)


def eval_truthfulqa(model, tokenizer, device, n, seed):
    """TruthfulQA MC1 et MC2"""
    try:
        ds = load_truthfulqa(n, seed)
    except Exception as e:
        logger.warning(f"TruthfulQA non disponible : {e}")
        return None, None

    mc1_correct = 0
    mc2_correct = 0.0  # accumulates normalized probability mass (continuous MC2 score)

    for ex in tqdm(ds, desc="TruthfulQA", leave=False):
        q = ex["question"]
        mc = ex["mc1_targets"]

        # MC1 : une seule bonne réponse
        choices_mc1 = mc["choices"]
        labels_mc1  = mc["labels"]
        scores = [conditional_log_likelihood(model, tokenizer, device, q + " ", c)
                  for c in choices_mc1]
        pred_mc1 = int(np.argmax(scores))
        if labels_mc1[pred_mc1] == 1:
            mc1_correct += 1

        # MC2 : somme des probabilités softmax assignées aux bonnes réponses
        mc2 = ex["mc2_targets"]
        choices_mc2 = mc2["choices"]
        labels_mc2  = mc2["labels"]
        scores2 = [conditional_log_likelihood(model, tokenizer, device, q + " ", c)
                   for c in choices_mc2]
        # Standard MC2 metric: normalize scores via softmax, sum probs of correct answers
        import torch.nn.functional as F
        scores2_t = torch.tensor(scores2, dtype=torch.float32)
        probs2 = F.softmax(scores2_t, dim=0)
        mask2 = torch.tensor([bool(l) for l in labels_mc2])
        mc2_correct += probs2[mask2].sum().item()

    return mc1_correct / len(ds), mc2_correct / len(ds)


def eval_mmlu(model, tokenizer, device, n, seed):
    """MMLU — 4-way multiple choice"""
    try:
        ds = load_mmlu(n, seed)
    except Exception as e:
        logger.warning(f"MMLU non disponible : {e}")
        return None

    correct = 0
    choices_letters = ["A", "B", "C", "D"]

    for ex in tqdm(ds, desc="MMLU", leave=False):
        q = ex["question"]
        choices = ex["choices"]
        answer_idx = ex["answer"]  # 0-3

        # Build a prompt ending with "Answer:" and score each letter " A".." D".
        # This is the standard lm-eval-harness MMLU protocol.
        prompt = (
            f"{q}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Answer:"
        )
        scores = [
            conditional_log_likelihood(model, tokenizer, device, prompt, f" {letter}")
            for letter in choices_letters
        ]

        pred = int(np.argmax(scores))
        if pred == answer_idx:
            correct += 1

    return correct / len(ds)


def eval_winogrande(model, tokenizer, device, n, seed):
    """WinoGrande — complétion de phrase avec 2 options"""
    try:
        ds = load_winogrande(n, seed)
    except Exception as e:
        logger.warning(f"WinoGrande non disponible : {e}")
        return None

    correct = 0
    for ex in tqdm(ds, desc="WinoGrande", leave=False):
        sentence = ex["sentence"]
        opt1, opt2 = ex["option1"], ex["option2"]
        answer = ex["answer"]  # "1" ou "2"

        # Partial scoring: context = text before blank, completion = option + rest.
        # This avoids length bias from the two options having different token counts.
        blank_idx = sentence.index("_")
        prefix = sentence[:blank_idx]
        suffix = sentence[blank_idx + 1:]  # text after the blank

        s1 = conditional_log_likelihood(model, tokenizer, device, prefix, opt1 + suffix)
        s2 = conditional_log_likelihood(model, tokenizer, device, prefix, opt2 + suffix)

        pred = "1" if s1 > s2 else "2"
        if pred == answer:
            correct += 1

    return correct / len(ds)


def eval_wikitext_bpb(model, tokenizer, device):
    """WikiText BPB (bits per byte) — métrique de perplexité normalisée"""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as e:
        logger.warning(f"WikiText non disponible : {e}")
        return None

    # Concatène tout le texte de test
    full_text = "\n\n".join([ex["text"] for ex in ds if ex["text"].strip()])
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"]

    # Compute actual bytes-per-token for this tokenizer/text combination
    num_bytes = len(full_text.encode("utf-8"))
    num_tokens = input_ids.shape[1]
    bytes_per_token = num_bytes / num_tokens

    # Calcul NLL par blocs de 512 tokens avec stride 256 (standard)
    # Context tokens are masked so only new tokens contribute to the loss.
    stride = 256
    seq_len = 512
    nlls = []
    prev_end = 0

    for begin in tqdm(range(0, input_ids.shape[1] - 1, stride),
                      desc="WikiText BPB", leave=False):
        end = min(begin + seq_len, input_ids.shape[1])
        target_len = end - prev_end
        chunk = input_ids[:, begin:end].to(device)

        # Mask context tokens so loss is computed only over new tokens
        labels = chunk.clone()
        labels[:, :-target_len] = -100

        with torch.no_grad():
            out = model(chunk, labels=labels)
            nll = out.loss.item()

        nlls.append(nll * target_len)
        prev_end = end
        if end == input_ids.shape[1]:
            break

    total_nll = sum(nlls)
    total_tokens = input_ids.shape[1] - 1
    avg_nll = total_nll / total_tokens

    # BPB = avg_NLL_nats / (log(2) * bytes_per_token)
    bpb = avg_nll / math.log(2) / bytes_per_token
    return bpb


# ---------------------------------------------------------------------------
# Pipeline d'évaluation complète
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, device, n=1024, seed=42):
    import math
    results = {}

    logger.info("=== Évaluation en cours ===")

    # --- Hallucination benchmarks ---
    logger.info("HaluEval Dialogue ...")
    halu_dial = eval_halueval_dial(model, tokenizer, device, n, seed)
    if halu_dial is not None:
        results["HE-Dial"] = round(halu_dial, 4)
        logger.info(f"  HE-Dial     = {halu_dial:.4f}")

    logger.info("HaluEval QA ...")
    halu_qa = eval_halueval_qa(model, tokenizer, device, n, seed)
    if halu_qa is not None:
        results["HE-QA"] = round(halu_qa, 4)
        logger.info(f"  HE-QA       = {halu_qa:.4f}")

    logger.info("HaluEval Summarization ...")
    halu_summ = eval_halueval_summ(model, tokenizer, device, n, seed)
    if halu_summ is not None:
        results["HE-Summ"] = round(halu_summ, 4)
        logger.info(f"  HE-Summ     = {halu_summ:.4f}")

    logger.info("MemoTrap ...")
    memo = eval_memotrap(model, tokenizer, device, n, seed)
    if memo is not None:
        results["MemoTrap"] = round(memo, 4)
        logger.info(f"  MemoTrap    = {memo:.4f}")

    logger.info("TruthfulQA ...")
    tfqa_mc1, tfqa_mc2 = eval_truthfulqa(model, tokenizer, device, n, seed)
    if tfqa_mc1 is not None:
        results["TFQA-MC1"] = round(tfqa_mc1, 4)
        results["TFQA-MC2"] = round(tfqa_mc2, 4)
        logger.info(f"  TFQA-MC1    = {tfqa_mc1:.4f}")
        logger.info(f"  TFQA-MC2    = {tfqa_mc2:.4f}")

    # --- Knowledge benchmarks ---
    logger.info("MMLU ...")
    mmlu = eval_mmlu(model, tokenizer, device, n, seed)
    if mmlu is not None:
        results["MMLU"] = round(mmlu, 4)
        logger.info(f"  MMLU        = {mmlu:.4f}")

    logger.info("NQ ...")
    nq = eval_nq(model, tokenizer, device, n, seed)
    if nq is not None:
        results["NQ"] = round(nq, 4)
        logger.info(f"  NQ          = {nq:.4f}")

    logger.info("PopQA ...")
    popqa = eval_popqa(model, tokenizer, device, n, seed)
    if popqa is not None:
        results["PopQA"] = round(popqa, 4)
        logger.info(f"  PopQA       = {popqa:.4f}")

    logger.info("WinoGrande ...")
    wino = eval_winogrande(model, tokenizer, device, n, seed)
    if wino is not None:
        results["Winogrande"] = round(wino, 4)
        logger.info(f"  Winogrande  = {wino:.4f}")

    logger.info("WikiText BPB ...")
    wt_bpb = eval_wikitext_bpb(model, tokenizer, device)
    if wt_bpb is not None:
        results["WikiText_BPB"] = round(wt_bpb, 4)
        logger.info(f"  WikiText BPB= {wt_bpb:.4f}")

    return results


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Chemin vers le modèle fine-tuné ou base model HF")
    parser.add_argument("--base_model", default=None,
                        help="Si model_path est un dossier LoRA, spécifier le base model ici")
    parser.add_argument("--n_samples", type=int, default=EVAL_PARAMS["n_samples"])
    parser.add_argument("--seed", type=int, default=EVAL_PARAMS["seed"])
    parser.add_argument("--csv", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv"),
                        help="Path to results.csv (shared across experiments)")
    parser.add_argument("--csv_column", default=None,
                        help="Column name to update in results.csv (e.g. 'baseline', 'game-lora', 'exp1')")
    parser.add_argument(
        "--reference", choices=["baseline", "game_lora"], default="baseline",
        help="Colonne de référence de l'article (Table 1) : 'baseline' ou 'game_lora'",
    )
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_path, args.base_model)
    results = run_eval(model, tokenizer, device, args.n_samples, args.seed)

    logger.info("\n=== Résultats finaux ===")
    for k, v in results.items():
        logger.info(f"  {k:<20} = {v:.4f}")

    # Table 1 — colonne Baseline
    REFERENCE_BASELINE = {
        "HE-Dial":     0.458,
        "HE-QA":       0.376,
        "HE-Summ":     0.438,
        "MemoTrap":    0.642,
        "TFQA-MC1":    0.252,
        "TFQA-MC2":    0.401,
        "MMLU":        0.477,
        "NQ":          0.066,
        "PopQA":       0.111,
        "WikiText_BPB": 0.784,
        "Winogrande":  0.573,
    }

    # Table 1 — colonne GAME-LoRA
    REFERENCE_GAME_LORA = {
        "HE-Dial":     0.491,
        "HE-QA":       0.445,
        "HE-Summ":     0.500,
        "MemoTrap":    0.650,
        "TFQA-MC1":    0.263,
        "TFQA-MC2":    0.412,
        "MMLU":        0.469,
        "NQ":          0.067,
        "PopQA":       0.112,
        "WikiText_BPB": 0.786,
        "Winogrande":  0.565,
    }

    reference = REFERENCE_GAME_LORA if args.reference == "game_lora" else REFERENCE_BASELINE
    logger.info(f"\nRéférence utilisée : Table 1 — {args.reference}")
    # Comparaison avec les valeurs de référence de l'article

    # Lower-is-better tasks (negated in relative improvement formula)
    LOWER_IS_BETTER = {"WikiText_BPB"}

    # Category definitions for aggregate reporting
    CATEGORIES = {
        "Hallucination": ["HE-Dial", "HE-QA", "HE-Summ", "MemoTrap", "TFQA-MC1", "TFQA-MC2"],
        "Knowledge":     ["MMLU", "NQ", "PopQA", "WikiText_BPB", "Winogrande"],
    }

    logger.info("\n=== Comparaison avec l'article (Baseline) ===")
    deltas = {}
    for k, ref in reference.items():
        if k in results and ref != 0:
            s = results[k]
            if k in LOWER_IS_BETTER:
                pct = (ref - s) / ref * 100.0   # improvement = reduction in BPB
            else:
                pct = (s - ref) / ref * 100.0
            deltas[k] = pct
            logger.info(f"  {k:<20} : obtenu={s:.4f}  ref={ref:.4f}  Δ={pct:+.2f}%")

    logger.info("\n=== Agrégats par catégorie ===")
    for cat, tasks in CATEGORIES.items():
        cat_deltas = [deltas[t] for t in tasks if t in deltas]
        if cat_deltas:
            avg = sum(cat_deltas) / len(cat_deltas)
            logger.info(f"  {cat:<20} : {avg:+.1f}%")

    # --- Update results.csv if --csv_column is specified ---
    if args.csv_column:
        import csv
        csv_path = args.csv
        # Read existing CSV
        rows = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames) if reader.fieldnames else ["metric"]
                for row in reader:
                    rows[row["metric"]] = row
        else:
            fieldnames = ["metric"]

        # Ensure column exists
        col = args.csv_column
        if col not in fieldnames:
            fieldnames.append(col)

        # Update values
        for metric, value in results.items():
            if metric not in rows:
                rows[metric] = {"metric": metric}
            rows[metric][col] = str(value)

        # Canonical metric order
        METRIC_ORDER = [
            "HE-Dial", "HE-QA", "HE-Summ", "MemoTrap",
            "TFQA-MC1", "TFQA-MC2", "MMLU", "NQ", "PopQA",
            "Winogrande", "WikiText_BPB",
        ]
        ordered_metrics = [m for m in METRIC_ORDER if m in rows]
        ordered_metrics += [m for m in rows if m not in METRIC_ORDER]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in ordered_metrics:
                writer.writerow(rows[m])

        logger.info(f"Colonne '{col}' mise à jour dans {csv_path}")
