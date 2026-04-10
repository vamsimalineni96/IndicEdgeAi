"""
build_eval_set.py

Pulls samples from the IndicQA dataset on HuggingFace and builds
a clean eval set for benchmarking Sarvam-1 across Indic languages.

Usage:
    python eval/build_eval_set.py --langs hi te ta --n_samples 200
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# IndicQA language codes → HuggingFace dataset config names
LANG_MAP = {
    "hi": "indicqa.hi",   # Hindi
    "te": "indicqa.te",   # Telugu
    "ta": "indicqa.ta",   # Tamil
    "bn": "indicqa.bn",   # Bengali
    "mr": "indicqa.mr",   # Marathi
    "gu": "indicqa.gu",   # Gujarati
    "kn": "indicqa.kn",   # Kannada
    "ml": "indicqa.ml",   # Malayalam
    "pa": "indicqa.pa",   # Punjabi
    "or": "indicqa.or",   # Odia
}

LANG_NAMES = {
    "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
    "ml": "Malayalam", "pa": "Punjabi", "or": "Odia",
}

def build_prompt(context: str, question: str) -> str:
    """Format a QA prompt for Sarvam-1."""
    return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"


def load_indicqa_samples(lang_code: str, n: int) -> list[dict]:
    """Load n samples from IndicQA for a given language."""
    config = LANG_MAP.get(lang_code)
    if not config:
        raise ValueError(f"Unsupported language: {lang_code}. Choose from {list(LANG_MAP.keys())}")

    print(f"  Loading IndicQA [{LANG_NAMES[lang_code]}]...")
    try:
        dataset = load_dataset("ai4bharat/IndicQA", config, split="test", trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load {config}: {e}")
        print(f"  Trying fallback split 'validation'...")
        dataset = load_dataset("ai4bharat/IndicQA", config, split="validation", trust_remote_code=True)

    samples = []
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for i in tqdm(indices, desc=f"  Processing {LANG_NAMES[lang_code]}", leave=False):
        row = dataset[i]

        # IndicQA format: context, question, answers (list)
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {})

        # Extract answer text
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not answer_texts:
            continue

        reference_answer = answer_texts[0]

        samples.append({
            "id": f"{lang_code}_{i}",
            "lang": lang_code,
            "lang_name": LANG_NAMES[lang_code],
            "context": context,
            "question": question,
            "reference_answer": reference_answer,
            "prompt": build_prompt(context, question),
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Build eval set from IndicQA")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["hi", "te", "ta"],
        help=f"Language codes. Options: {list(LANG_MAP.keys())}",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Total number of samples (split evenly across languages)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/eval_set.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Samples per language
    n_per_lang = args.n_samples // len(args.langs)
    print(f"\nBuilding eval set:")
    print(f"  Languages : {args.langs}")
    print(f"  Samples   : {n_per_lang} per language ({n_per_lang * len(args.langs)} total)")
    print(f"  Output    : {args.output}\n")

    all_samples = []
    for lang in args.langs:
        samples = load_indicqa_samples(lang, n_per_lang)
        all_samples.extend(samples)
        print(f"  ✓ {LANG_NAMES[lang]}: {len(samples)} samples loaded")

    # Shuffle
    random.shuffle(all_samples)

    # Write to JSONL
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n✅ Eval set saved: {out_path}")
    print(f"   Total samples : {len(all_samples)}")

    # Print a quick sample preview
    print(f"\n--- Sample preview ---")
    sample = all_samples[0]
    print(f"Lang    : {sample['lang_name']}")
    print(f"Question: {sample['question']}")
    print(f"Answer  : {sample['reference_answer']}")


if __name__ == "__main__":
    main()
