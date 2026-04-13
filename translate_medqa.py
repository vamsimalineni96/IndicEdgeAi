"""
translate_medqa.py
------------------
Translates a MedQA dataset (JSONL or JSON) from English to Hindi and Telugu
using AI4Bharat's IndicTrans2.

Supports two MedQA formats:
  1. JSONL  — one JSON object per line
  2. JSON   — a list of objects

Each record is expected to have:
    {
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "answer":  "A"          # just the key, not translated
    }

Output: two files
    <stem>_hindi.jsonl
    <stem>_telugu.jsonl

Usage:
    python translate_medqa.py --input data/medqa_test.jsonl
    python translate_medqa.py --input data/medqa_test.jsonl --batch_size 8 --lang both
    python translate_medqa.py --input data/medqa_test.jsonl --lang hindi
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME  = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG    = "eng_Latn"
LANG_MAP    = {
    "hindi":  "hin_Deva",
    "telugu": "tel_Telu",
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(model_name: str = MODEL_NAME):
    print(f"[INFO] Loading model: {model_name}  (device={DEVICE})")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)
    print("[INFO] Model ready.\n")
    return tokenizer, model, ip


# ── Core translation ───────────────────────────────────────────────────────────
def translate_batch(
    sentences: list[str],
    tgt_lang: str,
    tokenizer,
    model,
    ip: IndicProcessor,
    max_length: int = 512,
    num_beams: int = 4,
) -> list[str]:
    """Translate a list of English strings to tgt_lang."""
    # skip empty / None strings
    safe = [s if s else "" for s in sentences]

    batch = ip.preprocess_batch(safe, src_lang=SRC_LANG, tgt_lang=tgt_lang)

    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
        )

    raw = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return ip.postprocess_batch(raw, lang=tgt_lang)


# ── Dataset I/O ────────────────────────────────────────────────────────────────
def load_dataset(path: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON list first, fall back to JSONL
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"[INFO] Loaded {len(data)} records from JSON list.")
            return data
    except json.JSONDecodeError:
        pass

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    print(f"[INFO] Loaded {len(records)} records from JSONL.")
    return records


def save_dataset(records: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(records)} records → {path}")


# ── Field extraction helpers ───────────────────────────────────────────────────
def extract_texts(records: list[dict]) -> dict[str, list[str]]:
    """
    Returns a dict of field_key -> list of strings (one per record).
    Handles flat options or nested dict options.
    """
    questions = [r.get("question", "") for r in records]

    # Collect all option keys across the dataset
    option_keys = set()
    for r in records:
        opts = r.get("options", {})
        if isinstance(opts, dict):
            option_keys.update(opts.keys())
        elif isinstance(opts, list):
            option_keys.update(range(len(opts)))

    option_keys = sorted(option_keys)

    texts = {"question": questions}
    for key in option_keys:
        if isinstance(key, int):
            texts[f"option_{key}"] = [
                r.get("options", [])[key] if isinstance(r.get("options"), list) and key < len(r.get("options", [])) else ""
                for r in records
            ]
        else:
            texts[f"option_{key}"] = [
                r.get("options", {}).get(key, "") if isinstance(r.get("options"), dict) else ""
                for r in records
            ]

    return texts, option_keys


def build_translated_records(
    records: list[dict],
    translations: dict[str, list[str]],
    option_keys,
    lang_label: str,
) -> list[dict]:
    """Merge translated fields back into records."""
    output = []
    for i, rec in enumerate(records):
        new_rec = {
            "question": translations["question"][i],
            "answer":   rec.get("answer", ""),          # answer key unchanged
            f"question_{lang_label}": translations["question"][i],
            "question_en": rec.get("question", ""),
        }

        # Rebuild options dict / list in translated form
        orig_opts = rec.get("options", {})
        if isinstance(orig_opts, dict):
            new_opts = {}
            for key in option_keys:
                field = f"option_{key}"
                new_opts[key] = translations.get(field, [""] * len(records))[i]
            new_rec["options"] = new_opts
            new_rec["options_en"] = orig_opts
        elif isinstance(orig_opts, list):
            new_opts = []
            for key in option_keys:
                field = f"option_{key}"
                new_opts.append(translations.get(field, [""] * len(records))[i])
            new_rec["options"] = new_opts
            new_rec["options_en"] = orig_opts

        output.append(new_rec)
    return output


# ── Main translation pipeline ──────────────────────────────────────────────────
def translate_dataset(
    records: list[dict],
    tgt_lang_code: str,
    lang_label: str,
    tokenizer,
    model,
    ip: IndicProcessor,
    batch_size: int = 8,
) -> list[dict]:

    texts, option_keys = extract_texts(records)
    translations = {}

    total_fields = len(texts)
    for field_idx, (field_name, sentences) in enumerate(texts.items(), 1):
        print(f"  [{field_idx}/{total_fields}] Translating field: '{field_name}' ({len(sentences)} sentences)")

        translated = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            t0 = time.time()
            result = translate_batch(batch, tgt_lang_code, tokenizer, model, ip)
            elapsed = time.time() - t0
            translated.extend(result)

            done = min(start + batch_size, len(sentences))
            print(f"    {done}/{len(sentences)}  ({elapsed:.1f}s)")

        translations[field_name] = translated

    return build_translated_records(records, translations, option_keys, lang_label)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Translate MedQA to Hindi / Telugu")
    parser.add_argument("--input",      required=True,  help="Path to input JSONL or JSON file")
    parser.add_argument("--output_dir", default=None,   help="Output directory (default: same as input)")
    parser.add_argument("--batch_size", type=int, default=8, help="Sentences per batch (default: 8)")
    parser.add_argument("--lang",       default="both", choices=["hindi", "telugu", "both"],
                        help="Target language(s) (default: both)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    # Load dataset
    records = load_dataset(args.input)

    # Load model once
    tokenizer, model, ip = load_model()

    # Determine target languages
    targets = []
    if args.lang in ("hindi", "both"):
        targets.append(("hindi",  LANG_MAP["hindi"]))
    if args.lang in ("telugu", "both"):
        targets.append(("telugu", LANG_MAP["telugu"]))

    # Translate and save
    for lang_label, lang_code in targets:
        print(f"\n{'='*60}")
        print(f" Translating to {lang_label.upper()}  ({lang_code})")
        print(f"{'='*60}")

        translated = translate_dataset(
            records, lang_code, lang_label,
            tokenizer, model, ip,
            batch_size=args.batch_size,
        )

        out_path = out_dir / f"{stem}_{lang_label}.jsonl"
        save_dataset(translated, str(out_path))

    print("\n[DONE] All translations complete.")


if __name__ == "__main__":
    main()
