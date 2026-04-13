"""
test.py — Sanity check for the IndicEdgeAI pipeline.

Runs a single MedQA-style record through both translation and transliteration
to verify the full pipeline is working before processing a large dataset.

Usage:
    python test.py
    python test.py --lang telugu --mode translation
"""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test")


SAMPLE_RECORD = {
    "id": "sanity-0",
    "question": "A 45-year-old patient presents with a history of hypertension and type 2 diabetes. Which medication is most appropriate?",
    "options": {
        "A": "Metformin",
        "B": "Lisinopril",
        "C": "Atorvastatin",
        "D": "Amoxicillin",
    },
    "answer": "B",
}


def run_translation(lang: str) -> None:
    from src.model import login_hf, load_translation_model
    from src.translation import LANG_CODES, translate_fields

    if lang not in LANG_CODES:
        log.error("Unsupported lang '%s'. Valid: %s", lang, sorted(LANG_CODES))
        sys.exit(1)

    login_hf()
    load_translation_model()  # warm up

    texts = {
        "question": [SAMPLE_RECORD["question"]],
        **{f"option_{k}": [v] for k, v in SAMPLE_RECORD["options"].items()},
    }

    results = translate_fields(texts, lang=lang, batch_size=1)

    print(f"\n{'=' * 60}")
    print(f"TRANSLATION → {lang.upper()}")
    print(f"{'=' * 60}")
    print(f"  Question EN : {SAMPLE_RECORD['question']}")
    print(f"  Question    : {results['question'][0]}")
    for k in sorted(SAMPLE_RECORD["options"]):
        print(f"  Option {k}  EN : {SAMPLE_RECORD['options'][k]}")
        print(f"  Option {k}     : {results[f'option_{k}'][0]}")


def run_transliteration(lang: str) -> None:
    from src.transliteration import SCRIPT_MAP, transliterate_fields

    if lang not in SCRIPT_MAP:
        log.error("Unsupported lang '%s'. Valid: %s", lang, sorted(SCRIPT_MAP))
        sys.exit(1)

    texts = {
        "question": [SAMPLE_RECORD["question"]],
        **{f"option_{k}": [v] for k, v in SAMPLE_RECORD["options"].items()},
    }

    results = transliterate_fields(texts, lang=lang)

    print(f"\n{'=' * 60}")
    print(f"TRANSLITERATION → {lang.upper()}")
    print(f"{'=' * 60}")
    print(f"  Question EN : {SAMPLE_RECORD['question']}")
    print(f"  Question    : {results['question'][0]}")
    for k in sorted(SAMPLE_RECORD["options"]):
        print(f"  Option {k}  EN : {SAMPLE_RECORD['options'][k]}")
        print(f"  Option {k}     : {results[f'option_{k}'][0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check for the IndicEdgeAI pipeline")
    parser.add_argument("--lang", default="hindi", help="Target language (default: hindi)")
    parser.add_argument(
        "--mode", default="both", choices=["translation", "transliteration", "both"],
        help="Which mode to test (default: both)",
    )
    args = parser.parse_args()

    print(f"\nSample record:\n{json.dumps(SAMPLE_RECORD, indent=2, ensure_ascii=False)}\n")

    if args.mode in ("translation", "both"):
        run_translation(args.lang)

    if args.mode in ("transliteration", "both"):
        run_transliteration(args.lang)

    print("\n[PASS] Sanity check complete.")


if __name__ == "__main__":
    main()
