"""
pipeline.py
-----------
Main CLI for the IndicEdgeAI MedQA dataset builder.

Translates and/or transliterates a MedQA dataset (JSONL or JSON list) into
one or more Indic languages using AI4Bharat models.

Output schema per record
────────────────────────
{
    "id":                    0,
    "answer":                "A",
    "question_en":           "What is the primary ...",
    "options_en":            {"A": "...", "B": "...", "C": "...", "D": "..."},

    # For each requested language + mode:
    "question_hindi_trans":      "...",   # IndicTrans2 semantic translation
    "options_hindi_trans":       {"A": "...", ...},
    "question_hindi_translit":   "...",   # phonetic English in Devanagari
    "options_hindi_translit":    {"A": "...", ...},

    "question_telugu_trans":     "...",
    "options_telugu_trans":      {"A": "...", ...},
    "question_telugu_translit":  "...",
    "options_telugu_translit":   {"A": "...", ...},
}

Usage examples
──────────────
# Full pipeline — both languages, both modes (default)
python pipeline.py --input data/medqa_test.jsonl

# Translation only, Hindi only
python pipeline.py --input data/medqa_test.jsonl --langs hindi --modes translation

# Transliteration only, both languages
python pipeline.py --input data/medqa_test.jsonl --modes transliteration

# Custom output dir, larger batch size for GPU
python pipeline.py --input data/medqa_test.jsonl --output_dir out/ --batch_size 32

# Disable resume (always re-run from scratch)
python pipeline.py --input data/medqa_test.jsonl --no_resume

Supported languages : hindi  telugu  tamil  kannada  bengali  marathi
Supported modes     : translation  transliteration
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data import get_texts, load_records, merge_outputs, save_records, select_lang_fields
from src.model import login_hf
from src.translation import LANG_CODES, translate_fields
from src.transliteration import SCRIPT_MAP, transliterate_fields

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

SUPPORTED_LANGS = sorted(set(LANG_CODES) & set(SCRIPT_MAP))  # langs valid for both modes


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IndicEdgeAI — Translate & transliterate MedQA to Indic languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input JSONL or JSON file",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Output directory (default: <input_parent>/output/)",
    )
    parser.add_argument(
        "--langs", nargs="+", default=["hindi", "telugu"],
        choices=SUPPORTED_LANGS, metavar="LANG",
        help=f"Target languages. Default: hindi telugu. Options: {SUPPORTED_LANGS}",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["translation", "transliteration"],
        choices=["translation", "transliteration"], metavar="MODE",
        help="Pipeline modes to run (default: both)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Sentences per translation batch (default: 8; use 32+ on GPU)",
    )
    parser.add_argument(
        "--num_beams", type=int, default=4,
        help="Beam search width for translation (default: 4)",
    )
    parser.add_argument(
        "--max_length", type=int, default=512,
        help="Max output token length for translation (default: 512)",
    )
    parser.add_argument(
        "--model_name", default="ai4bharat/indictrans2-en-indic-1B",
        help="IndicTrans2 HuggingFace model ID",
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Disable field-level checkpoint resume (always start fresh)",
    )
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Auth
    login_hf()

    # Paths
    input_path = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    log.info("=" * 60)
    log.info("Input      : %s", input_path)
    log.info("Output dir : %s", out_dir)
    log.info("Languages  : %s", ", ".join(args.langs))
    log.info("Modes      : %s", ", ".join(args.modes))
    log.info("=" * 60)

    # Load data
    records = load_records(input_path)
    texts, option_keys = get_texts(records)
    log.info(
        "Records: %d | Fields: %s | Options: %s",
        len(records), list(texts.keys()), option_keys,
    )

    # ── Run pipeline ────────────────────────────────────────────────────────────
    # lang_outputs[lang][mode_key] = {field: [str, ...]}
    lang_outputs: dict[str, dict[str, dict[str, list[str]]]] = {}

    for lang in args.langs:
        lang_outputs[lang] = {}

        if "translation" in args.modes:
            log.info("")
            log.info("── TRANSLATION → %s ──", lang.upper())
            ckpt = (
                None if args.no_resume
                else out_dir / f".ckpt_{stem}_{lang}_trans.json"
            )
            lang_outputs[lang]["trans"] = translate_fields(
                texts,
                lang=lang,
                batch_size=args.batch_size,
                max_length=args.max_length,
                num_beams=args.num_beams,
                checkpoint_path=ckpt,
                model_name=args.model_name,
            )

        if "transliteration" in args.modes:
            log.info("")
            log.info("── TRANSLITERATION → %s ──", lang.upper())
            lang_outputs[lang]["translit"] = transliterate_fields(texts, lang=lang)

    # ── Merge and save ──────────────────────────────────────────────────────────
    log.info("")
    log.info("Merging outputs...")
    merged = merge_outputs(records, option_keys, lang_outputs)

    # Combined file with all languages and modes
    combined_path = out_dir / f"{stem}_indic.jsonl"
    save_records(merged, combined_path)

    # Per-language files for convenience
    for lang in args.langs:
        lang_records = select_lang_fields(merged, lang)
        save_records(lang_records, out_dir / f"{stem}_{lang}.jsonl")

    # Clean up checkpoint files on success
    if not args.no_resume:
        for lang in args.langs:
            ckpt = out_dir / f".ckpt_{stem}_{lang}_trans.json"
            if ckpt.exists():
                ckpt.unlink()
                log.debug("Removed checkpoint: %s", ckpt)

    log.info("")
    log.info("=" * 60)
    log.info("Done.")
    log.info("  Combined : %s", combined_path)
    for lang in args.langs:
        log.info("  %-10s: %s", lang, out_dir / f"{stem}_{lang}.jsonl")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
