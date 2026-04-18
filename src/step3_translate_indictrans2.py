"""
Step 3: Translate to Hindi and Telugu using IndicTrans2
========================================================
Uses AI4Bharat's IndicTrans2 model with IndicTransToolkit processor
for proper pre/post-processing.

Setup (run once on H100):
    pip install IndicTransToolkit
    pip install mosestokenizer
    pip install indic-nlp-library
    python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

Usage:
    python step3_translate_indictrans2.py

Output:
    data/translated/medqa_hindi.json
    data/translated/medqa_telugu.json
    data/translated/healthsearchqa_hindi.json
    data/translated/healthsearchqa_telugu.json
"""

import json
import os
import time
import torch
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────
INPUT_DIR = "data/curated"
OUTPUT_DIR = "data/translated"
BATCH_SIZE = 32  # IndicTrans2 works best with small batches
MAX_SEQ_LEN = 256  # Max tokens per sentence for translation

INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LANGS = {
    "hin_Deva": "hindi",
    "tel_Telu": "telugu",
}

SRC_LANG = "eng_Latn"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def initialize_model():
    """Load IndicTrans2 model, tokenizer, and IndicProcessor."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit.processor import IndicProcessor

    print("   Loading IndicTrans2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        INDICTRANS_MODEL, trust_remote_code=True
    )

    print("   Loading IndicTrans2 model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    print("   Initializing IndicProcessor...")
    ip = IndicProcessor(inference=True)

    print("   ✓ All components loaded!")
    return model, tokenizer, ip


def translate_batch(sentences, src_lang, tgt_lang, model, tokenizer, ip):
    """
    Translate a batch of sentences using the correct IndicTrans2 pipeline:
    1. IndicProcessor.preprocess_batch() — handles script normalization & placeholders
    2. Tokenize & generate
    3. IndicProcessor.postprocess_batch() — restores entities & formatting
    """
    # Step 1: Preprocess with IndicProcessor
    preprocessed = ip.preprocess_batch(
        sentences, src_lang=src_lang, tgt_lang=tgt_lang
    )

    # Step 2: Tokenize
    inputs = tokenizer(
        preprocessed,
        truncation=True,
        padding="longest",
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Step 3: Generate translations
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=MAX_SEQ_LEN,
            num_beams=1,
            num_return_sequences=1,
            do_sample=False
        )

    # Step 4: Decode tokens to text
    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # Step 5: Postprocess with IndicProcessor (entity replacement etc.)
    translations = ip.postprocess_batch(decoded, lang=tgt_lang)

    del inputs
    torch.cuda.empty_cache()

    return translations


def translate_text_list(texts, src_lang, tgt_lang, model, tokenizer, ip, desc=""):
    """Translate a list of texts in batches with progress bar."""
    all_translations = []
    errors = 0

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
        batch = texts[i:i + BATCH_SIZE]

        # Skip empty strings
        batch_cleaned = [t if t.strip() else "." for t in batch]

        try:
            translated = translate_batch(
                batch_cleaned, src_lang, tgt_lang, model, tokenizer, ip
            )
            all_translations.extend(translated)
        except Exception as e:
            print(f"\n   ⚠ Error at batch {i}: {e}")
            # Keep original English on error
            all_translations.extend(batch)
            errors += 1

    if errors:
        print(f"   ⚠ {errors} batch errors (kept English for those)")

    return all_translations


def translate_dataset(data, tgt_lang_code, tgt_lang_name, model, tokenizer, ip):
    """Translate an entire dataset (instructions + responses) to target language."""
    print(f"\n{'='*50}")
    print(f"   Translating to {tgt_lang_name} ({tgt_lang_code})")
    print(f"{'='*50}")

    instructions = [ex["instruction"] for ex in data]
    responses = [ex["response"] for ex in data]

    print(f"   Translating {len(instructions)} instructions...")
    translated_instructions = translate_text_list(
        instructions, SRC_LANG, tgt_lang_code, model, tokenizer, ip,
        desc=f"Instructions → {tgt_lang_name}"
    )

    print(f"   Translating {len(responses)} responses...")
    translated_responses = translate_text_list(
        responses, SRC_LANG, tgt_lang_code, model, tokenizer, ip,
        desc=f"Responses → {tgt_lang_name}"
    )

    # Assemble translated dataset
    translated = []
    for i, ex in enumerate(data):
        translated.append({
            "instruction": translated_instructions[i] if i < len(translated_instructions) else ex["instruction"],
            "response": translated_responses[i] if i < len(translated_responses) else ex["response"],
            "subject": ex["subject"],
            "topic": ex.get("topic", ""),
            "source": ex["source"],
            "language": tgt_lang_name,
            "original_instruction_en": ex["instruction"],
            "original_response_en": ex["response"],
        })

    return translated


def main():
    print("=" * 60)
    print("  STEP 3: Translate to Hindi & Telugu (IndicTrans2)")
    print("=" * 60)

    # ── Load source data ────────────────────────────────────────────
    print("\n[1/4] Loading curated datasets...")
    datasets_to_translate = {}

    medqa_path = os.path.join(INPUT_DIR, "medqa_curated_15k.json")
    healthqa_path = os.path.join(INPUT_DIR, "healthsearchqa_with_answers.json")

    if os.path.exists(medqa_path):
        with open(medqa_path, "r", encoding="utf-8") as f:
            datasets_to_translate["medqa"] = json.load(f)
        print(f"   ✓ MedQA: {len(datasets_to_translate['medqa'])} examples")
    else:
        print(f"   ✗ MedQA not found: {medqa_path}")

    if os.path.exists(healthqa_path):
        with open(healthqa_path, "r", encoding="utf-8") as f:
            datasets_to_translate["healthsearchqa"] = json.load(f)
        print(f"   ✓ HealthSearchQA: {len(datasets_to_translate['healthsearchqa'])} examples")
    else:
        print(f"   ✗ HealthSearchQA not found: {healthqa_path}")

    if not datasets_to_translate:
        print("   No datasets found! Run steps 1 and 2 first.")
        return

    # ── Install dependencies check ──────────────────────────────────
    print("\n[2/4] Checking dependencies...")
    try:
        from IndicTransToolkit.processor import IndicProcessor
        print("   ✓ IndicTransToolkit installed")
    except ImportError:
        print("   ✗ IndicTransToolkit not found. Install it:")
        print("     pip install IndicTransToolkit")
        return

    try:
        from mosestokenizer import MosesSentenceSplitter
        print("   ✓ mosestokenizer installed")
    except ImportError:
        print("   ⚠ mosestokenizer not found. Installing...")
        os.system("pip install mosestokenizer")

    # ── Load model ──────────────────────────────────────────────────
    print("\n[3/4] Loading IndicTrans2 model...")
    model, tokenizer, ip = initialize_model()

    # ── Quick test ──────────────────────────────────────────────────
    print("\n   Quick translation test...")
    test_sentences = ["What are the symptoms of diabetes?"]
    for tgt_code, tgt_name in TARGET_LANGS.items():
        test_result = translate_batch(
            test_sentences, SRC_LANG, tgt_code, model, tokenizer, ip
        )
        print(f"   EN → {tgt_name}: '{test_result[0]}'")

    # ── Translate all datasets ──────────────────────────────────────
    print("\n[4/4] Translating datasets...")

    for tgt_lang_code, tgt_lang_name in TARGET_LANGS.items():
        for ds_name, ds_data in datasets_to_translate.items():
            start_time = time.time()

            translated = translate_dataset(
                ds_data, tgt_lang_code, tgt_lang_name, model, tokenizer, ip
            )

            elapsed = time.time() - start_time
            output_path = os.path.join(OUTPUT_DIR, f"{ds_name}_{tgt_lang_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)

            print(f"   ✓ Saved {len(translated)} {tgt_lang_name} examples to {output_path}")
            print(f"     Time: {elapsed:.1f}s ({elapsed/len(translated):.3f}s per example)")

    # ── Save English originals with language tag ────────────────────
    print("\n   Tagging English originals...")
    for ds_name, ds_data in datasets_to_translate.items():
        english_tagged = []
        for ex in ds_data:
            english_tagged.append({
                "instruction": ex["instruction"],
                "response": ex["response"],
                "subject": ex["subject"],
                "topic": ex.get("topic", ""),
                "source": ex["source"],
                "language": "english",
            })

        output_path = os.path.join(OUTPUT_DIR, f"{ds_name}_english.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(english_tagged, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Saved {len(english_tagged)} English examples to {output_path}")

    # ── Cleanup ─────────────────────────────────────────────────────
    del model, tokenizer
    torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRANSLATION COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}/")
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        if f_name.endswith(".json"):
            f_path = os.path.join(OUTPUT_DIR, f_name)
            with open(f_path, "r") as f:
                count = len(json.load(f))
            print(f"    {f_name}: {count} examples")
    print("=" * 60)
    print("\n  Next: Run step4_merge_and_split.py")


if __name__ == "__main__":
    main()