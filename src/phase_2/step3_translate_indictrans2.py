"""
Step 3: Translate to Hindi and Telugu using IndicTrans2
========================================================
Uses AI4Bharat's IndicTrans2 model to translate the curated English
medical QA pairs into Hindi and Telugu.

Setup (run once):
    pip install sentencepiece protobuf
    git clone https://github.com/AI4Bharat/IndicTrans2.git
    cd IndicTrans2/huggingface_interface
    pip install -e .

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
BATCH_SIZE = 16  # IndicTrans2 translation batch size

# IndicTrans2 model — the En→Indic direction
INDICTRANS_MODEL = "ai4bharat/indictrans2-en-indic-1B"
# Use the distilled 1B model for speed; switch to the non-distilled
# "ai4bharat/indictrans2-en-indic-dist-200M" for even faster but lower quality

TARGET_LANGS = {
    "hin_Deva": "hindi",    # Hindi in Devanagari
    "tel_Telu": "telugu",   # Telugu in Telugu script
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_indictrans2():
    """Load IndicTrans2 model and tokenizer."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print("   Loading IndicTrans2 model...")
    tokenizer = AutoTokenizer.from_pretrained(
        INDICTRANS_MODEL, trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        INDICTRANS_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda")
    model.eval()
    print("   Model loaded successfully!")
    return model, tokenizer


def translate_batch_indictrans2(model, tokenizer, texts, src_lang, tgt_lang):
    """Translate a batch of texts using IndicTrans2."""
    # IndicTrans2 uses special language tags
    # Format: >>tgt_lang<< source text
    tagged_texts = [f">>{'hin_Deva' if tgt_lang == 'hin_Deva' else 'tel_Telu'}<< {t}" for t in texts]

    inputs = tokenizer(
        tagged_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=5,
            num_return_sequences=1,
        )

    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return translations


def try_ct2_translate(texts, src_lang, tgt_lang):
    """
    Alternative: Use the IndicTrans2 CT2 interface if available.
    This is faster but requires additional setup.
    Falls back to HuggingFace if not available.
    """
    try:
        from IndicTransToolkit.processor import IndicProcessor
        import ctranslate2

        ct2_model_path = f"indictrans2-en-indic-1B-ct2"
        if not os.path.exists(ct2_model_path):
            return None

        processor = IndicProcessor(inference=True)
        translator = ctranslate2.Translator(ct2_model_path, device="cuda")

        processed = processor.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
        # ... CT2 translation logic
        return None  # Placeholder — use HF interface instead

    except ImportError:
        return None


def translate_dataset(model, tokenizer, data, tgt_lang_code, tgt_lang_name):
    """Translate an entire dataset to the target language."""
    print(f"\n   Translating to {tgt_lang_name} ({tgt_lang_code})...")
    translated = []
    errors = 0

    # Collect all texts to translate (instructions + responses)
    instructions = [ex["instruction"] for ex in data]
    responses = [ex["response"] for ex in data]

    # Translate instructions in batches
    print(f"   Translating {len(instructions)} instructions...")
    translated_instructions = []
    for i in tqdm(range(0, len(instructions), BATCH_SIZE), desc="Instructions"):
        batch = instructions[i:i + BATCH_SIZE]
        try:
            trans = translate_batch_indictrans2(
                model, tokenizer, batch, "eng_Latn", tgt_lang_code
            )
            translated_instructions.extend(trans)
        except Exception as e:
            print(f"   Warning: Translation error at batch {i}: {e}")
            # On error, keep original English
            translated_instructions.extend(batch)
            errors += 1

    # Translate responses in batches
    print(f"   Translating {len(responses)} responses...")
    translated_responses = []
    for i in tqdm(range(0, len(responses), BATCH_SIZE), desc="Responses"):
        batch = responses[i:i + BATCH_SIZE]
        try:
            trans = translate_batch_indictrans2(
                model, tokenizer, batch, "eng_Latn", tgt_lang_code
            )
            translated_responses.extend(trans)
        except Exception as e:
            print(f"   Warning: Translation error at batch {i}: {e}")
            translated_responses.extend(batch)
            errors += 1

    # Assemble translated dataset
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

    if errors:
        print(f"   ⚠ {errors} batch errors (kept English for those)")

    return translated


def main():
    print("=" * 60)
    print("  STEP 3: Translate to Hindi & Telugu (IndicTrans2)")
    print("=" * 60)

    # ── Load source data ────────────────────────────────────────────
    print("\n[1/4] Loading curated datasets...")

    medqa_path = os.path.join(INPUT_DIR, "medqa_curated_15k.json")
    healthqa_path = os.path.join(INPUT_DIR, "healthsearchqa_with_answers.json")

    datasets_to_translate = {}

    if os.path.exists(medqa_path):
        with open(medqa_path, "r", encoding="utf-8") as f:
            datasets_to_translate["medqa"] = json.load(f)
        print(f"   Loaded MedQA: {len(datasets_to_translate['medqa'])} examples")
    else:
        print(f"   ⚠ MedQA file not found: {medqa_path}")
        print(f"     Run step1_download_and_curate.py first!")

    if os.path.exists(healthqa_path):
        with open(healthqa_path, "r", encoding="utf-8") as f:
            datasets_to_translate["healthsearchqa"] = json.load(f)
        print(f"   Loaded HealthSearchQA: {len(datasets_to_translate['healthsearchqa'])} examples")
    else:
        print(f"   ⚠ HealthSearchQA file not found: {healthqa_path}")
        print(f"     Run step2_generate_healthsearchqa.py first!")
        print(f"     (Proceeding with available datasets)")

    if not datasets_to_translate:
        print("   No datasets to translate! Exiting.")
        return

    # ── Load IndicTrans2 ────────────────────────────────────────────
    print("\n[2/4] Loading IndicTrans2 model...")
    model, tokenizer = load_indictrans2()

    # ── Translate ───────────────────────────────────────────────────
    print("\n[3/4] Translating datasets...")

    for tgt_lang_code, tgt_lang_name in TARGET_LANGS.items():
        for ds_name, ds_data in datasets_to_translate.items():
            start_time = time.time()
            translated = translate_dataset(
                model, tokenizer, ds_data, tgt_lang_code, tgt_lang_name
            )
            elapsed = time.time() - start_time

            output_path = os.path.join(OUTPUT_DIR, f"{ds_name}_{tgt_lang_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(translated, f, ensure_ascii=False, indent=2)

            print(f"   ✓ Saved {len(translated)} {tgt_lang_name} examples to {output_path}")
            print(f"     Time: {elapsed:.1f}s ({elapsed/len(translated):.3f}s per example)")

    # ── Also keep English originals tagged ──────────────────────────
    print("\n[4/4] Tagging English originals...")
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
    del model
    torch.cuda.empty_cache()

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRANSLATION COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}")
    for f_name in sorted(os.listdir(OUTPUT_DIR)):
        if f_name.endswith(".json"):
            f_path = os.path.join(OUTPUT_DIR, f_name)
            with open(f_path, "r") as f:
                count = len(json.load(f))
            print(f"    {f_name}: {count} examples")
    print("=" * 60)


if __name__ == "__main__":
    main()
