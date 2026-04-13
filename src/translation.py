"""
src/translation.py
------------------
Batch translation of text fields using AI4Bharat IndicTrans2.

Features:
  - Field-level checkpointing: if the process is interrupted, re-running
    resumes from the last completed field (not from record 0).
  - tqdm progress bars per field.
  - Configurable batch size, beam width, and max output length.
"""

import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from .model import DEVICE, load_translation_model

log = logging.getLogger(__name__)

SRC_LANG = "eng_Latn"

# Supported target languages — extend here to add more
LANG_CODES: dict[str, str] = {
    "hindi":   "hin_Deva",
    "telugu":  "tel_Telu",
    "tamil":   "tam_Taml",
    "kannada": "kan_Knda",
    "bengali": "ben_Beng",
    "marathi": "mar_Deva",
}


# ── Core batch function ────────────────────────────────────────────────────────

def _translate_batch(
    sentences: list[str],
    tgt_lang: str,
    tokenizer,
    model,
    ip,
    max_length: int = 512,
    num_beams: int = 4,
) -> list[str]:
    """Translate one batch. Returns translated strings in the same order."""
    # Guard against empty strings so the model never sees a blank input
    safe = [s if s and s.strip() else "." for s in sentences]

    batch = ip.preprocess_batch(safe, src_lang=SRC_LANG, tgt_lang=tgt_lang)
    inputs = tokenizer(
        batch, truncation=True, padding="longest", return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
        )

    raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    results = ip.postprocess_batch(raw, lang=tgt_lang)

    # Restore empty strings where the original was empty
    return [
        "" if not sentences[i] or not sentences[i].strip() else results[i]
        for i in range(len(sentences))
    ]


# ── Field-level translation with resume ───────────────────────────────────────

def translate_fields(
    texts: dict[str, list[str]],
    lang: str,
    batch_size: int = 8,
    max_length: int = 512,
    num_beams: int = 4,
    checkpoint_path: Path | None = None,
    model_name: str = "ai4bharat/indictrans2-en-indic-1B",
) -> dict[str, list[str]]:
    """
    Translate every field in `texts` to `lang`.

    Args:
        texts           — {field_name: [str, ...]} from data.get_texts()
        lang            — target language label (e.g. "hindi")
        batch_size      — sentences per GPU/CPU batch
        max_length      — max tokens in generated output
        num_beams       — beam search width
        checkpoint_path — if set, saves progress after each field so the run
                          can be resumed after a crash
        model_name      — HuggingFace model ID

    Returns:
        {field_name: [translated_str, ...]}
    """
    lang_code = LANG_CODES.get(lang)
    if not lang_code:
        raise ValueError(
            f"Unsupported language '{lang}'. Supported: {sorted(LANG_CODES)}"
        )

    tokenizer, model, ip = load_translation_model(model_name)

    # Load checkpoint (previously completed fields)
    done: dict[str, list[str]] = {}
    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            done = json.load(f)
        log.info(
            "Resuming translation for '%s': %d/%d fields already done",
            lang, len(done), len(texts),
        )

    results: dict[str, list[str]] = {}

    for field_name, sentences in texts.items():
        if field_name in done:
            log.info("  [skip] '%s' (checkpoint)", field_name)
            results[field_name] = done[field_name]
            continue

        log.info(
            "  Translating '%s' → %s  (%d sentences, batch=%d)",
            field_name, lang, len(sentences), batch_size,
        )

        translated: list[str] = []
        batches = range(0, len(sentences), batch_size)

        for start in tqdm(batches, desc=f"{field_name}→{lang}", unit="batch", leave=False):
            batch = sentences[start : start + batch_size]
            translated.extend(
                _translate_batch(batch, lang_code, tokenizer, model, ip, max_length, num_beams)
            )

        results[field_name] = translated

        # Persist checkpoint after every field
        if checkpoint_path:
            done[field_name] = translated
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=2)

    return results
