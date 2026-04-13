"""
src/data.py
-----------
Dataset I/O and field extraction for the MedQA pipeline.

MedQA record format expected:
    {
        "question": "...",
        "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
        "answer": "A"
    }
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_records(path: str | Path) -> list[dict]:
    """Load JSONL or JSON-list file. Returns list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    content = path.read_text(encoding="utf-8").strip()

    # Try JSON list first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            log.info("Loaded %d records (JSON list) from %s", len(data), path.name)
            return data
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    records = [json.loads(line) for line in content.splitlines() if line.strip()]
    log.info("Loaded %d records (JSONL) from %s", len(records), path.name)
    return records


def save_records(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Saved %d records → %s", len(records), path)


# ── Field extraction ──────────────────────────────────────────────────────────

def get_texts(records: list[dict]) -> tuple[dict[str, list[str]], list[str]]:
    """
    Extract all translatable text fields from records.

    Returns:
        texts      — {field_name: [str, ...]}  one entry per record
        option_keys — sorted list of option keys found (e.g. ["A","B","C","D"])
    """
    texts: dict[str, list[str]] = {
        "question": [r.get("question", "") for r in records]
    }

    # Discover option keys across the whole dataset
    option_keys: set[str] = set()
    for r in records:
        opts = r.get("options", {})
        if isinstance(opts, dict):
            option_keys.update(opts.keys())

    option_keys_sorted = sorted(option_keys)

    for key in option_keys_sorted:
        texts[f"option_{key}"] = [
            r.get("options", {}).get(key, "") if isinstance(r.get("options"), dict) else ""
            for r in records
        ]

    return texts, option_keys_sorted


# ── Merge outputs ─────────────────────────────────────────────────────────────

def merge_outputs(
    records: list[dict],
    option_keys: list[str],
    lang_outputs: dict[str, dict[str, dict[str, list[str]]]],
) -> list[dict]:
    """
    Merge source records with translation/transliteration outputs.

    Args:
        records      — original English records
        option_keys  — option keys discovered by get_texts()
        lang_outputs — {lang: {mode: {field: [str]}}}
                       mode is "trans" (translation) or "translit" (transliteration)

    Returns list of merged dicts with schema:
        id, answer,
        question_en, options_en,
        question_{lang}_trans, options_{lang}_trans,      (if translation ran)
        question_{lang}_translit, options_{lang}_translit  (if transliteration ran)
    """
    n = len(records)
    merged = []

    for i, rec in enumerate(records):
        row: dict = {
            "id":          rec.get("id", i),
            "answer":      rec.get("answer", ""),
            "question_en": rec.get("question", ""),
            "options_en":  rec.get("options", {}),
        }

        for lang, modes in lang_outputs.items():
            for mode, fields in modes.items():
                # question field
                row[f"question_{lang}_{mode}"] = (
                    fields.get("question", [""] * n)[i]
                )
                # options dict
                row[f"options_{lang}_{mode}"] = {
                    key: fields.get(f"option_{key}", [""] * n)[i]
                    for key in option_keys
                }

        merged.append(row)

    return merged


def select_lang_fields(merged: list[dict], lang: str) -> list[dict]:
    """Return records with only the source fields + fields for a single language."""
    base_keys = {"id", "answer", "question_en", "options_en"}
    lang_keys = {k for k in merged[0] if f"_{lang}_" in k}
    keep = base_keys | lang_keys
    return [{k: r[k] for k in keep if k in r} for r in merged]
