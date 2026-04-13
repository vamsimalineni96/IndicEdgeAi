"""
src/transliteration.py
-----------------------
Transliterate English text into Indic scripts.

What transliteration means here:
    English word "hypertension"  →  "हाइपरटेंशन"  (Devanagari, same sound)
    English word "diabetes"      →  "డయాబెటిస్"    (Telugu, same sound)

This is useful for building datasets where medical English terminology is
written in Indic script (for script-reading evaluation / mixed-language NLP).

Backend preference (tries in order):
  1. ai4bharat-transliteration  — neural word-level Roman→Indic  (best quality)
  2. indic_transliteration       — rule-based ITRANS scheme        (fallback)

Install the preferred backend with:
    pip install ai4bharat-transliteration
"""

import logging
import re

log = logging.getLogger(__name__)

# Supported languages and their codes for each backend
SCRIPT_MAP = {
    "hindi":   "hi",
    "telugu":  "te",
    "tamil":   "ta",
    "kannada": "kn",
    "bengali": "bn",
    "marathi": "mr",
}

# Cache: lang_code → XlitEngine instance (or None if unavailable)
_xlit_engines: dict[str, object] = {}


# ── Backend loading ───────────────────────────────────────────────────────────

def _get_xlit_engine(lang_code: str):
    """Return a cached AI4Bharat XlitEngine, or None if unavailable."""
    if lang_code in _xlit_engines:
        return _xlit_engines[lang_code]

    try:
        from ai4bharat.transliteration import XlitEngine
        engine = XlitEngine(lang_code, beam_width=4, rescore=True)
        _xlit_engines[lang_code] = engine
        log.info("Loaded AI4Bharat XlitEngine for '%s'", lang_code)
        return engine
    except ImportError:
        log.warning(
            "ai4bharat-transliteration not installed. "
            "Run `pip install ai4bharat-transliteration` for best results. "
            "Falling back to rule-based ITRANS transliteration."
        )
        _xlit_engines[lang_code] = None
    except Exception as exc:
        log.warning("Failed to load XlitEngine for '%s': %s. Using fallback.", lang_code, exc)
        _xlit_engines[lang_code] = None

    return None


def _fallback_transliterate(text: str, lang: str) -> str:
    """
    Rule-based transliteration via indic_transliteration (ITRANS scheme).
    Works well for Romanized Indic text; approximate for raw English.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        script_map = {
            "hindi":   sanscript.DEVANAGARI,
            "telugu":  sanscript.TELUGU,
            "tamil":   sanscript.TAMIL,
            "kannada": sanscript.KANNADA,
            "bengali": sanscript.BENGALI,
            "marathi": sanscript.DEVANAGARI,
        }
        target = script_map.get(lang, sanscript.DEVANAGARI)
        return transliterate(text.lower(), sanscript.ITRANS, target)
    except Exception as exc:
        log.debug("Fallback transliteration failed: %s", exc)
        return text  # return original if all else fails


# ── Single-string transliteration ─────────────────────────────────────────────

# Regex: a "word token" is a sequence of letters; everything else is punctuation
_WORD_RE = re.compile(r"([A-Za-z]+)|([^A-Za-z]+)")


def transliterate_text(text: str, lang: str) -> str:
    """
    Transliterate a single English string into the target Indic script.

    Punctuation, numbers, and whitespace are preserved in place.
    Each alphabetic word token is transliterated independently.
    """
    if not text or not text.strip():
        return text

    lang_code = SCRIPT_MAP.get(lang)
    if not lang_code:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {sorted(SCRIPT_MAP)}")

    engine = _get_xlit_engine(lang_code)

    if engine is not None:
        # AI4Bharat path: tokenize, transliterate each word, reassemble
        parts = []
        for match in _WORD_RE.finditer(text):
            word, punct = match.group(1), match.group(2)
            if word:
                try:
                    result = engine.translit_word(word, topk=1)
                    # result is {word: [candidate1, ...]}
                    parts.append(result.get(word, [word])[0] if result else word)
                except Exception:
                    parts.append(word)
            else:
                parts.append(punct)
        return "".join(parts)

    # Fallback path
    return _fallback_transliterate(text, lang)


# ── Batch field transliteration ───────────────────────────────────────────────

def transliterate_fields(
    texts: dict[str, list[str]],
    lang: str,
) -> dict[str, list[str]]:
    """
    Transliterate all fields in `texts` to `lang`.

    Args:
        texts — {field_name: [str, ...]} from data.get_texts()
        lang  — target language label (e.g. "hindi")

    Returns:
        {field_name: [transliterated_str, ...]}
    """
    if lang not in SCRIPT_MAP:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {sorted(SCRIPT_MAP)}")

    log.info("Transliterating to %s...", lang)
    results: dict[str, list[str]] = {}

    for field_name, sentences in texts.items():
        log.info("  Field '%s' (%d sentences)", field_name, len(sentences))
        results[field_name] = [transliterate_text(s, lang) for s in sentences]

    return results
