"""
src/model.py
------------
Lazy-load and cache the IndicTrans2 seq2seq model.
Call login_hf() once at startup to authenticate with HuggingFace Hub.
"""

import logging
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

log = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level cache: model_name → (tokenizer, model, processor)
_cache: dict = {}


def login_hf() -> None:
    """Authenticate with HuggingFace Hub using HF_TOKEN env var (optional)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return
    try:
        from huggingface_hub import login
        login(token)
        log.info("Authenticated with HuggingFace Hub")
    except Exception as exc:
        log.warning("HF login failed (continuing without auth): %s", exc)


def load_translation_model(
    model_name: str = "ai4bharat/indictrans2-en-indic-1B",
) -> tuple:
    """
    Load IndicTrans2 model, tokenizer, and IndicProcessor.
    Returns cached instance on subsequent calls.

    Returns:
        (tokenizer, model, IndicProcessor)
    """
    if model_name in _cache:
        log.debug("Using cached translation model: %s", model_name)
        return _cache[model_name]

    log.info("Loading translation model: %s  (device=%s)", model_name, DEVICE)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()

    ip = IndicProcessor(inference=True)

    _cache[model_name] = (tokenizer, model, ip)
    log.info("Translation model ready")
    return tokenizer, model, ip
