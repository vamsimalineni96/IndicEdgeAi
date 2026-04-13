import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(HF_TOKEN)
        print("logged in to hf hub")
    except Exception:
        pass

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading model...")
model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model      = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float32,   # float32 for CPU stability
).to(DEVICE)
model.eval()
ip = IndicProcessor(inference=True)
print("Model ready.\n")

# ── Sample sentence ────────────────────────────────────────────────────────────
sentence = "The patient has a history of hypertension and diabetes."


# ── 1. TRANSLATION (meaning converted to target language) ──────────────────────
def translate(text: str, tgt_lang: str) -> str:
    batch  = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang=tgt_lang)
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        tokens = model.generate(**inputs, max_length=256, num_beams=4)
    raw = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    return ip.postprocess_batch(raw, lang=tgt_lang)[0]

hindi_translation  = translate(sentence, "hin_Deva")
telugu_translation = translate(sentence, "tel_Telu")

print("=" * 60)
print("TRANSLATION")
print("=" * 60)
print(f"English : {sentence}")
print(f"Hindi   : {hindi_translation}")
print(f"Telugu  : {telugu_translation}")


# ── 2. TRANSLITERATION (English phonetics written in target script) ─────────────
# Uses ITRANS scheme: write English phonetically then convert to script
# e.g.  "doctor ne tablet diya" → "डॉक्टर ने टैबलेट दिया"
itrans_text = "patient ko hypertension aur diabetes ki history hai"

hindi_translit  = transliterate(itrans_text, sanscript.ITRANS, sanscript.DEVANAGARI)
telugu_translit = transliterate(itrans_text, sanscript.ITRANS, sanscript.TELUGU)

print()
print("=" * 60)
print("TRANSLITERATION  (ITRANS phonetic → script)")
print("=" * 60)
print(f"ITRANS  : {itrans_text}")
print(f"Hindi   : {hindi_translit}")
print(f"Telugu  : {telugu_translit}")
