# sarvam1-edge-benchmark

Benchmarking [Sarvam-1](https://huggingface.co/sarvamai/sarvam-1) — India's first open-source Indic LLM — across quantization levels for on-device / edge deployment.

**Goal:** Measure how much quality you lose (and how much speed/memory you gain) when quantizing Sarvam-1 from fp16 down to Q2_K, specifically across Indic languages.

> This is an ongoing project. Results and findings will be updated as experiments run.

---

## Why This Matters

Most quantization benchmarks are English-only. Indic languages have:
- Non-latin scripts (Devanagari, Telugu, Tamil...)
- Custom tokenizers with different fertility rates
- Uneven representation in training data

Quality drop after quantization may be **language-dependent** — and nobody has measured this for Sarvam-1 yet.

---

## Experiments

| Variant | Size (disk) | RAM (peak) | Latency (tok/s) | Avg BLEU (Hindi) | Avg BLEU (Telugu) | Avg BLEU (Tamil) |
|---|---|---|---|---|---|---|
| fp16 (baseline) | ~4.0 GB | ~5.0 GB | TBD | TBD | TBD | TBD |
| Q8_0 | ~2.1 GB | ~2.5 GB | TBD | TBD | TBD | TBD |
| Q4_K_M | ~1.2 GB | ~1.5 GB | TBD | TBD | TBD | TBD |
| Q2_K | ~0.8 GB | ~1.0 GB | TBD | TBD | TBD | TBD |

*Table will be updated as experiments complete.*

---

## Key Findings

> To be updated after benchmarks run.

---

## Repo Structure

```
sarvam1-edge-benchmark/
├── scripts/
│   ├── convert_to_gguf.sh       # Convert HF model → GGUF format
│   ├── quantize.sh              # Run llama.cpp quantization
│   ├── benchmark.py             # Main benchmark script
│   ├── eval_quality.py          # BLEU / exact-match scoring
│   └── run_inference.py         # Simple inference runner
├── eval/
│   ├── build_eval_set.py        # Pull & prep IndicQA samples
│   └── eval_set.jsonl           # 200-sample eval set (3 languages)
├── results/
│   └── benchmark_results.csv   # Raw benchmark numbers
├── notebooks/
│   └── analysis.ipynb           # Results analysis & plots
├── docs/
│   └── findings.md              # Detailed write-up of findings
└── models/                      # (gitignored) local model files
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt

# Build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc)
```

### 2. Download Sarvam-1

```bash
huggingface-cli download sarvamai/sarvam-1 --local-dir models/sarvam-1-hf
```

### 3. Convert & quantize

```bash
bash scripts/convert_to_gguf.sh
bash scripts/quantize.sh
```

### 4. Build eval set

```bash
python eval/build_eval_set.py --langs hi te ta --n_samples 200
```

### 5. Run benchmarks

```bash
python scripts/benchmark.py --model_path models/ --eval_set eval/eval_set.jsonl
```

---

## Hardware Tested

| Device | CPU | RAM | GPU |
|---|---|---|---|
| TBD | TBD | TBD | TBD |

---

## Requirements

- Python 3.10+
- llama.cpp (built from source)
- HuggingFace CLI
- ~10 GB free disk space for all model variants

---

## Related Work

- [Sarvam-1 HuggingFace](https://huggingface.co/sarvamai/sarvam-1)
- [Indic-Gemma-2b (Navarasa)](https://huggingface.co/Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa-2.0)
- [IndicQA dataset](https://huggingface.co/datasets/ai4bharat/IndicQA)
- [llama.cpp quantization guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)

---

## Author

Built as a side project to explore edge AI for Indic languages.
Contributions and feedback welcome — open an issue!
