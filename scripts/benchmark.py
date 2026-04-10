"""
benchmark.py

Main benchmarking script for Sarvam-1 quantization experiments.
Measures latency, memory usage, and quality (BLEU) across model variants.

Usage:
    # Benchmark a HuggingFace (fp16) model
    python scripts/benchmark.py \
        --model_type hf \
        --model_path models/sarvam-1-hf \
        --eval_set eval/eval_set.jsonl \
        --variant fp16

    # Benchmark a GGUF quantized model
    python scripts/benchmark.py \
        --model_type gguf \
        --model_path models/sarvam-1-Q4_K_M.gguf \
        --eval_set eval/eval_set.jsonl \
        --variant Q4_K_M \
        --llama_cpp_path llama.cpp/build/bin/llama-cli
"""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil
import sacrebleu
import torch
from tqdm import tqdm


# ─────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────

def get_ram_usage_mb() -> float:
    """Returns current process RAM usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def get_gpu_vram_mb() -> float:
    """Returns current GPU VRAM usage in MB (0 if no GPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


# ─────────────────────────────────────────────
# HuggingFace fp16 inference
# ─────────────────────────────────────────────

def load_hf_model(model_path: str):
    """Load Sarvam-1 from HuggingFace format."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading HF model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def run_hf_inference(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> tuple[str, float, int]:
    """
    Run inference with HF model.
    Returns: (generated_text, tokens_per_second, n_tokens_generated)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_input_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,        # greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - start

    n_generated = outputs.shape[1] - n_input_tokens
    tokens_per_sec = n_generated / elapsed if elapsed > 0 else 0.0

    generated_ids = outputs[0][n_input_tokens:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip(), tokens_per_sec, n_generated


# ─────────────────────────────────────────────
# GGUF (llama.cpp) inference
# ─────────────────────────────────────────────

def run_gguf_inference(
    model_path: str,
    llama_cpp_path: str,
    prompt: str,
    max_new_tokens: int = 64,
    n_threads: int = 4,
) -> tuple[str, float, int]:
    """
    Run inference using llama.cpp CLI.
    Returns: (generated_text, tokens_per_second, n_tokens)
    """
    cmd = [
        llama_cpp_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_new_tokens),
        "-t", str(n_threads),
        "--no-display-prompt",
        "-e",           # escape special chars
    ]

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.perf_counter() - start
        output = result.stdout.strip()

        # Parse token speed from llama.cpp stderr output
        # llama.cpp logs: "llama_perf_sampler_print: ... eval time = X ms / Y tokens"
        tokens_per_sec = 0.0
        for line in result.stderr.split("\n"):
            if "eval time" in line and "tokens" in line:
                try:
                    parts = line.split("/")
                    tokens_str = parts[-1].strip().split()[0]
                    ms_str = parts[-2].strip().split()[-2]
                    n_tok = float(tokens_str)
                    ms = float(ms_str)
                    tokens_per_sec = (n_tok / ms) * 1000
                except Exception:
                    pass

        if tokens_per_sec == 0.0 and elapsed > 0:
            # Fallback: estimate from wall time
            tokens_per_sec = max_new_tokens / elapsed

        return output, tokens_per_sec, max_new_tokens

    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", 0.0, 0
    except Exception as e:
        return f"[ERROR: {e}]", 0.0, 0


# ─────────────────────────────────────────────
# Quality scoring
# ─────────────────────────────────────────────

def compute_bleu(hypothesis: str, reference: str) -> float:
    """Compute sentence-level BLEU score (0-100)."""
    if not hypothesis or not reference:
        return 0.0
    try:
        bleu = sacrebleu.corpus_bleu([hypothesis], [[reference]])
        return bleu.score
    except Exception:
        return 0.0


def exact_match(hypothesis: str, reference: str) -> bool:
    """Check if hypothesis exactly matches reference (lowercased, stripped)."""
    return hypothesis.strip().lower() == reference.strip().lower()


# ─────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────

def run_benchmark(args):
    print(f"\n{'='*60}")
    print(f"  Sarvam-1 Edge Benchmark")
    print(f"  Variant  : {args.variant}")
    print(f"  Model    : {args.model_path}")
    print(f"  Eval set : {args.eval_set}")
    print(f"{'='*60}\n")

    # Load eval set
    samples = []
    with open(args.eval_set, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} eval samples\n")

    # Load HF model if needed
    hf_model, hf_tokenizer = None, None
    if args.model_type == "hf":
        hf_model, hf_tokenizer = load_hf_model(args.model_path)

    # Track results
    results = []
    ram_before = get_ram_usage_mb()

    for sample in tqdm(samples, desc=f"Benchmarking [{args.variant}]"):
        prompt = sample["prompt"]
        reference = sample["reference_answer"]
        lang = sample["lang"]

        # Run inference
        ram_start = get_ram_usage_mb()

        if args.model_type == "hf":
            generated, tok_per_sec, n_tokens = run_hf_inference(
                hf_model, hf_tokenizer, prompt, args.max_new_tokens
            )
        else:
            generated, tok_per_sec, n_tokens = run_gguf_inference(
                args.model_path,
                args.llama_cpp_path,
                prompt,
                args.max_new_tokens,
                args.n_threads,
            )

        ram_end = get_ram_usage_mb()
        peak_ram_mb = max(ram_end, ram_start)

        # Score
        bleu = compute_bleu(generated, reference)
        em = exact_match(generated, reference)

        results.append({
            "variant": args.variant,
            "sample_id": sample["id"],
            "lang": lang,
            "lang_name": sample["lang_name"],
            "bleu": round(bleu, 2),
            "exact_match": int(em),
            "tokens_per_sec": round(tok_per_sec, 2),
            "n_tokens_generated": n_tokens,
            "peak_ram_mb": round(peak_ram_mb, 1),
            "generated": generated[:200],       # truncate for CSV
            "reference": reference[:200],
        })

    # ── Summary ──────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Results Summary — {args.variant}")
    print(f"{'─'*60}")

    langs = sorted(set(r["lang"] for r in results))
    for lang in langs:
        lang_results = [r for r in results if r["lang"] == lang]
        avg_bleu = sum(r["bleu"] for r in lang_results) / len(lang_results)
        avg_tps = sum(r["tokens_per_sec"] for r in lang_results) / len(lang_results)
        avg_ram = sum(r["peak_ram_mb"] for r in lang_results) / len(lang_results)
        em_rate = sum(r["exact_match"] for r in lang_results) / len(lang_results) * 100

        lang_name = lang_results[0]["lang_name"]
        print(f"  {lang_name:<10} BLEU={avg_bleu:.1f}  EM={em_rate:.1f}%  "
              f"Speed={avg_tps:.1f} tok/s  RAM={avg_ram:.0f} MB")

    overall_bleu = sum(r["bleu"] for r in results) / len(results)
    overall_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    print(f"\n  Overall   BLEU={overall_bleu:.1f}  Speed={overall_tps:.1f} tok/s")
    print(f"{'─'*60}\n")

    # ── Save results ─────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        fieldnames = list(results[0].keys()) + ["timestamp"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        ts = datetime.now().isoformat()
        for row in results:
            row["timestamp"] = ts
            writer.writerow(row)

    print(f"✅ Results appended to: {out_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Sarvam-1 quantization variants"
    )
    parser.add_argument(
        "--model_type",
        choices=["hf", "gguf"],
        required=True,
        help="'hf' for HuggingFace fp16, 'gguf' for llama.cpp quantized",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to HF model dir or .gguf file",
    )
    parser.add_argument(
        "--eval_set",
        default="eval/eval_set.jsonl",
        help="Path to eval JSONL file",
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="Label for this run e.g. fp16, Q8_0, Q4_K_M, Q2_K",
    )
    parser.add_argument(
        "--output",
        default="results/benchmark_results.csv",
        help="Output CSV file (results are appended)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Max tokens to generate per sample",
    )
    parser.add_argument(
        "--llama_cpp_path",
        default="llama.cpp/build/bin/llama-cli",
        help="Path to llama.cpp CLI binary (for gguf mode)",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=4,
        help="CPU threads for llama.cpp inference",
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
