"""
Phase 2: Data Pipeline Setup
==============================
Run this script first to install dependencies and create directory structure.

Usage:
    python setup_phase2.py
"""

import subprocess
import os
import sys


def run_cmd(cmd, desc=""):
    """Run a shell command and print output."""
    if desc:
        print(f"\n  {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ Warning: {result.stderr[:200]}")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  Phase 2: Data Pipeline Setup")
    print("=" * 60)

    # ── Create directory structure ──────────────────────────────────
    print("\n[1/3] Creating directory structure...")
    dirs = [
        "data/curated",
        "data/translated",
        "data/final",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"   ✓ {d}/")

    # ── Install Python dependencies ─────────────────────────────────
    print("\n[2/3] Installing Python dependencies...")

    packages = [
        "datasets",          # HuggingFace datasets
        "transformers",      # Model loading
        "sentencepiece",     # IndicTrans2 tokenizer
        "protobuf",          # IndicTrans2 dependency
        "torch",             # PyTorch (should already be installed)
        "tqdm",              # Progress bars
        "accelerate",        # For model loading with device_map
    ]

    for pkg in packages:
        run_cmd(f"pip install -q {pkg}", f"Installing {pkg}")

    # ── Verify GPU access ───────────────────────────────────────────
    print("\n[3/3] Verifying GPU access...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"   ✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("   ⚠ No GPU detected! Translation will be very slow.")
    except ImportError:
        print("   ⚠ PyTorch not installed yet. Install it first.")

    # ── Print run order ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SETUP COMPLETE — Run scripts in this order:")
    print("=" * 60)
    print("""
  1. python step1_download_and_curate.py
     → Downloads MedQA+MedMCQA, curates 15K examples
     → Output: data/curated/medqa_curated_15k.json
     → Time: ~2 minutes

  2. python step2_generate_healthsearchqa.py --backend transformers
     → Generates answers for 3.1K health questions using Qwen2.5-7B
     → Output: data/curated/healthsearchqa_with_answers.json
     → Time: ~30-60 minutes (transformers) or ~5 min (vllm)

  3. python step3_translate_indictrans2.py
     → Translates English data to Hindi & Telugu
     → Output: data/translated/*.json
     → Time: ~1-2 hours (depends on dataset size)

  4. python step4_merge_and_split.py
     → Merges all languages, generates code-mixed examples
     → Splits into train/val/test ChatML JSONL
     → Output: data/final/train.jsonl, val.jsonl, test.jsonl
     → Time: ~1 minute

  Total estimated time: 2-4 hours
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
