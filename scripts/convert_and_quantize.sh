#!/usr/bin/env bash
# convert_and_quantize.sh
#
# Converts Sarvam-1 from HuggingFace format to GGUF,
# then produces Q8_0, Q4_K_M, and Q2_K quantized variants.
#
# Prerequisites:
#   - llama.cpp cloned and built at ./llama.cpp
#   - Sarvam-1 HF weights downloaded to ./models/sarvam-1-hf
#
# Usage:
#   bash scripts/convert_and_quantize.sh

set -e

# ── Config ───────────────────────────────────────────
HF_MODEL_DIR="models/sarvam-1-hf"
GGUF_DIR="models"
LLAMA_CPP_DIR="llama.cpp"
PYTHON="python3"

# ── Checks ───────────────────────────────────────────
echo ""
echo "=== Sarvam-1 → GGUF Conversion & Quantization ==="
echo ""

if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "❌ HF model not found at $HF_MODEL_DIR"
    echo "   Run: huggingface-cli download sarvamai/sarvam-1 --local-dir $HF_MODEL_DIR"
    exit 1
fi

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "❌ llama.cpp not found at $LLAMA_CPP_DIR"
    echo "   Run: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make -j\$(nproc)"
    exit 1
fi

QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    # Fallback to older build path
    QUANTIZE_BIN="$LLAMA_CPP_DIR/quantize"
fi

if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "❌ llama-quantize binary not found. Did you build llama.cpp?"
    echo "   Run: cd llama.cpp && cmake -B build && cmake --build build --config Release"
    exit 1
fi

# ── Step 1: Convert to fp16 GGUF ─────────────────────
GGUF_FP16="$GGUF_DIR/sarvam-1-fp16.gguf"

if [ -f "$GGUF_FP16" ]; then
    echo "✓ fp16 GGUF already exists, skipping conversion"
else
    echo "Step 1: Converting HF → GGUF (fp16)..."
    $PYTHON $LLAMA_CPP_DIR/convert_hf_to_gguf.py \
        "$HF_MODEL_DIR" \
        --outtype f16 \
        --outfile "$GGUF_FP16"
    echo "✓ Saved: $GGUF_FP16"
fi

# ── Step 2: Quantize to Q8_0 ─────────────────────────
GGUF_Q8="$GGUF_DIR/sarvam-1-Q8_0.gguf"

if [ -f "$GGUF_Q8" ]; then
    echo "✓ Q8_0 already exists, skipping"
else
    echo ""
    echo "Step 2: Quantizing → Q8_0 (high quality, ~2x smaller)..."
    $QUANTIZE_BIN "$GGUF_FP16" "$GGUF_Q8" Q8_0
    echo "✓ Saved: $GGUF_Q8"
fi

# ── Step 3: Quantize to Q4_K_M ───────────────────────
GGUF_Q4="$GGUF_DIR/sarvam-1-Q4_K_M.gguf"

if [ -f "$GGUF_Q4" ]; then
    echo "✓ Q4_K_M already exists, skipping"
else
    echo ""
    echo "Step 3: Quantizing → Q4_K_M (balanced, ~3.5x smaller)..."
    $QUANTIZE_BIN "$GGUF_FP16" "$GGUF_Q4" Q4_K_M
    echo "✓ Saved: $GGUF_Q4"
fi

# ── Step 4: Quantize to Q2_K ─────────────────────────
GGUF_Q2="$GGUF_DIR/sarvam-1-Q2_K.gguf"

if [ -f "$GGUF_Q2" ]; then
    echo "✓ Q2_K already exists, skipping"
else
    echo ""
    echo "Step 4: Quantizing → Q2_K (smallest, highest loss)..."
    $QUANTIZE_BIN "$GGUF_FP16" "$GGUF_Q2" Q2_K
    echo "✓ Saved: $GGUF_Q2"
fi

# ── Summary ──────────────────────────────────────────
echo ""
echo "=== Done! Model sizes ==="
echo ""
for f in "$GGUF_FP16" "$GGUF_Q8" "$GGUF_Q4" "$GGUF_Q2"; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  $size  $(basename $f)"
    fi
done

echo ""
echo "Next: run benchmarks"
echo "  python scripts/benchmark.py --model_type hf --model_path $HF_MODEL_DIR --variant fp16"
echo "  python scripts/benchmark.py --model_type gguf --model_path $GGUF_Q4 --variant Q4_K_M"
echo ""
