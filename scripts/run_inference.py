"""
run_inference.py

Interactive inference runner — test any Sarvam-1 variant manually.
Useful for quick sanity checks before running the full benchmark.

Usage:
    # HuggingFace (fp16)
    python scripts/run_inference.py --model_type hf --model_path models/sarvam-1-hf

    # GGUF quantized
    python scripts/run_inference.py --model_type gguf --model_path models/sarvam-1-Q4_K_M.gguf
"""

import argparse
import subprocess
import time

import psutil
import torch


EXAMPLE_PROMPTS = {
    "hi": "भारत की राजधानी क्या है?",
    "te": "భారతదేశ రాజధాని ఏమిటి?",
    "ta": "இந்தியாவின் தலைநகரம் என்ன?",
    "en": "What is the capital of India?",
}


def run_hf(model_path: str, prompt: str, max_new_tokens: int):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.\n")

    while True:
        if prompt:
            user_input = prompt
            prompt = None  # only use provided prompt once
        else:
            user_input = input("Prompt (or 'q' to quit): ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue

        print("\nGenerating...")
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start

        n_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        tok_per_sec = n_generated / elapsed if elapsed > 0 else 0
        ram_mb = psutil.Process().memory_info().rss / (1024 ** 2)

        print(f"\n{'─'*50}")
        print(f"Response : {response.strip()}")
        print(f"{'─'*50}")
        print(f"Tokens   : {n_generated} generated in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
        print(f"RAM      : {ram_mb:.0f} MB\n")


def run_gguf(model_path: str, llama_cpp_path: str, prompt: str, max_new_tokens: int, n_threads: int):
    print(f"\nUsing GGUF model: {model_path}")
    print(f"llama.cpp path  : {llama_cpp_path}\n")

    while True:
        if prompt:
            user_input = prompt
            prompt = None
        else:
            user_input = input("Prompt (or 'q' to quit): ").strip()
            if user_input.lower() in ("q", "quit", "exit"):
                break
            if not user_input:
                continue

        cmd = [
            llama_cpp_path,
            "-m", model_path,
            "-p", user_input,
            "-n", str(max_new_tokens),
            "-t", str(n_threads),
            "--no-display-prompt",
        ]

        print("\nGenerating...")
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.perf_counter() - start

        output = result.stdout.strip()

        # Parse speed from stderr
        tok_per_sec = 0.0
        for line in result.stderr.split("\n"):
            if "eval time" in line and "tokens" in line:
                try:
                    parts = line.split("/")
                    n_tok = float(parts[-1].strip().split()[0])
                    ms = float(parts[-2].strip().split()[-2])
                    tok_per_sec = (n_tok / ms) * 1000
                except Exception:
                    pass

        print(f"\n{'─'*50}")
        print(f"Response : {output}")
        print(f"{'─'*50}")
        print(f"Time     : {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Interactive inference with Sarvam-1")
    parser.add_argument("--model_type", choices=["hf", "gguf"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default=None, help="Single prompt (optional, runs interactive if not set)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--llama_cpp_path", default="llama.cpp/build/bin/llama-cli")
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo with example prompts in all 4 languages",
    )
    args = parser.parse_args()

    if args.demo:
        print("\n=== Demo: Testing across languages ===\n")
        for lang, prompt in EXAMPLE_PROMPTS.items():
            print(f"[{lang.upper()}] {prompt}")
        print()

    if args.model_type == "hf":
        run_hf(args.model_path, args.prompt, args.max_new_tokens)
    else:
        run_gguf(args.model_path, args.llama_cpp_path, args.prompt, args.max_new_tokens, args.n_threads)


if __name__ == "__main__":
    main()
