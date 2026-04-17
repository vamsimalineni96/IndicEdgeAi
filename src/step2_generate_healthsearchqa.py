"""
Step 2: Generate Answers for HealthSearchQA
=============================================
Uses Qwen2.5-7B (teacher model) to generate high-quality answers
for the 3.1K consumer health questions from HealthSearchQA.

Usage:
    python step2_generate_healthsearchqa.py [--backend transformers|vllm]

Output:
    data/curated/healthsearchqa_with_answers.json
"""

import json
import os
import argparse
import time
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "data/curated"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "healthsearchqa_with_answers.json")
BATCH_SIZE = 8  # For vLLM batching

SYSTEM_PROMPT = """You are a knowledgeable and empathetic medical assistant. 
Answer the patient's health question clearly and accurately. 
Provide practical advice while noting when professional medical consultation is needed.
Keep your response concise (2-4 paragraphs) and use simple language that a patient can understand.
Always include a brief disclaimer that this is general health information, not a substitute for professional medical advice."""

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_prompt(question):
    """Build the chat prompt for answer generation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def generate_with_transformers(questions):
    """Generate answers using HuggingFace transformers pipeline."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("   Loading model with transformers backend...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    results = []
    total = len(questions)

    for i, question in enumerate(questions):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"   Generating answer {i+1}/{total}...")

        messages = build_prompt(question)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        results.append(answer)

    del model
    torch.cuda.empty_cache()
    return results


def generate_with_vllm(questions):
    """Generate answers using vLLM for faster batched inference."""
    from vllm import LLM, SamplingParams

    print("   Loading model with vLLM backend...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )

    # Build all prompts
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    prompts = []
    for q in questions:
        messages = build_prompt(q)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    print(f"   Generating answers for {len(prompts)} questions (batched)...")
    outputs = llm.generate(prompts, sampling_params)

    results = [output.outputs[0].text.strip() for output in outputs]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    args = parser.parse_args()

    print("=" * 60)
    print("  STEP 2: Generate HealthSearchQA Answers")
    print("=" * 60)

    # ── Download HealthSearchQA ─────────────────────────────────────
    print("\n[1/3] Downloading HealthSearchQA...")
    ds = load_dataset("katielink/healthsearchqa")
    
    # The dataset might have different structures, let's handle both
    if "train" in ds:
        data = ds["train"]
    else:
        # Try to get the first available split
        split_name = list(ds.keys())[0]
        data = ds[split_name]

    # Extract questions
    questions = []
    for example in data:
        q = example.get("question", "") or example.get("text", "")
        if q and q.strip():
            questions.append(q.strip())

    print(f"   Found {len(questions)} health questions")
    print(f"   Sample: '{questions[0]}'")

    # ── Generate answers ────────────────────────────────────────────
    print(f"\n[2/3] Generating answers using {args.backend} backend...")
    start_time = time.time()

    if args.backend == "vllm":
        answers = generate_with_vllm(questions)
    else:
        answers = generate_with_transformers(questions)

    elapsed = time.time() - start_time
    print(f"   Generated {len(answers)} answers in {elapsed:.1f}s")
    print(f"   ({elapsed/len(answers):.2f}s per question)")

    # ── Format and save ─────────────────────────────────────────────
    print(f"\n[3/3] Formatting and saving...")

    formatted = []
    skipped = 0
    for q, a in zip(questions, answers):
        # Skip if answer is too short or empty
        if not a or len(a.strip()) < 50:
            skipped += 1
            continue

        formatted.append({
            "instruction": q,
            "response": a,
            "subject": "Consumer Health",
            "topic": "",
            "source": "healthsearchqa",
            "original_question": q,
            "has_explanation": True,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    avg_a_len = sum(len(ex["response"]) for ex in formatted) / len(formatted) if formatted else 0

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Questions processed:  {len(questions)}")
    print(f"  Answers generated:    {len(formatted)}")
    print(f"  Skipped (too short):  {skipped}")
    print(f"  Avg answer length:    {avg_a_len:.0f} chars")
    print(f"  Time taken:           {elapsed:.1f}s")
    print(f"  Output:               {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
