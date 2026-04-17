"""
Step 2: Generate Answers for HealthSearchQA (via NIM Endpoint)
===============================================================
Uses Qwen2.5-7B deployed as a NIM service to generate high-quality
answers for the 3.1K consumer health questions from HealthSearchQA.

Usage:
    python step2_generate_healthsearchqa.py --endpoint <YOUR_NIM_URL>

    Example:
    python step2_generate_healthsearchqa.py \
        --endpoint http://0.0.0.21:8000/v1/chat/completions \
        --api_key <KEY>

    Optional:
    --api_key <KEY>          API key if your NIM requires auth
    --concurrency 10         Number of parallel requests (default: 10)
    --batch_size 50          Save checkpoint every N responses (default: 50)
    --model_name <NAME>      Model name to send in request (default: Qwen/Qwen2.5-7B-Instruct)

Output:
    data/curated/healthsearchqa_with_answers.json
"""

import json
import os
import argparse
import time
import asyncio
import aiohttp
from datasets import load_dataset

# ── Config ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/curated"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "healthsearchqa_with_answers.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "healthsearchqa_checkpoint.json")

SYSTEM_PROMPT = """You are a knowledgeable and empathetic medical assistant. 
Answer the patient's health question clearly and accurately. 
Provide practical advice while noting when professional medical consultation is needed.
Keep your response concise (2-4 paragraphs) and use simple language that a patient can understand.
Always include a brief disclaimer that this is general health information, not a substitute for professional medical advice."""

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_request_body(question, model_name):
    """Build the OpenAI-compatible chat completion request."""
    return {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }


async def call_nim(session, endpoint, question, model_name, api_key=None, retries=3):
    """Call the NIM endpoint for a single question with retries."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = build_request_body(question, model_name)

    for attempt in range(retries):
        try:
            async with session.post(
                endpoint, json=body, headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
                ssl=False,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    answer = data["choices"][0]["message"]["content"].strip()
                    return answer
                elif resp.status == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"   ⚠ Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                else:
                    error_text = await resp.text()
                    print(f"   ⚠ HTTP {resp.status}: {error_text[:150]}")
                    await asyncio.sleep(1)
        except asyncio.TimeoutError:
            print(f"   ⚠ Timeout on attempt {attempt + 1}")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"   ⚠ Error: {type(e).__name__}: {e}")
            await asyncio.sleep(1)

    return None  # Failed after all retries


async def generate_all(endpoint, questions, model_name, api_key=None, concurrency=10, batch_size=50):
    """Generate answers for all questions with concurrency control."""
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(questions)
    completed = 0
    failed = 0

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        for i, ans in enumerate(checkpoint):
            if i < len(results) and ans is not None:
                results[i] = ans
        completed = sum(1 for r in results if r is not None)
        print(f"   Resuming from checkpoint: {completed}/{len(questions)} already done")

    async def process_one(idx, question):
        nonlocal completed, failed
        if results[idx] is not None:
            return

        async with semaphore:
            answer = await call_nim(session, endpoint, question, model_name, api_key)
            if answer:
                results[idx] = answer
            else:
                results[idx] = ""
                failed += 1
            completed += 1

            if completed % 100 == 0:
                pct = completed / len(questions) * 100
                print(f"   Progress: {completed}/{len(questions)} ({pct:.1f}%) | Failed: {failed}")

            if completed % batch_size == 0:
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(results, f)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_one(i, q) for i, q in enumerate(questions)]
        await asyncio.gather(*tasks)

    # Final checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f)

    return results, failed


def test_endpoint(endpoint, model_name, api_key=None):
    """Quick connectivity test before starting bulk generation."""
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = build_request_body("What causes headaches?", model_name)

    try:
        resp = requests.post(endpoint, json=body, headers=headers, timeout=30, verify=False)
        if resp.status_code == 200:
            test_answer = resp.json()["choices"][0]["message"]["content"]
            print(f"   ✓ Endpoint working!")
            print(f"     Test response: '{test_answer[:100]}...'")
            return True
        else:
            print(f"   ✗ HTTP {resp.status_code}: {resp.text[:300]}")
            return False
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate HealthSearchQA answers via NIM endpoint")
    parser.add_argument("--endpoint", required=True,
                        help="NIM endpoint URL (e.g. https://your-nim/v1/chat/completions)")
    parser.add_argument("--api_key", default=None,
                        help="API key for NIM endpoint (or set NIM_API_KEY env var)")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name to send in the request body")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of parallel requests (default: 10)")
    parser.add_argument("--batch_size", type=int, default=50,
                        help="Checkpoint save frequency (default: 50)")
    args = parser.parse_args()

    # Check env var for API key
    api_key = args.api_key or os.environ.get("NIM_API_KEY")

    print("=" * 60)
    print("  STEP 2: Generate HealthSearchQA Answers (NIM)")
    print("=" * 60)
    print(f"  Endpoint:    {args.endpoint}")
    print(f"  Model:       {args.model_name}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Auth:        {'Yes' if api_key else 'No'}")

    # ── Download HealthSearchQA ─────────────────────────────────────
    print("\n[1/3] Downloading HealthSearchQA...")
    ds = load_dataset("katielink/healthsearchqa", "all_data")

    if "train" in ds:
        data = ds["train"]
    else:
        split_name = list(ds.keys())[0]
        data = ds[split_name]

    questions = []
    for example in data:
        q = example.get("question", "") or example.get("text", "")
        if q and q.strip():
            questions.append(q.strip())

    print(f"   Found {len(questions)} health questions")
    print(f"   Sample: '{questions[0]}'")

    # ── Test endpoint ───────────────────────────────────────────────
    print(f"\n[2/3] Testing NIM endpoint...")
    if not test_endpoint(args.endpoint, args.model_name, api_key):
        print("\n   Endpoint test failed. Please check:")
        print("   - Is the NIM service running?")
        print("   - Is the URL correct? (should end with /v1/chat/completions)")
        print("   - Is the API key correct?")
        resp = input("   Continue anyway? (y/n): ").strip().lower()
        if resp != "y":
            return

    # ── Generate answers ────────────────────────────────────────────
    print(f"\n[3/3] Generating answers for {len(questions)} questions...")
    start_time = time.time()

    answers, failed = asyncio.run(
        generate_all(
            args.endpoint, questions, args.model_name,
            api_key, args.concurrency, args.batch_size
        )
    )

    elapsed = time.time() - start_time

    # ── Format and save ─────────────────────────────────────────────
    formatted = []
    skipped = 0
    for q, a in zip(questions, answers):
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

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    avg_a_len = sum(len(ex["response"]) for ex in formatted) / len(formatted) if formatted else 0

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Questions processed:  {len(questions)}")
    print(f"  Answers generated:    {len(formatted)}")
    print(f"  Failed requests:      {failed}")
    print(f"  Skipped (too short):  {skipped}")
    print(f"  Avg answer length:    {avg_a_len:.0f} chars")
    print(f"  Time taken:           {elapsed:.1f}s ({elapsed/len(questions):.2f}s per question)")
    print(f"  Throughput:           {len(questions)/elapsed:.1f} questions/sec")
    print(f"  Output:               {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()