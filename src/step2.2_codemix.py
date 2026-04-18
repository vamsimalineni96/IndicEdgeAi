import json
import asyncio
import aiohttp
import time

ENDPOINT = "https://your-nim-endpoint.com/v1/chat/completions"  # ← change this
API_KEY = "your-key-here"  # ← change this
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = """You are a helpful multilingual medical assistant. 
Answer health questions accurately in the same language as the question. 
If the question is in Hinglish or Tenglish (mixed language), respond in the same mixed style.
Always advise consulting a doctor for serious symptoms."""

with open("data/curated/codemixed_need_answers.json", "r") as f:
    data = json.load(f)

questions = [ex["instruction"] for ex in data]
print(f"Generating answers for {len(questions)} code-mixed questions...")

async def call_nim(session, question, sem):
    async with sem:
        body = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "max_tokens": 512, "temperature": 0.7, "top_p": 0.9,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
        async with session.post(ENDPOINT, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status == 200:
                r = await resp.json()
                return r["choices"][0]["message"]["content"].strip()
            return ""

async def main():
    sem = asyncio.Semaphore(10)
    async with aiohttp.ClientSession() as session:
        tasks = [call_nim(session, q, sem) for q in questions]
        answers = await asyncio.gather(*tasks)
    return answers

start = time.time()
answers = asyncio.run(main())

# Update the data with answers
for ex, ans in zip(data, answers):
    ex["response"] = ans

# Save back
with open("data/curated/codemixed_with_answers.json", "w") as f:
    json.dump([ex for ex in data if len(ex["response"].strip()) > 50], f, ensure_ascii=False, indent=2)

print(f"Done in {time.time()-start:.1f}s")
print(f"Saved {sum(1 for ex in data if len(ex['response'].strip()) > 50)} code-mixed QA pairs")