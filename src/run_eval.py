"""
IndicHealthAI — Model Evaluation Runner
========================================
Runs the eval prompt set against candidate models and computes:
  - MCQ accuracy (exact match)
  - BLEU and ROUGE-L for conversational / code-switching prompts
  - Aggregated comparison table

Usage:
  python run_eval.py --model_name qwen2.5-7b --model_path /path/to/model
  python run_eval.py --model_name llama3.1-8b --model_path /path/to/model
  python run_eval.py --model_name qwen2.5-3b --model_path /path/to/model
  python run_eval.py --model_name smollm2-1.7b --model_path /path/to/model

After running all models:
  python run_eval.py --compare
"""

import json
import os
import re
import argparse
import time
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# You can swap this backend depending on your H100 setup
# Option 1: vLLM (recommended for H100)
# Option 2: transformers + torch
# Option 3: llama.cpp server
# ---------------------------------------------------------------------------

BACKEND = "vllm"  # change to "transformers" or "llamacpp" as needed

EVAL_SET_PATH = "eval_prompt_set.json"
RESULTS_DIR = "results"
TEMPERATURE = 0.3
MAX_TOKENS = 512


def load_eval_set():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def init_model_vllm(model_path: str):
    """Initialize model with vLLM for fast batched inference on H100."""
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # single H100
        trust_remote_code=True,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return llm, sampling_params


def init_model_transformers(model_path: str):
    """Initialize model with HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def generate_response_vllm(llm, sampling_params, prompt: str) -> str:
    """Generate a single response using vLLM."""
    system_msg = (
        "You are a helpful medical assistant for Indian patients. "
        "Respond in the same language as the user's query. "
        "Provide safe, accurate medical information."
    )
    full_prompt = f"<|system|>\n{system_msg}\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    outputs = llm.generate([full_prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_response_transformers(model, tokenizer, prompt: str) -> str:
    """Generate a single response using transformers."""
    import torch
    
    system_msg = (
        "You are a helpful medical assistant for Indian patients. "
        "Respond in the same language as the user's query. "
        "Provide safe, accurate medical information."
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
        )
    
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def batch_generate_vllm(llm, sampling_params, prompts: list[str]) -> list[str]:
    """Batch generate for efficiency on H100."""
    system_msg = (
        "You are a helpful medical assistant for Indian patients. "
        "Respond in the same language as the user's query. "
        "Provide safe, accurate medical information."
    )
    full_prompts = [
        f"<|system|>\n{system_msg}\n<|user|>\n{p}\n<|assistant|>\n"
        for p in prompts
    ]
    
    outputs = llm.generate(full_prompts, sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def extract_mcq_answer(response: str) -> str:
    """Extract the chosen option letter from model response."""
    # Try common patterns
    patterns = [
        r'\b([A-D])\)',        # A)
        r'\b([A-D])\.',        # A.
        r'\b([A-D])\b',        # standalone A
        r'answer\s*(?:is)?\s*([A-D])',  # answer is A
        r'option\s*([A-D])',   # option A
        r'correct.*?([A-D])',  # correct ... A
        r'उत्तर.*?([A-D])',     # Hindi: answer ... A
        r'సమాధానం.*?([A-D])',   # Telugu: answer ... A
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Last resort: first capital letter A-D in response
    match = re.search(r'([A-D])', response)
    return match.group(1) if match else "X"


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute sentence-level BLEU score."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        smoothie = SmoothingFunction().method1
        return sentence_bleu(
            [ref_tokens], hyp_tokens,
            weights=(0.5, 0.5),  # bigram BLEU — more lenient for Indic
            smoothing_function=smoothie,
        )
    except ImportError:
        print("WARNING: nltk not installed. Skipping BLEU. pip install nltk")
        return -1.0


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except ImportError:
        print("WARNING: rouge-score not installed. Skipping ROUGE-L. pip install rouge-score")
        return -1.0


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(model_name: str, model_path: str):
    eval_set = load_eval_set()
    
    # Initialize model
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Backend: {BACKEND}")
    print(f"{'='*60}\n")
    
    if BACKEND == "vllm":
        llm, sampling_params = init_model_vllm(model_path)
    elif BACKEND == "transformers":
        model, tokenizer = init_model_transformers(model_path)
    
    results = []
    
    # Gather all prompts
    all_prompts = []
    all_meta = []
    
    for cat_key in ["category_1_medical_knowledge_mcq", "category_2_conversational_medical_dialogue", "category_3_code_switching_robustness"]:
        category = eval_set[cat_key]
        for p in category["prompts"]:
            all_prompts.append(p["prompt"])
            all_meta.append({"category": cat_key, **p})
    
    print(f"Total prompts: {len(all_prompts)}")
    
    # Generate responses
    if BACKEND == "vllm":
        print("Running batched inference with vLLM...")
        start = time.time()
        responses = batch_generate_vllm(llm, sampling_params, all_prompts)
        elapsed = time.time() - start
        print(f"Batch inference completed in {elapsed:.1f}s ({elapsed/len(all_prompts):.2f}s per prompt)")
    else:
        responses = []
        for i, prompt in enumerate(all_prompts):
            print(f"  [{i+1}/{len(all_prompts)}] Generating response...")
            start = time.time()
            if BACKEND == "transformers":
                resp = generate_response_transformers(model, tokenizer, prompt)
            responses.append(resp)
            elapsed = time.time() - start
            print(f"    Done in {elapsed:.1f}s")
    
    # Score responses
    print("\nScoring responses...")
    
    mcq_correct = 0
    mcq_total = 0
    bleu_scores = []
    rouge_scores = []
    refusals = 0
    
    for meta, response in zip(all_meta, responses):
        result = {
            "prompt_id": meta["id"],
            "model": model_name,
            "category": meta["category"],
            "language": meta["language"],
            "prompt": meta["prompt"],
            "raw_response": response,
            "refused": 0,
            "language_coherence": None,  # manual scoring
            "medical_relevance": None,   # manual scoring
        }
        
        # Check for refusal
        refusal_keywords = [
            "I cannot", "I can't", "I'm not able", "as an AI",
            "मैं यह नहीं", "నేను చేయలేను", "sorry", "apologies"
        ]
        if any(kw.lower() in response.lower() for kw in refusal_keywords):
            result["refused"] = 1
            refusals += 1
        
        # MCQ scoring
        if meta["category"] == "category_1_medical_knowledge_mcq":
            extracted = extract_mcq_answer(response)
            correct = meta["correct_answer"]
            is_correct = int(extracted == correct)
            result["extracted_answer"] = extracted
            result["correct_answer"] = correct
            result["mcq_correct"] = is_correct
            mcq_correct += is_correct
            mcq_total += 1
        
        # BLEU/ROUGE for conversational & code-switching
        if meta["category"] in ["category_2_conversational_medical_dialogue", "category_3_code_switching_robustness"]:
            ref = meta.get("reference_answer", "")
            if ref:
                bleu = compute_bleu(ref, response)
                rouge = compute_rouge_l(ref, response)
                result["bleu_score"] = bleu
                result["rouge_l_score"] = rouge
                if bleu >= 0:
                    bleu_scores.append(bleu)
                if rouge >= 0:
                    rouge_scores.append(rouge)
        
        results.append(result)
    
    # Save results
    out_dir = Path(RESULTS_DIR) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "responses.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Summary
    summary = {
        "model": model_name,
        "total_prompts": len(all_prompts),
        "mcq_accuracy": f"{mcq_correct}/{mcq_total} ({100*mcq_correct/mcq_total:.1f}%)" if mcq_total > 0 else "N/A",
        "avg_bleu": f"{sum(bleu_scores)/len(bleu_scores):.4f}" if bleu_scores else "N/A",
        "avg_rouge_l": f"{sum(rouge_scores)/len(rouge_scores):.4f}" if rouge_scores else "N/A",
        "refusal_rate": f"{refusals}/{len(all_prompts)} ({100*refusals/len(all_prompts):.1f}%)",
        "note": "language_coherence and medical_relevance require manual scoring — see responses.json",
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS: {model_name}")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to {out_dir}/")
    
    return summary


def compare_models():
    """Compare results across all evaluated models."""
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print("No results directory found. Run evaluations first.")
        return
    
    summaries = []
    for model_dir in sorted(results_path.iterdir()):
        summary_file = model_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summaries.append(json.load(f))
    
    if not summaries:
        print("No model summaries found.")
        return
    
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'MCQ Acc':>12} {'Avg BLEU':>12} {'Avg ROUGE-L':>14} {'Refusals':>12}")
    print(f"{'-'*70}")
    
    for s in summaries:
        print(f"{s['model']:<20} {s['mcq_accuracy']:>12} {s['avg_bleu']:>12} {s['avg_rouge_l']:>14} {s['refusal_rate']:>12}")
    
    print(f"\n  NOTE: Manual scores (language_coherence, medical_relevance) not included.")
    print(f"  Review individual responses.json files for manual scoring.\n")


# ---------------------------------------------------------------------------
# Manual scoring helper
# ---------------------------------------------------------------------------

def manual_scoring_helper(model_name: str):
    """Interactive helper for manual scoring of language coherence and medical relevance."""
    responses_file = Path(RESULTS_DIR) / model_name / "responses.json"
    if not responses_file.exists():
        print(f"No responses found for {model_name}")
        return
    
    with open(responses_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print(f"\nManual Scoring for: {model_name}")
    print("Score each response on two dimensions:")
    print("  Language Coherence (1-5): 1=broken, 3=understandable, 5=fluent native quality")
    print("  Medical Relevance  (1-5): 1=dangerous/wrong, 3=partially correct, 5=complete and safe")
    print("  Press Enter to skip, 'q' to quit and save progress.\n")
    
    for i, r in enumerate(results):
        if r.get("language_coherence") is not None:
            continue  # already scored
        
        print(f"\n--- [{i+1}/{len(results)}] {r['prompt_id']} ({r['language']}) ---")
        print(f"PROMPT: {r['prompt'][:200]}...")
        print(f"RESPONSE: {r['raw_response'][:400]}...")
        
        lc = input("  Language Coherence (1-5): ").strip()
        if lc == 'q':
            break
        if lc in ['1','2','3','4','5']:
            r["language_coherence"] = int(lc)
        
        mr = input("  Medical Relevance  (1-5): ").strip()
        if mr == 'q':
            break
        if mr in ['1','2','3','4','5']:
            r["medical_relevance"] = int(mr)
    
    with open(responses_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nProgress saved to {responses_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicHealthAI Model Evaluation")
    parser.add_argument("--model_name", type=str, help="Name for this model (e.g., qwen2.5-7b)")
    parser.add_argument("--model_path", type=str, help="HuggingFace model ID or local path")
    parser.add_argument("--compare", action="store_true", help="Compare all evaluated models")
    parser.add_argument("--manual_score", type=str, help="Run manual scoring for a model")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "transformers", "llamacpp"])
    
    args = parser.parse_args()
    
    if args.backend:
        BACKEND = args.backend
    
    if args.compare:
        compare_models()
    elif args.manual_score:
        manual_scoring_helper(args.manual_score)
    elif args.model_name and args.model_path:
        run_evaluation(args.model_name, args.model_path)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python run_eval.py --model_name qwen2.5-7b --model_path Qwen/Qwen2.5-7B-Instruct")
        print("  python run_eval.py --model_name llama3.1-8b --model_path meta-llama/Llama-3.1-8B-Instruct")
        print("  python run_eval.py --model_name qwen2.5-3b --model_path Qwen/Qwen2.5-3B-Instruct")
        print("  python run_eval.py --model_name smollm2-1.7b --model_path HuggingFaceTB/SmolLM2-1.7B-Instruct")
        print("  python run_eval.py --compare")
        print("  python run_eval.py --manual_score qwen2.5-7b")
