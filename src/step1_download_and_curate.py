"""
Step 1: Download and Curate Medical QA Dataset
================================================
Downloads JKumar11/MedQA_Medmcqa_combined (93K examples)
Filters for high-quality examples and curates ~15K for fine-tuning.

Usage:
    python step1_download_and_curate.py

Output:
    data/curated/medqa_curated_15k.json
"""

import json
import os
import random
from datasets import load_dataset

random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/curated"
TARGET_SIZE = 15000  # Target number of curated examples
MIN_QUESTION_LEN = 30  # Minimum question length in characters
MIN_EXPLANATION_LEN = 50  # Minimum explanation length in characters

# Subject distribution targets (approximate percentages)
# We want broad medical coverage, not just one specialty
PRIORITY_SUBJECTS = [
    "Medicine", "Pharmacology", "Pathology", "Anatomy",
    "Physiology", "Biochemistry", "Microbiology", "Surgery",
    "Pediatrics", "Obstetrics and Gynecology", "Ophthalmology",
    "ENT", "Psychiatry", "Dermatology", "Forensic Medicine",
    "Preventive Medicine", "Radiology", "Anesthesia",
    "Orthopedics", "Dental",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def format_qa_pair(example):
    """Convert MCQ format into a conversational QA instruction pair."""
    question = example["question"].strip()
    options = example.get("options", [])
    correct_idx = example.get("answer_idx", None)
    correct_answer = example.get("answer", "")
    explanation = example.get("explanation", "") or ""
    subject = example.get("subject", "General Medicine") or "General Medicine"
    topic = example.get("topic", "") or ""
    source = example.get("source", "unknown")

    # Build the options string
    option_labels = ["A", "B", "C", "D"]
    options_text = ""
    if options:
        for i, opt in enumerate(options):
            if i < len(option_labels):
                options_text += f"\n{option_labels[i]}. {opt}"

    # Build the instruction (user message)
    instruction = question
    if options_text:
        instruction += f"\n{options_text}"

    # Build the response (assistant message)
    # Include the correct answer and explanation
    response_parts = []
    if correct_answer:
        response_parts.append(f"The correct answer is: {correct_answer}")
    if explanation.strip():
        response_parts.append(f"\nExplanation: {explanation.strip()}")

    response = "\n".join(response_parts)

    return {
        "instruction": instruction,
        "response": response,
        "subject": subject,
        "topic": topic,
        "source": source,
        "original_question": question,
        "has_explanation": bool(explanation.strip()),
    }


def quality_filter(example):
    """Filter for high-quality examples."""
    question = example.get("question", "")
    explanation = example.get("explanation", "") or ""
    options = example.get("options", [])
    answer = example.get("answer", "")

    # Must have a question of minimum length
    if len(question.strip()) < MIN_QUESTION_LEN:
        return False

    # Must have a correct answer
    if not answer or not answer.strip():
        return False

    # Must have at least 2 options
    if len(options) < 2:
        return False

    # Prefer examples WITH explanations (but don't require all)
    # We'll handle the ratio later

    # Filter out questions that look like they have encoding issues
    bad_markers = ["â€™", "â€œ", "â€", "Â", "\x00"]
    for marker in bad_markers:
        if marker in question:
            return False

    return True


def main():
    print("=" * 60)
    print("  STEP 1: Download and Curate Medical QA Dataset")
    print("=" * 60)

    # ── Download ────────────────────────────────────────────────────
    print("\n[1/4] Downloading JKumar11/MedQA_Medmcqa_combined...")
    ds = load_dataset("JKumar11/MedQA_Medmcqa_combined")

    train_ds = ds["train"]
    val_ds = ds.get("validation", None)
    test_ds = ds.get("test", None)

    total = len(train_ds)
    if val_ds:
        total += len(val_ds)
    if test_ds:
        total += len(test_ds)
    print(f"   Total examples across all splits: {total}")
    print(f"   Train: {len(train_ds)}")
    if val_ds:
        print(f"   Validation: {len(val_ds)}")
    if test_ds:
        print(f"   Test: {len(test_ds)}")

    # ── Inspect schema ──────────────────────────────────────────────
    print("\n[2/4] Inspecting dataset schema...")
    sample = train_ds[0]
    print(f"   Fields: {list(sample.keys())}")
    print(f"   Sample question: {sample.get('question', 'N/A')[:100]}...")

    # ── Quality filter ──────────────────────────────────────────────
    print("\n[3/4] Applying quality filters...")

    # Use only train split for fine-tuning data
    # Keep val/test separate for evaluation
    all_examples = list(train_ds)
    print(f"   Starting with {len(all_examples)} training examples")

    filtered = [ex for ex in all_examples if quality_filter(ex)]
    print(f"   After quality filter: {len(filtered)}")

    # Separate into with-explanation and without-explanation
    with_exp = [ex for ex in filtered if (ex.get("explanation", "") or "").strip()
                and len((ex.get("explanation", "") or "").strip()) >= MIN_EXPLANATION_LEN]
    without_exp = [ex for ex in filtered if ex not in with_exp]
    print(f"   With explanations (>={MIN_EXPLANATION_LEN} chars): {len(with_exp)}")
    print(f"   Without explanations: {len(without_exp)}")

    # ── Curate balanced subset ──────────────────────────────────────
    print("\n[4/4] Curating balanced subset...")

    # Prioritize examples WITH explanations (80% target)
    target_with_exp = int(TARGET_SIZE * 0.8)
    target_without_exp = TARGET_SIZE - target_with_exp

    random.shuffle(with_exp)
    random.shuffle(without_exp)

    selected_with = with_exp[:target_with_exp]
    selected_without = without_exp[:target_without_exp]

    # If we don't have enough with explanations, backfill
    if len(selected_with) < target_with_exp:
        shortfall = target_with_exp - len(selected_with)
        selected_without = without_exp[:target_without_exp + shortfall]

    selected = selected_with + selected_without
    random.shuffle(selected)

    print(f"   Selected: {len(selected)} examples")
    print(f"     - With explanations: {len(selected_with)}")
    print(f"     - Without explanations: {len(selected_without)}")

    # ── Subject distribution ────────────────────────────────────────
    subject_counts = {}
    for ex in selected:
        subj = ex.get("subject", "Unknown") or "Unknown"
        subject_counts[subj] = subject_counts.get(subj, 0) + 1

    print("\n   Subject distribution:")
    for subj, count in sorted(subject_counts.items(), key=lambda x: -x[1])[:15]:
        pct = count / len(selected) * 100
        print(f"     {subj}: {count} ({pct:.1f}%)")
    if len(subject_counts) > 15:
        print(f"     ... and {len(subject_counts) - 15} more subjects")

    # ── Format and save ─────────────────────────────────────────────
    formatted = [format_qa_pair(ex) for ex in selected]

    output_path = os.path.join(OUTPUT_DIR, "medqa_curated_15k.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"\n   Saved {len(formatted)} curated QA pairs to: {output_path}")

    # ── Also save the validation/test sets for evaluation ───────────
    if val_ds:
        val_formatted = [format_qa_pair(ex) for ex in val_ds if quality_filter(ex)]
        val_path = os.path.join(OUTPUT_DIR, "medqa_val.json")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(val_formatted, f, ensure_ascii=False, indent=2)
        print(f"   Saved {len(val_formatted)} validation examples to: {val_path}")

    if test_ds:
        test_formatted = [format_qa_pair(ex) for ex in test_ds if quality_filter(ex)]
        test_path = os.path.join(OUTPUT_DIR, "medqa_test.json")
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_formatted, f, ensure_ascii=False, indent=2)
        print(f"   Saved {len(test_formatted)} test examples to: {test_path}")

    # ── Quick stats ─────────────────────────────────────────────────
    avg_q_len = sum(len(ex["instruction"]) for ex in formatted) / len(formatted)
    avg_r_len = sum(len(ex["response"]) for ex in formatted) / len(formatted)
    with_exp_count = sum(1 for ex in formatted if ex["has_explanation"])

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Curated examples:     {len(formatted)}")
    print(f"  With explanations:    {with_exp_count} ({with_exp_count/len(formatted)*100:.1f}%)")
    print(f"  Avg question length:  {avg_q_len:.0f} chars")
    print(f"  Avg response length:  {avg_r_len:.0f} chars")
    print(f"  Unique subjects:      {len(subject_counts)}")
    print(f"  Output:               {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
