"""
Step 4: Generate Code-Mixed Examples + Final Merge & Split
============================================================
1. Creates Hinglish and Tenglish code-mixed examples from English data
2. Merges all languages (EN + HI + TE + code-mixed)
3. Splits into train/val/test (80/10/10)
4. Formats as ChatML JSONL (Qwen's instruction format)

Usage:
    python step4_merge_and_split.py

Output:
    data/final/train.jsonl
    data/final/val.jsonl
    data/final/test.jsonl
    data/final/data_stats.json
"""

import json
import os
import random
from collections import Counter

random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
TRANSLATED_DIR = "data/translated"
OUTPUT_DIR = "data/final"
CODEMIX_COUNT = 2000  # Number of code-mixed examples to generate
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Code-mixing templates ───────────────────────────────────────────────
# These simulate how real Indian users ask medical questions

HINGLISH_TEMPLATES = [
    "Doctor sahab, {symptom_en} ho raha hai, kya karna chahiye?",
    "Mujhe {symptom_en} hai aur bahut takleef ho rahi hai. Koi medicine batao?",
    "Mere {body_part_en} mein {symptom_en} hai, should I see a doctor?",
    "Kya {medicine_en} safe hai? Main {condition_en} ke liye le raha hoon.",
    "Meri age {age} hai, {symptom_en} ho raha hai 2 din se. Is it serious?",
    "{condition_en} mein kya diet follow karni chahiye?",
    "Bacche ko {symptom_en} hai, koi home remedy batao please?",
    "Doctor ne {medicine_en} prescribe kiya hai, side effects kya hain?",
    "Pregnancy mein {symptom_en} normal hai ya doctor ko dikhana chahiye?",
    "Mera blood pressure {bp} aa raha hai, yeh high hai kya?",
]

TENGLISH_TEMPLATES = [
    "Doctor garu, {symptom_en} vastundi, em cheyali?",
    "Naku {symptom_en} undi, chala baadha ga undi. Emi medicine teesukovali?",
    "Na {body_part_en} lo {symptom_en} undi, doctor ki chupinchala?",
    "{medicine_en} safe aa? Nenu {condition_en} ki teesukuntunna.",
    "Na age {age}, {symptom_en} 2 rojulu nundi undi. Serious aa?",
    "{condition_en} lo em diet follow avali?",
    "Pillalaki {symptom_en} vachindi, emi home remedy cheppandi?",
    "Doctor {medicine_en} raasaaru, side effects emiti?",
    "Pregnancy lo {symptom_en} normal aa leda doctor ki chupinchala?",
    "Na blood pressure {bp} vastundi, idi high aa?",
]

# Medical vocabulary for template filling
SYMPTOMS = [
    "fever", "headache", "stomach pain", "cough", "cold",
    "body pain", "vomiting", "diarrhea", "chest pain", "back pain",
    "joint pain", "sore throat", "dizziness", "fatigue", "nausea",
    "breathing problem", "skin rash", "acidity", "constipation",
    "weakness", "weight loss", "insomnia", "anxiety",
]

BODY_PARTS = [
    "chest", "stomach", "back", "knee", "shoulder", "head",
    "throat", "ear", "eye", "leg", "arm", "neck",
]

MEDICINES = [
    "paracetamol", "ibuprofen", "metformin", "amoxicillin",
    "omeprazole", "cetirizine", "azithromycin", "aspirin",
    "crocin", "dolo 650", "combiflam",
]

CONDITIONS = [
    "diabetes", "hypertension", "thyroid", "asthma", "PCOD",
    "arthritis", "migraine", "gastritis", "anemia",
]

AGES = ["25", "30", "35", "40", "45", "50", "55", "60", "65"]
BPS = ["140/90", "150/95", "160/100", "130/85", "145/92"]


def generate_codemixed_examples():
    """Generate synthetic code-mixed medical questions."""
    examples = []

    for _ in range(CODEMIX_COUNT // 2):
        # Hinglish
        template = random.choice(HINGLISH_TEMPLATES)
        question = template.format(
            symptom_en=random.choice(SYMPTOMS),
            body_part_en=random.choice(BODY_PARTS),
            medicine_en=random.choice(MEDICINES),
            condition_en=random.choice(CONDITIONS),
            age=random.choice(AGES),
            bp=random.choice(BPS),
        )
        examples.append({
            "instruction": question,
            "response": "",  # Will be filled by teacher model in Step 5
            "subject": "Consumer Health",
            "topic": "code-mixed",
            "source": "synthetic-hinglish",
            "language": "hinglish",
        })

        # Tenglish
        template = random.choice(TENGLISH_TEMPLATES)
        question = template.format(
            symptom_en=random.choice(SYMPTOMS),
            body_part_en=random.choice(BODY_PARTS),
            medicine_en=random.choice(MEDICINES),
            condition_en=random.choice(CONDITIONS),
            age=random.choice(AGES),
            bp=random.choice(BPS),
        )
        examples.append({
            "instruction": question,
            "response": "",
            "subject": "Consumer Health",
            "topic": "code-mixed",
            "source": "synthetic-tenglish",
            "language": "tenglish",
        })

    return examples


def format_chatml(example):
    """Format a single example as ChatML (Qwen format)."""
    system_msg = (
        "You are a helpful multilingual medical assistant. "
        "Answer health questions accurately in the same language as the question. "
        "Always advise consulting a doctor for serious symptoms."
    )

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]},
        ],
        "metadata": {
            "language": example.get("language", "english"),
            "subject": example.get("subject", ""),
            "source": example.get("source", ""),
        }
    }


def main():
    print("=" * 60)
    print("  STEP 4: Code-Mixed Generation + Merge & Split")
    print("=" * 60)

    # ── Load all translated datasets ────────────────────────────────
    print("\n[1/5] Loading translated datasets...")

    all_data = []
    lang_counts = Counter()

    for f_name in sorted(os.listdir(TRANSLATED_DIR)):
        if not f_name.endswith(".json"):
            continue

        f_path = os.path.join(TRANSLATED_DIR, f_name)
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"   Loaded {f_name}: {len(data)} examples")

        for ex in data:
            # Skip examples with empty responses
            if not ex.get("response", "").strip():
                continue
            all_data.append(ex)
            lang = ex.get("language", "english")
            lang_counts[lang] += 1

    print(f"\n   Total loaded: {len(all_data)}")
    for lang, count in lang_counts.most_common():
        print(f"     {lang}: {count}")

    # ── Generate code-mixed examples ────────────────────────────────
    print(f"\n[2/5] Generating {CODEMIX_COUNT} code-mixed examples...")

    codemixed = generate_codemixed_examples()
    # Note: These have empty responses — they need to be answered by the teacher
    # We'll handle that in a separate step or you can run step2-style generation
    codemixed_with_responses = [ex for ex in codemixed if ex["response"].strip()]
    codemixed_need_answers = [ex for ex in codemixed if not ex["response"].strip()]

    print(f"   Generated {len(codemixed)} code-mixed questions")
    if codemixed_need_answers:
        print(f"   ⚠ {len(codemixed_need_answers)} need answers from teacher model")
        print(f"     Saving to data/curated/codemixed_need_answers.json")
        print(f"     Run step2 pattern with these to generate answers,")
        print(f"     then re-run this script.")

        os.makedirs("data/curated", exist_ok=True)
        with open("data/curated/codemixed_need_answers.json", "w", encoding="utf-8") as f:
            json.dump(codemixed_need_answers, f, ensure_ascii=False, indent=2)

    # For now, only include code-mixed examples that have responses
    # (In practice, you'd generate answers first)
    all_data.extend(codemixed_with_responses)

    # ── Shuffle and split ───────────────────────────────────────────
    print(f"\n[3/5] Shuffling and splitting...")

    random.shuffle(all_data)
    total = len(all_data)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_data = all_data[:train_end]
    val_data = all_data[train_end:val_end]
    test_data = all_data[val_end:]

    print(f"   Total: {total}")
    print(f"   Train: {len(train_data)} ({len(train_data)/total*100:.1f}%)")
    print(f"   Val:   {len(val_data)} ({len(val_data)/total*100:.1f}%)")
    print(f"   Test:  {len(test_data)} ({len(test_data)/total*100:.1f}%)")

    # ── Format as ChatML JSONL ──────────────────────────────────────
    print(f"\n[4/5] Formatting as ChatML JSONL...")

    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in split_data:
                chatml = format_chatml(ex)
                f.write(json.dumps(chatml, ensure_ascii=False) + "\n")
        print(f"   ✓ Saved {split_name}.jsonl ({len(split_data)} examples)")

    # ── Compute and save stats ──────────────────────────────────────
    print(f"\n[5/5] Computing dataset statistics...")

    def compute_stats(data, name):
        lang_dist = Counter(ex.get("language", "unknown") for ex in data)
        source_dist = Counter(ex.get("source", "unknown") for ex in data)
        subject_dist = Counter(ex.get("subject", "unknown") for ex in data)
        avg_inst_len = sum(len(ex["instruction"]) for ex in data) / len(data) if data else 0
        avg_resp_len = sum(len(ex["response"]) for ex in data) / len(data) if data else 0

        return {
            "split": name,
            "total": len(data),
            "language_distribution": dict(lang_dist.most_common()),
            "source_distribution": dict(source_dist.most_common()),
            "subject_distribution": dict(subject_dist.most_common(10)),
            "avg_instruction_length_chars": round(avg_inst_len),
            "avg_response_length_chars": round(avg_resp_len),
        }

    stats = {
        "train": compute_stats(train_data, "train"),
        "val": compute_stats(val_data, "val"),
        "test": compute_stats(test_data, "test"),
        "total_examples": total,
        "languages": list(lang_counts.keys()),
        "codemixed_questions_pending_answers": len(codemixed_need_answers),
    }

    stats_path = os.path.join(OUTPUT_DIR, "data_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ── Print summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Output directory: {OUTPUT_DIR}/")
    print(f"    train.jsonl:     {len(train_data)} examples")
    print(f"    val.jsonl:       {len(val_data)} examples")
    print(f"    test.jsonl:      {len(test_data)} examples")
    print(f"    data_stats.json: dataset statistics")
    print(f"\n  Language distribution (full dataset):")
    for lang, count in lang_counts.most_common():
        pct = count / total * 100
        print(f"    {lang}: {count} ({pct:.1f}%)")

    if codemixed_need_answers:
        print(f"\n  ⚠ ACTION NEEDED:")
        print(f"    {len(codemixed_need_answers)} code-mixed questions need answers.")
        print(f"    Generate them with the teacher model (similar to step2),")
        print(f"    then re-run this script to include them.")

    print("\n  Next: Run Phase 3 (fine-tuning) with:")
    print(f"    python finetune.py --train_data {OUTPUT_DIR}/train.jsonl \\")
    print(f"                       --val_data {OUTPUT_DIR}/val.jsonl")
    print("=" * 60)


if __name__ == "__main__":
    main()
