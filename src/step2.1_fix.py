"""
Step 2.5: Clean HealthSearchQA — Randomize Disclaimers + Fix Flagged Responses
================================================================================
Fixes two issues identified in the judge report:
  1. Boilerplate disclaimers are nearly identical → randomize across 8 variants
  2. Critical error at idx 1314 (pyoderma gangrenosum) → remove it

Usage:
    python step2_5_clean_healthsearchqa.py

Input:
    data/curated/healthsearchqa_with_answers.json

Output:
    data/curated/healthsearchqa_with_answers.json  (overwritten with cleaned version)
    data/curated/healthsearchqa_removed.json       (removed/flagged entries for reference)
"""

import json
import os
import re
import random

random.seed(42)

INPUT_FILE = "data/curated/healthsearchqa_with_answers.json"
OUTPUT_FILE = "data/curated/healthsearchqa_with_answers.json"
REMOVED_FILE = "data/curated/healthsearchqa_removed.json"

# ── Flagged indices from judge report ───────────────────────────────────
# These had critical medical accuracy issues (score ≤ 2)
CRITICAL_REMOVE_INDICES = {1314}

# ── Disclaimer variants ────────────────────────────────────────────────
# These replace the repetitive boilerplate the model tends to produce
DISCLAIMER_VARIANTS = [
    "Please remember that this information is for general awareness only. "
    "For personalized medical advice, consult a qualified healthcare provider.",

    "This is meant as general health information and should not replace "
    "a consultation with your doctor. Please seek professional medical advice "
    "for your specific situation.",

    "Note: This response provides general guidance only. "
    "Always consult a healthcare professional before making any medical decisions.",

    "Keep in mind that every individual's health situation is different. "
    "It's best to discuss your specific concerns with a qualified medical professional.",

    "Important: This is general health information, not a diagnosis or treatment plan. "
    "Please visit a healthcare provider for advice tailored to your needs.",

    "While this information may be helpful, it is not a substitute for professional "
    "medical evaluation. Consider scheduling an appointment with your doctor.",

    "This overview is intended for informational purposes. "
    "A qualified healthcare professional can provide guidance specific to your condition.",

    "Remember, online health information has its limits. "
    "For accurate diagnosis and treatment, please consult a medical professional.",
]

# ── Common boilerplate patterns to detect ───────────────────────────────
BOILERPLATE_PATTERNS = [
    r"(?i)it(?:'s| is) (?:always |)(?:important|essential|crucial|recommended|advisable|best) "
    r"to (?:consult|see|visit|speak with|talk to|seek) (?:a |your |with a |)"
    r"(?:healthcare|medical|health care|health-care) (?:professional|provider|practitioner|expert|doctor)",

    r"(?i)(?:please |)(?:note|remember|keep in mind) that this (?:information|response|answer|) "
    r"(?:is |)(?:not |)(?:a substitute|meant to replace|intended to replace)",

    r"(?i)(?:this |the above |)(?:information|content|response|advice) "
    r"(?:is |)(?:provided |meant |intended |)(?:for |as |)"
    r"(?:general|informational|educational) (?:purposes|information|awareness|guidance)",

    r"(?i)(?:always |)(?:consult|see|visit|speak with) "
    r"(?:a |your |)(?:doctor|physician|healthcare provider|medical professional|qualified) "
    r"(?:before |for |if |when )",

    r"(?i)this is not (?:a substitute for |meant to replace |intended as )"
    r"professional medical advice",

    r"(?i)(?:i am|i'm) (?:an AI|not a doctor|not a medical professional)",
]


def find_disclaimer_paragraph(text):
    """
    Find the last paragraph that matches boilerplate disclaimer patterns.
    Returns (text_before, disclaimer_paragraph) or (text, None) if not found.
    """
    # Split into paragraphs
    paragraphs = text.strip().split("\n\n")

    if len(paragraphs) < 2:
        # Single paragraph — check if the last 1-2 sentences are a disclaimer
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) >= 2:
            last_two = " ".join(sentences[-2:])
            for pattern in BOILERPLATE_PATTERNS:
                if re.search(pattern, last_two):
                    main_text = " ".join(sentences[:-2])
                    return main_text, last_two
        return text, None

    # Check last paragraph (most common location for disclaimer)
    last_para = paragraphs[-1].strip()
    for pattern in BOILERPLATE_PATTERNS:
        if re.search(pattern, last_para):
            main_text = "\n\n".join(paragraphs[:-1])
            return main_text, last_para

    # Check second-to-last if last paragraph is very short
    if len(paragraphs) >= 3 and len(last_para.split()) < 15:
        second_last = paragraphs[-2].strip()
        for pattern in BOILERPLATE_PATTERNS:
            if re.search(pattern, second_last):
                main_text = "\n\n".join(paragraphs[:-2])
                remaining = last_para
                return main_text + "\n\n" + remaining, second_last

    return text, None


def replace_disclaimer(response):
    """Replace boilerplate disclaimer with a random variant."""
    main_text, old_disclaimer = find_disclaimer_paragraph(response)

    if old_disclaimer is None:
        # No disclaimer found — append one (the model should always have one)
        new_disclaimer = random.choice(DISCLAIMER_VARIANTS)
        return response.strip() + "\n\n" + new_disclaimer, "added"

    # Replace with a random variant
    new_disclaimer = random.choice(DISCLAIMER_VARIANTS)
    return main_text.strip() + "\n\n" + new_disclaimer, "replaced"


def main():
    print("=" * 60)
    print("  STEP 2.5: Clean HealthSearchQA Responses")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"   Loaded {len(data)} entries")

    # ── Remove critical entries ─────────────────────────────────────
    print(f"\n[2/4] Removing {len(CRITICAL_REMOVE_INDICES)} critically flagged entries...")
    removed = []
    cleaned = []

    for i, entry in enumerate(data):
        if i in CRITICAL_REMOVE_INDICES:
            entry["removal_reason"] = "Critical medical accuracy error (judge score ≤ 2)"
            removed.append(entry)
            print(f"   ✗ Removed idx {i}: '{entry['instruction'][:60]}...'")
        else:
            cleaned.append(entry)

    print(f"   Remaining: {len(cleaned)} entries")

    # ── Randomize disclaimers ───────────────────────────────────────
    print(f"\n[3/4] Randomizing disclaimers...")
    stats = {"replaced": 0, "added": 0, "unchanged": 0}

    for entry in cleaned:
        original = entry["response"]
        new_response, action = replace_disclaimer(original)
        entry["response"] = new_response
        stats[action] = stats.get(action, 0) + 1

    print(f"   Disclaimers replaced:  {stats.get('replaced', 0)}")
    print(f"   Disclaimers added:     {stats.get('added', 0)}")

    # ── Save ────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"   ✓ Saved {len(cleaned)} cleaned entries to {OUTPUT_FILE}")

    if removed:
        with open(REMOVED_FILE, "w", encoding="utf-8") as f:
            json.dump(removed, f, ensure_ascii=False, indent=2)
        print(f"   ✓ Saved {len(removed)} removed entries to {REMOVED_FILE}")

    # ── Verify with examples ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  SAMPLE — Before vs After disclaimer replacement:")
    print("-" * 60)

    sample = cleaned[0]
    response_lines = sample["response"].strip().split("\n\n")
    if len(response_lines) >= 2:
        print(f"\n  Question: {sample['instruction'][:80]}...")
        print(f"\n  Last paragraph (new disclaimer):")
        print(f"  \"{response_lines[-1][:120]}...\"")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Input:                {len(data)} entries")
    print(f"  Removed (critical):   {len(removed)}")
    print(f"  Output:               {len(cleaned)} entries")
    print(f"  Disclaimers varied:   {stats.get('replaced', 0) + stats.get('added', 0)}")
    print(f"  Disclaimer variants:  {len(DISCLAIMER_VARIANTS)} templates in rotation")
    print(f"  Output file:          {OUTPUT_FILE}")
    print("=" * 60)
    print("\n  Next: Run step3_translate_indictrans2.py")


if __name__ == "__main__":
    main()