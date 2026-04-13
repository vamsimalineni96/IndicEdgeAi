"""
translate_medqa.py — DEPRECATED
--------------------------------
This script has been superseded by pipeline.py which supports both
translation AND transliteration, field-level resume, progress bars,
and multiple Indic languages.

Replace any usage of:
    python translate_medqa.py --input data/medqa.jsonl [options]

With:
    python pipeline.py --input data/medqa.jsonl [options]

Key differences
───────────────
  Old                                  New
  -----------------------------------  -------------------------------------------
  translation only                     translation + transliteration
  no resume on crash                   field-level checkpoint resume
  Hindi / Telugu hardcoded             --langs hindi telugu tamil kannada bengali marathi
  output overwrites question field     clean schema: question_en / question_{lang}_trans
  no progress bars                     tqdm bars per field

Run `python pipeline.py --help` for full usage.
"""

import sys

print(__doc__, file=sys.stderr)
print("Launching pipeline.py with the same arguments...\n", file=sys.stderr)

# Pass-through to pipeline.py so existing scripts don't hard-break
import runpy
sys.argv[0] = "pipeline.py"
runpy.run_path("pipeline.py", run_name="__main__")
