"""
batch_emails.py
---------------
Splits filtered_emails.jsonl into batches of 50 emails each.
Output files are named: batch_001.jsonl, batch_002.jsonl, ...

Usage:
  python batch_emails.py                              # uses default path
  python batch_emails.py filtered_emails.jsonl        # custom input
  python batch_emails.py filtered_emails.jsonl 30     # custom batch size
"""

import json
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATA_RAW    = Path(__file__).resolve().parents[1] / "data" / "raw"
INPUT_FILE  = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_RAW / "filtered_emails.jsonl"
BATCH_SIZE  = int(sys.argv[2]) if len(sys.argv) > 2 else 50
OUTPUT_DIR  = DATA_RAW / "batches"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load all records
    records = []
    with INPUT_FILE.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {lineno}: {exc}")

    total        = len(records)
    num_batches  = (total + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division

    for batch_idx in range(num_batches):
        start  = batch_idx * BATCH_SIZE
        end    = min(start + BATCH_SIZE, total)
        chunk  = records[start:end]

        out_path = OUTPUT_DIR / f"batch_{batch_idx + 1:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for rec in chunk:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  batch_{batch_idx + 1:03d}.jsonl  →  emails {start + 1}–{end}  ({len(chunk)} records)")

    print(f"\nDone — {total} emails split into {num_batches} batches of ≤{BATCH_SIZE}.")
    print(f"Output folder: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
