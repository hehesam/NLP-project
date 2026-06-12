"""
filter_emails.py
----------------
Reads the raw emails JSONL export and produces two slimmed-down output files:
  - filtered_emails.jsonl   (one JSON object per line)
  - filtered_emails.csv     (spreadsheet-friendly)

Output columns:
  id, imap_msg_id, from, sender_email, subject, body_plain,
  topic_label (empty), priority_label (empty)

Usage:
  python filter_emails.py                          # uses default paths below
  python filter_emails.py my_emails.jsonl          # custom input file
"""

import json
import csv
import re
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_RAW    = Path(__file__).resolve().parents[1] / "data" / "raw"
INPUT_FILE  = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_RAW / "emails_export.jsonl"
OUTPUT_JSON = DATA_RAW / "filtered_emails.jsonl"
OUTPUT_CSV  = DATA_RAW / "filtered_emails.csv"

# ── Helpers ──────────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"<([^>]+)>")          # extract address from "Name <addr>"
_BARE_RE  = re.compile(r"[\w.+-]+@[\w.-]+")   # fallback: bare address anywhere

def extract_email(from_field: str) -> str:
    """Return just the email address from a 'Display Name <addr>' string."""
    m = _EMAIL_RE.search(from_field)
    if m:
        return m.group(1).strip()
    m = _BARE_RE.search(from_field)
    if m:
        return m.group(0).strip()
    return from_field.strip()   # give back whatever we got if nothing matched


KEEP_FIELDS = ["id", "imap_msg_id", "from", "sender_email",
               "subject", "body_plain", "topic_label", "priority_label"]

def filter_record(raw: dict) -> dict:
    return {
        "id"             : raw.get("id", ""),
        "imap_msg_id"    : raw.get("imap_msg_id", ""),
        "from"           : raw.get("from", ""),
        "sender_email"   : extract_email(raw.get("from", "")),
        "subject"        : raw.get("subject", ""),
        "body_plain"     : raw.get("body_plain", ""),
        "topic_label"    : "",   # to be filled during annotation
        "priority_label" : "",   # to be filled during annotation
    }

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    records = []
    with INPUT_FILE.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(filter_record(json.loads(line)))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed line {lineno}: {exc}")

    # ── Write JSONL ──────────────────────────────────────────────────────────
    with OUTPUT_JSON.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Write CSV ────────────────────────────────────────────────────────────
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=KEEP_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"Done — {len(records)} records written.")
    print(f"  JSONL : {OUTPUT_JSON}")
    print(f"  CSV   : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
