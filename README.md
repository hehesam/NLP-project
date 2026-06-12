# Multilingual University Email Triage — NLP Project

An end-to-end NLP pipeline that classifies university emails by **topic** (5 classes) and **priority** (High / Medium / Low), progressing from raw IMAP export through LLM-assisted annotation, classical baselines, hybrid feature engineering, and multilingual transformer fine-tuning.

> **Dataset privacy:** The corpus consists of real UniGe institutional emails and is **not distributed** with this repository. All dataset files are gitignored. See [`data/`](#data-folder) below for the expected layout.

---

## Repository layout

```
NLP-project/
├── 1_extraction/          # Stage 1 — IMAP export & filtering
├── 2_annotation/          # Stage 2 — LLM annotation & cleaning
├── 3_preprocessing_eda/   # Stage 3 — Preprocessing pipeline & EDA
├── 4_modeling/            # Stage 4 — Baselines → hybrid → transformers
├── data/                  # All datasets (gitignored — see below)
│   ├── raw/               # IMAP export, filtered emails, attachments
│   ├── annotation/        # Merged & cleaned LLM annotation CSVs
│   ├── processed/         # gold_dataset.csv, frozen_dataset.csv
│   └── labels/            # Per-model batch label files (batchNN_{ge,cl,gpt}.jsonl)
└── docs/
    ├── report/            # Deep-research background report (Markdown)
    ├── proposal/          # Project proposal (PDF + DOCX)
    ├── notes/             # Workflow notes, update logs, master prompts
    └── obsidian/          # Obsidian knowledge-base vault
```

---

## Pipeline overview

### Stage 1 — Extraction (`1_extraction/`)

| File | Purpose |
|------|---------|
| `export_unige_emails.py` | Connects to a UniGe Zimbra mailbox over IMAP, exports emails to `data/raw/emails_export.{jsonl,csv}` + attachments. Handles TLS quirks and decodes MIME. |
| `filter_emails.py` | Reads the raw JSONL export and produces a slim `filtered_emails.jsonl` / `.csv` with only the fields needed for annotation. |
| `batch_emails.py` | Splits `filtered_emails.jsonl` into fixed-size batches (default 50) for LLM annotation. |
| `email_query.ipynb` | Interactive notebook for ad-hoc IMAP queries and exploring the export. |

**Run order:**
```bash
python 1_extraction/export_unige_emails.py   # → data/raw/emails_export.*
python 1_extraction/filter_emails.py         # → data/raw/filtered_emails.jsonl
python 1_extraction/batch_emails.py          # → data/raw/batches/batch_NNN.jsonl
```

---

### Stage 2 — Annotation (`2_annotation/`)

Three LLMs (Gemini, Claude, GPT-4) independently annotated each email. Labels were merged by majority vote.

| File | Purpose |
|------|---------|
| `build_annotation_dataset.py` | Merges per-model batch JSONL files from `data/labels/` into a single `annotation_dataset.{jsonl,csv}`. |
| `Topic_annotation_cleaning.ipynb` | Cleans topic labels: resolves disagreements, drops the residual `Other` class, produces `annotation_dataset_clean.csv`. |
| `Piority_annotation_cleaning.ipynb` | Cleans priority labels (High / Medium / Low), produces `annotation_dataset_clean_3label.csv`. |
| `Finalizing_labels.ipynb` | Merges cleaned labels with the raw email text to produce `gold_dataset.csv` — the input to preprocessing. |

**Topic classes:** `course_exam`, `administrative`, `IT_technical`, `financial_billing`, `general_info`

**Run order:**
```bash
python 2_annotation/build_annotation_dataset.py
# then run the three notebooks in order
```

---

### Stage 3 — Preprocessing & EDA (`3_preprocessing_eda/`)

| File | Purpose |
|------|---------|
| `Preprocessing_pipeline.ipynb` | Reads `gold_dataset.csv`, applies multilingual text cleaning (lowercase, URL/email tokenisation, HTML stripping, boilerplate removal), and writes three text variants (`text_subject`, `text_body`, `text_subject_body`) to `frozen_dataset.csv`. |
| `eda.ipynb` | Exploratory analysis of `frozen_dataset.csv`: label distributions, text-length profiles, subject-template repetition (motivates the grouped-by-subject split), and language proxy. |

The output `data/processed/frozen_dataset.csv` is the single artifact that all modeling notebooks load.

---

### Stage 4 — Modeling (`4_modeling/`)

All four notebooks share the same two evaluation splits built once from `frozen_dataset.csv`:

- **Stratified split** — `train_test_split(..., stratify=y, random_state=42)` — the optimistic upper bound.
- **Grouped-by-subject split** — `GroupShuffleSplit(groups=text_subject, random_state=42)` — the honest estimate: no subject template seen at both train and test time.

| Notebook | What it does |
|----------|-------------|
| `03_text_baselines.ipynb` | TF-IDF (word 1–2 grams) + Logistic Regression / Linear SVM across the three input fields. Establishes the baseline macro-F1 numbers. |
| `04_text_feature_diagnostics.ipynb` | Shortcut-baseline checks (length, language proxy) and representation ablation (word vs character n-grams). Confirms `word_1_2gram` as the backbone. |
| `05_hybrid_feature_engineering.ipynb` | Adds four handcrafted feature families on top of the TF-IDF backbone: **length**, **language**, **regex/rules** (date/urgency patterns), **metadata** (sender domain, attachments, recipient count, time-of-day). Key finding: hybrid features do not improve the stratified score but improve the grouped-by-subject score by > +5 macro-F1 points — meaning they help generalisation to unseen email templates. |
| `06_transformer_finetune.ipynb` | Fine-tunes `xlm-roberta-base` and `Musixmatch/umberto-commoncrawl-cased-v1` on both tasks. The best transformer beats the hybrid system on both splits. |

---

## Data folder

The `data/` tree is **entirely gitignored** because the corpus consists of real institutional emails. To reproduce the pipeline from scratch you need access to the original UniGe mailbox and LLM API keys.

Expected layout after running the pipeline:

```
data/
├── raw/
│   ├── emails_export.jsonl       # full IMAP export
│   ├── emails_export.csv         # slim CSV version
│   ├── emails_export_full.pkl    # full dataframe with attachments metadata
│   ├── filtered_emails.jsonl     # filtered subset for annotation
│   └── attachments/              # downloaded email attachments
├── annotation/
│   ├── annotation_dataset.{jsonl,csv}      # merged LLM labels (raw)
│   ├── annotation_dataset_clean.csv        # after topic cleaning
│   └── annotation_dataset_clean_3label.csv # after priority cleaning
│   └── final_annotation_dataset.csv        # majority-vote labels
├── processed/
│   ├── gold_dataset.csv     # labels + email text (input to preprocessing)
│   └── frozen_dataset.csv   # cleaned text variants (input to modeling)
└── labels/
    └── batchNN_{ge,cl,gpt}.jsonl  # per-model LLM annotation batches
```

---

## Setup

**Python 3.10+** is required.

```bash
pip install pandas scikit-learn notebook tqdm joblib \
            transformers torch accelerate \
            python-dotenv
```

For the IMAP export script only:

```bash
pip install tqdm  # already above — no extra deps
```

---

## Reproducing the results

1. **Export emails** — run `1_extraction/export_unige_emails.py` with your IMAP credentials.
2. **Filter & batch** — run `filter_emails.py` then `batch_emails.py`.
3. **Annotate** — submit each `data/raw/batches/batch_NNN.jsonl` to Gemini / Claude / GPT-4 with the master prompt in `docs/notes/master_prompt.txt`. Save responses as `data/labels/batchNN_{ge,cl,gpt}.jsonl`.
4. **Build annotation dataset** — `python 2_annotation/build_annotation_dataset.py`.
5. **Clean annotations** — run the three notebooks in `2_annotation/` in order.
6. **Preprocess** — run `3_preprocessing_eda/Preprocessing_pipeline.ipynb`.
7. **Model** — run the four notebooks in `4_modeling/` in order (03 → 04 → 05 → 06).

---

## Key results (summary)

| Model | Task | Stratified macro-F1 | Grouped macro-F1 |
|-------|------|--------------------:|----------------:|
| TF-IDF + LinearSVC (text only) | Topic | 0.861 | 0.804 |
| TF-IDF + LinearSVC (hybrid) | Topic | 0.854 | 0.853 |
| XLM-RoBERTa fine-tuned | Topic | 0.858 | 0.815 |
| TF-IDF + LinearSVC (text only) | Priority | 0.765 | 0.705 |
| TF-IDF + LinearSVC (hybrid) | Priority | 0.747 | 0.739 |
| XLM-RoBERTa fine-tuned | Priority | 0.752 | 0.803 |

---

## Privacy & security

- **No email data is stored in this repository.** The entire `data/` tree is gitignored.
- All credentials (IMAP passwords, API keys, Telegram tokens) are loaded from `.env` files that are also gitignored.
- SSH keys in `5_mvp_bot/` (a removed sub-project) are gitignored via `*.key`.
- Before pushing to a public remote, **rewrite git history** to purge any data files that were committed in earlier commits (use `git filter-repo` — see the scrubbing note below).

### Scrubbing git history before publishing

```bash
# 1. Back up the repo first
cp -r NLP-project NLP-project.bak

# 2. Install git-filter-repo (once)
pip install git-filter-repo

# 3. Remove all CSV / JSONL / PKL files from history
git filter-repo --path-glob '*.csv' --invert-paths
git filter-repo --path-glob '*.jsonl' --invert-paths
git filter-repo --path-glob '*.pkl' --invert-paths

# 4. Force-push (rewrites all commit hashes — coordinate with any collaborators first)
git push origin --force --all
```

> After `git filter-repo` the remote tracking branch is gone; re-add the remote with `git remote add origin <url>` before pushing.

---

