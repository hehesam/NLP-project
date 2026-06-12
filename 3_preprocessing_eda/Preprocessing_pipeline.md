---
title: Preprocessing Pipeline
tags:
  - nlp
  - preprocessing
  - dataset
  - multilingual
aliases:
  - Preprocessing
  - Frozen Dataset
created: 2026-04-29
---

# Preprocessing Pipeline

> [!info] Companion to
> [[Preprocessing_pipeline.ipynb]] — this note documents the *why* and the *take‑aways*. The notebook is the *what* (executable code).

## TL;DR

The pipeline turns the manually-adjudicated **`gold_dataset.csv`** into a frozen experimental artifact, **`frozen_dataset.csv`**, ready to be loaded by every downstream notebook (EDA, TF‑IDF, transformers, etc.).

It performs four things, in order:

1. Loads the gold CSV without dropping anything.
2. Runs an integrity audit (rows, missing / invalid labels, duplicates, empty fields).
3. Removes only **justified** rows, with each removal explicitly logged.
4. Applies *light* multilingual cleaning and emits three text variants.

> [!success] Result on the current run
> **919 rows in → 919 rows out**, no rows had to be dropped. All labels valid, no duplicate ids, no rows with both subject and body empty.

---

## Inputs and outputs

| Kind   | Path                                | Shape   | Notes                                                                 |
| ------ | ----------------------------------- | ------- | --------------------------------------------------------------------- |
| Input  | `Preprocessing/gold_dataset.csv`    | 919 × 5 | Columns: `id, subject, body_plain, topic, priority`                   |
| Output | `Preprocessing/frozen_dataset.csv`  | 919 × 8 | Adds `text_subject`, `text_body`, `text_subject_body`                 |

> [!note] Naming note
> The original brief mentions `body_clean`, but the gold dataset ships with `body_plain`. We treat `body_plain` as the body source and produce `text_body` from it. No content change.

---

## Label spaces

Two independent supervised tasks — see [[EDA]] for distributions.

- **Task A — Topic** (5 classes, lowercase): `administrative`, `course-exam`, `event`, `deadline-action`, `advertisement`.
- **Task B — Priority** (3 classes, capitalized): `High`, `Medium`, `Low`.

Both label sets are validated against canonical sets (`VALID_TOPICS`, `VALID_PRIORITIES`) defined in cell 2 of the notebook; rows outside them would be dropped (and reported), but on the current run none exist.

---

## Step-by-step walkthrough

### 1 — Imports and configuration

Pure stdlib + `numpy` and `pandas`. Paths and canonical label sets are defined once so the rest of the notebook stays declarative.

### 2 — Load the gold dataset

The CSV is read as-is, **nothing is dropped yet**. This is so the audit in step 3 can describe the input faithfully.

### 3 — Sanity / integrity checks

A pure audit cell. It computes everything required by the spec, with no side-effects:

- number of rows
- missing topic / priority labels
- invalid topic / priority labels (outside the canonical sets)
- duplicate `id`s
- empty subjects (using a NaN-or-whitespace-only check, not just `isna()`)
- empty bodies (same definition)
- rows where **both** subject and body are missing/empty

> [!example] Current-run audit
> ```
> rows                          : 919
> missing_topic                 : 0
> missing_priority              : 0
> invalid_topic                 : 0
> invalid_priority              : 0
> duplicate_ids                 : 0
> empty_subjects                : 0
> empty_bodies                  : 45
> both_subject_and_body_empty   : 0
> ```

### 4 — Justified row removals

The brief is strict: *do not silently drop rows*. The notebook only allows these removals, each with an explicit reason and counter:

| Reason                              | Drop? | Why                                          |
| ----------------------------------- | :---: | -------------------------------------------- |
| Missing or invalid `topic`          |  ✅    | Unusable for Task A                          |
| Missing or invalid `priority`       |  ✅    | Unusable for Task B                          |
| Duplicate `id` (keep first)         |  ✅    | Preserve a clean primary key                 |
| Both subject **and** body empty     |  ✅    | No textual signal at all                     |
| Only body empty (subject present)   |  ❌    | Subject alone is enough                      |
| Only subject empty (body present)   |  ❌    | Body alone is enough                         |

> [!success] Current run
> 0 rows removed.

### 5 — Light multilingual text preprocessing

The cleaning is intentionally **conservative** because the corpus is mixed English / Italian and downstream models (TF‑IDF in particular) need both vocabularies intact.

| What we **do**                                                  | What we **don't**                              |
| --------------------------------------------------------------- | ---------------------------------------------- |
| Coerce to `str` safely (NaN → `""`)                             | Remove punctuation                             |
| Lowercase                                                        | Remove digits                                  |
| Strip simple HTML leftovers (`<...>`, `&nbsp;`, `&amp;`, …)     | Stem                                           |
| Replace emails with the token `EMAIL`                           | Lemmatize                                      |
| Replace URLs (http / https / www / ftp) with `URL`              | Remove English stopwords                       |
| Collapse repeated whitespace into single spaces                 | Strip Italian stopwords                        |
| Trim                                                             | Detect / translate language                    |

The order matters: emails are replaced **before** URLs to avoid `info@unige.it` accidentally matching a URL pattern.

### 6 — Build `text_subject` and `text_body`

`clean_text` is applied to `subject` and `body_plain`. The cell also prints char-length descriptive statistics so we can spot anomalies.

### 7 — Build `text_subject_body`

Concatenation with explicit segment markers, so a model can still tell where the subject ends and the body begins:

```text
[SUBJECT] <cleaned subject> [BODY] <cleaned body>
```

If a side is empty after cleaning, its marker is omitted (we do not want spurious `[BODY]` tokens in the vocabulary).

### 8 — Final post-cleaning audit

The same integrity properties are recomputed on the cleaned dataframe so we have a closing report next to the opening one. Everything should be 0 except `rows`.

### 9 — Freeze and save

Persist the final 8-column dataframe to `frozen_dataset.csv`:

```text
id, subject, body_plain, topic, priority, text_subject, text_body, text_subject_body
```

This is the artifact every downstream notebook loads — never re-read `gold_dataset.csv`.

---

## Take-aways

> [!tip] Things to remember when building on top of this artifact
>
> - **Use `text_subject_body` for most baselines.** It carries the strongest signal and the `[SUBJECT]` / `[BODY]` markers are intentionally kept distinguishable.
> - **Keep punctuation and digits.** They are signal here (course codes like `90535`, exam numbers, dates), not noise.
> - **Do not strip stopwords blindly.** A common English stopword list deletes Italian content words and vice versa. If you want to filter, build a *bilingual* list.
> - **`URL` and `EMAIL` are tokens.** Treat them as features (their frequency differs sharply between `advertisement` and `deadline-action`).
> - **45 rows have empty bodies.** Their `text_body` is `""`, but `text_subject_body` is non-empty for all rows — so `text_subject_body` is the safest default input.

> [!warning] Reproducibility
> The notebook is deterministic — same input → same output. Re-running it overwrites `frozen_dataset.csv`. If you need to compare experiments across changes to the cleaning logic, snapshot the file under a different name first.

---

## Related

- [[EDA]] — exploratory analysis of the frozen artifact (run this *next*).
- [[Preprocessing_pipeline.ipynb]] — the executable notebook this note describes.
