---
title: Exploratory Data Analysis
tags:
  - nlp
  - eda
  - dataset
  - leakage
  - class-balance
aliases:
  - EDA
created: 2026-04-29
---

# Exploratory Data Analysis — Frozen Email Dataset

> [!info] Companion to
> [[eda.ipynb]] — this note documents the analysis sections, the actual numbers from the latest run, and the take‑aways that should drive the modeling notebook. The notebook is the source of truth; this note is the readable summary.

## TL;DR

> [!summary] Headline findings
> - **Shape**: 919 rows × 8 columns. No NaN columns; 45 rows have empty bodies but non-empty subjects, so `text_subject_body` is non-empty for **every** row.
> - **Topic balance** (Task A): mild imbalance, `course-exam` ≈ 29% to `advertisement` ≈ 12%, ratio **≈ 2.5×**.
> - **Priority balance** (Task B): nearly uniform (`Low` 37% / `Medium` 34% / `High` 30%), ratio **≈ 1.24×**.
> - **Topic ↔ Priority coupling**: Cramér's V = **0.556** — *strong* dependence. Some topics almost determine the priority.
> - **Templates**: **22.4%** of rows live in groups that share an identical cleaned subject; **3.8%** share an identical cleaned body. This is the most actionable leakage risk.
> - **Languages**: ~58% Italian, ~39% English, ~3% mixed by a function-word proxy.

---

## What this notebook covers

Twelve sections, top-to-bottom, mirroring the analysis flow:

1. Imports and configuration.
2. Load + shape + dtypes.
3. Missing values (raw NaN and "logically empty").
4. **Topic distribution bar chart** (Task A).
5. **Priority distribution bar chart** (Task B).
6. **Topic × Priority heatmaps** (counts and row-normalized %).
   1. Cramér's V — single-number measure of dependence.
7. **Text length distributions** — chars and words for the three text fields.
   1. Length per class — boxplots and per-class medians.
8. **Top repeated subjects** — counts and label-purity per template.
9. Body-level near-duplicates.
10. Vocabulary fingerprint per topic — top tokens via smoothed log-odds.
11. Quick language proxy (IT / EN function-word hits).
12. Discussion — class balance and possible leakage.

---

## 1 — Shape and dtypes

```text
Shape : 919 rows  x  8 columns
Memory: 6.81 MB
```

All columns are `object` (text). The eight columns are: `id`, `subject`, `body_plain`, `topic`, `priority`, `text_subject`, `text_body`, `text_subject_body`.

## 2 — Missing values

| Column              | NaN | Empty-or-NaN | %    |
| ------------------- | :-: | :----------: | :--: |
| `id`                |  0  |      0       | 0.0  |
| `subject`           |  0  |      0       | 0.0  |
| `body_plain`        | 45  |      45      | 4.9  |
| `topic`             |  0  |      0       | 0.0  |
| `priority`          |  0  |      0       | 0.0  |
| `text_subject`      |  0  |      0       | 0.0  |
| `text_body`         | 45  |      45      | 4.9  |
| `text_subject_body` |  0  |      0       | 0.0  |

> [!tip] Why we use `text_subject_body` as the default model input
> It is the only text field that is non-empty for 100% of rows.

## 3 — Topic distribution (Task A)

| Topic              | Count | %    |
| ------------------ | :---: | :--: |
| `course-exam`      |  267  | 29.1 |
| `deadline-action`  |  197  | 21.4 |
| `event`            |  189  | 20.6 |
| `administrative`   |  159  | 17.3 |
| `advertisement`    |  107  | 11.6 |

Imbalance ratio (max / min) = **2.5×**. Manageable with `class_weight="balanced"` or stratified sampling — no resampling needed at this scale.

## 4 — Priority distribution (Task B)

| Priority   | Count | %    |
| ---------- | :---: | :--: |
| `Low`      |  338  | 36.8 |
| `Medium`   |  309  | 33.6 |
| `High`     |  272  | 29.6 |

Imbalance ratio = **1.24×**. Effectively balanced.

## 5 — Topic × Priority joint distribution

```text
priority         High  Medium  Low
topic
administrative     22      70   67
course-exam       142     117    8
event               1      36  152
deadline-action   107      80   10
advertisement       0       6  101
```

> [!warning] The two tasks are **not** independent
> - `event` is overwhelmingly `Low` (152 / 189 ≈ **80%**).
> - `advertisement` is **never** `High` and almost always `Low` (101 / 107 ≈ **94%**).
> - `course-exam` and `deadline-action` skew `High` (53% and 54% respectively).

### Cramér's V

$$
V = 0.556
$$

Per the rule of thumb (`<0.1` negligible, `0.1–0.3` weak, `0.3–0.5` moderate, `>0.5` strong), this is a **strong** association. Practically: knowing the topic gives you a substantial prior on the priority. Treat the two tasks as **correlated**, not independent — this matters for error analysis and for any joint multitask model.

## 6 — Text length distributions

5-number summary plus P90 / P99 (in characters and whitespace tokens):

| Stat   | subj_chars | subj_words | body_chars | body_words | sb_chars | sb_words |
| ------ | :--------: | :--------: | :--------: | :--------: | :------: | :------: |
| count  |     919    |     919    |     919    |     919    |    919   |    919   |
| mean   |    62.1    |     8.7    |   1254.0   |    189.1   |  1333.8  |   199.7  |
| std    |    28.1    |     4.1    |   1148.8   |    184.7   |  1154.3  |   185.4  |
| min    |      6     |      2     |      0     |      0     |    40    |     6    |
| 25%    |      42    |      6     |     578    |     82     |    645   |    90    |
| 50%    |      59    |      8     |     916    |    137     |   1000   |    148   |
| 75%    |      79    |     11     |    1595    |    234     |   1680   |   246.5  |
| 90%    |      97    |     14     |    2598    |    386     |   2682   |    399   |
| 99%    |    158.5   |    21.8    |    5307    |    820     |   5401   |  829.6   |
| max    |     171    |     27     |    13789   |    2382    |   13878  |   2395   |

> [!tip] Practical implications
> - For TF‑IDF baselines, the body dominates length — use `text_subject_body` and let `min_df` / `max_df` tame vocabulary size.
> - For transformer baselines, the **median** subject+body fits comfortably in 512 tokens, but the long tail (P99 ≈ 830 words ≈ 1100+ subword tokens) will be truncated. Truncate from the **end**, since the subject and the lead of the body usually carry the topic.

### Length per class

Median words in `text_subject_body` per class:

| Topic              | Median words |
| ------------------ | :----------: |
| `administrative`   |      148     |
| `course-exam`      |      101     |
| `event`            |      188     |
| `deadline-action`  |      177     |
| `advertisement`    |      180     |

| Priority | Median words |
| -------- | :----------: |
| `High`   |      113     |
| `Medium` |      184     |
| `Low`    |      156     |

> [!warning] Length is a partial shortcut
> `course-exam` is markedly **shorter** than the rest, and `High`-priority emails are markedly shorter than `Medium`/`Low`. A model can pick up class identity from length alone — useful, but fragile on out-of-distribution mail.

## 7 — Top repeated subjects

> [!example] Repeat statistics
> ```
> Unique non-empty subjects   : 775
> Subjects appearing >1 times : 62
> Emails living in such groups: 206 (22.4% of all rows)
> ```

The 15 most frequent cleaned subjects:

| # | text_subject (truncated)                                                      | count | dominant topic   | topic purity | dominant priority | priority purity |
| - | ----------------------------------------------------------------------------- | :---: | ---------------- | :----------: | ----------------- | :-------------: |
| 1 | `[studenti] mercoledi' scienza`                                               |  25   | `event`          |     1.00     | `Low`             |      1.00       |
| 2 | `re: about the auk`                                                           |  10   | `course-exam`    |     0.80     | `Medium`          |      0.60       |
| 3 | `prenotazione esame`                                                          |   8   | `course-exam`    |     1.00     | `High`            |      0.75       |
| 4 | `nuove scelte personalizzate solo per te!`                                    |   8   | `advertisement`  |     1.00     | `Low`             |      1.00       |
| 5 | `universita' degli studi di genova - registrazione esame`                     |   7   | `course-exam`    |     0.86     | `Medium`          |      1.00       |
| 6 | `re: auk activity: drl`                                                       |   6   | `course-exam`    |     1.00     | `High`            |      0.50       |
| 7 | `scopri le nostre ultime offerte ora!`                                        |   6   | `advertisement`  |     1.00     | `Low`             |      1.00       |
| 8 | `esito esame`                                                                 |   6   | `course-exam`    |     1.00     | `High`            |      1.00       |
| 9 | `aulaweb2024 80412: ri: exam 16/7`                                            |   5   | `course-exam`    |     1.00     | `Medium`          |      0.80       |
|10 | `non hai trovato il prodotto high-tech adatto a te?`                          |   4   | `advertisement`  |     1.00     | `Low`             |      1.00       |
|11 | `aulaweb2024 info-10852: re: short course: preparing effective scientif…`     |   4   | `course-exam`    |     1.00     | `Medium`          |      1.00       |
|12 | `[studenticorsilaurea] promemoria scadenza 15 dicembre: rilevazione del…`     |   4   | `deadline-action`|     1.00     | `Medium`          |      1.00       |
|13 | `il prezzo del prodotto che ti interessa è diminuito!`                        |   4   | `advertisement`  |     1.00     | `Low`             |      1.00       |
|14 | `[studenti] comitato potenziamento attività sportive`                         |   3   | `administrative` |     1.00     | …                 |       …         |
|15 | `elevate la vostra esperienza di gioco - acquistate ora.`                     |   3   | `advertisement`  |     1.00     | `Low`             |      1.00       |

> [!danger] This is the single biggest leakage risk in the corpus
> Most templated subjects have **topic purity = 1.00** — the same subject always carries the same label. With a random row-level split, train and test will share template instances and metrics will be optimistic.

## 8 — Body-level near-duplicates

```text
Non-empty bodies           : 874
Unique non-empty bodies    : 852
Bodies with >1 occurrences : 13
Emails living in body dups : 35 (3.8% of rows)
```

The duplicates are mass mailings — promotional emails (Lenovo offers), survey reminders, scholarship calls, and standard administrative notices. Smaller magnitude than subject-level repetition but stricter (whole-document copies).

## 9 — Vocabulary fingerprint per topic

Top 10 unigrams per topic by smoothed log-odds:

$$
\text{score}(w, c) = \log\frac{p(w \mid c) + \epsilon}{p(w \mid \neg c) + \epsilon}
$$

| Rank | administrative   | course-exam | event           | deadline-action | advertisement |
| :--: | ---------------- | ----------- | --------------- | --------------- | ------------- |
|  #1  | quorum           | 114470      | acquario        | soddisfazione   | lenovo        |
|  #2  | seggi            | lesson      | mercoledi       | rilevazione     | prodotti      |
|  #3  | ranking          | 108871      | cineversity     | scholarships    | promozioni    |
|  #4  | elettorale       | 90535       | auditorium      | questionnaire   | windows       |
|  #5  | rappresentanza   | commento    | fb              | survey          | thinkvision   |
|  #6  | cnsu             | mock        | lucia           | compilazione    | smarter       |
|  #7  | collegio         | today's     | edutainment     | universale      | logo          |
|  #8  | programmazione   | contents    | pusillo         | esprimere       | thinksystem   |
|  #9  | aggiornata       | seems       | dell'acquario   | anonima         | thinkstation  |
| #10  | 0102099232       | yes         | proiezione      | esprimere       | thinkedge     |

> [!success] Encouraging signal
> Even a quick log-odds eyeball test produces clean, semantically coherent fingerprints per class. A simple linear model on TF‑IDF features will already be a strong baseline.

> [!warning] But also visible: signature & sender shortcuts
> `lenovo`, `thinkvision`, `thinksystem`, `thinkstation`, `thinkedge` are *brand* terms — the model partly learns the *sender*, not the content. Course codes (`114470`, `108871`, `90535`) are similar: they perfectly identify a course, but they also act as a near-id, which can hurt generalization to new courses.

## 10 — Language proxy

A coarse function-word check (no language detector loaded):

```text
it         57.8 %
en         39.0 %
mixed       2.6 %
unknown     0.7 %
```

Per-topic language mix (% within topic):

| Topic              | en   | it   | mixed | unknown |
| ------------------ | :--: | :--: | :---: | :-----: |
| `administrative`   | 37.7 | 60.4 |  1.9  |   0.0   |
| `course-exam`      | 70.8 | 25.5 |  3.7  |   0.0   |
| `event`            | 18.0 | 81.5 |  0.5  |   0.0   |
| `deadline-action`  | 31.5 | 63.5 |  5.1  |   0.0   |
| `advertisement`    | 12.1 | 82.2 |  0.0  |   5.6   |

> [!warning] Language is partially correlated with topic
> `course-exam` skews English (course materials, exam announcements), while `event` and `advertisement` skew Italian. A monolingual TF‑IDF will silently rank by language. **Multilingual** features (or a multilingual transformer like `xlm-roberta-base`) are the right baseline.

---

## 11 — Discussion: class balance and possible leakage

### Class balance — interpretation

- **Topic (Task A)**: ratio ≈ 2.5×, dominated by `course-exam`. Stratified splits + `class_weight="balanced"` are sufficient. Report **macro‑F1** plus per-class F1; *do not* report plain accuracy as the headline metric.
- **Priority (Task B)**: near-uniform. Accuracy is more meaningful here, but macro‑F1 stays the safer headline metric to enable apples-to-apples comparisons across experiments.

### Stratification

With ~107 examples in the smallest topic class, a non-stratified split risks too-small per-class test counts. **Stratification on `topic` is mandatory** for Task A; for Task B you can stratify on `priority` directly, or jointly on `(topic, priority)`.

### Leakage risks, ranked

| # | Risk                                       | Magnitude        | Suggested mitigation                                                                                       |
| - | ------------------------------------------ | ---------------- | ---------------------------------------------------------------------------------------------------------- |
| 1 | **Templated subjects** with constant label | 22.4% of rows    | Group-aware split (`GroupShuffleSplit` keyed on `text_subject` or a normalized template hash); also report a "no-templates" ablation. |
| 2 | **Body duplicates**                         | 3.8% of rows     | Deduplicate by body hash for one evaluation pass; compare to the row-level metrics.                         |
| 3 | **Cross-task coupling** (Cramér's V = 0.56) | strong           | Not train↔test leakage, but worth flagging: a Task-A win can mechanically lift Task-B numbers. Report both.|
| 4 | **Length as a shortcut**                    | systematic       | Add a length-only baseline; the gap between TF‑IDF and length-only tells you how much real signal exists.   |
| 5 | **Signatures / sender giveaways**           | visible in tokens| Robustness check: re-train on bodies with the last *k* lines stripped; expect *some* drop, accept it.       |

### Recommendations to carry into the modeling notebook

> [!tip] Carry-forward checklist
> - Use **stratified** splits (and consider **grouped-by-subject** splits for the templated-traffic ablation).
> - Report **macro‑F1** plus per-class F1; do not headline accuracy.
> - Run a **dedup ablation** (drop body duplicates) and compare; the delta is a credible upper bound on template-induced optimism.
> - Add a **length-only** and a **lang-proxy-only** baseline; treat any TF‑IDF win over both as the "real" gain from text content.
> - Prefer **multilingual features** (`TfidfVectorizer` with no English stopwords, or a multilingual transformer) given the IT/EN mix.

---

## Related

- [[Preprocessing]] — how the frozen artifact analyzed here was built.
- [[eda.ipynb]] — the executable notebook this note describes.
- [[Preprocessing_pipeline.ipynb]] — the upstream pipeline.
