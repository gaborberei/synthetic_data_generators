---
name: data-quality-gate
description: Validate a dataset CSV against its dataset_brief.yaml BEFORE building any analysis on it. Use on the first load of a dataset in a session, before computing any metric, when the user asks to analyze a dataset that ships with a brief, or whenever the CSV or brief changes.
---

# Data-quality gate

## Purpose

Catch a broken, substituted, or mis-contracted data file before any analysis
is built on it — and decide, from the brief's grain, which analyses are even
meaningful on this data.

## When to use

The first time the dataset is loaded in a session. Re-run if the CSV or the
brief changes. Do not start metrics, charts, or modeling before the gate has
passed once.

## Instructions

Run the bundled checker (all logic lives in Python — do not re-derive checks
in prose). Run it from the repo root, passing the brief's path:

```bash
python .claude/skills/data-quality-gate/validate.py path/to/dataset_brief.yaml
```

It validates the brief↔CSV contract (declared columns, primary-key
uniqueness, core action, documented value sets, row/user counts, time
coverage) and summarises missing periods at the brief's granularity. All logic
is self-contained in the bundled `validate.py` — no extra packages required.

## Outcome policy

- **All PASS** → proceed; at most one short line ("data verified against the
  brief"). Do not paste the full report unless asked.
- **Any FAIL** → **HALT analysis.** Show which check failed with expected vs
  actual, and ask the user whether the data changed intentionally. If it did,
  the brief must be updated first — never silently adapt the analysis to data
  that contradicts its brief.

## Decision rules after a pass

- `schema.granularity: weekly` → day-granularity analyses are off-limits
  (within-week timing does not exist); never build a daily dense grid. Work in
  weekly windows and weekly retention.
- Event grain **with gaps** → any rolling-window or streak metric requires a
  zero-filled dense grid first (`rolling(7)` on sparse rows means "last 7
  active days", not "last 7 calendar days").

## Remember

- Verification ≠ conclusions: a passing gate says the file is intact and
  matches its contract, not that the data is unbiased.
- Gaps under a declared `data_quality.sparsity` are a quirk to handle, not a
  quality failure.
- **NEVER open `solutions.yaml`** (or any answer key next to the brief) — it
  spoils the exercise the dataset exists for.
