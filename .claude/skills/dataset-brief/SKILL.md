---
name: dataset-brief
description: Generate an analyst-facing dataset_brief.yaml (canonical brief_version 1 structure) for a generated dataset or AI-handoff challenge folder. Use when the user asks to create a dataset brief, package a dataset as a challenge, document a CSV for another AI session, or build a challenge folder with brief + solutions.
---

# Generate a dataset_brief.yaml

Produce an analyst-facing brief for a dataset CSV, following the **canonical
structure** (brief_version 1) defined by `template.yaml` in this skill
directory. The brief is consumed cold by another AI session, so every fact must
be in the file — and every fact must be true of the actual CSV.

## Inputs to establish first

1. **The CSV to document** (usually an analyst-facing export at weekly user
   grain). If only a generator output dir is given (`output/<name>/`), the
   analyst CSV may still need to be built: weekly events + user dimensions
   merged, ground-truth `segment` dropped, experiment variants pivoted into
   `exp_<name>` columns.
2. **The generating config** (`configs/<name>.yaml`) — the source for
   known-context facts and for what must stay secret.
3. **Output location** — brief sits next to the CSV, with `solutions.yaml` as
   the sibling answer key.

## Structure (canonical — do not deviate silently)

Use `template.yaml` in this skill directory. Top-level keys, in this order:
`brief_version`, `answer_key`, `schema`, `analysis`, `data_quality`,
`dataset`, `task`, `grain`, `retention_metric`, `time_coverage`, `columns`,
`known_context`, `hints`.

If a dataset genuinely needs a structural deviation (key added/removed/renamed),
**explicitly tell the user what changed and why** — this is a standing request.
Bump `brief_version` only for structural changes.

## Spoiler policy

The brief contains only what a real-life data team would know:
- YES: that experiments ran (name, window, what was changed, target metric,
  assignment scope/ratio, variant column), releases and rough ship week, plan
  tiers, product beliefs ("page views driven by activity"), heavy-tail warning.
- NO: effect-size multipliers, shock existence/windows/magnitudes, hidden
  segment structure, retention formulas' parameters, anything from the
  `shocks:`/`segments:` config blocks. Those go in `solutions.yaml` only.

## Process

1. **Profile the CSV with pandas, don't assume**: columns and dtypes, unique
   values per categorical, min/max of `week`/`abs_week`, row/user counts,
   event_type list, `event_count` min, primary-key uniqueness.
2. **Fill the template** from the profile + the safe parts of the config.
   Derive the machine-readable layer (`schema`, `analysis`, `data_quality`)
   from the measured data. Document every column with type, description, and
   possible values. Keep the retention-metric definitions intact (activity =
   any row that week; cohort = first active week; classic N-week retention;
   churn = absence, returns allowed; WAU).
3. **Known gotchas to carry over**: version-like columns parse as float unless
   read as string (note it in the column description and `schema.dtypes`);
   blank CSV cells read as NaN in pandas, so `unenrolled_value: ""` needs
   `keep_default_na=False` to compare literally.
4. **User review checkpoint (required before writing the final brief).**
   The derived facts need validation, not approval — but the editorial
   sections are the user's call. Present the draft values of `task`,
   `known_context` (especially the spoiler boundary: what the analyst is told
   exists vs. must discover), `beliefs`, and `hints`, then ask the user
   (AskUserQuestion) whether they agree or want to add/remove anything.
   Datasets whose design is deception-based (e.g. notion_hard) ALWAYS need
   this question — how much context is "known" is the exercise's difficulty
   dial. Skip the checkpoint only if the user has already specified these
   sections for this dataset in the same conversation.
5. **Validate before finishing** (all from the written files):
   - both YAMLs parse with `yaml.safe_load`;
   - `schema.primary_key` is actually unique in the CSV;
   - every `analysis.segment_cols` and `variant_column` exists;
   - every `values:` list matches the CSV's actual uniques;
   - `data_quality.event_count_min` matches the data;
   - spot-check one strong prose claim (e.g. "every active week contains a
     page_created row") against the CSV;
   - confirm the brief leaks nothing from the `shocks:`/`segments:` blocks.
6. If building the full challenge package, also write `solutions.yaml`
   (provenance with config+seed, exact shock windows/multipliers/signatures,
   hidden structure, configured + measured experiment readouts, grader
   checklist) and verify the planted signals are visible from the handed CSV
   alone.
7. If the brief/CSV folder will be shared OUTSIDE this repo, copy the bundled
   `.claude/skills/data-quality-gate/` (in this repo) alongside it so the
   validation gate travels with the data.
