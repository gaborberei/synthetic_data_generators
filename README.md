# Synthetic Data Generator — Causal Shocks

Generates realistic, event-level **product-analytics datasets** for a
Notion-like app. The data embeds known user segments, retention dynamics, a
causal link between actions and page views, per-user dimensions
(channel/country/platform/app version/plan), time-boxed "shocks", and A/B
experiments — all of which you can try to detect and diagnose in the data.

## Install

```bash
git clone https://github.com/gaborberei/synthetic_data_generators.git
cd synthetic_data_generators
pip install -r requirements.txt   # pandas, numpy, pyyaml (+ matplotlib/seaborn for analyze --save)
```

Generated data lands in `output/<config name>/`, which is **gitignored** — every
dataset is fully reproducible from its `configs/*.yaml` + the `--seed`, so the
(multi-GB) outputs are never committed. Regenerate any dataset with the
Quickstart commands below.

## Layout

| File / dir          | Purpose                                                       |
|---------------------|---------------------------------------------------------------|
| `configs/notion_easy.yaml` | **Easy** dataset: a Notion-like app with three obvious step-change shocks. |
| `configs/notion_easy_ab.yaml` | Easy dataset plus **two honest A/B tests**, scheduled so neither overlaps the other or any shock window. |
| `configs/notion_hard.yaml` | **Hard** dataset: every shock invites a wrong first diagnosis, plus two A/B tests (spoilers inside — don't share). |
| `configs/chess_medium.yaml` | **Medium**, single-event dataset: a chess.com-like app emitting only `game_played`, with Duolingo-style lifecycle retention and time-series variance (seasonality, AR(1) noise, a news spike). |
| `configs/duolingo_hard.yaml` | **Hard**, daily-grain dataset (`grain: daily`): a Duolingo-like language app whose core loop is the daily streak — habit-driven daily retention, streak freezes, reactivation notifications, and a notification A/B test. See "The daily grain" below. |
| `generator.py`      | `CausalShockGenerator` — turns a config into events.          |
| `challenge.py`      | Builds the analyst CSV + `dataset_brief.yaml` + `solutions.yaml` from a config. |
| `analysis.py`       | Config-driven dashboards (cohort heatmap, 6-panel, experiments); needs `--raw`. |
| `validate.py`       | Checks generated data against the config it came from; needs `--raw`. |
| `main.py`           | CLI: `generate` (3-file handoff; `--raw` for the rest) / `analyze` / `validate`. |

## Quickstart

```bash
pip install -r requirements.txt

# Each generate writes exactly THREE files (see Outputs below), reproducible
# from config + seed:
python main.py generate --config configs/notion_hard.yaml --seed 42    # hard
python main.py generate --config configs/notion_easy.yaml --seed 42    # easy
python main.py generate --config configs/chess_medium.yaml --seed 42   # medium, single-event
python main.py generate --config configs/duolingo_hard.yaml --seed 42  # hard, daily-grain

# Add --raw to ALSO dump the event log + ground_truth files (needed only by the
# `analyze` / `validate` commands below):
python main.py generate --config configs/notion_hard.yaml --seed 42 --raw

# Dashboards (PNGs written next to the data) and compliance checks — require --raw
python main.py analyze output/notion_hard --save
python main.py validate output/notion_hard
python main.py validate output/duolingo_hard        # runs the daily compliance suite
```

## Outputs (`output/<config name>/`)

By default `generate` writes exactly **three** files per dataset — the analyst
handoff package. The analyst CSV is prefixed with the config's `dataset_name`
(`notion_hard_*`, `notion_easy_*`):

| File | Audience | Contents |
|------|----------|----------|
| `<name>_analyst_weekly.csv` / `_analyst_daily.csv` | analysts | the file to analyse: one row per (week\|date, user, event_type) with `event_count`, plus per-user dimensions and one `exp_<name>` column per experiment (segment is never included) |
| `dataset_brief.yaml` | analysts | spoiler-free context: schema, grain, retention definitions, columns, the analysis tasks, and known context |
| `solutions.yaml` | answer key | `tasks_to_find` (what to discover) + exact shock windows/magnitudes, configured **and measured** experiment readouts, hidden structure, grader checklist |

The brief and solutions are generated in code from the config (the editorial
text lives in each config's `meta:` block). The brief validates against the
analyst CSV via the bundled data-quality gate
(`.claude/skills/data-quality-gate/validate.py`).

To run the dataset as a whodunit, hand out only `<name>_analyst_*.csv` +
`dataset_brief.yaml`; keep `solutions.yaml` as the answer key.

### `--raw` (debugging / dashboards)

`generate --raw` additionally writes the previous artifacts — the event-level
`<name>_events.csv`, the bare `<name>_events_weekly.csv` / `_events_daily.csv`,
`<name>_experiment_assignments.csv`, and `ground_truth_users.csv` /
`ground_truth_config.yaml` / `ground_truth_timeseries.csv`. These are what the
`analyze` and `validate` commands read; `--hide-truth` drops `segment` from the
raw event CSV.

## The model in one paragraph

Users arrive in weekly cohorts growing ~2.5%/week. Each user gets a segment
(**casual/core/power** — sets retention decay, weekly page creation, feature
usage), plus signup-time dimensions: acquisition channel (with real quality
differences — referral retains best, paid worst), country, platform, plan tier
(free/pro/team, with weekly upgrade/downgrade events), and an app version
(new releases are adopted gradually). All datasets carry all dimensions.
**Plan tier causally drives behaviour**: pro/team users create more pages
(×1.2 / ×1.35) and retain slightly better — on top of the compositional
correlation (power users skew paid at signup). A good sanity exercise: paid
vs free engagement compared *across* all users mixes composition with
causation; compared *within* one segment it isolates the real ~1.2× effect. **Page views are causal**: a weighted
function of each user's actions that week, so anything suppressing a feature
also suppresses views. **Shocks** are time-boxed multipliers on acquisition,
retention, activity, one feature, a plan transition, or *instrumentation*
(logged events silently dropped while true behaviour continues). A
`where: {column: [values]}` filter scopes any shock; `cohort_scope: new` makes
it follow signup cohorts for life. **Experiments** split users 50/50 by a
deterministic hash and multiply treatment behaviour inside their window —
optionally with traps (novelty decay, heterogeneous effects). All week indices
are absolute, 0-indexed simulation weeks.

## The datasets

**Easy (`notion_easy`)** — clearly visible on weekly charts: a marketing
pause (weeks 10–14), an AI outage (weeks 25–28, drags page views down
causally), and a competitor-launch churn bump (weeks 40+).

**Easy + experiments (`notion_easy_ab`)** — the same world
with two clean, trap-free A/B tests slotted into the shock-free gaps:
`share_button_redesign` (weeks 15–24, `shared` ×1.15, new users) and
`comment_prompts` (weeks 29–39, `commented` ×1.20, all users). Windows and
target features are disjoint, so the readouts never interact; variants are in
`*_experiment_assignments.csv` (join on `user_id`).

**Hard (`notion_hard`)** — each anomaly's obvious diagnosis is wrong;
see the YAML for full spoilers. Roughly: a data problem that looks like a
product problem, a price change whose churn is masked by a mix shift, a cohort
regression that looks like slow drift but has a findable root cause, plus one
honest A/B test and one whose launch lift overstates the steady state.

**Medium, single-event (`chess_medium`)** — one event type (`game_played`), so all the
signal is in the time series and the lifecycle. Uses two optional engine
modes: `transitions` swaps the plateau/decay retention curve for the Duolingo
growth-model state machine (weekly active probability depends on the gap
since last play and the previous bucket — CURR/NURR/RURR/SURR/REAC/RESU), and
`time_series` layers annual seasonality (chess peaks in winter), AR(1)
autocorrelated weekly noise, a one-off World Championship acquisition spike
(week 46), and a persistent per-player engagement multiplier (exported per user
as `activity_mult` in `ground_truth_users.csv`). The realized *weekly*
seasonality/noise/spike multipliers are exported separately to
`ground_truth_timeseries.csv` (with `--raw`) so `validate` can still check
cohort sizes exactly. Timestamps span all 7 days and skew toward evening play.

## The daily grain (`grain: daily`)

Every config above simulates **weekly**: a user is active-or-not per week. A
config with `grain: daily` instead runs a separate per-day engine
(`generator.py::_simulate_daily`) for products whose core loop is daily — the
streak. Users still *arrive* in weekly cohorts (`periods_to_simulate` is still
in weeks), but each is then stepped day by day:

- **Habit curve** (`daily.return`): a user active yesterday returns today with a
  probability that rises from `new_user` (a 1-day streak) toward `habit_max` as
  the streak grows (rate `habit_gain`). Longer streaks are stickier — the signal
  an analyst recovers.
- **Lapse / resurrection** (`daily.lapse`): once a streak breaks, the daily
  return probability is `base · decay^(idle-1)` (floored) — the daily analogue
  of the weekly REAC/RESU arrows. Keep `base` high: real retention is weekly-ish,
  so a learner who studies a few times a week misses days without churning.
- **Streak freezes** (`streak`): a freeze is earned every `freeze_earn_every`
  active days (capped at `max_freezes`) and auto-spent to save a streak across a
  missed day (`streak_freeze_used`). `streak_milestone` fires at the configured
  `milestones`; `streak_lost` on a break of a streak ≥ `emit_lost_min_streak`.
- **Notifications** (`notifications`): a reminder is sent to users idle between
  `idle_days_trigger` and `idle_days_max` days (`notification_sent`); opened with
  `open_rate` (`notification_opened`), an open multiplies the user's reactivation
  probability by `reactivation_boost`. A `type: notification` experiment effect
  scales that boost for the treatment group — a readable reactivation A/B test.
- **Per-segment stickiness** (`segments[*].retention_mult`): committed learners
  retain better *and* do more lessons/day, so early activity predicts retention.

Dimensions, shocks (`retention` / `activity`), experiments, `time_series`
seasonality/noise, and per-user variance all work exactly as in the weekly
engine. Output is aggregated to `*_events_daily.csv` (one row per user/day/
event_type with a count), which preserves each user's active-day calendar —
everything streak and activation analysis needs — without an event-level log.
`python main.py validate output/duolingo_hard` runs a daily-specific compliance suite
(habit curve, segment/channel retention ordering, streak/freeze/milestone
mechanics, notification window, and the reactivation A/B readout).

## Editing or extending

Change a config and re-run — user counts, growth, segments, dimensions,
releases, plans, shock timing/strength, and experiments are all YAML. New
shock scopes need no code: any user attribute (segment, dimension, app_version,
or plan) works in `where:`. After any
change, `python main.py validate output/<name>` confirms the data still
matches the config.

## License

MIT — see [`LICENSE`](./LICENSE).
