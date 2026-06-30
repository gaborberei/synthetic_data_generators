from __future__ import annotations

"""Build the analyst handoff package from a generated dataset.

A single `generate` run produces exactly three artifacts per dataset, all
derived deterministically from the config + seed:

  1. the merged analyst CSV (the file to analyse),
  2. ``dataset_brief.yaml``  — analyst-facing context, NO spoilers,
  3. ``solutions.yaml``      — the answer key, including the tasks the analyst
     is expected to find.

The brief follows the canonical structure of
``.claude/skills/dataset-brief/template.yaml``; the data-quality gate
(``.claude/skills/data-quality-gate/validate.py``) validates the brief against
the CSV. Every value-bearing fact in the brief is derived from the actual
analyst DataFrame so the gate passes by construction; editorial prose comes
from the config's optional ``meta:`` block.
"""

import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd

# Per-user dimensions that are constant across a user's lifetime.
STATIC_DIMS = ["acquisition_channel", "country", "platform", "course"]

_DIM_DESC = {
    "acquisition_channel": "How the user was acquired (fixed at signup).",
    "country": "User country (fixed at signup).",
    "platform": "Primary platform (fixed at signup).",
    "course": "Course / track the user is on (fixed at signup).",
}


# ---------------------------------------------------------------------------
# 1. The analyst CSV
# ---------------------------------------------------------------------------
def build_analyst_frame(
    df: pd.DataFrame, assignments_df: pd.DataFrame | None, config: dict
) -> pd.DataFrame:
    """Merge the enriched event df into the analyst-facing weekly/daily frame.

    Drops the hidden ``segment``; aggregates to grain; merges static per-user
    dimensions; keeps ``app_version`` per (user, period); pivots experiment
    assignments into ``exp_<name>`` columns (blank = unenrolled).
    """
    grain = _output_grain(config)
    period = "date" if grain == "daily" else "week"

    agg = (
        df.groupby([period, "user_id", "event_type"]).size().reset_index(name="event_count")
    )

    dims_present = [c for c in STATIC_DIMS if c in df.columns]
    if dims_present:
        user_dims = df[["user_id", *dims_present]].drop_duplicates("user_id")
        agg = agg.merge(user_dims, on="user_id", how="left")

    if "app_version" in df.columns:
        ev = df.sort_values("event_time")
        ver = ev.groupby([period, "user_id"])["app_version"].last().reset_index()
        ver["app_version"] = ver["app_version"].astype(str)
        agg = agg.merge(ver, on=[period, "user_id"], how="left")

    if assignments_df is not None and not assignments_df.empty:
        wide = assignments_df.pivot_table(
            index="user_id", columns="experiment", values="group", aggfunc="first"
        )
        wide.columns = [f"exp_{c}" for c in wide.columns]
        agg = agg.merge(wide.reset_index(), on="user_id", how="left")
        for col in (c for c in agg.columns if c.startswith("exp_")):
            agg[col] = agg[col].fillna("")

    lead = [period, "user_id", "event_type", "event_count"]
    rest = [c for c in agg.columns if c not in lead]
    return agg[lead + rest].sort_values([period, "user_id", "event_type"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _period_col(grain: str) -> str:
    return "date" if grain == "daily" else "week"


def _output_grain(config: dict) -> str:
    """Aggregation/output grain, decoupled from the engine `grain`.

    Defaults to the engine grain, but a weekly-engine config can set
    `output_grain: daily` to emit daily rows (per-event day-of-week placement)
    while keeping the weekly simulation — see generator.py `day_of_week`.
    """
    return config.get("output_grain", config.get("grain", "weekly"))


def _exp_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("exp_")]


def _dim_cols(df: pd.DataFrame, period: str) -> list[str]:
    skip = {period, "user_id", "event_type", "event_count"}
    return [c for c in df.columns if c not in skip and not c.startswith("exp_")]


def _start_monday(config: dict) -> pd.Timestamp:
    start = config.get("global_settings", {}).get("start_date", "2024-01-01")
    d = pd.Timestamp(start)
    return d - pd.Timedelta(days=d.weekday())


def _abs_week(series: pd.Series, config: dict) -> pd.Series:
    return ((pd.to_datetime(series) - _start_monday(config)).dt.days // 7).astype(int)


def _retention_metric(grain: str) -> dict:
    if grain == "daily":
        return {
            "activity_definition": (
                "A user is ACTIVE on a day if they have a core-action row that day. "
                "notification_*/streak_* rows describe the day; the core action is the substantive activity."
            ),
            "cohort_definition": (
                "Signup cohort = first active day (min date), optionally rolled up to the weekly "
                "signup cohort. First active day = signup day."
            ),
            "retention_rate": (
                "Classic N-day (or N-week) retention: fraction of a cohort active exactly N "
                "days/weeks after signup; each (cohort, age) cell computed independently."
            ),
            "churn_and_resurrection": (
                "Churn = a run of idle days; users frequently return (lapse/resurrection is "
                "central), so model returns explicitly and compute cells independently."
            ),
            "weekly_active_users": (
                "DAU = distinct users active on a day; WAU = distinct users active in a calendar "
                "week. The daily/weekly ratio is itself a habit signal."
            ),
        }
    return {
        "activity_definition": "A user is ACTIVE in a week if they have at least one row (any event_type) that week.",
        "cohort_definition": "Signup cohort = first active week (min week). First active week = signup week.",
        "retention_rate": (
            "Classic N-week retention: fraction of a cohort active exactly N weeks after signup; "
            "each (cohort, age) cell computed independently."
        ),
        "churn_and_resurrection": (
            "Churn = absence in a week; users MAY return, so compute each retention cell "
            "independently rather than as a survival product."
        ),
        "weekly_active_users": "WAU = count of distinct user_ids with at least one row in the week.",
    }


def _grain_text(output_grain: str, engine_grain: str) -> str:
    """Row-format note (output grain) + analysis note (engine grain).

    Daily *output* from a weekly *engine* (day-of-week placement) is honest
    about there being NO day-over-day streak signal — activity is decided
    weekly and only scattered across days.
    """
    if output_grain == "daily":
        if engine_grain == "daily":
            return (
                "One row per (date, user_id, event_type) with event_count. A user appears on a day "
                "only if they did something that day; absence = an idle day. First observed date = "
                "signup day. Reconstruct each user's dense daily active calendar (zero-fill idle days) "
                "before computing streaks or rolling-window metrics — gaps are idle days, not missing data."
            )
        return (
            "One row per (date, user_id, event_type) with event_count. A user appears on a day only "
            "if they had that event that day; absence of a row means no events of that type that day. "
            "First observed date = signup day. Activity is decided weekly and distributed across "
            "working days, so the daily pattern carries day-of-week structure but NO day-over-day "
            "streak dependence; derive retention from the weekly pattern of activity."
        )
    return (
        "One row per (week, user_id, event_type) with event_count. A user appears in a week only "
        "if active that week; absence of a row means no events of that type. First observed week = "
        "signup week. Retention must be derived from the pattern of weeks in which a user appears."
    )


# ---------------------------------------------------------------------------
# 2. The brief
# ---------------------------------------------------------------------------
def build_brief(config: dict, df: pd.DataFrame, csv_name: str) -> dict:
    meta = config.get("meta", {}) or {}
    grain = config.get("grain", "weekly")   # engine grain -> analytical narrative
    out_grain = _output_grain(config)       # output grain -> aggregation / format
    period = _period_col(out_grain)
    base_event = config.get("base_event", "page_created")
    dims = _dim_cols(df, period)
    exp_cols = _exp_cols(df)
    experiments = config.get("experiments", []) or []

    # --- schema ---
    dtypes = {}
    if "app_version" in df.columns:
        dtypes["app_version"] = "str"
    for c in exp_cols:
        dtypes[c] = "str"
    schema = {
        "granularity": out_grain,
        "time_column": period,
        "count_column": "event_count",
        "primary_key": [period, "user_id", "event_type"],
    }
    if dtypes:
        schema["dtypes"] = dtypes

    analysis = {
        "core_action": base_event,
        "natural_frequency": out_grain,
        "segment_cols": dims,
    }
    if exp_cols:
        analysis["variant_columns"] = exp_cols

    data_quality = {
        "sparsity": "active_periods_only" if out_grain == "daily" else "active_weeks_only",
        "event_count_min": int(df["event_count"].min()),
    }

    dataset = {
        "file": csv_name,
        "product": meta.get("product", "A software product emitting user-action events."),
        "description": meta.get(
            "description",
            "One year of product-analytics events aggregated per user per period. Unknown "
            "incidents and/or experiments may have left marks in the data — none are named here.",
        ),
        "rows": int(len(df)),
        "users": int(df["user_id"].nunique()),
    }

    # --- columns (values derived from the data so the gate passes) ---
    t = pd.to_datetime(df[period])
    columns: dict = {
        period: {
            "type": "date",
            "description": ("Calendar day (YYYY-MM-DD)." if out_grain == "daily"
                            else "Monday date of the weekly period (YYYY-MM-DD)."),
            "values": f"{df[period].nunique()} dates from {t.min().date()} to {t.max().date()}",
        },
        "user_id": {
            "type": "string",
            "description": "Stable per-user identifier.",
            "values": f"{df['user_id'].nunique():,} distinct users",
        },
        "event_type": {
            "type": "string",
            "description": "What the user did; one row per (user, period, type) with event_count.",
            "values": {
                str(et): (meta.get("events", {}) or {}).get(str(et), f"The '{et}' event.")
                for et in sorted(df["event_type"].unique())
            },
        },
        "event_count": {
            "type": "int",
            "description": "Number of events of this type for this user in this period (>= 1).",
            "values": f"{int(df['event_count'].min())} .. {int(df['event_count'].max())}",
        },
    }
    for c in dims:
        if c == "app_version":
            columns[c] = {
                "type": "string",
                "description": (
                    "App version the user was on during that period (can change over time as "
                    "releases roll out). Stored as a string — read as str to avoid float mis-parse."
                ),
                "values": sorted(str(v) for v in df[c].unique()),
            }
        else:
            columns[c] = {
                "type": "string",
                "description": _DIM_DESC.get(c, f"{c} (fixed at signup)."),
                "values": sorted(str(v) for v in df[c].unique()),
            }
    for c in exp_cols:
        name = c[len("exp_"):]
        columns[c] = {
            "type": "string",
            "description": (
                f"Assignment for the '{name}' A/B test. Blank = not enrolled. Read the CSV with "
                "keep_default_na=False so blanks compare as \"\" not NaN."
            ),
            "values": sorted(str(v) for v in df[c].unique()),
        }

    # --- known_context (spoiler-safe) ---
    known: dict = {}
    if experiments:
        known["experiments"] = [
            {
                "name": e["name"],
                "variant_column": f"exp_{e['name']}",
                "control_label": "control",
                "unenrolled_value": "",
                "assignment": _assignment_text(e),
                "note": "Window, target metric, and effect size are NOT given — recover them from the data.",
            }
            for e in experiments
        ]
    beliefs = list(meta.get("beliefs", []) or [])
    if beliefs:
        known["beliefs"] = beliefs

    # --- time_coverage ---
    time_coverage = {"start_week": str(t.min().date()), "end_week": str(t.max().date())}
    if out_grain != "daily":
        time_coverage["n_weeks"] = int(df[period].nunique())

    brief = {
        "brief_version": 1,
        "answer_key": "solutions.yaml — do NOT open before your analysis is written down",
        "schema": schema,
        "analysis": analysis,
        "data_quality": data_quality,
        "dataset": dataset,
        "task": meta.get("task", _default_task(bool(experiments))),
        "grain": _grain_text(out_grain, grain),
        "retention_metric": _retention_metric(grain),
        "time_coverage": time_coverage,
        "columns": columns,
        "known_context": known,
        "hints": list(meta.get("hints", []) or _default_hints(grain, bool(exp_cols))),
    }
    return brief


def _assignment_text(e: dict) -> str:
    a = e.get("assignment", {}) or {}
    ratio = a.get("ratio", 0.5)
    scope = a.get("scope", "all_users")
    who = "new users only" if scope == "new_users" else "all users eligible"
    return f"{int(ratio*100)}/{100-int(ratio*100)} randomized; {who}."


def _default_task(has_exp: bool) -> list[str]:
    tasks = [
        "Find any anomalies/shocks: WHEN (weeks), WHICH metric(s), MAGNITUDE, plausible ROOT "
        "CAUSE. The number and type of shocks is not given.",
    ]
    if has_exp:
        tasks.append(
            "Read out each A/B experiment: recover its window from the data, measure the lift on "
            "the metric it plausibly targeted (with a CI), run an SRM check, give a ship/no-ship call."
        )
    tasks.append("Summarize overall product health: growth, cohort retention, engagement mix.")
    return tasks


def _default_hints(grain: str, has_exp: bool) -> list[str]:
    hints = [
        "Aggregate to a per-period active-user series and a new-signup series first.",
        "Separate calendar-time movements from cohort-age movements with a cohort (triangle/heatmap) view.",
    ]
    if has_exp:
        hints.append(
            "A variant column marks WHO is enrolled and their arm across all their periods — it does "
            "NOT mark the active window. Recover the window from when the treatment/control gap opens."
        )
    return hints


# ---------------------------------------------------------------------------
# 3. The solutions / answer key
# ---------------------------------------------------------------------------
def _srm_chi2(c: int, t: int) -> float:
    tot = c + t
    if tot == 0:
        return 0.0
    m = tot / 2
    return round(((c - m) ** 2 + (t - m) ** 2) / m, 2)


def _measure_experiment(df: pd.DataFrame, config: dict, e: dict, grain: str) -> dict:
    period = _period_col(grain)
    vcol = f"exp_{e['name']}"
    out: dict = {}
    if vcol not in df.columns:
        return {"note": "variant column not present in analyst CSV"}
    d = df.copy()
    d["_aw"] = _abs_week(d[period], config)
    w0, w1 = e["start_week"], e["end_week"]
    enrolled = d[d[vcol].isin(["control", "treatment"])]
    nc = int(enrolled[enrolled[vcol] == "control"]["user_id"].nunique())
    nt = int(enrolled[enrolled[vcol] == "treatment"]["user_id"].nunique())
    out["enrolled_users"] = {"control": nc, "treatment": nt}
    out["srm_chi2_df1"] = _srm_chi2(nc, nt)
    out["srm_flag"] = out["srm_chi2_df1"] > 3.84

    effects = e.get("effects", []) or [{}]
    eff = effects[0]
    win = enrolled[(enrolled["_aw"] >= w0) & (enrolled["_aw"] <= w1)]

    if eff.get("type") == "notification":
        # reactivation effect: shows up as more active days, not a higher open rate.
        base = config.get("base_event", "lesson_completed")
        act = win[win["event_type"] == base]
        days = act.groupby([vcol, "user_id"]).size().groupby(vcol).mean()
        c, t = days.get("control", np.nan), days.get("treatment", np.nan)
        sent = win[win["event_type"] == "notification_sent"].groupby(vcol)["event_count"].sum()
        opened = win[win["event_type"] == "notification_opened"].groupby(vcol)["event_count"].sum()
        out["target_metric"] = "reactivation -> active days / lesson volume (NOT open rate)"
        out["mean_active_days_per_user"] = {"control": round(float(c), 2), "treatment": round(float(t), 2)}
        out["active_days_lift"] = f"{(t - c) / c * 100:+.1f}%"
        if not sent.empty:
            out["open_rate"] = {
                "control": round(float(opened.get("control", 0) / sent.get("control", 1)), 3),
                "treatment": round(float(opened.get("treatment", 0) / sent.get("treatment", 1)), 3),
                "note": "unchanged across arms — the effect is post-open, not on opening",
            }
        lift = (t - c) / c * 100
    else:
        target = eff.get("target")
        out["target_metric"] = target
        tgt = win[win["event_type"] == target]
        means = tgt.groupby(vcol)["event_count"].mean()
        c, t = means.get("control", np.nan), means.get("treatment", np.nan)
        out["mean_per_active_period"] = {"control": round(float(c), 4), "treatment": round(float(t), 4)}
        lift = (t - c) / c * 100
        out["lift"] = f"{lift:+.1f}%"
        # novelty trap: report lift by exposure sub-window
        if any(tr.get("type") == "novelty" for tr in (e.get("traps") or [])):
            buckets = {}
            span = w1 - w0
            for lab, lo, hi in [("launch", w0, w0 + span // 3),
                                ("mid", w0 + span // 3 + 1, w0 + 2 * span // 3),
                                ("late", w0 + 2 * span // 3 + 1, w1)]:
                sub = tgt[(tgt["_aw"] >= lo) & (tgt["_aw"] <= hi)].groupby(vcol)["event_count"].mean()
                if "control" in sub and "treatment" in sub:
                    buckets[f"{lab}_wk{lo}_{hi}"] = f"{(sub['treatment'] - sub['control']) / sub['control'] * 100:+.1f}%"
            out["lift_by_exposure"] = buckets
            out["trap"] = "novelty decay — a single window average is misleading; plot lift vs weeks-since-exposure"

    out["verdict"] = (
        "SHIP — positive effect"
        + (", but mind the trap" if e.get("traps") else "")
        + ("; NOTE mild SRM imbalance" if out["srm_flag"] else "; SRM clean")
        if lift > 0 else "NO-SHIP — no positive effect"
    )
    return out


def build_solutions(config: dict, df: pd.DataFrame, seed, csv_name: str) -> dict:
    grain = config.get("grain", "weekly")   # engine grain
    out_grain = _output_grain(config)       # output grain (matches the analyst frame)
    experiments = config.get("experiments", []) or []
    shocks = config.get("shocks", []) or []
    ts = config.get("time_series", {}) or {}

    # --- tasks_to_find (the headline: what the analyst should discover) ---
    tasks: list[str] = []
    for s in shocks:
        scope = ""
        if s.get("where"):
            scope = " scoped to " + ", ".join(f"{k}={v}" for k, v in s["where"].items())
        if s.get("cohort_scope") == "new":
            scope += " (follows signup cohorts)"
        tgt = f" on {s['target_feature']}" if s.get("target_feature") else ""
        tasks.append(
            f"Detect the {s['type']} anomaly in abs weeks {s['start_week']}-{s['end_week']}{tgt}{scope}: "
            f"{s.get('description', '').strip()}"
        )
    for e in experiments:
        eff = (e.get("effects") or [{}])[0]
        target = eff.get("target", "reactivation")
        tasks.append(
            f"Read out the '{e['name']}' A/B test (target {target}, abs weeks "
            f"{e['start_week']}-{e['end_week']}): recover window, lift, SRM, ship decision."
        )
    if ts.get("seasonality"):
        tasks.append("Detect annual seasonality (peak at week 0 / January, trough mid-year).")
    for sp in ts.get("spikes", []) or []:
        tasks.append(f"Detect the one-off spike at week {sp['week']}: {sp.get('description', '')}.")
    if grain == "daily":
        tasks.append("Recover the habit curve (longer streak -> higher next-day return) and lapse/resurrection dynamics.")
    tasks.append("Summarize overall product health: growth, cohort retention, engagement mix; handle heavy-tailed activity.")

    # --- detailed shock answers ---
    shock_ans = []
    for s in shocks:
        entry = {
            "type": s["type"],
            "window_abs_weeks": [s["start_week"], s["end_week"]],
            "multiplier": s.get("multiplier"),
            "signature": s.get("description", "").strip(),
        }
        if s.get("target_feature"):
            entry["target"] = s["target_feature"]
        if s.get("where"):
            entry["scope"] = s["where"]
        if s.get("cohort_scope"):
            entry["cohort_scope"] = s["cohort_scope"]
        shock_ans.append(entry)

    # --- experiment answers (configured + measured) ---
    exp_ans = []
    for e in experiments:
        eff = (e.get("effects") or [{}])[0]
        exp_ans.append({
            "name": e["name"],
            "variant_column": f"exp_{e['name']}",
            "window_abs_weeks": [e["start_week"], e["end_week"]],
            "assignment": _assignment_text(e),
            "configured_effect": {k: v for k, v in eff.items()} | (
                {"traps": e["traps"]} if e.get("traps") else {}
            ),
            "measured_readout": _measure_experiment(df, config, e, out_grain),
        })

    # --- hidden structure ---
    hidden: dict = {}
    if config.get("segments"):
        hidden["segments"] = {
            name: {k: v for k, v in spec.items()} for name, spec in config["segments"].items()
        }
    ch = (config.get("dimensions", {}).get("acquisition_channel", {}) or {}).get("retention_multiplier")
    if ch:
        hidden["channel_retention_multipliers"] = ch
    if config.get("releases"):
        hidden["releases"] = config["releases"].get("versions")
    if config.get("plans"):
        hidden["plans"] = {
            "note": "Plans drive engagement but are NOT observable (no plan column/events).",
            "multipliers": config["plans"].get("multipliers"),
        }
    for block in ("time_series", "daily", "streak", "notifications"):
        if config.get(block):
            hidden[block] = config[block]

    grader = [t for t in tasks]
    grader.append("Uses a cohort (age) view, not just calendar series, to separate dated incidents from cohort-aging.")
    grader.append("Handles heavy-tailed engagement (median / log scale), not raw means.")

    g = config.get("global_settings", {})
    return {
        "provenance": {
            "config": config.get("dataset_name", csv_name),
            "seed": seed,
            "generator": f"CausalShockGenerator (generator.py), grain: {grain}",
            "analyst_csv": csv_name,
            "rows": int(len(df)),
            "users": int(df["user_id"].nunique()),
            "periods_simulated_weeks": g.get("periods_to_simulate"),
            "generated_at": _dt.date.today().isoformat(),
        },
        "tasks_to_find": tasks,
        "shocks": shock_ans,
        "experiments": exp_ans,
        "hidden_structure": hidden,
        "grader_checklist": grader,
    }
