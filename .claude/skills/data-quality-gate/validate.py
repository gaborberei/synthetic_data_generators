#!/usr/bin/env python3
"""
Data-quality gate: validate a dataset CSV against its dataset_brief.yaml.

Usage:
    python validate.py [path/to/dataset_brief.yaml]   # default ./dataset_brief.yaml

Reads ONLY the brief and the CSV it describes. Checks the machine-readable
contract (columns, primary key, core action, value sets, counts, time
coverage) and summarises missing periods at the brief's granularity.

Exit code 0 = all contract checks pass; 1 = at least one failure.
Self-contained: needs only pandas + pyyaml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

FAILURES: list[str] = []


def check(name: str, ok: bool, expected=None, actual=None) -> None:
    if ok:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name} — expected {expected!r}, got {actual!r}")


def main(brief_path: str) -> int:
    brief_path = Path(brief_path)
    if not brief_path.exists():
        print(f"[FAIL] brief not found: {brief_path}")
        return 1
    brief = yaml.safe_load(brief_path.read_text()) or {}

    schema = brief.get("schema") or {}
    analysis = brief.get("analysis") or {}
    dq = brief.get("data_quality") or {}
    dataset = brief.get("dataset") or {}
    columns_doc = brief.get("columns") or {}
    experiments = (brief.get("known_context") or {}).get("experiments") or []

    csv_file = dataset.get("file")
    if not csv_file:
        print("[FAIL] brief has no dataset.file")
        return 1
    csv_path = brief_path.parent / csv_file
    if not csv_path.exists():
        print(f"[FAIL] CSV not found: {csv_path}")
        return 1

    dtypes = {k: str for k in (schema.get("dtypes") or {})}
    # keep_default_na=False so blank variant cells stay "" and compare literally
    df = pd.read_csv(csv_path, dtype=dtypes or None, keep_default_na=False, na_values=[])
    print(f"loaded {csv_path.name}: {len(df):,} rows × {len(df.columns)} columns\n")

    time_col = schema.get("time_column", "event_time")
    count_col = schema.get("count_column")
    granularity = schema.get("granularity", "event")

    # --- declared columns exist -------------------------------------------
    declared = [time_col, "user_id", "event_type"]
    declared += [count_col] if count_col else []
    declared += analysis.get("segment_cols") or []
    declared += [e["variant_column"] for e in experiments if e.get("variant_column")]
    missing = [c for c in declared if c not in df.columns]
    check("declared columns exist", not missing, declared, f"missing {missing}")

    # --- primary key unique -------------------------------------------------
    pk = schema.get("primary_key")
    if pk and all(c in df.columns for c in pk):
        dupes = int(df.duplicated(pk).sum())
        check(f"primary key {pk} unique", dupes == 0, "0 duplicates", f"{dupes:,} duplicates")

    # --- core action present -------------------------------------------------
    core = analysis.get("core_action")
    if core and "event_type" in df.columns:
        ets = set(df["event_type"].unique())
        check(f"core_action {core!r} present", core in ets, core, sorted(ets)[:10])

    # --- event_count minimum --------------------------------------------------
    if count_col and count_col in df.columns and dq.get("event_count_min") is not None:
        counts = pd.to_numeric(df[count_col], errors="coerce")
        check(f"{count_col} >= {dq['event_count_min']}",
              counts.min() >= dq["event_count_min"], f">= {dq['event_count_min']}", counts.min())

    # --- documented value sets match -------------------------------------------
    for col, spec in columns_doc.items():
        if col not in df.columns or not isinstance(spec, dict):
            continue
        values = spec.get("values")
        if isinstance(values, dict):
            expected = set(values.keys())
        elif isinstance(values, list):
            expected = {"" if v is None else str(v) for v in values}
        else:
            continue  # prose description — not enforceable
        actual = {str(v) for v in df[col].unique()}
        check(f"columns.{col}.values match data", actual == expected,
              sorted(expected), sorted(actual))

    # --- row / user counts -------------------------------------------------------
    if dataset.get("rows") is not None:
        check("dataset.rows", len(df) == int(dataset["rows"]), int(dataset["rows"]), len(df))
    if dataset.get("users") is not None:
        raw = str(dataset["users"]).replace("~", "").replace(",", "").strip()
        approx = "~" in str(dataset["users"])
        n_users = df["user_id"].nunique()
        target = int(raw)
        ok = abs(n_users - target) <= 0.05 * target if approx else n_users == target
        check(f"dataset.users ({'~5% tolerance' if approx else 'exact'})", ok, dataset["users"], n_users)

    # --- time coverage ---------------------------------------------------------------
    tc = brief.get("time_coverage") or {}
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col])
        if tc.get("start_week"):
            check("time_coverage.start_week", str(t.min().date()) == str(tc["start_week"]),
                  str(tc["start_week"]), str(t.min().date()))
        if tc.get("end_week"):
            check("time_coverage.end_week", str(t.max().date()) == str(tc["end_week"]),
                  str(tc["end_week"]), str(t.max().date()))
        if tc.get("n_weeks") and granularity == "weekly":
            check("time_coverage.n_weeks", t.nunique() == int(tc["n_weeks"]),
                  int(tc["n_weeks"]), t.nunique())

    # --- missing periods (info, judged against the declared sparsity) -----------------
    if time_col in df.columns:
        t = pd.to_datetime(df[time_col]).dt.normalize()  # event grain: count calendar days, not timestamps
        period_days = 7 if granularity == "weekly" else 1
        unit = "week" if granularity == "weekly" else "day"
        per = pd.DataFrame({"user_id": df["user_id"], "p": t})
        agg = per.groupby("user_id")["p"].agg(["min", "max", "nunique"])
        span = ((agg["max"] - agg["min"]).dt.days // period_days) + 1
        missing_periods = int((span - agg["nunique"]).sum())
        pct = missing_periods / int(span.sum()) if span.sum() else 0.0
        users_with_gaps = int(((span - agg["nunique"]) > 0).sum())
        print(f"\n[INFO] missing {unit}s between each user's first and last activity: "
              f"{missing_periods:,} ({pct:.1%}); users with gaps: {users_with_gaps:,}")
        sparsity = dq.get("sparsity")
        if sparsity in ("active_weeks_only", "active_periods_only"):
            print(f"[INFO] gaps are DECLARED ({sparsity}) — expected, not a failure. "
                  f"Zero-fill before any rolling-{unit} metric.")
        elif missing_periods > 0:
            print(f"[WARN] gaps present but the brief declares no sparsity — confirm whether "
                  f"this is by design, and zero-fill before rolling-{unit} metrics.")

    print()
    if FAILURES:
        print(f"VERDICT: FAIL — {len(FAILURES)} check(s) failed: {FAILURES}")
        return 1
    print("VERDICT: PASS — data matches the brief")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./dataset_brief.yaml"))
