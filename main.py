from __future__ import annotations

"""CLI for the synthetic data generator.

Examples
--------
    # Generate the hard dataset (reproducible), hide the answer key
    python main.py generate --config configs/notion_hard.yaml --seed 42 --hide-truth

    # Generate the easy dataset
    python main.py generate --config configs/notion_easy.yaml --seed 42

    # Render the analysis dashboards for a generated dataset
    python main.py analyze output/notion_hard --save

    # Check that the generated data matches the config it was built from
    python main.py validate output/notion_hard
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from generator import CausalShockGenerator


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def grouped_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (week, user, event_type) with an event_count."""
    return (
        df.groupby(["week", "user_id", "event_type"])
        .size()
        .reset_index(name="event_count")
    )


def grouped_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (date, user, event_type) with an event_count.

    The right aggregate for daily-grain (streak/habit) datasets: it preserves
    each user's active-day calendar — everything streak and activation analysis
    needs — without the bulk of an event-level log.
    """
    return (
        df.groupby(["date", "user_id", "event_type"])
        .size()
        .reset_index(name="event_count")
    )


def cmd_generate(args: argparse.Namespace) -> None:
    from challenge import build_analyst_frame, build_brief, build_solutions

    config = load_config(args.config)
    out_dir = Path(args.output_dir or Path("output") / Path(args.config).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = config.get("dataset_name", Path(args.config).stem)
    # Output grain (filename suffix + raw dump) may differ from the engine grain:
    # a weekly-engine config can set output_grain: daily (day-of-week placement).
    grain = config.get("output_grain", config.get("grain", "weekly"))

    print("Generating events...")
    generator = CausalShockGenerator(config, seed=args.seed)
    df = generator.run()
    print(f"  {len(df):,} events for {df['user_id'].nunique():,} users")

    # --- the three handoff artifacts (the default output) --------------------
    analyst = build_analyst_frame(df, generator.assignments_df, config)
    suffix = "daily" if grain == "daily" else "weekly"
    csv_name = f"{prefix}_analyst_{suffix}.csv"
    analyst.to_csv(out_dir / csv_name, index=False)
    print(f"  wrote {out_dir / csv_name}  ({len(analyst):,} rows)")

    with open(out_dir / "dataset_brief.yaml", "w") as f:
        yaml.safe_dump(build_brief(config, analyst, csv_name), f, sort_keys=False)
    with open(out_dir / "solutions.yaml", "w") as f:
        yaml.safe_dump(
            build_solutions(config, analyst, args.seed, csv_name), f, sort_keys=False
        )
    print(f"  wrote dataset_brief.yaml + solutions.yaml to {out_dir}/")

    if not args.raw:
        return

    # --- raw dump (only with --raw): event log + ground truth ---------------
    # Lets the `analyze` and `validate` commands run against this dataset.
    with open(out_dir / "ground_truth_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    generator.users_df.to_csv(out_dir / "ground_truth_users.csv", index=False)
    if generator.timeseries_df is not None:
        generator.timeseries_df.to_csv(out_dir / "ground_truth_timeseries.csv", index=False)
    if not generator.assignments_df.empty:
        generator.assignments_df.to_csv(
            out_dir / f"{prefix}_experiment_assignments.csv", index=False
        )
    public_df = df.drop(columns=["segment"]) if args.hide_truth else df
    public_df.to_csv(out_dir / f"{prefix}_events.csv", index=False)
    if grain == "daily":
        grouped_daily(public_df).to_csv(out_dir / f"{prefix}_events_daily.csv", index=False)
    else:
        grouped_weekly(public_df).to_csv(out_dir / f"{prefix}_events_weekly.csv", index=False)
    print(f"  --raw: wrote event log + ground truth to {out_dir}/")


def cmd_analyze(args: argparse.Namespace) -> None:
    from analysis import render_all

    render_all(Path(args.data_dir), save=args.save)


def cmd_validate(args: argparse.Namespace) -> None:
    from validate import run_checks

    raise SystemExit(run_checks(Path(args.data_dir)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic product-analytics generator")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate a dataset from a config")
    p.add_argument("--config", default="configs/notion_hard.yaml")
    p.add_argument("--output-dir", default=None,
                   help="default: output/<config name>")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (reproducible)")
    p.add_argument("--raw", action="store_true",
                   help="also dump the raw event log + ground_truth files "
                        "(needed by `analyze` / `validate`); default writes only "
                        "the analyst CSV + dataset_brief.yaml + solutions.yaml")
    p.add_argument("--hide-truth", action="store_true",
                   help="with --raw, drop the segment column from the raw events "
                        "CSV (the analyst CSV never contains segment)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("analyze", help="render dashboards for a generated dataset")
    p.add_argument("data_dir", help="a dataset directory under output/")
    p.add_argument("--save", action="store_true",
                   help="save PNGs into the dataset directory instead of showing")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("validate", help="check generated data against its config")
    p.add_argument("data_dir", help="a dataset directory under output/")
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
