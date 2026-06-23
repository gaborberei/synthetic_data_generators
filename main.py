from __future__ import annotations

"""CLI for the synthetic data generator.

Examples
--------
    # Generate the advanced dataset (reproducible), hide the answer key
    python main.py generate --config configs/sneaky_shocks.yaml --seed 42 --hide-truth

    # Generate the beginner dataset
    python main.py generate --config configs/causal_shocks.yaml --seed 42

    # Render the analysis dashboards for a generated dataset
    python main.py analyze output/sneaky_shocks --save

    # Check that the generated data matches the config it was built from
    python main.py validate output/sneaky_shocks
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
    config = load_config(args.config)
    out_dir = Path(args.output_dir or Path("output") / Path(args.config).stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating events...")
    generator = CausalShockGenerator(config, seed=args.seed)
    df = generator.run()
    print(f"  {len(df):,} events for {df['user_id'].nunique():,} users")

    # Ground truth: full config copy + per-user truth (segment, dimensions,
    # plan/version at signup). Analysts get only the prefixed files.
    prefix = config.get("dataset_name", Path(args.config).stem)
    with open(out_dir / "ground_truth_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    generator.users_df.to_csv(out_dir / "ground_truth_users.csv", index=False)
    if generator.timeseries_df is not None:
        # Realized weekly multipliers (seasonality * noise * spikes) — part of
        # the answer key; validate uses them to check cohort sizes exactly.
        generator.timeseries_df.to_csv(
            out_dir / "ground_truth_timeseries.csv", index=False
        )
    if not generator.assignments_df.empty:
        generator.assignments_df.to_csv(
            out_dir / f"{prefix}_experiment_assignments.csv", index=False
        )
    print(f"  wrote ground truth + config to {out_dir}/")

    public_df = df.drop(columns=["segment"]) if args.hide_truth else df
    public_df.to_csv(out_dir / f"{prefix}_events.csv", index=False)
    print(f"  wrote {out_dir / f'{prefix}_events.csv'}")
    # Daily-grain configs get a daily aggregate; everything else, the weekly one.
    if config.get("grain") == "daily":
        grouped_daily(public_df).to_csv(
            out_dir / f"{prefix}_events_daily.csv", index=False
        )
        print(f"  wrote {out_dir / f'{prefix}_events_daily.csv'}")
    else:
        grouped_weekly(public_df).to_csv(
            out_dir / f"{prefix}_events_weekly.csv", index=False
        )
        print(f"  wrote {out_dir / f'{prefix}_events_weekly.csv'}")


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
    p.add_argument("--config", default="configs/sneaky_shocks.yaml")
    p.add_argument("--output-dir", default=None,
                   help="default: output/<config name>")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (reproducible)")
    p.add_argument("--hide-truth", action="store_true",
                   help="drop the segment column from the events CSVs "
                        "(it stays in ground_truth_users.csv)")
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
