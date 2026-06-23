from __future__ import annotations

"""Unified analysis & visualization for generated datasets.

Everything is driven by the ground-truth files a `generate` run writes into
its output directory — shock windows, labels, and experiments come from
`ground_truth_config.yaml`, so the same code annotates any config correctly.

Reads only the small files (events_weekly.csv, ground truth), never the big
event-level CSV. matplotlib / seaborn are imported lazily.

Usage:  python main.py analyze output/sneaky_shocks --save
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_SHOCK_COLORS = ["red", "orange", "purple", "brown", "teal", "magenta"]


# --------------------------------------------------------------------- #
# Loading + shared tables
# --------------------------------------------------------------------- #
def load(data_dir: Path) -> dict:
    with open(data_dir / "ground_truth_config.yaml") as f:
        config = yaml.safe_load(f)
    prefix = config.get("dataset_name", data_dir.name)
    weekly = pd.read_csv(data_dir / f"{prefix}_events_weekly.csv", parse_dates=["week"])
    weekly["abs_week"] = ((weekly["week"] - weekly["week"].min()).dt.days // 7).astype(int)
    users = pd.read_csv(data_dir / "ground_truth_users.csv")
    assignments_path = data_dir / f"{prefix}_experiment_assignments.csv"
    assignments = (
        pd.read_csv(assignments_path) if assignments_path.exists() else None
    )
    return {
        "weekly": weekly,
        "config": config,
        "users": users,
        "assignments": assignments,
    }


def retention_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """Retention rate per (cohort_week, cohort_age) from user-week activity."""
    active = weekly[["user_id", "abs_week"]].drop_duplicates()
    cohort = active.groupby("user_id")["abs_week"].min().rename("cohort_week")
    active = active.merge(cohort, on="user_id")
    active["age"] = active["abs_week"] - active["cohort_week"]
    counts = active.groupby(["cohort_week", "age"])["user_id"].nunique().unstack()
    return counts.div(counts[0], axis=0)


def _week_shocks(config: dict) -> list[dict]:
    return [s for s in config.get("shocks", []) if s.get("cohort_scope") != "new"]


def _cohort_shocks(config: dict) -> list[dict]:
    return [s for s in config.get("shocks", []) if s.get("cohort_scope") == "new"]


def _shade_shocks(ax, shocks: list[dict]) -> None:
    for i, shock in enumerate(shocks):
        ax.axvspan(
            shock["start_week"], shock["end_week"],
            color=_SHOCK_COLORS[i % len(_SHOCK_COLORS)], alpha=0.10,
            label=shock.get("description", shock["type"]),
        )


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #
def plot_heatmap(data: dict, save_path: str | None = None) -> None:
    """Classic cohort retention heatmap (signup cohort x weeks since signup).

    Cohort-scoped shocks appear as horizontal breaks; calendar-week retention
    shocks as diagonal bands (cohort + age = shock week)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    rates = retention_table(data["weekly"])
    fig, ax = plt.subplots(figsize=(16, 11))
    sns.heatmap(rates, vmin=0, vmax=1, cmap="RdYlGn", ax=ax,
                cbar_kws={"label": "retention rate"})
    for shock in _cohort_shocks(data["config"]):
        ax.axhline(shock["start_week"], color="black", linewidth=1.5, linestyle="--")
        ax.text(rates.shape[1] - 1, shock["start_week"] - 0.6,
                shock.get("description", "") + " ", ha="right",
                fontsize=10, fontweight="bold")
    ax.set_title("Cohort retention heatmap")
    ax.set_xlabel("Weeks since signup (cohort age)")
    ax.set_ylabel("Signup cohort (week)")
    fig.tight_layout()
    _output(fig, save_path)


def plot_dashboard(data: dict, save_path: str | None = None) -> None:
    """6 panels covering every shock type: topline, per-event-type volume,
    signups, per-segment users & intensity, retention by cohort."""
    import matplotlib.pyplot as plt

    weekly, config, users = data["weekly"], data["config"], data["users"]
    week_shocks = _week_shocks(config)
    by_type = (
        weekly.groupby(["abs_week", "event_type"])["event_count"]
        .sum().unstack(fill_value=0)
    )
    d = weekly.merge(users[["user_id", "segment"]], on="user_id")
    seg_palette = {"casual": "gray", "core": "#1f77b4", "power": "#d62728"}

    fig, axes = plt.subplots(2, 3, figsize=(22, 11))

    # 1. Topline volume with every calendar shock window shaded.
    ax = axes[0, 0]
    ax.plot(by_type.index, by_type.sum(axis=1), color="black", linewidth=2)
    _shade_shocks(ax, week_shocks)
    ax.set_title("1. Topline: total weekly events")
    ax.set_xlabel("Week"); ax.set_ylabel("Events")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. Volume by event type, indexed (feature + instrumentation shocks).
    ax = axes[0, 1]
    base_week = 5
    for col in by_type.columns:
        base = by_type[col].loc[base_week]
        # Skip rare event types (plan changes): indexing a tiny base just
        # produces noise spikes that drown the informative lines.
        if base >= 50:
            ax.plot(by_type.index, by_type[col] / base * 100, linewidth=1.6, label=col)
    _shade_shocks(ax, week_shocks)
    ax.set_title(f"2. Events by type (week {base_week} = 100)")
    ax.set_xlabel("Week"); ax.set_ylabel("Index")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. New signups per week (acquisition shocks).
    ax = axes[0, 2]
    cohort_sizes = users.groupby("cohort_week")["user_id"].nunique()
    ax.plot(cohort_sizes.index, cohort_sizes.values, marker="o", markersize=3,
            color="#1f77b4")
    _shade_shocks(ax, week_shocks)
    ax.set_title("3. New signups per week")
    ax.set_xlabel("Cohort week"); ax.set_ylabel("New users")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. Active users by segment (retention shocks, mix shifts).
    ax = axes[1, 0]
    act = d.groupby(["abs_week", "segment"])["user_id"].nunique().unstack()
    for seg in act.columns:
        ax.plot(act.index, act[seg], linewidth=2, label=seg,
                color=seg_palette.get(seg))
    _shade_shocks(ax, week_shocks)
    ax.set_title("4. Weekly active users by segment")
    ax.set_xlabel("Week"); ax.set_ylabel("Active users")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 5. Events per active user by segment (activity shocks, survivor effects).
    ax = axes[1, 1]
    vol = d.groupby(["abs_week", "segment"])["event_count"].sum().unstack()
    epu = vol / act
    for seg in epu.columns:
        ax.plot(epu.index, epu[seg], linewidth=2, label=seg,
                color=seg_palette.get(seg))
    _shade_shocks(ax, week_shocks)
    ax.set_title("5. Events per active user by segment")
    ax.set_xlabel("Week"); ax.set_ylabel("Events / active user")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 6. Week-4 retention by signup cohort (cohort-scoped shocks).
    ax = axes[1, 2]
    rates = retention_table(weekly)
    if 4 in rates.columns:
        ret4 = rates[4].dropna()
        ax.plot(ret4.index, ret4.values, color="purple", linewidth=2.5,
                marker="o", markersize=4)
    for shock in _cohort_shocks(config):
        ax.axvline(shock["start_week"], color="black", linestyle="--",
                   linewidth=1.5, label=shock.get("description", ""))
    ax.set_title('6. "Month 1" retention by signup cohort')
    ax.set_xlabel("Signup cohort (week)"); ax.set_ylabel("Retention @ week 4")
    ax.set_ylim(0, 1)
    if _cohort_shocks(config):
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _output(fig, save_path)


def experiment_lift(data: dict, exp: dict) -> pd.DataFrame:
    """Observed treatment/control lift on the target metric, by weeks of
    exposure.

    Metric: target events *per page created* — the quantity the treatment
    multiplier acts on. Normalizing by pages removes the (heavy-tailed)
    activity variance that a per-user metric would carry."""
    weekly, assignments = data["weekly"], data["assignments"]
    target = exp["effects"][0]["target"]
    a = assignments[assignments["experiment"] == exp["name"]]

    sub = weekly[
        weekly["event_type"].isin([target, "page_created"])
        & weekly["abs_week"].between(exp["start_week"], exp["end_week"])
    ]
    sub = sub.merge(a, on="user_id")
    sub["weeks_exposed"] = sub["abs_week"] - sub["first_exposed_week"]
    per_user = (
        sub.groupby(["weeks_exposed", "group", "user_id", "event_type"])
        ["event_count"].sum().unstack(fill_value=0)
    )
    cells = (
        per_user.groupby(level=["weeks_exposed", "group"])
        .apply(_ratio_stats, target=target).unstack("group")
    )
    lift = cells[("rate", "treatment")] / cells[("rate", "control")]
    se_log = np.sqrt(
        cells[("var", "treatment")] / cells[("rate", "treatment")] ** 2
        + cells[("var", "control")] / cells[("rate", "control")] ** 2
    )
    out = pd.DataFrame(
        {
            "lift": lift,
            "lo": lift * np.exp(-1.96 * se_log),
            "hi": lift * np.exp(1.96 * se_log),
            "n": cells["pages"].sum(axis=1),
        }
    )
    return out[out["n"] >= 1000]  # drop noisy tail cells


def _ratio_stats(g: pd.DataFrame, target: str) -> pd.Series:
    """Ratio-of-sums rate with a cluster-robust (per-user) variance.

    Treating pages as independent trials understates uncertainty: pages
    cluster within users, and heavy users (power segment) dominate the
    denominator, so arm-composition luck moves the pooled rate. The standard
    fix is the delta-method variance over user-level sums:
        Var(R) = sum_u (x_u - R * n_u)^2 / (sum_u n_u)^2
    """
    x, n = g[target], g["page_created"]
    rate = x.sum() / n.sum()
    var = ((x - rate * n) ** 2).sum() / n.sum() ** 2
    return pd.Series({"rate": rate, "var": var, "pages": n.sum()})


def _p_two_sided(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def experiment_summary(data: dict, exp: dict) -> dict:
    """Pooled readout over the whole window: lift, 95% CI (cluster-robust at
    the user level — see `_ratio_stats`), p-value vs lift=1, and a
    sample-ratio-mismatch (SRM) test on the assignment counts.
    """
    weekly, assignments = data["weekly"], data["assignments"]
    target = exp["effects"][0]["target"]
    a = assignments[assignments["experiment"] == exp["name"]]

    sub = weekly[
        weekly["event_type"].isin([target, "page_created"])
        & weekly["abs_week"].between(exp["start_week"], exp["end_week"])
    ].merge(a, on="user_id")
    per_user = (
        sub.groupby(["group", "user_id", "event_type"])["event_count"]
        .sum().unstack(fill_value=0)
    )
    stats = per_user.groupby(level="group").apply(_ratio_stats, target=target)
    lift = stats.at["treatment", "rate"] / stats.at["control", "rate"]
    se_log = math.sqrt(
        stats.at["treatment", "var"] / stats.at["treatment", "rate"] ** 2
        + stats.at["control", "var"] / stats.at["control", "rate"] ** 2
    )
    z = math.log(lift) / se_log

    # SRM: chi-square (1 df) on observed vs configured assignment split.
    ratio = exp["assignment"]["ratio"]
    n_t = (a["group"] == "treatment").sum()
    n_all = len(a)
    z_srm = (n_t - n_all * ratio) / math.sqrt(n_all * ratio * (1 - ratio))

    return {
        "lift": lift,
        "lo": lift * math.exp(-1.96 * se_log),
        "hi": lift * math.exp(1.96 * se_log),
        "se_log": se_log,
        "p": _p_two_sided(z),
        "srm_p": _p_two_sided(z_srm),
        "n_users": n_all,
        "n_pages": int(stats["pages"].sum()),
    }


def plot_experiments(data: dict, save_path: str | None = None) -> None:
    """One panel per experiment: observed lift vs weeks of exposure."""
    import matplotlib.pyplot as plt

    experiments = data["config"].get("experiments", [])
    if not experiments or data["assignments"] is None:
        return
    fig, axes = plt.subplots(1, len(experiments), figsize=(9 * len(experiments), 6),
                             squeeze=False)
    for ax, exp in zip(axes[0], experiments):
        lift = experiment_lift(data, exp)
        s = experiment_summary(data, exp)
        m = exp["effects"][0]["multiplier"]
        ax.fill_between(lift.index, lift["lo"], lift["hi"], color="#1f77b4",
                        alpha=0.18, label="95% CI")
        ax.plot(lift.index, lift["lift"], marker="o", linewidth=2, color="#1f77b4",
                label="observed lift")
        ax.axhline(m, color="green", linestyle="--", label=f"configured x{m}")
        ax.axhline(1.0, color="gray", linewidth=1)
        p_txt = "p < 0.001" if s["p"] < 0.001 else f"p = {s['p']:.3f}"
        srm_txt = f"SRM p = {s['srm_p']:.2f}" + (" !!" if s["srm_p"] < 0.001 else " ok")
        ax.text(0.02, 0.02,
                f"pooled lift {s['lift']:.3f} (95% CI {s['lo']:.3f}-{s['hi']:.3f}), {p_txt}\n"
                f"{s['n_users']:,} users, {s['n_pages']:,} pages, {srm_txt}",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        target = exp["effects"][0]["target"]
        ax.set_title(f"{exp['name']}\n{target} per page created, treatment / control")
        ax.set_xlabel("Weeks of exposure"); ax.set_ylabel("Lift")
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _output(fig, save_path)


def render_all(data_dir: Path, save: bool = False) -> None:
    data = load(data_dir)
    sp = (lambda name: str(data_dir / name)) if save else (lambda name: None)
    plot_heatmap(data, save_path=sp("retention_heatmap.png"))
    plot_dashboard(data, save_path=sp("dashboard.png"))
    plot_experiments(data, save_path=sp("experiments.png"))


def _output(fig, save_path: str | None) -> None:
    import matplotlib.pyplot as plt

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  saved plot -> {save_path}")
        plt.close(fig)
    else:
        plt.show()
