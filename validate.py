from __future__ import annotations

"""Compliance checks: does a generated dataset match the config it came from?

Every check compares an observed statistic in the data against the value the
config promises, with tolerances sized for ~10k users. Checks for features a
config doesn't use (dimensions, plans, experiments) are skipped automatically.

Usage:  python main.py validate output/sneaky_shocks      (exit 1 on failure)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis import experiment_lift, experiment_summary, load, retention_table

_results: list[tuple[bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    _results.append((ok, name))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def _norm_sf(z: float) -> float:
    """Two-sided p-value for a z-score (no scipy dependency)."""
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _expected_cohort_sizes(config: dict, ts_acq: pd.Series | None) -> dict[int, int]:
    """base * (1+growth)^w * acquisition multipliers — shared weekly/daily."""
    g = config["global_settings"]
    acq = {
        w: np.prod([
            s["multiplier"] for s in config.get("shocks", [])
            if s["type"] == "acquisition" and s["start_week"] <= w <= s["end_week"]
        ]) * (1.0 if ts_acq is None else ts_acq[w])
        for w in range(g["periods_to_simulate"])
    }
    return {w: int(g["base_users"] * (1 + g["growth_rate"]) ** w * acq[w]) for w in acq}


def _markov_retention_curve(t: dict, ages: int, mult: float = 1.0) -> np.ndarray:
    """Expected P(active) by cohort age under the Duolingo transition engine.

    Evolves the state distribution week by week: active as New / Current /
    Reactivated / Resurrected, or inactive with a gap of 1-4 weeks (REAC
    window) / 5+ (RESU window). `mult` scales every transition probability
    (capped at 0.98, exactly like the generator).
    """
    def p(x: float) -> float:
        return min(x * mult, 0.98)

    a_new, a_cur, a_rea, a_res = 1.0, 0.0, 0.0, 0.0
    gap = [0.0] * 5  # inactive for 1, 2, 3, 4, 5+ weeks
    curve = [1.0]
    for _ in range(1, ages):
        active = a_new + a_cur + a_rea + a_res
        cur = (a_new * p(t["NURR"]) + a_cur * p(t["CURR"])
               + a_rea * p(t["RURR"]) + a_res * p(t["SURR"]))
        rea = (gap[0] + gap[1] + gap[2]) * p(t["REAC"])
        res = (gap[3] + gap[4]) * p(t["RESU"])
        gap = [active - cur,
               gap[0] * (1 - p(t["REAC"])),
               gap[1] * (1 - p(t["REAC"])),
               gap[2] * (1 - p(t["REAC"])),
               (gap[3] + gap[4]) * (1 - p(t["RESU"]))]
        a_new, a_cur, a_rea, a_res = 0.0, cur, rea, res
        curve.append(cur + rea + res)
    return np.array(curve)


def _clean_weeks(config: dict, feature: str) -> list[int]:
    """Weeks where nothing (shock or experiment) touches a feature's rate."""
    periods = config["global_settings"]["periods_to_simulate"]
    dirty: set[int] = set()
    for s in config.get("shocks", []):
        if s["type"] in ("feature", "instrumentation") and s.get("target_feature") == feature:
            dirty.update(range(s["start_week"], s["end_week"] + 1))
    for e in config.get("experiments", []):
        if any(eff.get("target") == feature for eff in e.get("effects", [])):
            dirty.update(range(e["start_week"], e["end_week"] + 1))
    return [w for w in range(periods) if w not in dirty]


def _streaks(active_days: list[np.ndarray]):
    """Yield (streak_length_today, returned_tomorrow) for each active day.

    `active_days` is a list of per-user sorted arrays of 0-indexed day numbers.
    """
    for days in active_days:
        s = set(days.tolist())
        streak = 1
        for i, day in enumerate(days):
            streak = streak + 1 if i > 0 and day - days[i - 1] == 1 else 1
            yield streak, (day + 1) in s


def run_checks_daily(data_dir: Path, config: dict) -> int:
    """Compliance checks for a `grain: daily` streak/habit dataset.

    The weekly suite's per-page / causal / plateau checks don't apply; instead
    we verify the daily habit curve, the streak/notification mechanics, and the
    segment / channel / experiment effects that drive retention.
    """
    _results.clear()
    g = config["global_settings"]
    segments = config["segments"]
    prefix = config.get("dataset_name", data_dir.name)
    daily = pd.read_csv(data_dir / f"{prefix}_events_daily.csv", parse_dates=["date"])
    users = pd.read_csv(data_dir / "ground_truth_users.csv")
    ts_path = data_dir / "ground_truth_timeseries.csv"
    ts_acq = (
        pd.read_csv(ts_path).set_index("week")["acquisition_mult"]
        if ts_path.exists() else None
    )
    daily["daynum"] = (daily["date"].values.astype("datetime64[D]")
                       - np.datetime64(g["start_date"])).astype(int)
    daily["abs_week"] = daily["daynum"] // 7
    base = config.get("base_event", "lesson_completed")
    les = daily[daily["event_type"] == base]
    active_by_user = les.groupby("user_id")["daynum"].apply(lambda s: np.array(sorted(set(s))))
    lesson_days = {u: set(d.tolist()) for u, d in active_by_user.items()}
    # A streak freeze bridges a non-lesson day: the engine keeps the streak and
    # resets days_idle to 0, so for streak/idle reconstruction a frozen day
    # counts as "covered" exactly like a lesson day.
    fz_days = daily[daily["event_type"] == "streak_freeze_used"].groupby("user_id")["daynum"].apply(set).to_dict()
    covered = {u: lesson_days[u] | fz_days.get(u, set()) for u in lesson_days}

    # --- 1. Cohort growth ---
    expected = _expected_cohort_sizes(config, ts_acq)
    sizes = users.groupby("cohort_week")["user_id"].nunique()
    mismatch = [w for w in expected if sizes.get(w, 0) != expected[w]]
    check("cohort growth", not mismatch,
          f"{len(expected) - len(mismatch)}/{len(expected)} weeks exact")

    # --- 2. Segment mix ---
    mix = users["segment"].value_counts(normalize=True)
    worst = max(abs(mix.get(s, 0) - segments[s]["prob"]) for s in segments)
    check("segment mix", worst < 0.03, f"max deviation {worst:.3f} (tol 0.03)")

    # --- 3. Dimension mixes ---
    for dim, spec in config.get("dimensions", {}).items():
        mix = users[dim].value_counts(normalize=True)
        worst = max(abs(mix.get(v, 0) - p) for v, p in spec["values"].items())
        check(f"dimension mix: {dim}", worst < 0.03, f"max deviation {worst:.3f} (tol 0.03)")

    # --- 4. Habit curve: next-day return rises with streak length ---
    ret_by_streak: dict[int, list] = {}
    for streak, returned in _streaks(list(active_by_user)):
        ret_by_streak.setdefault(min(streak, 10), []).append(returned)
    r1 = np.mean(ret_by_streak[1])
    r10 = np.mean(ret_by_streak[10])
    new_user = config["daily"]["return"]["new_user"]
    check("habit curve: streak-1 ≈ new_user", abs(r1 - new_user) < 0.06,
          f"streak-1 next-day return {r1:.3f} vs configured new_user {new_user}")
    check("habit curve: longer streaks stickier", r10 > r1 + 0.10,
          f"streak-1 {r1:.3f} -> streak-10 {r10:.3f}")

    # --- 5. Segment retention ordering follows retention_mult ---
    seg_of = users.set_index("user_id")["segment"].to_dict()
    coh = users.set_index("user_id")["cohort_week"].to_dict()
    age12 = les.assign(age=lambda x: x["abs_week"] - x["user_id"].map(coh))
    ret12 = age12[age12["age"] == 12].groupby("user_id").size().index
    ret12_set = set(ret12)
    seg_ret = {}
    for sg in segments:
        uu = [u for u in users["user_id"] if seg_of[u] == sg and coh[u] <= 39]
        seg_ret[sg] = np.mean([u in ret12_set for u in uu]) if uu else 0.0
    ordered = sorted(segments, key=lambda s: segments[s].get("retention_mult", 1.0))
    ok = all(seg_ret[ordered[i]] <= seg_ret[ordered[i + 1]] for i in range(len(ordered) - 1))
    check("segment retention ordering", ok,
          "wk-12 retention " + ", ".join(f"{s}={seg_ret[s]:.3f}" for s in ordered))

    # --- 6. Channel retention ordering (referral > organic > social > paid) ---
    chan = config.get("dimensions", {}).get("acquisition_channel", {})
    if chan.get("retention_multiplier"):
        ch = users.set_index("user_id")["acquisition_channel"].to_dict()
        n_all = users[users["cohort_week"] <= 39].groupby("acquisition_channel")["user_id"].nunique()
        r = {c: 0 for c in n_all.index}
        for u in ret12_set:
            if coh.get(u, 99) <= 39 and u in ch:
                r[ch[u]] = r.get(ch[u], 0) + 1
        rr = {c: r[c] / n_all[c] for c in n_all.index}
        ok = rr["referral"] > rr["organic"] > rr["social"] > rr["paid_search"]
        check("channel retention ordering", ok,
              "wk-12 retention " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(rr.items())))

    # --- 7. Notifications: idle-window respected + reactivation A/B lift ---
    notif = config.get("notifications", {})
    if notif:
        # notification_sent should only land on idle users within the configured
        # window. The engine's days_idle counter = (calendar days since the last
        # covered lesson/frozen day) - 1, and it notifies when days_idle is in
        # [trigger, max], so the calendar gap is in [trigger+1, max+1].
        sent = daily[daily["event_type"] == "notification_sent"][["user_id", "daynum"]]
        trig, mx = notif.get("idle_days_trigger", 1), notif.get("idle_days_max")
        bad = 0
        for u, dn in sent.head(5000).itertuples(index=False):
            s = covered.get(u, set())
            gap = next((k for k in range(1, (mx or 60) + 3) if (dn - k) in s), None)
            idle = None if gap is None else gap - 1
            if idle is None or idle < trig or (mx is not None and idle > mx):
                bad += 1
        check("notification idle window", bad == 0,
              f"{bad}/5000 sampled notifications outside idle [{trig},{mx}]")

    exp = next((e for e in config.get("experiments", [])
                if any(eff["type"] == "notification" for eff in e.get("effects", []))), None)
    asg_path = data_dir / f"{prefix}_experiment_assignments.csv"
    if exp and asg_path.exists():
        asg = pd.read_csv(asg_path)
        asg = asg[asg["experiment"] == exp["name"]][["user_id", "group"]]
        # SRM on the 50/50 split
        gc = asg["group"].value_counts()
        ratio = exp["assignment"]["ratio"]
        n = gc.sum()
        z_srm = (gc.get("treatment", 0) - n * ratio) / math.sqrt(n * ratio * (1 - ratio))
        check(f"no SRM: {exp['name']}", _norm_sf(z_srm) > 0.001,
              f"treatment share {gc.get('treatment', 0) / n:.3f} (n={n:,})")
        # reactivation within 2 days of an opened notification, by group
        active_sets = {u: set(d.tolist()) for u, d in active_by_user.items()}
        op = daily[daily["event_type"] == "notification_opened"][["user_id", "daynum"]].merge(
            asg, on="user_id")
        op["react"] = [
            any((dn + k) in active_sets.get(u, set()) for k in (1, 2))
            for u, dn in zip(op["user_id"], op["daynum"])
        ]
        gp = op.groupby("group")["react"].agg(["mean", "count"])
        pt, nt = gp.loc["treatment", "mean"], gp.loc["treatment", "count"]
        pc, nc = gp.loc["control", "mean"], gp.loc["control", "count"]
        se = math.sqrt(pt * (1 - pt) / nt + pc * (1 - pc) / nc)
        z = (pt - pc) / se
        check(f"reactivation lift: {exp['name']}", pt > pc and _norm_sf(z) < 0.05,
              f"treatment {pt:.3f} vs control {pc:.3f} "
              f"({(pt / pc - 1) * 100:+.1f}%, p={_norm_sf(z):.2g})")

    # --- 8. Streak mechanics: freezes bridge gaps, milestones at thresholds ---
    fz = daily[daily["event_type"] == "streak_freeze_used"][["user_id", "daynum"]]
    if len(fz):
        active_sets = {u: set(d.tolist()) for u, d in active_by_user.items()}
        bridged = 0
        for u, dn in fz.head(500).itertuples(index=False):
            s = active_sets.get(u, set())
            if dn not in s and (dn - 1) in s and (dn + 1) in s:
                bridged += 1
        check("streak freeze bridges a gap", bridged > 0.7 * min(len(fz), 500),
              f"{bridged}/{min(len(fz), 500)} sampled freezes sit between two active days")

    mile = set(config.get("streak", {}).get("milestones", []))
    if mile:
        # A streak_milestone fires when the streak (count of lesson days in the
        # current covered run, freezes bridging gaps) hits a configured length.
        ms = daily[daily["event_type"] == "streak_milestone"][["user_id", "daynum"]]
        ok_ms = 0
        for u, dn in ms.head(500).itertuples(index=False):
            cov, les_u = covered.get(u, set()), lesson_days.get(u, set())
            length = k = 0
            while (dn - k) in cov:        # walk back through the covered run
                if (dn - k) in les_u:     # count only lesson days toward the streak
                    length += 1
                k += 1
            if length in mile:
                ok_ms += 1
        check("milestones at configured lengths", ok_ms == min(len(ms), 500),
              f"{ok_ms}/{min(len(ms), 500)} milestone events at a configured streak length")

    failed = [name for ok, name in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed"
          + (f" — FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def run_checks(data_dir: Path) -> int:
    with open(data_dir / "ground_truth_config.yaml") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("grain") == "daily":
        return run_checks_daily(data_dir, cfg)
    _results.clear()
    data = load(data_dir)
    weekly, config, users = data["weekly"], data["config"], data["users"]
    g = config["global_settings"]
    segments = config["segments"]

    # --- 1. Cohort sizes are exactly base * (1+growth)^w * acquisition ---
    # With a time_series block the realized weekly multiplier (seasonality *
    # noise * spikes) is part of the answer key, so the check stays exact.
    ts_path = data_dir / "ground_truth_timeseries.csv"
    ts_acq = (
        pd.read_csv(ts_path).set_index("week")["acquisition_mult"]
        if ts_path.exists()
        else None
    )
    acq = {
        w: np.prod([
            s["multiplier"] for s in config.get("shocks", [])
            if s["type"] == "acquisition" and s["start_week"] <= w <= s["end_week"]
        ]) * (1.0 if ts_acq is None else ts_acq[w])
        for w in range(g["periods_to_simulate"])
    }
    expected = {
        w: int(g["base_users"] * (1 + g["growth_rate"]) ** w * acq[w])
        for w in acq
    }
    sizes = users.groupby("cohort_week")["user_id"].nunique()
    mismatch = [w for w in expected if sizes.get(w, 0) != expected[w]]
    check("cohort growth", not mismatch,
          f"{len(expected) - len(mismatch)}/{len(expected)} weeks exact")

    # --- 2. Segment mix ---
    mix = users["segment"].value_counts(normalize=True)
    worst = max(abs(mix.get(s, 0) - segments[s]["prob"]) for s in segments)
    check("segment mix", worst < 0.03, f"max deviation {worst:.3f} (tol 0.03)")

    # --- 3. Dimension mixes ---
    for dim, spec in config.get("dimensions", {}).items():
        mix = users[dim].value_counts(normalize=True)
        worst = max(abs(mix.get(v, 0) - p) for v, p in spec["values"].items())
        check(f"dimension mix: {dim}", worst < 0.03,
              f"max deviation {worst:.3f} (tol 0.03)")

    # --- 4. Plan signup mix per segment ---
    if config.get("plans"):
        for seg, dist in config["plans"]["signup_distribution"].items():
            mix = users[users["segment"] == seg]["signup_plan"].value_counts(normalize=True)
            worst = max(abs(mix.get(p, 0) - v) for p, v in dist.items())
            check(f"signup plan mix: {seg}", worst < 0.05,
                  f"max deviation {worst:.3f} (tol 0.05)")

    # --- 5. Feature rates per page in clean weeks, per segment ---
    # Two optional causal edges change the expected per-page feature rate:
    #   activity_weights spawn extra pages (used_ai -> pages); marginal_page_
    #   multiplier dilutes a feature's rate on those extra pages. Steady-state
    #   per-page rate for feature g in segment s:
    #       R_s  = Σ_f prob_{s,f} · activity_weights[f]        (extra/base pages)
    #       rate = prob_{s,g} · (1 + m_g · R_s) / (1 + R_s)
    #   Collapses to prob when no edges are configured.
    base_event = config.get("base_event", "page_created")
    causal = config.get("causality") or {}
    activity_weights = causal.get("activity_weights", {})
    marginal_mult = causal.get("marginal_page_multiplier", {})
    # When an activity-weight driver is itself shocked (e.g. the AI outage zeroes
    # used_ai) no extra pages spawn, so dilution is absent — exclude those weeks
    # when checking a diluted feature.
    all_weeks = set(range(g["periods_to_simulate"]))
    driver_dirty: set[int] = set()
    for drv in activity_weights:
        driver_dirty |= all_weeks - set(_clean_weeks(config, drv))

    d = weekly.merge(users[["user_id", "segment"]], on="user_id")
    for seg, spec in segments.items():
        fp = spec.get("feature_probabilities", {})
        R = sum(fp.get(f, 0) * w for f, w in activity_weights.items())
        for feat, prob in fp.items():
            wks = set(_clean_weeks(config, feat))
            m_g = marginal_mult.get(feat, 1.0)
            if m_g != 1.0:
                wks -= driver_dirty   # only diluted-active weeks
            expected = prob * (1 + m_g * R) / (1 + R) if R else prob
            sub = d[(d["segment"] == seg) & d["abs_week"].isin(list(wks))]
            piv = sub.groupby("event_type")["event_count"].sum()
            rate = piv.get(feat, 0) / max(piv.get(base_event, 1), 1)
            tol = max(0.02, 0.12 * expected)
            check(f"feature rate {seg}/{feat}", abs(rate - expected) < tol,
                  f"observed {rate:.3f} vs {expected:.3f} (tol {tol:.3f})")

    # --- 5b. Dilution edge is real: on AI-spawned pages a diluted feature fires
    #     below its raw configured rate (clearest in `power`, where R is largest). ---
    if activity_weights and marginal_mult:
        feat = next(iter(marginal_mult))
        prob = segments.get("power", {}).get("feature_probabilities", {}).get(feat)
        if prob:
            wks = list(set(_clean_weeks(config, feat)) - driver_dirty)
            sub = d[(d["segment"] == "power") & d["abs_week"].isin(wks)]
            piv = sub.groupby("event_type")["event_count"].sum()
            rate = piv.get(feat, 0) / max(piv.get("page_created", 1), 1)
            check(f"dilution edge: power/{feat}", rate < prob - 0.02,
                  f"per-page {rate:.3f} below undiluted {prob:.2f} "
                  "(AI-spawned pages share less)")

    # --- 6. Causality: views ≈ base + weighted actions (outside tracking bugs) ---
    # Skipped for single-event configs (no causality block -> no page views).
    if config.get("causality"):
        instr_weeks: set[int] = set()
        for s in config.get("shocks", []):
            if s["type"] == "instrumentation":
                instr_weeks.update(range(s["start_week"], s["end_week"] + 1))
        cw = weekly[~weekly["abs_week"].isin(instr_weeks)]
        uw = cw.pivot_table(index=["user_id", "abs_week"], columns="event_type",
                            values="event_count", aggfunc="sum", fill_value=0)
        weights = config["causality"]["weights"]
        pred = config["causality"]["base_views"] + sum(
            uw.get(k, 0) * v for k, v in weights.items()
        )
        ratio = uw["page_view"] / pred
        check("causal page views", 0.90 < ratio.mean() < 1.02 and ratio.std() < 0.2,
              f"observed/predicted mean {ratio.mean():.3f}, std {ratio.std():.3f}")

    # --- 7. Retention plateaus (early cohorts, late ages, shock-free cells) ---
    def cell_clean(cohort: int, age: int) -> bool:
        wk = cohort + age
        for s in config.get("shocks", []):
            if s["type"] != "retention":
                continue
            anchor = cohort if s.get("cohort_scope") == "new" else wk
            if s["start_week"] <= anchor <= s["end_week"]:
                return False
        for e in config.get("experiments", []):
            if any(eff["type"] == "retention" for eff in e.get("effects", [])):
                if e["start_week"] <= wk <= e["end_week"]:
                    return False
        return True

    # Note: retention_table ratios (active@age / active@age0) are invariant
    # to any *user-constant* retention multiplier (channel, plan) — the
    # multiplier scales numerator and denominator equally — so the expected
    # value is the raw segment decay formula, with no multiplier corrections.
    plans_cfg = config.get("plans") or {}
    rates = retention_table(weekly)
    seg_rates = {
        seg: retention_table(d[d["segment"] == seg]) for seg in segments
    }
    transitions = config.get("transitions")
    if transitions:
        # Duolingo engine: the expected curve is segment-independent, but a
        # per-user retention multiplier does NOT cancel out of the ratios
        # (the state evolution is nonlinear in p), so mix the Markov curve
        # over the dimension-value multipliers by their signup shares.
        combos = [(1.0, 1.0)]
        for spec in config.get("dimensions", {}).values():
            rm = spec.get("retention_multiplier")
            if not rm:
                continue
            combos = [
                (share_acc * share, m_acc * rm.get(v, 1.0))
                for share_acc, m_acc in combos
                for v, share in spec["values"].items()
            ]
        expected_curve = np.zeros(25)
        for share, m in combos:
            expected_curve += share * _markov_retention_curve(transitions, 25, m)

    for seg, spec in segments.items():
        obs, exp_vals = [], []
        for cohort in range(4):
            for age in range(20, 25):
                if not cell_clean(cohort, age):
                    continue
                v = seg_rates[seg].at[cohort, age] if age in seg_rates[seg].columns else np.nan
                if not np.isnan(v):
                    obs.append(v)
                    if transitions:
                        exp_vals.append(expected_curve[age])
                    else:
                        plateau = spec["retention"]["plateau"]
                        decay = spec["retention"]["decay"]
                        exp_vals.append(
                            (1 - plateau) * np.exp(-decay * age) + plateau
                        )
        if obs:
            diff = abs(np.mean(obs) - np.mean(exp_vals))
            check(f"retention {'curve' if transitions else 'plateau'}: {seg}",
                  diff < 0.05,
                  f"observed {np.mean(obs):.3f} vs expected {np.mean(exp_vals):.3f} (tol 0.05)")

    # --- 8. Instrumentation: logged share ≈ multiplier ---
    by_week = d.groupby(["abs_week", "event_type"])["event_count"].sum().unstack(fill_value=0)
    for s in config.get("shocks", []):
        if s["type"] != "instrumentation":
            continue
        feat = s["target_feature"]
        wks = list(range(s["start_week"], s["end_week"] + 1))
        clean = [w for w in _clean_weeks(config, feat) if w < s["start_week"]][-5:]
        rate_in = (by_week.loc[wks, feat] / by_week.loc[wks, "page_created"]).mean()
        rate_out = (by_week.loc[clean, feat] / by_week.loc[clean, "page_created"]).mean()
        share = rate_in / rate_out
        check(f"instrumentation {feat}", abs(share - s["multiplier"]) < 0.07,
              f"logged share {share:.3f} vs {s['multiplier']} (tol 0.07)")

    # --- 9. Channel retention ordering ---
    # Denominator must be ALL signups (from ground truth), not active-at-age-0
    # users: a per-user retention multiplier cancels out of n4/n0.
    chan = config.get("dimensions", {}).get("acquisition_channel", {})
    if chan.get("retention_multiplier"):
        act = weekly[["user_id", "abs_week"]].drop_duplicates().merge(
            users[["user_id", "acquisition_channel", "cohort_week"]], on="user_id"
        )
        n_all = users[users["cohort_week"] <= 19].groupby(
            "acquisition_channel")["user_id"].nunique()
        at4 = act[(act["cohort_week"] <= 19)
                  & (act["abs_week"] == act["cohort_week"] + 4)]
        n4 = at4.groupby("acquisition_channel")["user_id"].nunique()
        r = (n4 / n_all).to_dict()
        ok = r["referral"] > r["organic"] > r["social"] > r["paid_search"]
        check("channel retention ordering", ok,
              "wk-4 retention " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(r.items())))

    # --- 9b. Plan tier causally drives behaviour (within core, to isolate
    # the causal multiplier from segment composition) ---
    if plans_cfg.get("multipliers"):
        core_users = users.loc[
            users["segment"] == "core", ["user_id", "signup_plan", "cohort_week"]
        ].copy()

        # Engagement: pages per active user-week, signup-pro vs signup-free.
        # Early weeks only, to limit contamination from plan transitions.
        m_act = plans_cfg["multipliers"]["activity"]["pro"]
        wk = weekly[weekly["abs_week"] <= 26].merge(core_users, on="user_id")
        act_uw = (
            wk[["user_id", "abs_week", "signup_plan"]].drop_duplicates()
            .groupby("signup_plan").size()
        )
        pages = (
            wk[wk["event_type"] == "page_created"]
            .groupby("signup_plan")["event_count"].sum()
        )
        ratio = (pages["pro"] / act_uw["pro"]) / (pages["free"] / act_uw["free"])
        check("plan engagement: core", abs(ratio - m_act) < 0.12,
              f"pages/active-week pro vs free x{ratio:.3f} "
              f"vs configured x{m_act} (tol 0.12)")

        # Retention: weekly active rate over ages 1-8, paid vs free signups.
        act = weekly[["user_id", "abs_week"]].drop_duplicates().merge(
            core_users[core_users["cohort_week"] <= 25], on="user_id"
        )
        act["age"] = act["abs_week"] - act["cohort_week"]
        act["paid"] = act["signup_plan"] != "free"
        eligible = core_users[core_users["cohort_week"] <= 25].copy()
        eligible["paid"] = eligible["signup_plan"] != "free"
        n_users = eligible.groupby("paid")["user_id"].nunique()
        cells = act[act["age"].between(1, 8)].groupby("paid").size()
        rate = cells / (n_users * 8)
        check("plan retention: core", rate[True] > rate[False],
              f"weekly active rate (ages 1-8): paid {rate[True]:.3f} "
              f"vs free {rate[False]:.3f}")

        # Team retention edge, measured at ages 10-20 where the weekly
        # active probability is far from its 1.0 cap (early ages saturate,
        # compressing any multiplier below its configured value).
        m_team = plans_cfg["multipliers"]["retention"]["team"]
        late = act[act["age"].between(10, 20)].groupby("signup_plan").size()
        n_plan = eligible.groupby("signup_plan")["user_id"].nunique()
        late_rate = late / (n_plan * 11)
        edge = late_rate["team"] / late_rate["free"]
        check("plan retention edge: core team",
              m_team - 0.15 < edge < m_team + 0.15,
              f"weekly active rate ages 10-20: team {late_rate['team']:.3f} "
              f"vs free {late_rate['free']:.3f} (x{edge:.2f}, "
              f"configured x{m_team})")

    # --- 10. Experiments (statistical readout) ---
    for exp in config.get("experiments", []):
        s = experiment_summary(data, exp)
        check(f"no SRM: {exp['name']}", s["srm_p"] > 0.001,
              f"sample-ratio-mismatch p = {s['srm_p']:.3f} (n={s['n_users']:,})")

        m = exp["effects"][0]["multiplier"]
        if not exp.get("traps"):
            # Guard with a 99% CI: a 95% guard would fail 1-in-20 healthy
            # generations by construction.
            lo99 = s["lift"] * np.exp(-2.576 * s["se_log"])
            hi99 = s["lift"] * np.exp(2.576 * s["se_log"])
            ok = lo99 <= m <= hi99 and s["p"] < 0.05
            check(f"clean lift: {exp['name']}", ok,
                  f"pooled {s['lift']:.3f} (95% CI {s['lo']:.3f}-{s['hi']:.3f}, "
                  f"99% {lo99:.3f}-{hi99:.3f}) vs configured {m}, p = {s['p']:.2g}")
        elif any(t["type"] == "novelty" for t in exp.get("traps", [])):
            lift = experiment_lift(data, exp)
            early = lift["lift"].loc[lift.index <= 1].mean()
            late = lift["lift"].loc[lift.index >= 6].mean()
            check(f"novelty decay: {exp['name']}",
                  early > late + 0.05 and early > 1.1,
                  f"lift exposure 0-1: {early:.3f}, exposure 6+: {late:.3f} "
                  f"(pooled would claim {s['lift']:.3f}, p = {s['p']:.2g})")

    # --- 11. Downgrade spike (transition shocks) ---
    for s in config.get("shocks", []):
        if s["type"] != "transition":
            continue
        evt = "downgraded" if s.get("target") == "downgrade" else "upgraded"
        wk = by_week[evt] if evt in by_week.columns else pd.Series(dtype=float)
        in_w = wk.loc[s["start_week"]:s["end_week"]].mean()
        before = wk.loc[max(0, s["start_week"] - 8):s["start_week"] - 1].mean()
        spike = in_w / max(before, 0.1)
        check(f"transition spike: {evt}", spike > 3,
              f"{evt}/week {in_w:.1f} in window vs {before:.1f} before (x{spike:.1f})")

    failed = [name for ok, name in _results if not ok]
    print(f"\n{len(_results) - len(failed)}/{len(_results)} checks passed"
          + (f" — FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0
