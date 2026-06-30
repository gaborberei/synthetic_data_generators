from __future__ import annotations

"""Causal + shock synthetic event generator.

Simulates an event-level product-analytics dataset for a Notion-like app.

Model (see configs/*.yaml for the knobs):
  * Users arrive in weekly cohorts that grow over time (with optional
    acquisition shocks).
  * Each user belongs to a segment (casual / core / power) and, at signup, is
    assigned dimension values (acquisition_channel, country, platform, ...),
    a plan tier, and an app version (with later releases adopted gradually).
  * In every active week a user creates pages, fires feature events
    (used_ai, shared, ...), may upgrade/downgrade their plan, and generates
    page views. Page views are a *causal* function of the actions, so a
    feature shock ripples into page views.
  * Shocks are time-boxed multipliers on acquisition, retention, activity,
    a feature, a plan transition, or instrumentation (logged-event loss).
    A `where: {column: [values]}` filter scopes a shock to matching users;
    `cohort_scope: new` makes the window select *signup cohorts* instead of
    activity weeks (the effect then follows those cohorts for life).
  * Experiments assign users to treatment/control via a deterministic hash
    and apply feature / activity / retention multipliers to the treatment
    group inside the experiment window, with optional traps (novelty decay,
    heterogeneous per-segment effects).

Optional blocks (each absent -> behaviour identical to before they existed):
  * `transitions` swaps the per-segment plateau/decay retention curve for the
    Duolingo growth-model state machine: a user's weekly active probability
    depends on the gap since they were last active and their last bucket
    (Current/CURR, Reactivated/RURR, Resurrected/SURR, New/NURR; gaps of 2-4
    weeks draw REAC, longer gaps RESU).
  * `time_series` makes weekly curves lifelike: annual seasonality
    (1 + amp*cos(2*pi*week/period)) with separate amplitudes for acquisition /
    activity / retention, AR(1) multiplicative noise shared across metrics
    (good weeks cluster), one-off acquisition `spikes` (news moments), and a
    persistent lognormal per-user activity multiplier. The realized weekly
    multipliers are exposed as `timeseries_df` (saved as ground truth).
  * `base_event` renames the base activity event (default `page_created`).
  * Omitting `causality` disables derived `page_view` events entirely —
    single-event datasets (e.g. a chess app emitting only `game_played`)
    configure a base_event and no causality block.
  * `timestamps` controls event times: `days` per week (default 5, Mon-Fri)
    and an optional `peak_hour` for an evening-skewed hour distribution.
  * `grain: daily` switches to a separate per-day engine (_simulate_daily) for
    streak/habit products: a user's daily return probability rises with their
    streak (habit) and decays with days-since-active (lapse), with streak
    freezes and reactivation notifications. Users still arrive in weekly
    cohorts; dimensions, shocks, experiments and time_series apply unchanged.
    The `daily`, `streak` and `notifications` blocks configure it. Every
    non-daily config leaves the weekly path byte-for-byte identical.

The generator is fully seedable: pass a `seed` for reproducible output.
(Experiment assignment is hash-based, hence reproducible independent of seed.)
"""

import hashlib
import itertools
from typing import Callable

import numpy as np
import pandas as pd

# Event types that are not derived features (handled separately).
PAGE_CREATED = "page_created"
PAGE_VIEW = "page_view"
UPGRADED = "upgraded"
DOWNGRADED = "downgraded"


def hash_unit(user_id: str, salt: str) -> float:
    """Deterministic uniform [0, 1) value for a (user, salt) pair."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x100000000


class CausalShockGenerator:
    """Generate event-level synthetic data from a config dict."""

    def __init__(self, config: dict, seed: int | None = None) -> None:
        self.config = config
        self.segments: dict = config["segments"]
        self.shocks: list[dict] = config.get("shocks", [])
        # No causality block -> no derived page-view events at all.
        self.causality: dict | None = config.get("causality")
        self.base_event: str = config.get("base_event", PAGE_CREATED)
        self.transitions: dict | None = config.get("transitions")
        self.timestamps_cfg: dict = config.get("timestamps", {})
        # Grain: "weekly" (default, the original engine) or "daily" (a separate
        # per-day simulation path for streak/habit products — see _simulate_daily).
        # Any non-daily config leaves the weekly path byte-for-byte unchanged.
        self.grain: str = config.get("grain", "weekly")
        self.daily_cfg: dict = config.get("daily", {})
        self.streak_cfg: dict = config.get("streak", {})
        self.notifications_cfg: dict = config.get("notifications", {})
        self.dimensions: dict = config.get("dimensions", {})
        rel = config.get("releases", {})
        self.release_versions: list[dict] = sorted(
            rel.get("versions", []), key=lambda r: r["week"]
        )
        self.adoption_rate: float = rel.get("adoption_rate", 0.3)
        self.plans: dict | None = config.get("plans")
        self.experiments: list[dict] = config.get("experiments", [])
        self.rng = np.random.default_rng(seed)

        # Evening-skewed hour distribution (gaussian around peak_hour on a
        # 24h circle, with a floor so no hour is empty).
        self._hour_probs: np.ndarray | None = None
        peak = self.timestamps_cfg.get("peak_hour")
        if peak is not None:
            hours = np.arange(24)
            dist = np.minimum((hours - peak) % 24, (peak - hours) % 24)
            w = np.exp(-0.5 * dist**2 / 16) + 0.15
            self._hour_probs = w / w.sum()

        # Optional within-week day-of-week placement (weekly engine). Absent ->
        # each event falls uniformly across `timestamps.days` (legacy behavior,
        # identical RNG stream, so other configs are byte-for-byte unchanged).
        # Present -> the working day is drawn from a named `profile` selected per
        # (event_type, segment); see _day_probs / _emit_week.
        self.dow_cfg: dict | None = config.get("day_of_week")
        self._dow_offsets: np.ndarray | None = None
        self._dow_profiles: dict[str, np.ndarray] = {}
        self._dow_uniform: np.ndarray | None = None
        self._dow_cache: dict[tuple, np.ndarray] = {}
        if self.dow_cfg:
            day_idx = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
            days = self.dow_cfg.get("days", ["mon", "tue", "wed", "thu", "fri"])
            self._dow_offsets = np.array([day_idx[d.lower()[:3]] for d in days], dtype=int)
            for name, weights in (self.dow_cfg.get("profiles") or {}).items():
                w = np.asarray(weights, dtype=float)
                self._dow_profiles[name] = w / w.sum()
            self._dow_uniform = np.ones(len(days)) / len(days)

        # Time-series realism: per-week multiplier arrays, realized once at
        # init (seed-reproducible) and exposed via `timeseries_df`.
        self.ts_acq: np.ndarray | None = None
        self.ts_act: np.ndarray | None = None
        self.ts_ret: np.ndarray | None = None
        self.user_variance_sigma: float = 0.0
        self.timeseries_df: pd.DataFrame | None = None
        ts = config.get("time_series")
        if ts:
            periods = config["global_settings"]["periods_to_simulate"]
            season = ts.get("seasonality", {})
            period_weeks = season.get("period_weeks", 52)
            noise_cfg = ts.get("noise", {})
            phi = noise_cfg.get("phi", 0.0)
            sigma = noise_cfg.get("sigma", 0.0)
            x, noise = 0.0, np.empty(periods)
            for w_i in range(periods):
                x = phi * x + self.rng.normal(0.0, sigma)
                noise[w_i] = np.exp(x)
            weeks = np.arange(periods)

            def seasonal(amp: float) -> np.ndarray:
                return 1 + amp * np.cos(2 * np.pi * weeks / period_weeks)

            spikes = np.ones(periods)
            for s in ts.get("spikes", []):
                spikes[s["week"]] *= s["multiplier"]
            # Retention compounds week over week, so its noise is dampened
            # (noise ** retention_power) on top of a small amplitude.
            ret_power = noise_cfg.get("retention_power", 1.0)
            self.ts_acq = seasonal(season.get("acquisition_amp", 0.0)) * noise * spikes
            self.ts_act = seasonal(season.get("activity_amp", 0.0)) * noise
            self.ts_ret = seasonal(season.get("retention_amp", 0.0)) * noise**ret_power
            self.user_variance_sigma = ts.get("user_variance_sigma", 0.0)
            self.timeseries_df = pd.DataFrame(
                {
                    "week": weeks,
                    "noise": noise,
                    "acquisition_mult": self.ts_acq,
                    "activity_mult": self.ts_act,
                    "retention_mult": self.ts_ret,
                }
            )

        # Normalize the first cohort date back to its Monday so weekday
        # indexing (Mon-Fri) is consistent across cohorts.
        start = pd.Timestamp(config["global_settings"]["start_date"])
        self.start_monday = start - pd.Timedelta(days=start.weekday())

        # Filled by run(); exposed for export.
        self.users_df: pd.DataFrame | None = None
        self.assignments_df: pd.DataFrame | None = None
        self._events: list[tuple] = []
        self._users: list[dict] = []
        self._assignments: list[dict] = []

    # ------------------------------------------------------------------ #
    # Distributions and small helpers
    # ------------------------------------------------------------------ #
    def _activity_sampler(self, dist: dict) -> Callable[[int], np.ndarray]:
        """Return a function size -> array of pages-created counts (>= 1)."""
        dtype = dist["type"]
        rng = self.rng
        if dtype == "poisson":
            return lambda n: np.maximum(1, rng.poisson(dist["lambda"], n))
        if dtype == "normal":
            return lambda n: np.maximum(
                1, rng.normal(dist["loc"], dist["scale"], n).astype(int)
            )
        if dtype == "lognormal":
            return lambda n: np.maximum(
                1, rng.lognormal(dist["mean"], dist["sigma"], n).astype(int)
            )
        return lambda n: np.ones(n, dtype=int)

    @staticmethod
    def _feature_count(spec, pages: int, mult: float, rng) -> int:
        """Number of feature events over `pages` pages this week.

        `spec` (a `feature_probabilities` entry) is either:
          * a scalar per-page probability  -> Bernoulli, at most one per page
            (`rng.binomial(pages, p)`); or
          * `{type: poisson, lambda: λ}`    -> Poisson with mean `λ * pages`,
            which lets a feature occur MORE than once per page (e.g. used_ai
            fired several times on a page). `mult` folds in the shock /
            experiment / marginal-page multipliers.
        """
        if pages <= 0:
            return 0
        if isinstance(spec, dict):
            lam = float(spec.get("lambda", spec.get("lam", 0.0))) * mult
            return int(rng.poisson(max(0.0, lam) * pages))
        return int(rng.binomial(pages, min(1.0, spec * mult)))

    @staticmethod
    def _feature_count_vec(spec, pages: np.ndarray, mult, rng) -> np.ndarray:
        """Vectorised `_feature_count` over an array of `pages` (and `mult`).

        Used by the daily engine to draw a feature's per-active-day counts for
        all active users at once. Same two cases as the scalar version: poisson
        (mean `lambda * mult * pages`) or per-page Bernoulli (`binomial`).
        """
        pages = np.asarray(pages)
        if isinstance(spec, dict):
            lam = float(spec.get("lambda", spec.get("lam", 0.0)))
            return rng.poisson(np.maximum(0.0, lam * mult) * pages)
        p = np.clip(spec * mult, 0.0, 1.0)
        return rng.binomial(pages, p)

    def _sample_categorical(self, values: dict, n: int) -> np.ndarray:
        names = list(values)
        probs = np.array([values[k] for k in names], dtype=float)
        probs /= probs.sum()
        return self.rng.choice(names, size=n, p=probs)

    @staticmethod
    def _active_probability(week: int, plateau: float, decay: float) -> float:
        """Exponential decay from 1.0 toward `plateau` over time."""
        return (1 - plateau) * np.exp(-decay * week) + plateau

    def _random_timestamp(self, week_start: pd.Timestamp) -> pd.Timestamp:
        """A random timestamp within the given week.

        Defaults to weekdays (Mon-Fri) with a uniform time of day; the
        `timestamps` config can widen to 7 days and skew hours toward an
        evening `peak_hour`.
        """
        days = int(self.rng.integers(0, self.timestamps_cfg.get("days", 5)))
        if self._hour_probs is None:
            seconds = int(self.rng.integers(0, 86_400))
        else:
            hour = int(self.rng.choice(24, p=self._hour_probs))
            seconds = hour * 3_600 + int(self.rng.integers(0, 3_600))
        return week_start + pd.Timedelta(days=days, seconds=seconds)

    def _day_probs(self, event_type: str, segment: str) -> np.ndarray | None:
        """Within-week day distribution for (event_type, segment), or None.

        None when no `day_of_week` block is configured — callers then keep the
        legacy uniform placement (identical RNG stream). Otherwise resolves
        `assign[event_type]` (a profile name, or a `{default, <segment>}` map),
        falling back to `assign.default`, then to a uniform profile.
        """
        if self.dow_cfg is None:
            return None
        key = (event_type, segment)
        cached = self._dow_cache.get(key)
        if cached is not None:
            return cached
        assign = self.dow_cfg.get("assign", {})
        top = assign.get("default")
        entry = assign.get(event_type, top)
        if isinstance(entry, dict):
            name = entry.get(segment) or entry.get("default") or top
        else:
            name = entry or top
        probs = self._dow_profiles.get(name, self._dow_uniform)
        self._dow_cache[key] = probs
        return probs

    def _emit_week(
        self, uid: str, week_start: pd.Timestamp, event_type: str, n: int, segment: str
    ) -> None:
        """Append `n` events for one user in one week.

        With no `day_of_week` block this defers to per-event `_random_timestamp`
        (preserving the legacy RNG stream). With a block, the working day is
        drawn in a batch from the (event_type, segment) profile and timestamps
        are vectorised — both applying the profile and shedding the per-event
        Python-loop cost at daily volume.
        """
        if n <= 0:
            return
        day_probs = self._day_probs(event_type, segment)
        if day_probs is None:
            for _ in range(n):
                self._events.append((uid, self._random_timestamp(week_start), event_type))
            return
        idx = self.rng.choice(len(day_probs), size=n, p=day_probs)
        days = self._dow_offsets[idx]
        secs = self._day_seconds(n)
        base = week_start.to_datetime64()
        times = (
            base + days.astype("timedelta64[D]") + secs.astype("timedelta64[s]")
        ).astype("datetime64[us]").astype(object)
        self._events.extend(
            zip(itertools.repeat(uid), times.tolist(), itertools.repeat(event_type))
        )

    # ------------------------------------------------------------------ #
    # Scoping: `where` filters, shocks, experiments
    # ------------------------------------------------------------------ #
    @staticmethod
    def _where_mask(where: dict | None, attrs: dict, n: int) -> np.ndarray:
        """Boolean mask over n users for a {column: [values]} filter.

        `attrs` maps column name -> per-user array or group-constant scalar.
        An unknown column matches nobody (fail closed).
        """
        mask = np.ones(n, dtype=bool)
        if not where:
            return mask
        for col, allowed in where.items():
            v = attrs.get(col)
            if v is None:
                return np.zeros(n, dtype=bool)
            if isinstance(v, np.ndarray):
                mask &= np.isin(v, list(allowed))
            elif v not in allowed:
                return np.zeros(n, dtype=bool)
        return mask

    def _shock_vec(
        self,
        shock_type: str,
        abs_week: int,
        cohort_week: int,
        attrs: dict,
        n: int,
        target: str | None = None,
    ) -> np.ndarray:
        """Per-user combined multiplier of all matching shocks."""
        mult = np.ones(n)
        for shock in self.shocks:
            if shock["type"] != shock_type:
                continue
            if shock.get("cohort_scope") == "new":
                if not (shock["start_week"] <= cohort_week <= shock["end_week"]):
                    continue
            elif not (shock["start_week"] <= abs_week <= shock["end_week"]):
                continue
            if target is not None:
                shock_target = shock.get("target_feature", shock.get("target"))
                if shock_target != target:
                    continue
            mask = self._where_mask(shock.get("where"), attrs, n)
            mult = np.where(mask, mult * shock["multiplier"], mult)
        return mult

    def _experiment_vec(
        self,
        effect_type: str,
        abs_week: int,
        exp_state: list[tuple],
        segment: str,
        n: int,
        target: str | None = None,
    ) -> np.ndarray:
        """Per-user multiplier from experiment treatment effects.

        exp_state holds (experiment, treated_mask, exposure_start_week)
        triples for this user group.
        """
        mult = np.ones(n)
        for exp, treated, exposure_start in exp_state:
            if not (exp["start_week"] <= abs_week <= exp["end_week"]):
                continue
            for eff in exp.get("effects", []):
                if eff["type"] != effect_type:
                    continue
                if target is not None and eff.get("target") != target:
                    continue
                m = eff["multiplier"]
                for trap in exp.get("traps", []):
                    if trap["type"] == "heterogeneous":
                        m = trap["multipliers"].get(segment, m)
                    elif trap["type"] == "novelty":
                        weeks_exposed = abs_week - exposure_start
                        decay = 0.5 ** (weeks_exposed / trap["half_life_weeks"])
                        m = 1 + (m - 1) * decay
                mult = np.where(treated, mult * m, mult)
        return mult

    # ------------------------------------------------------------------ #
    # Cohort simulation
    # ------------------------------------------------------------------ #
    def _generate_cohort(
        self,
        n_users: int,
        duration_weeks: int,
        start_user_id: int,
        cohort_week: int,
    ) -> None:
        if n_users <= 0:
            return
        rng = self.rng

        user_ids = np.array(
            [f"user_{start_user_id + i:06d}" for i in range(n_users)]
        )
        seg_keys = list(self.segments)
        seg_probs = {k: self.segments[k]["prob"] for k in seg_keys}
        user_segments = self._sample_categorical(seg_probs, n_users)

        # --- Dimensions (assigned at signup, fixed for life) ---
        dim_values = {
            dim: self._sample_categorical(spec["values"], n_users)
            for dim, spec in self.dimensions.items()
        }
        user_ret_mult = np.ones(n_users)
        for dim, spec in self.dimensions.items():
            rm = spec.get("retention_multiplier")
            if rm:
                user_ret_mult *= np.array(
                    [rm.get(v, 1.0) for v in dim_values[dim]]
                )

        # Persistent per-user engagement multiplier (time_series block).
        user_act_mult = (
            rng.lognormal(0.0, self.user_variance_sigma, n_users)
            if self.user_variance_sigma
            else None
        )

        # --- App version: signup version + adoption week per release ---
        signup_version = None
        adoption: dict[str, np.ndarray] = {}
        for rel in self.release_versions:
            if rel["week"] <= cohort_week:
                signup_version = rel["version"]
                adoption[rel["version"]] = np.full(n_users, cohort_week)
            else:
                adoption[rel["version"]] = rel["week"] + rng.geometric(
                    self.adoption_rate, n_users
                )

        # --- Plan tier at signup ---
        plan_state = None
        levels: list[str] = []
        if self.plans:
            levels = self.plans["levels"]
            plan_state = np.zeros(n_users, dtype=int)
            for seg_name in seg_keys:
                pos = np.where(user_segments == seg_name)[0]
                if len(pos) == 0:
                    continue
                dist = self.plans["signup_distribution"][seg_name]
                sampled = self._sample_categorical(dist, len(pos))
                plan_state[pos] = [levels.index(p) for p in sampled]

        # --- Experiment assignment (deterministic hash) ---
        cohort_exp_state: list[tuple] = []
        for exp in self.experiments:
            scope = exp["assignment"].get("scope", "new_users")
            if scope == "new_users":
                eligible = exp["start_week"] <= cohort_week <= exp["end_week"]
                exposure_start = cohort_week
            else:  # all_users
                eligible = cohort_week <= exp["end_week"]
                exposure_start = max(exp["start_week"], cohort_week)
            if not eligible:
                continue
            ratio = exp["assignment"]["ratio"]
            treated = np.array(
                [hash_unit(uid, exp["name"]) < ratio for uid in user_ids]
            )
            cohort_exp_state.append((exp, treated, exposure_start))
            for uid, t in zip(user_ids, treated):
                self._assignments.append(
                    {
                        "experiment": exp["name"],
                        "user_id": uid,
                        "group": "treatment" if t else "control",
                        "first_exposed_week": exposure_start,
                    }
                )

        # --- Ground-truth user records ---
        for i in range(n_users):
            rec = {
                "user_id": user_ids[i],
                "cohort_week": cohort_week,
                "segment": user_segments[i],
            }
            for dim in self.dimensions:
                rec[dim] = dim_values[dim][i]
            if signup_version is not None:
                rec["signup_app_version"] = signup_version
            for ver, weeks in adoption.items():
                rec[f"adopted_{ver}_week"] = int(weeks[i])
            if plan_state is not None:
                rec["signup_plan"] = levels[plan_state[i]]
            if user_act_mult is not None:
                rec["activity_mult"] = round(float(user_act_mult[i]), 4)
            self._users.append(rec)

        # --- Daily simulation path (grain: daily) ---
        # The user setup above (ids, segments, dimensions, retention/activity
        # multipliers, experiment assignment, ground-truth records) is shared.
        # The daily streak/habit model is a self-contained branch; the weekly
        # loop below is left untouched for every other config.
        if self.grain == "daily":
            # Fold plan engagement multipliers into the per-user activity /
            # retention multipliers so paid users stay more engaged within a
            # segment (the weekly path applies these via act_by_level/ret_by_level;
            # the daily path has no plan loop). No plans -> no change, so
            # chess_daily / duolingo_hard are unaffected.
            daily_act_mult = user_act_mult
            daily_ret_mult = user_ret_mult
            if self.plans and plan_state is not None:
                pm = self.plans.get("multipliers", {})
                act_lv = np.array([pm.get("activity", {}).get(lv, 1.0) for lv in levels])
                ret_lv = np.array([pm.get("retention", {}).get(lv, 1.0) for lv in levels])
                pa = act_lv[plan_state]
                daily_act_mult = pa if user_act_mult is None else user_act_mult * pa
                daily_ret_mult = user_ret_mult * ret_lv[plan_state]
            self._simulate_daily(
                user_ids=user_ids,
                user_segments=user_segments,
                dim_values=dim_values,
                user_ret_mult=daily_ret_mult,
                user_act_mult=daily_act_mult,
                cohort_week=cohort_week,
                duration_weeks=duration_weeks,
                cohort_exp_state=cohort_exp_state,
            )
            return

        # --- Weekly simulation, per segment ---
        causality = self.causality
        weights = causality["weights"] if causality else {}
        base_views = causality["base_views"] if causality else 0
        noise = causality["noise_scale"] if causality else 0.0
        # Optional causal edges (absent -> no effect, keeps other configs
        # byte-for-byte identical):
        #   activity_weights      feature -> extra pages_created (e.g. used_ai)
        #   marginal_page_mult    feature rate on those AI-spawned (marginal),
        #                         lower-quality pages (e.g. shared diluted)
        activity_weights = causality.get("activity_weights", {}) if causality else {}
        marginal_page_mult = (
            causality.get("marginal_page_multiplier", {}) if causality else {}
        )
        cohort_start = self.start_monday + pd.DateOffset(weeks=cohort_week)
        transitions = (self.plans or {}).get("weekly_transitions", {})

        # Per-plan behaviour multipliers (pro/team users are causally more
        # engaged), indexed by plan level for cheap per-user lookup.
        plan_mults = (self.plans or {}).get("multipliers", {})
        act_by_level = np.array(
            [plan_mults.get("activity", {}).get(lv, 1.0) for lv in levels]
        )
        ret_by_level = np.array(
            [plan_mults.get("retention", {}).get(lv, 1.0) for lv in levels]
        )

        for seg_name in seg_keys:
            idx = np.where(user_segments == seg_name)[0]
            n_seg = len(idx)
            if n_seg == 0:
                continue

            seg = self.segments[seg_name]
            feat_probs = seg.get("feature_probabilities", {})
            sample_pages = self._activity_sampler(seg["activity_distribution"])
            if self.transitions is None:
                retention = seg["retention"]
                base_active = [
                    self._active_probability(w, retention["plateau"], retention["decay"])
                    for w in range(duration_weeks)
                ]
            else:
                # Duolingo state machine: per-user weeks-since-active and last
                # bucket (0 New, 1 Current, 2 Reactivated, 3 Resurrected).
                last_active = np.zeros(n_seg, dtype=int)
                state = np.zeros(n_seg, dtype=int)
            seg_ret_mult = user_ret_mult[idx]
            seg_ids = user_ids[idx]
            attrs: dict = {"segment": seg_name, "app_version": signup_version}
            for dim in self.dimensions:
                attrs[dim] = dim_values[dim][idx]
            exp_state = [
                (exp, treated[idx], exposure_start)
                for exp, treated, exposure_start in cohort_exp_state
            ]

            for rel_week in range(duration_weeks):
                abs_week = cohort_week + rel_week
                if self.plans:
                    attrs["plan"] = np.array(levels)[plan_state[idx]]

                if self.transitions is None:
                    p_active = (
                        base_active[rel_week]
                        * seg_ret_mult
                        * self._shock_vec("retention", abs_week, cohort_week, attrs, n_seg)
                        * self._experiment_vec("retention", abs_week, exp_state, seg_name, n_seg)
                    )
                    if self.plans and plan_mults:
                        p_active = p_active * ret_by_level[plan_state[idx]]
                    if self.ts_ret is not None:
                        p_active = p_active * self.ts_ret[abs_week]
                    act = np.where(rng.random(n_seg) < p_active)[0]
                elif rel_week == 0:
                    # New users are active by definition in their signup week.
                    act = np.arange(n_seg)
                else:
                    t = self.transitions
                    gap = rel_week - last_active
                    p_base = np.where(
                        gap == 1,
                        np.select(
                            [state == 1, state == 2, state == 3],
                            [t["CURR"], t["RURR"], t["SURR"]],
                            default=t["NURR"],
                        ),
                        np.where(gap <= 4, t["REAC"], t["RESU"]),
                    )
                    p_active = (
                        p_base
                        * seg_ret_mult
                        * self._shock_vec("retention", abs_week, cohort_week, attrs, n_seg)
                        * self._experiment_vec("retention", abs_week, exp_state, seg_name, n_seg)
                    )
                    if self.plans and plan_mults:
                        p_active = p_active * ret_by_level[plan_state[idx]]
                    if self.ts_ret is not None:
                        p_active = p_active * self.ts_ret[abs_week]
                    act = np.where(rng.random(n_seg) < np.minimum(p_active, 0.98))[0]
                    state[act] = np.where(gap[act] == 1, 1, np.where(gap[act] <= 4, 2, 3))
                    last_active[act] = rel_week
                if len(act) == 0:
                    continue
                n_act = len(act)
                act_attrs = {
                    k: (v[act] if isinstance(v, np.ndarray) else v)
                    for k, v in attrs.items()
                }
                act_exp = [(e, t[act], x) for e, t, x in exp_state]

                # Pages created, with per-user activity multipliers
                # (stochastic rounding keeps fractional multipliers unbiased).
                pages = sample_pages(n_act).astype(float)
                a_mult = self._shock_vec(
                    "activity", abs_week, cohort_week, act_attrs, n_act
                ) * self._experiment_vec("activity", abs_week, act_exp, seg_name, n_act)
                if self.plans and plan_mults:
                    a_mult = a_mult * act_by_level[plan_state[idx[act]]]
                if self.ts_act is not None:
                    a_mult = a_mult * self.ts_act[abs_week]
                if user_act_mult is not None:
                    a_mult = a_mult * user_act_mult[idx[act]]
                scaled = pages * a_mult
                floor = np.floor(scaled).astype(int)
                pages = np.maximum(1, floor + (rng.random(n_act) < scaled - floor))

                # Per-user, per-feature probability multipliers for this week.
                feat_mult = {
                    feat: self._shock_vec(
                        "feature", abs_week, cohort_week, act_attrs, n_act, target=feat
                    )
                    * self._experiment_vec(
                        "feature", abs_week, act_exp, seg_name, n_act, target=feat
                    )
                    for feat in feat_probs
                }

                week_start = cohort_start + pd.DateOffset(weeks=rel_week)

                # Plan transitions for active users (emits upgrade/downgrade
                # events; plan state persists across weeks).
                if self.plans and transitions:
                    glob = idx[act]
                    cur = plan_state[glob]
                    up_p = transitions.get("upgrade", 0.0) * self._shock_vec(
                        "transition", abs_week, cohort_week, act_attrs, n_act,
                        target="upgrade",
                    )
                    down_p = transitions.get("downgrade", 0.0) * self._shock_vec(
                        "transition", abs_week, cohort_week, act_attrs, n_act,
                        target="downgrade",
                    )
                    r = rng.random(n_act)
                    up = (r < up_p) & (cur < len(levels) - 1)
                    down = (~up) & (r > 1 - down_p) & (cur > 0)
                    plan_state[glob[up]] += 1
                    plan_state[glob[down]] -= 1
                    for uid in seg_ids[act[up]]:
                        self._emit_week(uid, week_start, UPGRADED, 1, seg_name)
                    for uid in seg_ids[act[down]]:
                        self._emit_week(uid, week_start, DOWNGRADED, 1, seg_name)

                for j, pos in enumerate(act):
                    uid = seg_ids[pos]
                    base_pages = int(pages[j])

                    # 1. Features on the BASE pages, at the full per-page rate.
                    #    Counts only here; emission is deferred to step 4 so the
                    #    AI-spawned pages (step 2) can feed back first.
                    base_feat = {}
                    for feat, prob in feat_probs.items():
                        base_feat[feat] = self._feature_count(
                            prob, base_pages, feat_mult[feat][j], rng
                        )

                    # 2. Causal edge: feature usage spawns EXTRA pages this week
                    #    (e.g. used_ai makes a user more productive). Stochastic
                    #    rounding keeps fractional weights unbiased.
                    extra = sum(
                        base_feat.get(f, 0) * w for f, w in activity_weights.items()
                    )
                    extra_floor = int(np.floor(extra))
                    extra_pages = extra_floor + int(rng.random() < extra - extra_floor)

                    # 3. Features on the EXTRA pages, at a DILUTED rate: the
                    #    AI-spawned pages are lower-quality, so they are less
                    #    likely to be shared/commented (marginal_page_multiplier).
                    action_counts = {}
                    for feat, prob in feat_probs.items():
                        n_extra = self._feature_count(
                            prob,
                            extra_pages,
                            feat_mult[feat][j] * marginal_page_mult.get(feat, 1.0),
                            rng,
                        )
                        action_counts[feat] = base_feat[feat] + n_extra

                    n_pages = base_pages + extra_pages
                    action_counts[self.base_event] = n_pages

                    # 4. Emit base + feature events.
                    self._emit_week(uid, week_start, self.base_event, n_pages, seg_name)
                    for feat, n_feat in action_counts.items():
                        if feat == self.base_event:
                            continue
                        self._emit_week(uid, week_start, feat, n_feat, seg_name)

                    # 5. Causal page views: driven by the (amplified) actions.
                    if causality is not None:
                        weighted = sum(
                            count * weights.get(action, 0)
                            for action, count in action_counts.items()
                        )
                        n_views = max(
                            1, int((base_views + weighted) * rng.normal(1.0, noise))
                        )
                        self._emit_week(uid, week_start, PAGE_VIEW, n_views, seg_name)

    # ------------------------------------------------------------------ #
    # Daily simulation path (grain: daily) — streak / habit products
    # ------------------------------------------------------------------ #
    def _day_seconds(self, n: int) -> np.ndarray:
        """`n` random within-day second offsets in [0, 86400).

        Evening-skewed toward `timestamps.peak_hour` when set (study sessions
        cluster after work/school), otherwise uniform across the day.
        """
        if self._hour_probs is None:
            return self.rng.integers(0, 86_400, n)
        hours = self.rng.choice(24, size=n, p=self._hour_probs)
        return hours * 3_600 + self.rng.integers(0, 3_600, n)

    def _emit_daily(
        self,
        uids: np.ndarray,
        day_start: pd.Timestamp,
        event_type: str,
        counts: np.ndarray | None = None,
    ) -> None:
        """Append `(uid, timestamp, event_type)` rows for users on one day.

        With `counts`, user *i* emits `counts[i]` events (several lessons in a
        day); without it, exactly one event per user (a notification, a
        milestone, a streak break). Timestamps are random within the day. This
        is vectorised — the weekly path's per-event Python loop would be far too
        slow at daily volume.
        """
        if len(uids) == 0:
            return
        if counts is not None:
            uids = np.repeat(uids, counts)
            if len(uids) == 0:
                return
        secs = self._day_seconds(len(uids))
        base = day_start.to_datetime64()
        times = (base + secs.astype("timedelta64[s]")).astype("datetime64[us]").astype(object)
        self._events.extend(
            zip(uids.tolist(), times.tolist(), itertools.repeat(event_type))
        )

    def _simulate_daily(
        self,
        *,
        user_ids: np.ndarray,
        user_segments: np.ndarray,
        dim_values: dict,
        user_ret_mult: np.ndarray,
        user_act_mult: np.ndarray | None,
        cohort_week: int,
        duration_weeks: int,
        cohort_exp_state: list[tuple],
    ) -> None:
        """Per-day streak/habit simulation for one weekly signup cohort.

        Each user is stepped day by day from signup to the end of the horizon.
        The daily return probability rises with the current streak (habit
        forming, saturating toward `daily.return.habit_max`) for users who were
        active yesterday, and decays with days-since-active for lapsed users
        (`daily.lapse`, the daily analogue of the weekly REAC/RESU arrows).
        Notifications lift the reactivation probability of idle users; a streak
        freeze protects a streak across a single missed day. Shocks, experiment
        effects, seasonality and per-user variance reuse the weekly machinery.

        Events emitted: `base_event` (lessons, counted per active day),
        `streak_milestone` (at configured thresholds), `streak_lost` (on a
        break of a streak >= `streak.emit_lost_min_streak`), `streak_freeze_used`,
        and `notification_sent` / `notification_opened`.
        """
        rng = self.rng
        ret = self.daily_cfg.get("return", {})
        lap = self.daily_cfg.get("lapse", {})
        p_new = ret.get("new_user", 0.5)
        p_habit_max = ret.get("habit_max", 0.9)
        habit_gain = ret.get("habit_gain", 0.06)
        lap_base = lap.get("base", 0.15)
        lap_decay = lap.get("decay", 0.5)
        lap_floor = lap.get("floor", 0.0)
        # weekday_multipliers may be a single list (global, the chess/duolingo
        # case) OR a {segment: list} map — resolved per segment below.
        weekday_cfg = self.daily_cfg.get("weekday_multipliers")

        # Causal event-mix edges (same as the weekly engine), applied per active
        # day. With no `feature_probabilities` and no `causality` block the day
        # stays single-event — identical to before (chess_daily, duolingo_hard).
        causality = self.causality
        weights = causality["weights"] if causality else {}
        base_views = causality["base_views"] if causality else 0
        noise = causality["noise_scale"] if causality else 0.0
        activity_weights = causality.get("activity_weights", {}) if causality else {}
        marginal_page_mult = (
            causality.get("marginal_page_multiplier", {}) if causality else {}
        )

        freeze_every = self.streak_cfg.get("freeze_earn_every", 0)
        max_freezes = self.streak_cfg.get("max_freezes", 0)
        freezes_on = bool(freeze_every and max_freezes)
        milestones = sorted(self.streak_cfg.get("milestones", []))
        lost_min = self.streak_cfg.get("emit_lost_min_streak", 1)

        notif_trigger = self.notifications_cfg.get("idle_days_trigger")
        notif_max = self.notifications_cfg.get("idle_days_max")  # taper off after this
        notif_open = self.notifications_cfg.get("open_rate", 0.0)
        notif_boost = self.notifications_cfg.get("reactivation_boost", 1.0)

        total_days = duration_weeks * 7
        cohort_start = self.start_monday + pd.DateOffset(weeks=cohort_week)

        for seg_name in self.segments:
            idx = np.where(user_segments == seg_name)[0]
            n = len(idx)
            if n == 0:
                continue
            seg = self.segments[seg_name]
            sample_lessons = self._activity_sampler(seg["activity_distribution"])
            feat_probs = seg.get("feature_probabilities", {})
            # Per-segment weekday rhythm (different user groups, different days).
            if isinstance(weekday_cfg, dict):
                seg_weekday = weekday_cfg.get(seg_name, weekday_cfg.get("default"))
            else:
                seg_weekday = weekday_cfg
            # Optional per-segment stickiness: committed learners retain better,
            # so early activity (which tracks segment) predicts retention — the
            # signal the activation methodology is meant to recover.
            seg_sticky = seg.get("retention_mult", 1.0)
            seg_ids = user_ids[idx]
            seg_ret = user_ret_mult[idx] * seg_sticky
            seg_act = user_act_mult[idx] if user_act_mult is not None else None
            attrs = {"segment": seg_name}
            for dim in self.dimensions:
                attrs[dim] = dim_values[dim][idx]
            exp_state = [(e, t[idx], x) for e, t, x in cohort_exp_state]

            streak = np.zeros(n, dtype=int)
            days_idle = np.zeros(n, dtype=int)
            freezes = np.zeros(n, dtype=int)

            for d in range(total_days):
                abs_week = cohort_week + d // 7
                day_start = cohort_start + pd.Timedelta(days=d)

                if d == 0:
                    # Signup day: every new user is active by definition.
                    active = np.ones(n, dtype=bool)
                else:
                    was_active = days_idle == 0
                    # Habit curve: streak == 1 -> p_new, long streak -> habit_max.
                    p_streak = p_habit_max - (p_habit_max - p_new) * np.exp(
                        -habit_gain * np.maximum(streak - 1, 0)
                    )
                    p_lap = np.maximum(
                        lap_floor, lap_base * lap_decay ** np.maximum(days_idle - 1, 0)
                    )
                    # Notifications to idle users lift their reactivation odds.
                    if notif_trigger is not None:
                        notify = (~was_active) & (days_idle >= notif_trigger)
                        if notif_max is not None:
                            # Apps stop nudging the long-dormant; this also keeps
                            # notification volume from swamping the lesson signal.
                            notify &= days_idle <= notif_max
                        self._emit_daily(seg_ids[notify], day_start, "notification_sent")
                        opened = notify & (rng.random(n) < notif_open)
                        self._emit_daily(seg_ids[opened], day_start, "notification_opened")
                        exp_notif = self._experiment_vec(
                            "notification", abs_week, exp_state, seg_name, n
                        )
                        p_lap = np.where(opened, p_lap * notif_boost * exp_notif, p_lap)
                    p_today = np.where(was_active, p_streak, p_lap)
                    p_today = p_today * seg_ret
                    p_today = p_today * self._shock_vec(
                        "retention", abs_week, cohort_week, attrs, n
                    )
                    p_today = p_today * self._experiment_vec(
                        "retention", abs_week, exp_state, seg_name, n
                    )
                    if self.ts_ret is not None:
                        p_today = p_today * self.ts_ret[abs_week]
                    if seg_weekday is not None:
                        p_today = p_today * seg_weekday[day_start.weekday()]
                    active = rng.random(n) < np.minimum(p_today, 0.99)

                inactive = ~active
                if d > 0:
                    missed = inactive & (streak > 0)
                    if freezes_on:
                        use_freeze = missed & (freezes > 0)
                        freezes[use_freeze] -= 1
                        self._emit_daily(seg_ids[use_freeze], day_start, "streak_freeze_used")
                        broke = missed & ~use_freeze
                    else:
                        use_freeze = np.zeros(n, dtype=bool)
                        broke = missed
                    self._emit_daily(
                        seg_ids[broke & (streak >= lost_min)], day_start, "streak_lost"
                    )
                    streak[broke] = 0
                    # Frozen days are "covered": they do not deepen the idle gap.
                    days_idle[inactive & ~use_freeze] += 1

                act = np.where(active)[0]
                if len(act) == 0:
                    continue
                streak[act] += 1
                days_idle[act] = 0
                if freezes_on:
                    earned = act[streak[act] % freeze_every == 0]
                    freezes[earned] = np.minimum(freezes[earned] + 1, max_freezes)
                for m in milestones:
                    self._emit_daily(seg_ids[act[streak[act] == m]], day_start, "streak_milestone")

                # Lessons completed this day, with activity multipliers
                # (stochastic rounding keeps fractional multipliers unbiased).
                act_attrs = {
                    k: (v[act] if isinstance(v, np.ndarray) else v)
                    for k, v in attrs.items()
                }
                act_exp = [(e, t[act], x) for e, t, x in exp_state]
                lessons = sample_lessons(len(act)).astype(float)
                a_mult = self._shock_vec(
                    "activity", abs_week, cohort_week, act_attrs, len(act)
                ) * self._experiment_vec("activity", abs_week, act_exp, seg_name, len(act))
                if self.ts_act is not None:
                    a_mult = a_mult * self.ts_act[abs_week]
                if seg_act is not None:
                    a_mult = a_mult * seg_act[act]
                scaled = lessons * a_mult
                floor = np.floor(scaled).astype(int)
                counts = np.maximum(1, floor + (rng.random(len(act)) < scaled - floor))

                # Single-event day (chess/duolingo): emit the base event and stop
                # — no extra RNG draws, so those datasets are byte-for-byte stable.
                if not feat_probs and causality is None:
                    self._emit_daily(seg_ids[act], day_start, self.base_event, counts)
                    continue

                # --- Full event mix + causal page views on this active day, the
                #     vectorised analogue of the weekly engine's per-user block. ---
                n_act = len(act)
                feat_mult = {
                    feat: self._shock_vec(
                        "feature", abs_week, cohort_week, act_attrs, n_act, target=feat
                    )
                    * self._experiment_vec(
                        "feature", abs_week, act_exp, seg_name, n_act, target=feat
                    )
                    for feat in feat_probs
                }

                # 1. Features on the BASE events at the full per-event rate.
                base_feat = {
                    feat: self._feature_count_vec(prob, counts, feat_mult[feat], rng)
                    for feat, prob in feat_probs.items()
                }

                # 2. Causal edge: feature usage spawns EXTRA base events (e.g.
                #    used_ai -> more shares). Stochastic rounding stays unbiased.
                if activity_weights:
                    extra = np.zeros(n_act, dtype=float)
                    for f, w in activity_weights.items():
                        extra = extra + base_feat.get(f, 0) * w
                    extra_floor = np.floor(extra).astype(int)
                    extra_pages = extra_floor + (rng.random(n_act) < extra - extra_floor)
                else:
                    extra_pages = np.zeros(n_act, dtype=int)

                # 3. Features on the EXTRA events at a DILUTED rate (the spawned
                #    events are lower-quality — marginal_page_multiplier).
                action_counts = {}
                for feat, prob in feat_probs.items():
                    n_extra = self._feature_count_vec(
                        prob,
                        extra_pages,
                        feat_mult[feat] * marginal_page_mult.get(feat, 1.0),
                        rng,
                    )
                    action_counts[feat] = base_feat[feat] + n_extra

                total_base = counts + extra_pages
                action_counts[self.base_event] = total_base

                # 4. Emit base + feature events for the active users.
                self._emit_daily(seg_ids[act], day_start, self.base_event, total_base)
                for feat, fc in action_counts.items():
                    if feat == self.base_event:
                        continue
                    self._emit_daily(seg_ids[act], day_start, feat, fc)

                # 5. Causal page views, driven by the (amplified) actions.
                if causality is not None:
                    weighted = np.zeros(n_act)
                    for action, cnt in action_counts.items():
                        w = weights.get(action, 0)
                        if w:
                            weighted = weighted + cnt * w
                    views = np.maximum(
                        1, ((base_views + weighted) * rng.normal(1.0, noise, n_act)).astype(int)
                    )
                    self._emit_daily(seg_ids[act], day_start, PAGE_VIEW, views)

    # ------------------------------------------------------------------ #
    # Run + post-processing
    # ------------------------------------------------------------------ #
    def _acquisition_multiplier(self, week: int) -> float:
        mult = 1.0
        for shock in self.shocks:
            if shock["type"] != "acquisition":
                continue
            if shock["start_week"] <= week <= shock["end_week"]:
                mult *= shock["multiplier"]
        return mult

    def run(self) -> pd.DataFrame:
        """Run the full simulation and return an enriched event DataFrame."""
        g = self.config["global_settings"]
        base_users = g["base_users"]
        growth = g["growth_rate"]
        periods = g["periods_to_simulate"]

        self._events, self._users, self._assignments = [], [], []
        user_id_ptr = 0
        for week in range(periods):
            mult = self._acquisition_multiplier(week)
            if self.ts_acq is not None:
                mult = mult * self.ts_acq[week]
            n_users = int(base_users * ((1 + growth) ** week) * mult)
            self._generate_cohort(
                n_users,
                duration_weeks=periods - week,
                start_user_id=user_id_ptr,
                cohort_week=week,
            )
            user_id_ptr += n_users

        self.users_df = pd.DataFrame(self._users)
        self.assignments_df = pd.DataFrame(
            self._assignments,
            columns=["experiment", "user_id", "group", "first_exposed_week"],
        )

        df = pd.DataFrame(
            self._events, columns=["user_id", "event_time", "event_type"]
        )
        self._events = []
        df = self.enrich(df)
        df = self._attach_attributes(df)
        return self._apply_instrumentation(df)

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add date / week / abs_week helper columns used by analysis."""
        if df.empty:
            return df
        df = df.sort_values("event_time").reset_index(drop=True)
        df["date"] = df["event_time"].dt.floor("D")
        # Week start = the Monday of each event's week; abs_week is the
        # 0-indexed week from the simulation start, matching the shock windows.
        df["week"] = df["date"] - pd.to_timedelta(df["event_time"].dt.weekday, unit="D")
        df["abs_week"] = ((df["week"] - self.start_monday).dt.days // 7).astype(int)
        return df

    def _attach_attributes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge per-user columns onto events; derive per-event app_version."""
        if df.empty or self.users_df is None or self.users_df.empty:
            return df
        cols = ["user_id", "segment"] + list(self.dimensions)
        adoption_cols = [
            f"adopted_{r['version']}_week" for r in self.release_versions
        ]
        df = df.merge(self.users_df[cols + adoption_cols], on="user_id", how="left")
        if self.release_versions:
            version = np.full(len(df), self.release_versions[0]["version"], dtype=object)
            for rel in self.release_versions[1:]:
                adopted = df[f"adopted_{rel['version']}_week"].to_numpy()
                version = np.where(df["abs_week"] >= adopted, rel["version"], version)
            df["app_version"] = version
        return df.drop(columns=adoption_cols)

    def _apply_instrumentation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Silently drop logged events for `instrumentation` shocks.

        Runs *after* generation, so behaviour that depends on the true actions
        (causal page views) is untouched — only the recorded events go
        missing, exactly like a real tracking bug. `where` filters may use
        any event column (segment, dimensions, app_version, ...).
        """
        for shock in self.shocks:
            if shock["type"] != "instrumentation":
                continue
            mask = (df["event_type"] == shock["target_feature"]) & df[
                "abs_week"
            ].between(shock["start_week"], shock["end_week"])
            for col, allowed in (shock.get("where") or {}).items():
                mask &= df[col].isin(allowed)
            idx = df.index[mask]
            # `multiplier` is the share of events that still get logged.
            dropped = idx[self.rng.random(len(idx)) >= shock["multiplier"]]
            df = df.drop(dropped)
        return df.reset_index(drop=True)
