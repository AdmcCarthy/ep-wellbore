"""Place depth observations (points and intervals) against depth intervals of the same wellbore: which interval holds a sample, a test or a core, and how much of it.

The join is vectorised for large queries.  Intervals are sorted once on a composite key (wellbore rank, depth), and every observation finds its candidate intervals with two
binary searches (`numpy.searchsorted`), so the cost is O((n + m) log m) plus the number of matches, with no Python loop over wellbores or observations and no cross join.

Conventions (one for every caller):

* Depths are measured depths along the wellbore from one datum; the caller says which (the library does not convert).
* An interval holds a depth when ``top <= depth < base`` (top inclusive, base exclusive), so a point on a boundary belongs to the interval below it and never to both.
* An observation with ``base > top`` is an interval and matches every interval it overlaps by more than `min_overlap_m`; ``base == top`` (or a missing base) is a point.
* Intervals may overlap each other (several hierarchy levels, a repeated section): every match is returned, none is dropped.

Pure functions return `(result, events)` where they flag data; events are plain dictionaries (`rule`, `rows_affected`, `severity`, `detail`) that the host turns into its own log rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _event(rule: str, rows: int, severity: str, detail: str) -> dict:
    return {"rule": rule, "rows_affected": int(rows), "severity": severity, "detail": detail}


def overlap_pairs(obs_well, obs_top, obs_base, int_well, int_top, int_base, min_overlap_m: float = 0.0) -> tuple[pd.DataFrame, list[dict]]:
    """Every (observation, interval) pair in the same wellbore whose depths meet.

    Arguments are equal-length arrays per side: a wellbore key (any hashable, integers are fastest), the top and the base.  A missing or non-finite observation base makes it a point.
    Returns a frame with `obs_pos` and `int_pos` (positions in the inputs, so the caller keeps its own columns), `overlap_m` (0 for a point) and `obs_length_m`, sorted by
    `obs_pos`, `int_pos`.  Observations with no interval, intervals with a missing or inverted depth and observations with a missing top are not in the result; the events count them.
    """
    codes = pd.factorize(pd.concat([pd.Series(obs_well), pd.Series(int_well)], ignore_index=True))[0].astype(np.int64)      # one integer code per wellbore across both sides
    n_obs = len(obs_top)
    ow, iw = codes[:n_obs], codes[n_obs:]
    ot, ob = np.asarray(obs_top, dtype="float64"), np.asarray(obs_base, dtype="float64")
    it, ib = np.asarray(int_top, dtype="float64"), np.asarray(int_base, dtype="float64")
    events: list[dict] = []

    no_top = ~np.isfinite(ot)
    ob = np.where(np.isfinite(ob), ob, ot)                                  # a missing base: a point
    inverted_obs = np.isfinite(ot) & (ob < ot)
    ok_obs = ~no_top & ~inverted_obs
    if no_top.any():
        events.append(_event("depth_link_observation_without_depth", no_top.sum(), "info", "observation with no top depth: not placed"))
    if inverted_obs.any():
        events.append(_event("depth_link_observation_inverted", inverted_obs.sum(), "warning", "observation base above its top: not placed, kept in the source"))
    ok_int = np.isfinite(it) & np.isfinite(ib) & (ib > it)
    bad_int = int(len(it) - ok_int.sum())
    if bad_int:
        events.append(_event("depth_link_interval_unusable", bad_int, "warning", "interval with a missing, zero or inverted depth: not used"))

    int_idx = np.flatnonzero(ok_int)
    obs_idx = np.flatnonzero(ok_obs)
    empty = pd.DataFrame({"obs_pos": np.empty(0, np.int64), "int_pos": np.empty(0, np.int64), "overlap_m": np.empty(0), "obs_length_m": np.empty(0)})
    if not len(int_idx) or not len(obs_idx):
        return empty, events

    lo_all = float(min(it[int_idx].min(), ot[obs_idx].min()))
    stride = float(max(ib[int_idx].max(), ob[obs_idx].max()) - lo_all) + 1.0
    ckey_t = iw[int_idx] * stride + (it[int_idx] - lo_all)                  # the composite key keeps the wellbores apart on one axis
    ckey_b = iw[int_idx] * stride + (ib[int_idx] - lo_all)
    order = np.argsort(ckey_t, kind="stable")
    int_sorted, ct, cb = int_idx[order], ckey_t[order], ckey_b[order]
    cmax = np.maximum.accumulate(cb)                                        # first interval whose running maximum base lies below a depth cannot overlap it

    a = ow[obs_idx] * stride + (ot[obs_idx] - lo_all)
    b = ow[obs_idx] * stride + (ob[obs_idx] - lo_all)
    point = (ob[obs_idx] - ot[obs_idx]) <= 0.0
    first = np.searchsorted(cmax, a, side="right")                          # first candidate: running maximum base above the observation top
    last = np.where(point, np.searchsorted(ct, a, side="right"), np.searchsorted(ct, b, side="left"))     # one past the last candidate: interval top at or below / below the base
    count = np.maximum(last - first, 0)
    total = int(count.sum())
    if not total:
        return empty, events
    obs_rep = np.repeat(np.arange(len(obs_idx)), count)
    offset = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    cand = first[obs_rep] + offset                                          # position in the sorted intervals

    t_i, b_i = ct[cand], cb[cand]
    ov = np.minimum(b[obs_rep], b_i) - np.maximum(a[obs_rep], t_i)
    keep = np.where(point[obs_rep], (t_i <= a[obs_rep]) & (a[obs_rep] < b_i), ov > float(min_overlap_m))
    obs_rep, cand, ov = obs_rep[keep], cand[keep], np.where(point[obs_rep], 0.0, ov)[keep]
    out = pd.DataFrame({"obs_pos": obs_idx[obs_rep], "int_pos": int_sorted[cand], "overlap_m": ov, "obs_length_m": (ob[obs_idx] - ot[obs_idx])[obs_rep]})
    return out.sort_values(["obs_pos", "int_pos"], ignore_index=True), events


def coverage_range(int_well, int_top, int_base) -> pd.DataFrame:
    """Per wellbore the shallowest top and deepest base of its usable intervals (`well`, `top`, `base`): tells an observation above, below or in a gap of the intervals apart."""
    f = pd.DataFrame({"well": np.asarray(int_well), "top": np.asarray(int_top, dtype="float64"), "base": np.asarray(int_base, dtype="float64")})
    f = f[np.isfinite(f["top"]) & np.isfinite(f["base"]) & (f["base"] > f["top"])]
    return f.groupby("well", sort=False).agg(top=("top", "min"), base=("base", "max")).reset_index()
