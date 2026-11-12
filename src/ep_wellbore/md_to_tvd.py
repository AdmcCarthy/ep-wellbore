"""Measured depth to true vertical depth from a deviation survey: the one shared way, for any number of wellbores and depths at once.

Every caller (a formation top, a test, a sample, a core, a pretest, a log depth) asks the same question of the same survey and gets the same answer with the same provenance.

* The path between two stations is the **minimum-curvature arc** (AGENTS section 6B), not a straight line: the tangent is turned along the arc between the two stations and the depth
  increment is the minimum-curvature increment over the part of the arc that is used.  At a station the value is the station's own TVD, so the answer is consistent with the
  survey as delivered; between stations a straight-line interpolation is wrong by metres in a build section (a 400 m arc that builds 46 degrees is 15 m out at the middle).
* **Nothing is extrapolated.**  Above the first station (a riser, a template) and below the last one the answer is empty and `outside_by_m` says by how much, so a caller can tell a
  survey that stops short of TD from a wellbore with no survey.  A TVD is never fabricated (a vertical well is not assumed).
* The datum is the caller's: stations carry TVD in one datum (for example RT) and the answer is in that datum; the caller subtracts the rotary table elevation for TVDSS.
* Vectorised: stations and queries of all wellbores are joined with two binary searches on a composite wellbore-and-depth key (no loop over wellbores), so a million depths take well
  under a second.  Which survey belongs to a wellbore is decided before (one survey per wellbore key; `ep_wellbore.wellhead.select_trusted_surveys`).

Pure functions return `(result, events)`; events are plain dictionaries (`rule`, `rows_affected`, `severity`, `detail`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SMALL = 1.0e-9


def _event(rule: str, rows: int, severity: str, detail: str) -> dict:
    return {"rule": rule, "rows_affected": int(rows), "severity": severity, "detail": detail}


def tvd_at_md(station_well, station_md, station_tvd, station_inclination_deg, station_azimuth_deg, query_well, query_md,
              station_north=None, station_east=None) -> tuple[pd.DataFrame, list[dict]]:
    """TVD (and, when given, the north and east offsets) at the query depths, from the stations of the same wellbore key.

    Returns one row per query, in the order given: `tvd_m` (and `north_m`, `east_m` when the station offsets are given; they follow the azimuth reference of the stations, so grid offsets need grid
    azimuths), `inclination_deg` and `azimuth_deg` at the depth (the tangent of the arc; empty between stations without angles), `inside_survey`, `interp_method` (`station` when the depth is a station,
    `minimum_curvature` between two stations with angles, `linear` between two stations without angles, empty outside), `bracket_md_m` (the length of the station interval the depth lies
    in: a coarse bracket is a weaker answer), `nearest_station_m` (distance to the nearest station of that interval) and `outside_by_m` (signed: negative above the first station, positive
    below the last, empty inside).  Stations with a missing depth or TVD, and repeated depths, are left out and counted.
    """
    q_md = np.asarray(query_md, dtype="float64")
    n_q = len(q_md)
    events: list[dict] = []
    codes = pd.factorize(pd.concat([pd.Series(station_well), pd.Series(query_well)], ignore_index=True))[0].astype(np.int64)
    n_s = len(station_md)
    s_code, q_code = codes[:n_s], codes[n_s:]
    md, tvd = np.asarray(station_md, dtype="float64"), np.asarray(station_tvd, dtype="float64")
    inc, azi = np.asarray(station_inclination_deg, dtype="float64"), np.asarray(station_azimuth_deg, dtype="float64")
    north = np.asarray(station_north, dtype="float64") if station_north is not None else np.full(n_s, np.nan)
    east = np.asarray(station_east, dtype="float64") if station_east is not None else np.full(n_s, np.nan)

    usable = np.isfinite(md) & np.isfinite(tvd)
    if (~usable).any():
        events.append(_event("md_to_tvd_station_unusable", (~usable).sum(), "warning", "station with a missing depth or TVD: left out"))
    idx = np.flatnonzero(usable)
    order = np.lexsort((md[idx], s_code[idx]))
    idx = idx[order]
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = (s_code[idx][1:] != s_code[idx][:-1]) | (md[idx][1:] != md[idx][:-1])
    if (~keep).any():
        events.append(_event("md_to_tvd_station_repeated", (~keep).sum(), "info", "station at a depth already given for the wellbore: the first is used"))
    idx = idx[keep]
    out = pd.DataFrame({"tvd_m": np.full(n_q, np.nan), "north_m": np.full(n_q, np.nan), "east_m": np.full(n_q, np.nan), "inclination_deg": np.full(n_q, np.nan), "azimuth_deg": np.full(n_q, np.nan),
                        "inside_survey": np.zeros(n_q, dtype=bool),
                        "interp_method": np.full(n_q, "", dtype=object), "bracket_md_m": np.full(n_q, np.nan), "nearest_station_m": np.full(n_q, np.nan), "outside_by_m": np.full(n_q, np.nan)})
    if not len(idx) or not n_q:
        out["outside_by_m"] = np.nan
        return out, events

    sc, smd, stvd, sinc, sazi, snorth, seast = s_code[idx], md[idx], tvd[idx], inc[idx], azi[idx], north[idx], east[idx]
    lo = float(min(smd.min(), np.nanmin(q_md) if np.isfinite(q_md).any() else smd.min()))
    stride = float(max(smd.max(), np.nanmax(q_md) if np.isfinite(q_md).any() else smd.max()) - lo) + 1.0
    skey = sc * stride + (smd - lo)
    finite = np.isfinite(q_md)
    qkey = np.where(finite, q_code * stride + (np.where(finite, q_md, lo) - lo), -1.0)
    i = np.searchsorted(skey, qkey, side="right") - 1                       # the station at or before the query
    ic = np.clip(i, 0, len(skey) - 1)
    same = finite & (i >= 0) & (sc[ic] == q_code)                           # the query's wellbore has a station at or above it
    exact = same & (smd[ic] == q_md)
    nxt = np.clip(i + 1, 0, len(skey) - 1)
    between = same & ~exact & (i + 1 < len(skey)) & (sc[nxt] == q_code)
    inside = exact | between
    out["inside_survey"] = inside

    # ---- at a station: the station's own values --------------------------------------------------------------------------------------------------------------------------------
    e = np.flatnonzero(exact)
    out.loc[e, ["tvd_m", "north_m", "east_m", "inclination_deg", "azimuth_deg"]] = np.column_stack([stvd[ic[e]], snorth[ic[e]], seast[ic[e]], sinc[ic[e]], sazi[ic[e]]])
    out.loc[e, "interp_method"] = "station"
    out.loc[e, "nearest_station_m"] = 0.0

    # ---- between two stations: the minimum-curvature arc -----------------------------------------------------------------------------------------------------------------------
    b = np.flatnonzero(between)
    if len(b):
        a, c = ic[b], nxt[b]
        dmd = smd[c] - smd[a]
        length = q_md[b] - smd[a]
        s = length / dmd
        angles = np.isfinite(sinc[a]) & np.isfinite(sinc[c]) & np.isfinite(sazi[a]) & np.isfinite(sazi[c])
        i1, i2, a1, a2 = np.radians(sinc[a]), np.radians(sinc[c]), np.radians(sazi[a]), np.radians(sazi[c])
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_beta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
            beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))                  # the dogleg of the whole interval
            small = beta < _SMALL
            sb = np.where(small, 1.0, np.sin(beta))
            w1 = np.where(small, 1.0 - s, np.sin((1.0 - s) * beta) / sb)    # tangent at the fraction s of the arc (spherical interpolation of the two tangents)
            w2 = np.where(small, s, np.sin(s * beta) / sb)
            t1 = np.stack([np.sin(i1) * np.cos(a1), np.sin(i1) * np.sin(a1), np.cos(i1)])
            t2 = np.stack([np.sin(i2) * np.cos(a2), np.sin(i2) * np.sin(a2), np.cos(i2)])
            ts = w1 * t1 + w2 * t2
            norm = np.linalg.norm(ts, axis=0)
            ts = ts / np.where(norm > 0, norm, 1.0)
            i_s = np.arccos(np.clip(ts[2], -1.0, 1.0))
            a_s = np.arctan2(ts[1], ts[0])
            part = s * beta                                                 # the angle turned over the part of the arc that is used
            half = part / 2.0
            ratio = np.where(part < _SMALL, 1.0, np.tan(np.where(part < _SMALL, 1.0, half)) / np.where(part < _SMALL, 1.0, half))
            d_tvd = length / 2.0 * (np.cos(i1) + np.cos(i_s)) * ratio
            d_north = length / 2.0 * (np.sin(i1) * np.cos(a1) + np.sin(i_s) * np.cos(a_s)) * ratio
            d_east = length / 2.0 * (np.sin(i1) * np.sin(a1) + np.sin(i_s) * np.sin(a_s)) * ratio
        linear_tvd = stvd[a] + s * (stvd[c] - stvd[a])
        out.loc[b, "tvd_m"] = np.where(angles, stvd[a] + d_tvd, linear_tvd)
        out.loc[b, "north_m"] = np.where(angles, snorth[a] + d_north, snorth[a] + s * (snorth[c] - snorth[a]))
        out.loc[b, "east_m"] = np.where(angles, seast[a] + d_east, seast[a] + s * (seast[c] - seast[a]))
        out.loc[b, "inclination_deg"] = np.where(angles, np.degrees(i_s), np.nan)                      # the angles at the depth: the tangent of the arc there
        out.loc[b, "azimuth_deg"] = np.where(angles, np.degrees(a_s) % 360.0, np.nan)
        out.loc[b, "interp_method"] = np.where(angles, "minimum_curvature", "linear")
        out.loc[b, "bracket_md_m"] = dmd
        out.loc[b, "nearest_station_m"] = np.minimum(length, dmd - length)
        no_angles = int((~angles).sum())
        if no_angles:
            events.append(_event("md_to_tvd_linear_between_stations", no_angles, "info", "stations without inclination or azimuth: TVD interpolated linearly between them"))

    # ---- outside the survey: empty, and by how much ----------------------------------------------------------------------------------------------------------------------------
    outside = finite & ~inside
    first = pd.Series(smd).groupby(sc).min()
    last = pd.Series(smd).groupby(sc).max()
    has_survey = outside & pd.Series(q_code).isin(first.index).to_numpy()
    above = has_survey & (q_md < pd.Series(q_code).map(first).to_numpy(dtype="float64"))
    below = has_survey & ~above
    out.loc[above, "outside_by_m"] = (q_md - pd.Series(q_code).map(first).to_numpy(dtype="float64"))[above]
    out.loc[below, "outside_by_m"] = (q_md - pd.Series(q_code).map(last).to_numpy(dtype="float64"))[below]
    if outside.any():
        events.append(_event("md_to_tvd_outside_survey", outside.sum(), "info", f"{int(has_survey.sum()):,} of them in a wellbore with a survey (above the first or below the last station): left empty, never extrapolated; "
                             f"{int((outside & ~has_survey).sum()):,} in a wellbore with no survey"))
    out["interp_method"] = out["interp_method"].astype("category")
    return out, events


def interpolate_along_md(station_well, station_md, station_values: dict, query_well, query_md) -> tuple[pd.DataFrame, list[dict]]:
    """Linear interpolation along the hole of quantities that are not geometry (an uncertainty, a sigma): the same station rules as `tvd_at_md` (the wellbore's own stations, nothing
    extrapolated, `inside_survey` and `outside_by_m` returned), one column per entry of `station_values`.  A station with a missing value is left out of that column."""
    n_s = len(station_md)
    nan = np.full(n_s, np.nan)
    frames, events = [], []
    for name, values in station_values.items():
        res, more = tvd_at_md(station_well, station_md, values, nan, nan, query_well, query_md)
        frames.append(res[["tvd_m"]].rename(columns={"tvd_m": name}))
        events += more
        shape = res
    out = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=range(len(query_md)))
    if frames:
        out["inside_survey"], out["outside_by_m"] = shape["inside_survey"].to_numpy(), shape["outside_by_m"].to_numpy()
    return out, events