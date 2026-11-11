"""Well trajectory mathematics: the minimum-curvature method (AGENTS section 6B: the only method allowed for computing TVD from a survey).

Pure functions on arrays.  Units are the caller's: depths in, depths out in the same unit; `dls_basis` is the length the dogleg severity is quoted per, in that same unit.
Nothing is extrapolated and nothing is invented: the path starts from the start values given and runs through the stations in the order given."""
from __future__ import annotations

import numpy as np
import pandas as pd


def minimum_curvature(md, inclination_deg, azimuth_deg, start_tvd: float = 0.0, start_north: float = 0.0, start_east: float = 0.0, dls_basis: float = 30.0) -> pd.DataFrame:
    """Position of every station by the minimum-curvature method.

    Inputs are station arrays in order of increasing measured depth.  Returns a frame with `tvd`, `north`, `east`, `dogleg_deg` (the angle between the two tangents of the interval)
    and `dls` (dogleg severity in degrees per `dls_basis` of measured depth; null for the first station).  The azimuth must be on the reference the offsets are wanted in (grid or true north)."""
    md = np.asarray(md, dtype=float)
    inc = np.radians(np.asarray(inclination_deg, dtype=float))
    azi = np.radians(np.asarray(azimuth_deg, dtype=float))
    n = len(md)
    out = pd.DataFrame({"tvd": np.full(n, np.nan), "north": np.full(n, np.nan), "east": np.full(n, np.nan), "dogleg_deg": np.full(n, np.nan), "dls": np.full(n, np.nan)})
    if n == 0:
        return out
    out.loc[0, ["tvd", "north", "east"]] = [start_tvd, start_north, start_east]
    if n == 1:
        return out
    dmd = np.diff(md)
    i1, i2, a1, a2 = inc[:-1], inc[1:], azi[:-1], azi[1:]
    cos_beta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    small = beta < 1.0e-9
    ratio = np.where(small, 1.0, np.tan(np.where(small, 1.0, beta) / 2.0) / np.where(small, 1.0, beta / 2.0))      # ratio factor 2/beta * tan(beta/2); 1 on a straight interval
    d_north = dmd / 2.0 * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * ratio
    d_east = dmd / 2.0 * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * ratio
    d_tvd = dmd / 2.0 * (np.cos(i1) + np.cos(i2)) * ratio
    out.loc[1:, "tvd"] = start_tvd + np.cumsum(d_tvd)
    out.loc[1:, "north"] = start_north + np.cumsum(d_north)
    out.loc[1:, "east"] = start_east + np.cumsum(d_east)
    out.loc[1:, "dogleg_deg"] = np.degrees(beta)
    with np.errstate(divide="ignore", invalid="ignore"):
        out.loc[1:, "dls"] = np.where(dmd > 0, np.degrees(beta) / dmd * dls_basis, np.nan)
    return out


def compute_tvd_from_survey(md, inclination_deg, azimuth_deg, start_tvd: float = 0.0) -> np.ndarray:
    """TVD at every station by minimum curvature (the function named in the project rules; it is the `tvd` column of `minimum_curvature`)."""
    return minimum_curvature(md, inclination_deg, azimuth_deg, start_tvd=start_tvd)["tvd"].to_numpy()


def survey_quality(md, inclination_deg, azimuth_deg, max_gap: float = 150.0) -> list[str]:
    """Reasons a survey should not be trusted without a look: stations out of order or repeated, angles out of range, missing values, a gap larger than `max_gap` (md units) between stations."""
    md = np.asarray(md, dtype=float)
    inc = np.asarray(inclination_deg, dtype=float)
    azi = np.asarray(azimuth_deg, dtype=float)
    flags: list[str] = []
    if np.isnan(md).any() or np.isnan(inc).any() or np.isnan(azi).any():
        flags.append("missing_values")
    d = np.diff(md[~np.isnan(md)])
    if (d < 0).any():
        flags.append("md_not_increasing")
    if (d == 0).any():
        flags.append("repeated_md")
    if (np.nanmin(inc, initial=0) < 0) or (np.nanmax(inc, initial=0) > 180):
        flags.append("inclination_out_of_range")
    if (np.nanmin(azi, initial=0) < -1.0) or (np.nanmax(azi, initial=0) > 360.0):
        flags.append("azimuth_out_of_range")
    if len(d) and (d > max_gap).any():
        flags.append("large_station_gap")
    return flags


def position_at_md(station_md, station_values: dict[str, np.ndarray], query_md) -> pd.DataFrame:
    """Interpolate station quantities (for example `tvd`, `north`, `east`, `inclination`) to any measured depths.

    Linear between stations (the stations of a definitive survey are close enough that the curvature between two is small; use `minimum_curvature` on a denser set when it matters).
    Outside the surveyed interval the answer is NULL: a path is never extrapolated, and the interval above the first station (a riser) or below the last is not invented.
    Azimuth is not interpolated here (angles wrap); interpolate north and east instead."""
    md = np.asarray(station_md, dtype=float)
    q = np.asarray(query_md, dtype=float)
    out = pd.DataFrame(index=range(len(q)))
    inside = (q >= md.min()) & (q <= md.max()) if len(md) else np.zeros(len(q), dtype=bool)
    for name, values in station_values.items():
        v = np.interp(q, md, np.asarray(values, dtype=float))
        out[name] = np.where(inside, v, np.nan)
    out["inside_survey"] = inside
    return out
