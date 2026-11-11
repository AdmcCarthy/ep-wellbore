"""Wellhead, rotary-table and survey-uncertainty logic shared by every trajectory source (pure functions; thresholds come from the adapter or `config/position_uncertainty.yaml`).

Three ideas live here:
* WHERE A SURVEY STARTS tells what kind of drilling slot it is.  A subsea template slot is on the seabed: the first surveyed station lies at the water depth below mean sea level (the wellhead), and the
  part above (water column plus air gap) is a vertical riser that is not part of the surveyed path.  A platform slot is at or above mean sea level.  The logic is checked, not assumed.
* WHICH ELEVATION THE DEPTHS REFER TO: the rotary-table elevation that makes survey end and TD agree wins over one that does not (the depth reference of a TD is the elevation it was measured from).
  A difference between two sources is classified, never silently resolved.
* HOW WELL THE POSITION IS KNOWN: a covariance matrix becomes a one-sigma lateral and vertical position uncertainty with its basis stated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def classify_wellhead(origin_tvdss_m: float, water_depth_m: float | None, rt_m: float | None, tol_m: float = 2.0) -> tuple[str, str]:
    """`subsea` when the first surveyed station is at the seabed, `platform` when it is at or above mean sea level, otherwise `unknown`; with the evidence in words.

    `origin_tvdss_m` is the depth of the first station below mean sea level (positive down); `water_depth_m` below mean sea level; `rt_m` the rotary table above mean sea level."""
    if origin_tvdss_m is None or not np.isfinite(origin_tvdss_m):
        return "unknown", "no survey origin"
    if water_depth_m is not None and np.isfinite(water_depth_m) and abs(origin_tvdss_m - water_depth_m) <= tol_m:
        return "subsea", f"the survey starts {origin_tvdss_m:.1f} m below mean sea level, at the seabed (water depth {water_depth_m:.1f} m): a seabed wellhead; the {water_depth_m + (rt_m or 0):.0f} m above it is a riser, not surveyed"
    lowest = -(rt_m if rt_m is not None and np.isfinite(rt_m) else 0.0) - tol_m
    if lowest <= origin_tvdss_m <= tol_m:
        return "platform", f"the survey starts {origin_tvdss_m:.1f} m below mean sea level, at or above the sea surface: a platform slot"
    return "unknown", f"the survey starts {origin_tvdss_m:.1f} m below mean sea level, neither at the seabed (water depth {water_depth_m}) nor at the sea surface"


def choose_rt(sodir_rt: float, edm_rt: float, md_end_msl_m: float, td_md_m: float, tol_m: float = 0.5) -> tuple[float, str]:
    """The rotary-table elevation to add to below-sea-level depths.  The elevation implied by TD minus survey end decides when it agrees with one of the candidates within `tol_m`
    (official first: the Sodir value wins a tie); without a TD the Sodir value, then the source's own datum.  Returns (elevation [m], source label)."""
    implied = td_md_m - md_end_msl_m if np.isfinite(td_md_m) and np.isfinite(md_end_msl_m) else np.nan
    if np.isfinite(sodir_rt) and np.isfinite(implied) and abs(implied - sodir_rt) <= tol_m:
        return sodir_rt, "sodir"
    if np.isfinite(edm_rt) and np.isfinite(implied) and abs(implied - edm_rt) <= tol_m:
        return edm_rt, "source_datum_td_consistent"
    if np.isfinite(sodir_rt):
        return sodir_rt, "sodir"
    if np.isfinite(edm_rt):
        return edm_rt, "source_datum"
    return np.nan, "missing"


def elevation_verdict(sodir_rt: float, source_rt: float, source_default_rt: float, implied_rt: float, kb_above_rt_m: tuple[float, float] = (0.2, 3.0), tol_m: float = 0.5, same_m: float = 0.05) -> str:
    """Classify the difference between the Sodir elevation and the elevation the source calls the rotary table (the source names its datum).

    agree                       within `tol_m`
    sodir_may_be_kb             Sodir is HIGHER than the rotary table by an amount a kelly bushing above the table can explain (`kb_above_rt_m`, an assumption): Sodir may report the KB
    sodir_equals_source_default Sodir is lower and equals the source's planning default rig elevation: Sodir may carry the planning value while depths refer to the measured table
    sodir_lower_unexplained     Sodir is lower than the table by more than `tol_m`: a kelly bushing sits ABOVE the table, so a KB reading does not explain it
    sodir_higher_unexplained    Sodir is higher by more than the KB band
    unknown                     a value is missing"""
    if not (np.isfinite(sodir_rt) and np.isfinite(source_rt)):
        return "unknown"
    d = sodir_rt - source_rt
    if abs(d) <= tol_m:
        return "agree"
    if d > 0:
        return "sodir_may_be_kb" if kb_above_rt_m[0] <= d <= kb_above_rt_m[1] else "sodir_higher_unexplained"
    if np.isfinite(source_default_rt) and abs(sodir_rt - source_default_rt) <= same_m:
        return "sodir_equals_source_default"
    return "sodir_lower_unexplained"


def position_sigma(cov_xx, cov_yy, cov_zz, cov_xy, to_m: float) -> pd.DataFrame:
    """One-sigma position uncertainty from covariance columns: `sigma_east_m`, `sigma_north_m`, `sigma_lateral_m` (the semi-major axis of the horizontal ellipse, the largest eigenvalue's root)
    and `sigma_vertical_m`.  `to_m` converts the file's length unit (the covariances are in that unit squared).  The CONFIDENCE LEVEL of the file's covariance is not stated by the source:
    the caller records that as the basis; nothing here makes it one-sigma, it only takes the square root."""
    xx = np.asarray(cov_xx, dtype=float)
    yy = np.asarray(cov_yy, dtype=float)
    zz = np.asarray(cov_zz, dtype=float)
    xy = np.nan_to_num(np.asarray(cov_xy, dtype=float))
    half = (xx + yy) / 2.0
    root = np.sqrt(np.maximum(((xx - yy) / 2.0) ** 2 + xy ** 2, 0.0))
    major = np.sqrt(np.maximum(half + root, 0.0))
    return pd.DataFrame({"sigma_east_m": np.sqrt(np.maximum(xx, 0.0)) * to_m, "sigma_north_m": np.sqrt(np.maximum(yy, 0.0)) * to_m, "sigma_lateral_m": major * to_m, "sigma_vertical_m": np.sqrt(np.maximum(zz, 0.0)) * to_m})


def uncertainty_class(sigma_lateral_m: float, sigma_vertical_m: float, limits: dict) -> str:
    """Plain-words class of a position: `tight`, `normal`, `loose`, or `not stated` when a sigma is missing.  The limits are an ASSUMPTION in config/position_uncertainty.yaml."""
    if not (np.isfinite(sigma_lateral_m) and np.isfinite(sigma_vertical_m)):
        return "not stated"
    worst = max(sigma_lateral_m, sigma_vertical_m * float(limits.get("vertical_weight", 1.0)))
    if worst <= float(limits["tight_m"]):
        return "tight"
    return "normal" if worst <= float(limits["normal_m"]) else "loose"


def select_trusted_surveys(survey: pd.DataFrame, ranks: dict[str, int]) -> pd.DataFrame:
    """For every linked wellbore keep the survey set of the most trusted source class (lowest rank number).  Unlinked surveys (no `wellbore_id`) are left out: a survey that is not tied to a wellbore is not selected.

    `ranks` maps `survey_source_class` to a rank; a class not listed ranks last.  Within one rank the set with the most stations wins, then the first survey id (stable)."""
    s = survey[survey["wellbore_id"].notna()].copy()
    if s.empty:
        return s
    s["source_trust_rank"] = s["survey_source_class"].map(ranks).fillna(max(ranks.values(), default=0) + 1).astype(int)
    sets = s.groupby(["wellbore_id", "survey_id"]).agg(rank=("source_trust_rank", "first"), n=("station_no", "size")).reset_index().sort_values(["wellbore_id", "rank", "n", "survey_id"], ascending=[True, True, False, True])
    best = sets.drop_duplicates("wellbore_id")[["wellbore_id", "survey_id"]]
    return s.merge(best, on=["wellbore_id", "survey_id"]).reset_index(drop=True)
