"""Does the age of the tool or practice matter for the uncertainty of a measurement?  A hypothesis test per data type, with the evidence kept apart from the decision.

Hypothesis (decision of 2026-11-05): measurements and tools have improved over time, so older measurements carry more uncertainty than newer ones, but this depends on the data type: not all
leak-off tests are the same, it depends on how the test was executed AND on the tool.  The open data carry no tool or contractor field, so the DATE of the well is the only proxy for the tool
generation and the practice: this is a proxy for both, not a tool effect.  The test therefore reads 'did the indicator change with the year the well was drilled', never 'the tool causes it'.

An INDICATOR is a number per record that grows with uncertainty or falls with it:
    residual      the distance of the measurement from an independent expectation (a leak-off value from the modelled band, a test temperature from the bottom-hole temperature): a larger spread
                  can mean a noisier measurement, but also a more varied well population;
    resolution    the share of values reported at a coarse step (a depth at a multiple of 10 m, a density at a multiple of 0.05 sg): a coarser step means a less precise measurement or report;
    quality flag  the share of records the project itself rates `high` confidence.
Each trend is tested as a rank correlation of the indicator with the drilling year, with a 90 % interval from a bootstrap over WELLS (records of one well are not independent), against the
direction the hypothesis expects.  The verdict uses the interval; the decision status asks whether the change between the first and the last era is larger than a stated tolerance.
Pure functions.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_physics_base import stats as kv

WINDOW_YEARS, ANCHOR = 5, 1965          # the project's 5-year windows


def window_label(year) -> str:
    if pd.isna(year):
        return ""
    y0 = ANCHOR + ((int(year) - ANCHOR) // WINDOW_YEARS) * WINDOW_YEARS
    return f"{y0}–{str(y0 + WINDOW_YEARS - 1)[-2:]}"


def well_year(header: pd.DataFrame) -> pd.Series:
    """Drilling year per wellbore id from the header entry date (the spud year; the completion date is not always held)."""
    d = pd.to_datetime(header["entry_date"], errors="coerce")
    return pd.Series(d.dt.year.to_numpy(), index=header["wellbore_id"].to_numpy(), name="year")


def attach_year(frame: pd.DataFrame, header: pd.DataFrame, id_col: str = "wellbore_id") -> pd.DataFrame:
    y = well_year(header)
    out = frame.copy()
    out["year"] = out[id_col].map(y)
    out["window"] = out["year"].map(window_label)
    return out


def is_multiple(values: pd.Series, step: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    r = np.round(v / step)
    return (np.abs(v - r * step) < 1e-6).astype(float).where(v.notna())


def indicator_frames(lot: pd.DataFrame, header: pd.DataFrame, chk: pd.DataFrame, temp: pd.DataFrame, dst: pd.DataFrame, mud: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """The indicator per data type: {key: {data_type, title, frame (id, year, value), expected: 'falls'|'rises', kind, unit}}.  Records without a year or a value are dropped."""
    out: dict[str, dict[str, Any]] = {}

    def add(key, data_type, title, kind, expected, unit, df, value):
        f = attach_year(df, header)[["wellbore_id", "year", "window"]].assign(value=pd.to_numeric(value, errors="coerce").to_numpy()).dropna(subset=["year", "value"])
        out[key] = {"data_type": data_type, "title": title, "kind": kind, "expected": expected, "unit": unit, "frame": f}

    x = chk[(chk["check_id"] == "chk.lot_vs_fracture_band") & chk["status"].isin(["agree", "disagree"])]
    add("lot_residual", "leak-off and integrity tests", "Distance of the leak-off value from the modelled fracture band", "residual", "falls", "% of the band middle", x, 100 * x["difference"].abs())
    add("lot_resolution", "leak-off and integrity tests", "Leak-off value reported at a multiple of 0.05 sg", "resolution", "falls", "share", lot[lot["lot_emw_sg"].notna()], is_multiple(lot.loc[lot["lot_emw_sg"].notna(), "lot_emw_sg"], 0.05))
    add("shoe_resolution", "casing shoe depth", "Shoe depth reported at a multiple of 10 m", "resolution", "falls", "share", lot[lot["casing_depth_md_m"].notna()], is_multiple(lot.loc[lot["casing_depth_md_m"].notna(), "casing_depth_md_m"], 10.0))
    t = temp[temp["dst_minus_uncorrected_c"].notna()]
    add("temp_residual", "formation temperature", "Distance of the test temperature from the uncorrected bottom-hole temperature", "residual", "falls", "degC", t, t["dst_minus_uncorrected_c"].abs())
    d = dst[dst["pressure_confidence"].notna()]
    add("pressure_quality", "reservoir pressure tests", "Share of test pressures the project rates high confidence", "quality flag", "rises", "share", d, (d["pressure_confidence"].astype(str) == "high").astype(float))
    m = mud[mud["mud_weight_sg"].notna()]
    add("mud_resolution", "mud weight", "Mud weight reported at a multiple of 0.05 sg", "resolution", "falls", "share", m, is_multiple(m["mud_weight_sg"], 0.05))
    return out


def trend_by_window(frame: pd.DataFrame, min_n: int = 30) -> pd.DataFrame:
    """Mean of the indicator and records / wells per 5-year window; windows with fewer than `min_n` records are kept but marked `thin` (shown, not trusted)."""
    g = frame.groupby("window").agg(year=("year", "min"), records=("value", "size"), wells=("wellbore_id", "nunique"), mean=("value", "mean"), median=("value", "median"))
    g["thin"] = g["records"] < min_n
    return g.sort_values("year").reset_index()


def _stats(v: np.ndarray) -> tuple[float, float]:
    """(Spearman rank correlation of the indicator with the year, relative change of the mean over the 10th-to-90th percentile span of the years from a straight-line fit)."""
    x, y = v[:, 0], v[:, 1]
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    rho = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
    slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    span = float(np.quantile(x, 0.9) - np.quantile(x, 0.1))
    mean = float(y.mean())
    return rho, (slope * span / mean if mean > 0 else np.nan)


def trend_test(ind: dict[str, Any], tolerance: float, min_n: int = 30, min_abs_rho: float = 0.10) -> dict[str, Any]:
    """Trend of the indicator with the drilling year, bootstrap over wells: the verdict (against the expected direction, with a minimum effect) and the decision status of the change that the
    straight-line trend predicts between the 10th and the 90th percentile year (robust against thin first and last windows)."""
    f = ind["frame"]
    if len(f) < 100 or f["wellbore_id"].nunique() < 30:
        return {"verdict": "not_tested", "n": len(f), "n_wells": int(f["wellbore_id"].nunique()), "detail": "fewer than 100 records or 30 wells"}
    arr = np.column_stack([f["year"].to_numpy(dtype=float), f["value"].to_numpy(dtype=float)])
    rng = np.random.default_rng(kv.SEED)
    keys, inv = np.unique(f["wellbore_id"].to_numpy(), return_inverse=True)
    idx = [np.flatnonzero(inv == k) for k in range(len(keys))]
    rho, change = _stats(arr)
    sims = np.empty((400, 2))
    for i in range(len(sims)):
        pick = rng.integers(0, len(keys), len(keys))
        sims[i] = _stats(arr[np.concatenate([idx[j] for j in pick])])
    lo, hi = (float(np.nanquantile(sims[:, 0], q)) for q in (0.05, 0.95))
    clo, chi = (float(np.nanquantile(sims[:, 1], q)) for q in (0.05, 0.95))
    sign = -1.0 if ind["expected"] == "falls" else 1.0
    if sign * lo > 0 and abs(rho) >= min_abs_rho:
        verdict = "confirmed"
    elif sign * hi < 0 and abs(rho) >= min_abs_rho:
        verdict = "contradicted"
    elif (lo > 0 or hi < 0) and abs(rho) < min_abs_rho:
        verdict = "negligible"                      # detectable with this many records but too small to matter (a rank correlation below the minimum effect)
    else:
        verdict = "inconclusive"
    decision = kv.decision_status(change, clo, chi, tolerance) if np.isfinite(change) else "no_data"
    y10, y90 = (int(np.quantile(arr[:, 0], q)) for q in (0.1, 0.9))
    return {"verdict": verdict, "n": len(f), "n_wells": int(f["wellbore_id"].nunique()), "rho": rho, "ci_low": lo, "ci_high": hi, "first_window": y10, "last_window": y90,
            "change": change, "change_low": clo, "change_high": chi, "decision": decision,
            "detail": f"rank correlation with the drilling year {rho:+.2f} (90 % interval {lo:+.2f} to {hi:+.2f}) over {len(f):,} records in {f['wellbore_id'].nunique():,} wells; the straight-line trend "
                      f"changes the mean by {100 * change:+.0f} % between {y10} and {y90} (interval {100 * clo:+.0f} to {100 * chi:+.0f} %)"}

def run_tests(inds: dict[str, dict[str, Any]], tolerance: float = 0.25, min_abs_rho: float = 0.10) -> pd.DataFrame:
    rows = []
    for key, ind in inds.items():
        r = trend_test(ind, tolerance, min_abs_rho=min_abs_rho)
        rows.append({"indicator": key, "data_type": ind["data_type"], "title": ind["title"], "kind": ind["kind"], "expected": ind["expected"], "unit": ind["unit"], "decision_tolerance": tolerance, "min_abs_rho": min_abs_rho, **r})
    return pd.DataFrame(rows)


def trends_frame(inds: dict[str, dict[str, Any]]) -> pd.DataFrame:
    return pd.concat([trend_by_window(i["frame"]).assign(indicator=k, data_type=i["data_type"], title=i["title"], unit=i["unit"]) for k, i in inds.items()], ignore_index=True)
