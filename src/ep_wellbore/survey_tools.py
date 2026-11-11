"""Survey tools: how the positions of different tool classes and different times compare at the same depth, and what that says about quality.

The hypothesis of the analysis notebooks is that older and cruder tools carry more uncertainty than newer, better ones.  A survey does not state its own true error, so the evidence here is RELATIVE: where two surveys of different tool
classes cover the same wellbore depth, how far apart are their positions, and does that grow with depth, with the age of the survey, or with the tool class?  Positions are compared at equal measured depth after the surveys
are put on the same depth basis (the caller gives rows of `well_survey` on the RT basis).  A difference between two tools is not the error of either one; it is a LOWER BOUND of the larger of the two errors only when
both are unbiased, and is reported as a difference."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ep_wellbore.trajectory import position_at_md


def tool_overlap(survey: pd.DataFrame, step_m: float = 30.0) -> pd.DataFrame:
    """For every wellbore, every pair of survey sets (survey ids) of DIFFERENT tool classes that overlap in measured depth: the horizontal and vertical difference at depths `step_m` apart inside the overlap.

    Columns: wellbore, the two survey ids with tool class and year, md, horizontal and vertical difference.  Real-time rows are included when present."""
    s = survey[survey["tool_class"].notna() & survey["md_m"].notna()].copy()
    s["year"] = pd.to_datetime(s["survey_date"], errors="coerce").dt.year
    out = []
    for wid, g in s.groupby("source_wellbore_id"):
        sets = {sid: x.sort_values("md_m") for sid, x in g.groupby("survey_id") if len(x) >= 3}
        ids = list(sets)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                A, B = sets[a], sets[b]
                ca, cb = A["tool_class"].iloc[0], B["tool_class"].iloc[0]
                if ca == cb:
                    continue
                lo, hi = max(A["md_m"].min(), B["md_m"].min()), min(A["md_m"].max(), B["md_m"].max())
                if hi - lo < step_m:
                    continue
                md = np.arange(lo, hi + 1e-6, step_m)
                pa = position_at_md(A["md_m"].to_numpy(float), {"x": A["x_native"].to_numpy(float), "y": A["y_native"].to_numpy(float), "tvd": A["tvd_m"].to_numpy(float)}, md)
                pb = position_at_md(B["md_m"].to_numpy(float), {"x": B["x_native"].to_numpy(float), "y": B["y_native"].to_numpy(float), "tvd": B["tvd_m"].to_numpy(float)}, md)
                ok = pa["inside_survey"].to_numpy() & pb["inside_survey"].to_numpy()
                if not ok.any():
                    continue
                out.append(pd.DataFrame({"source_wellbore_id": wid, "survey_a": a, "tool_a": ca, "year_a": A["year"].iloc[0], "survey_b": b, "tool_b": cb, "year_b": B["year"].iloc[0], "md_m": md[ok],
                                         "horizontal_diff_m": np.hypot(pa["x"] - pb["x"], pa["y"] - pb["y"]).to_numpy()[ok], "vertical_diff_m": np.abs(pa["tvd"] - pb["tvd"]).to_numpy()[ok]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def overlap_summary(overlap: pd.DataFrame) -> pd.DataFrame:
    """Median and 90th percentile of the difference by tool pair (unordered), with the number of wellbores and depth points, and the Spearman correlation of the difference with depth (does it grow with depth?)."""
    if overlap.empty:
        return pd.DataFrame(columns=["tool_pair", "wellbores", "points", "horizontal_median_m", "horizontal_p90_m", "vertical_median_m", "rank_corr_with_depth"])
    o = overlap.copy()
    o["tool_pair"] = [" vs ".join(sorted([a, b])) for a, b in zip(o["tool_a"], o["tool_b"])]
    rows = []
    for pair, g in o.groupby("tool_pair"):
        rows.append({"tool_pair": pair, "wellbores": g["source_wellbore_id"].nunique(), "points": len(g), "horizontal_median_m": float(g["horizontal_diff_m"].median()), "horizontal_p90_m": float(g["horizontal_diff_m"].quantile(0.9)),
                     "vertical_median_m": float(g["vertical_diff_m"].median()), "rank_corr_with_depth": float(g["md_m"].rank().corr(g["horizontal_diff_m"].rank())) if len(g) > 3 else np.nan})
    return pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)


def era_table(run_check: pd.DataFrame, survey: pd.DataFrame) -> pd.DataFrame:
    """Survey runs by year and tool class: how many, and how far their stations lie from the definitive path (median of the run medians).  The Volve era is narrow (the 15/9-19 wells of 1992-1998 and the F wells of 2007-2014),
    so a trend with age is reported with its sample size and is not a test of the hypothesis for the North Sea."""
    y = survey[survey["survey_kind"] == "run"].assign(year=lambda d: pd.to_datetime(d["survey_date"], errors="coerce").dt.year).groupby("survey_id")["year"].min()
    r = run_check.copy()
    r["year"] = r["survey_id"].map(y)
    g = r.groupby(["tool_class", "year"], dropna=False).agg(runs=("survey_id", "size"), median_of_median_diff_m=("median_horizontal_diff_m", "median"), max_diff_m=("max_horizontal_diff_m", "max")).reset_index()
    return g.sort_values(["tool_class", "year"]).reset_index(drop=True)
