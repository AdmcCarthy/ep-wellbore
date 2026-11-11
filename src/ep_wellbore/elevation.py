"""Which elevation do the depths of a wellbore refer to: the rotary table (RT) or the kelly bushing (KB)?  A monitored judgement that is re-evaluated every time a source is added (docs/volve_edm_structure.md section 13).

Evidence comes in two kinds, in one long table (`well_elevation_evidence`: wellbore, source, kind, name used, value):
* `named`: an elevation a source states and NAMES (Sodir: kelly bushing; the EDM export: rotary table and a planning default rig elevation; the daily drilling reports: kelly; WITSML geometry: air gap);
* `implied`: the elevation the DEPTHS of a source refer to, derived by comparing two depth sources (Sodir TD against the end of the EDM survey, which is below sea level; the real-time trajectory, measured from the
  rig, against the EDM path).

The judgement never says "error".  Two elevations that differ by a kelly-bushing height are expected (a bushing sits above the table), rig moves and mobilisation can change the rig's elevations between wells, and
sources name the same number differently; every wellbore therefore gets a status `monitor`, a reading (`KB-like`, `RT-like`, `unresolved`), and the evidence that supports it.  Thresholds are adapter configuration."""
from __future__ import annotations

import numpy as np
import pandas as pd

SCHEMA_VERSIONS = {"well_elevation_evidence": "1.0", "well_elevation_judgement": "1.0"}
KB_LIKE_NAMES = ("kelly",)
RT_LIKE_NAMES = ("rotary",)


def judge_depth_datum(evidence: pd.DataFrame, tol_m: float = 0.5, kb_above_rt_m: tuple[float, float] = (0.2, 3.0)) -> pd.DataFrame:
    """One row per wellbore: the elevation the depths refer to (`depth_elevation_m`, the median of the implied values), how many implied values and how far apart, which named values agree with it and which do not,
    the pair of named values that looks like a KB above an RT, and the reading.

    Reading: `KB-like` when the depths agree with the HIGHER of two named values that differ by a kelly-bushing height, `RT-like` when they agree with the LOWER, `unresolved` otherwise (one value only, no
    implied value, or no agreement).  `naming_conflict` is True when the value the depths follow is named both Kelly and Rotary table by different sources.  Confidence: high with two or more agreeing implied values and a
    named value that agrees; medium with one implied value; low without.  Status is always `monitor`."""
    rows = []
    for wid, e in evidence.groupby("wellbore_id"):
        imp_all = e[e["kind"] == "implied"]
        strong = imp_all[imp_all["strength"].eq("strong")]["value_m"].astype(float).dropna() if "strength" in imp_all else imp_all.iloc[0:0]["value_m"]
        imp = strong if len(strong) else imp_all["value_m"].astype(float).dropna()                  # a precise fit outranks the TD-based value (the survey may stop short of TD)
        nam = e[e["kind"] == "named"].dropna(subset=["value_m"]).copy()
        nam["value_m"] = nam["value_m"].astype(float)
        depth_e = float(imp.median()) if len(imp) else np.nan
        spread = float(imp.max() - imp.min()) if len(imp) > 1 else (0.0 if len(imp) == 1 else np.nan)
        agree = nam[(nam["value_m"] - depth_e).abs() <= tol_m] if np.isfinite(depth_e) else nam.iloc[0:0]
        off = nam.drop(agree.index)
        hi = lo = np.nan
        vals = sorted(nam["value_m"].round(2).unique())
        pair = [(a, b) for a in vals for b in vals if a > b and kb_above_rt_m[0] <= a - b <= kb_above_rt_m[1]]
        if pair:
            lo, hi = max(pair, key=lambda p: p[0] - p[1])[1], max(pair, key=lambda p: p[0] - p[1])[0]
        reading = "unresolved"
        if np.isfinite(depth_e) and np.isfinite(hi):
            reading = "KB-like" if abs(depth_e - hi) <= tol_m else ("RT-like" if abs(depth_e - lo) <= tol_m else "unresolved")
        explanation = ("no depth evidence yet" if not np.isfinite(depth_e) else
                       f"the depths refer to {depth_e:.1f} m" + ((f"; the named values {lo:.1f} and {hi:.1f} m differ by {hi - lo:.1f} m, a bushing height, and the depths follow the {'higher (a kelly bushing sits above the table)' if reading == 'KB-like' else 'lower (the table)'}"
                                                                                                     if reading != "unresolved" else f"; the named values {lo:.1f} and {hi:.1f} m differ by {hi - lo:.1f} m but the depths agree with neither (the only implied value is weak or the survey stops short of TD)")
                                                                                                    if np.isfinite(hi) else "; no second named value of a bushing height is available, so KB and RT cannot be told apart"))
        names_agree = " ".join(agree["name_used"].astype(str)).lower()
        conflict = any(k in names_agree for k in KB_LIKE_NAMES) and any(k in names_agree for k in RT_LIKE_NAMES)
        n_imp = int(len(imp))
        n_agree_imp = int((imp_all["value_m"].astype(float).sub(depth_e).abs() <= tol_m).sum()) if np.isfinite(depth_e) else 0            # implied values of every strength that agree with the depth elevation
        conf = "high" if (n_agree_imp >= 2 and len(agree)) else ("medium" if n_imp >= 1 and len(strong) else ("low" if n_imp == 0 else "low"))
        rows.append({"wellbore_id": wid, "depth_elevation_m": depth_e, "n_implied": n_imp, "implied_spread_m": spread,
                     "agreeing_named": "; ".join(f"{r.name_used} {r.value_m:.1f} ({r.source})" for r in agree.itertuples()), "disagreeing_named": "; ".join(f"{r.name_used} {r.value_m:.1f} ({r.source})" for r in off.itertuples()),
                     "higher_named_m": hi, "lower_named_m": lo, "kb_rt_gap_m": hi - lo if np.isfinite(hi) else np.nan, "reading": reading, "explanation": explanation, "naming_conflict": bool(conflict), "confidence": conf, "status": "monitor",
                     "sources_used": ", ".join(sorted(set(e["source"].astype(str)))), "n_sources": int(e["source"].nunique())})
    return pd.DataFrame(rows)


def evidence_from_sources(wellbore_ids: list, sodir: pd.DataFrame | None, check: pd.DataFrame | None, ddr_check: pd.DataFrame | None, rt_traj: pd.DataFrame | None, wbgeometry: pd.DataFrame | None,
                          edm_values: dict[str, float], max_incl_diff_deg: float = 0.05, edm_wells: set | None = None, extra_named: pd.DataFrame | None = None) -> pd.DataFrame:
    """The long evidence table from whatever sources exist (a missing source adds no rows): Sodir (named kelly bushing), the EDM export (rotary table and default rig elevation, only for the wellbores of the rig the
    export describes, `edm_wells`; implied by Sodir TD and the survey end, a WEAK value because a survey can stop short of TD), the daily reports (named kelly), the real-time trajectory (implied by an inclination-profile
    fit, a STRONG value) and the WITSML wellbore geometry (named air gap, when positive)."""
    rows = []

    def add(w, source, kind, name, value, note="", strength=""):
        if value is not None and np.isfinite(value):
            rows.append({"wellbore_id": w, "source": source, "kind": kind, "name_used": name, "value_m": float(value), "note": note, "strength": strength})

    if sodir is not None:
        s = sodir.set_index("wellbore_id")["rt_elev_m"]
        for w in wellbore_ids:
            if w in s.index:
                add(w, "sodir", "named", "Kelly bushing (Sodir elevation field)", s[w])
    if check is not None and len(check):
        c = check[check["wellbore_id"].notna()]
        for r in c.itertuples():
            if edm_wells is None or int(r.wellbore_id) in edm_wells:
                add(int(r.wellbore_id), "edm export", "named", "Rotary table (EDM datum)", edm_values.get("rotary"))
                add(int(r.wellbore_id), "edm export", "named", "default rig elevation (EDM planning value)", edm_values.get("default"))
            add(int(r.wellbore_id), "sodir TD against edm survey", "implied", "depth reference implied", r.rt_implied_by_md_m, "Sodir TD minus the end of the survey below sea level", "weak")
    if ddr_check is not None and len(ddr_check):
        for r in ddr_check[ddr_check["wellbore_id"].notna()].itertuples():
            add(int(r.wellbore_id), "daily drilling reports", "named", "Kelly (daily report elevKelly)", r.elev_kelly_m)
    if rt_traj is not None and len(rt_traj):
        good = rt_traj[rt_traj["wellbore_id"].notna() & (rt_traj["median_inclination_diff_at_offset_deg"] <= max_incl_diff_deg)]            # only a precise inclination-profile fit is evidence
        for r in good.itertuples():
            add(int(r.wellbore_id), "real-time trajectory", "implied", "depth reference implied", r.datum_offset_m, "offset that aligns the inclination profile of the real-time trajectory with the EDM survey", "strong")
    if wbgeometry is not None and len(wbgeometry):
        for r in wbgeometry[wbgeometry["wellbore_id"].notna() & (wbgeometry["air_gap_m"] > 0)].itertuples():
            add(int(r.wellbore_id), "witsml wellbore geometry", "named", "air gap (WITSML wbGeometry)", r.air_gap_m)
    if extra_named is not None and len(extra_named):                                     # named values from other sources: columns wellbore_id, source, name_used, value_m
        for r in extra_named.dropna(subset=["wellbore_id", "value_m"]).itertuples():
            add(int(r.wellbore_id), r.source, "named", r.name_used, float(r.value_m))
    return pd.DataFrame(rows, columns=["wellbore_id", "source", "kind", "name_used", "value_m", "note", "strength"])
