"""Basic well design from the casing and leak-off table: strings, hole sizes, shoe depths, open hole below the shoe, and the telescoping check.  Pure functions (docs/well_construction.md).

What the open data hold: per casing string the type, the casing and hole diameter, the shoe depth (casing_depth) and the depth of the hole section when the string was run (hole_depth), and the
leak-off / integrity test at the shoe.  What they do NOT hold: cement type, volume, top of cement, bond logs, barrier status, test results of the shoe track.  The leak-off test at a shoe is the
only indirect evidence of the cement and the formation at the shoe, so the design figures say 'strings and shoes', never 'cement quality'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MM_PER_IN = 25.4
ORDER = ["CONDUCTOR", "SURF.COND.", "INTERM.", "LINER", "PROD."]          # the order of the strings of a well from the top (OPEN HOLE and PILOT HOLE are sections, not casing)
EIGHTHS = {0: "", 1: " 1/8", 2: " 1/4", 3: " 3/8", 4: " 1/2", 5: " 5/8", 6: " 3/4", 7: " 7/8"}


def inch_label(mm) -> str:
    """Nominal size in inches with fractions ('13 3/8') from a diameter in mm (nearest eighth); NaN gives ''."""
    if mm is None or pd.isna(mm):
        return ""
    eighths = int(round(float(mm) / MM_PER_IN * 8))
    return f"{eighths // 8}{EIGHTHS[eighths % 8]}"


def design_table(lot: pd.DataFrame) -> pd.DataFrame:
    """One row per string / section per wellbore, in depth order, with `string_no`, nominal sizes, the shoe depth, the open hole below the shoe (hole depth minus shoe depth, never negative: a negative
    value is flagged), and the test at the shoe."""
    d = lot.copy()
    for c in ("casing_depth_md_m", "hole_depth_md_m", "casing_diameter_mm", "hole_diameter_mm", "lot_emw_sg"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["casing_type"] = d["casing_type"].astype("string").str.strip().str.upper()
    d["_has"] = d["lot_emw_sg"].notna()
    d = d.sort_values(["wellbore_id", "casing_depth_md_m", "_has"], ascending=[True, True, False])
    d = d.drop_duplicates(["wellbore_id", "casing_depth_md_m", "casing_diameter_mm"], keep="first").reset_index(drop=True)      # several tests at one shoe are one string (the tested row is kept)
    d["string_no"] = d.groupby("wellbore_id").cumcount() + 1
    d["casing_in"] = d["casing_diameter_mm"].map(inch_label)
    d["hole_in"] = d["hole_diameter_mm"].map(inch_label)
    d["open_hole_below_shoe_m"] = d["hole_depth_md_m"] - d["casing_depth_md_m"]
    d["negative_open_hole"] = d["open_hole_below_shoe_m"] < 0
    d["has_test"] = d["lot_emw_sg"].notna()
    d["is_casing"] = d["casing_type"].isin(ORDER)
    return d[["wellbore_id", "well_name", "string_no", "casing_type", "is_casing", "casing_diameter_mm", "casing_in", "hole_diameter_mm", "hole_in", "casing_depth_md_m", "hole_depth_md_m",
              "open_hole_below_shoe_m", "negative_open_hole", "test_type", "lot_emw_sg", "has_test"]]


def common_sizes(design: pd.DataFrame, top: int = 8) -> pd.DataFrame:
    """The most common nominal casing sizes in the data with the number of strings and the median shoe depth: the sizes are read from the data, not assumed."""
    c = design[design["is_casing"] & design["casing_in"].ne("")]
    g = c.groupby("casing_in").agg(strings=("wellbore_id", "size"), wells=("wellbore_id", "nunique"), median_shoe_md_m=("casing_depth_md_m", "median"), median_hole_in_mm=("hole_diameter_mm", "median"),
                                   share_tested=("has_test", "mean"), casing_mm=("casing_diameter_mm", "median"))
    return g.sort_values("strings", ascending=False).head(top).sort_values("casing_mm", ascending=False)


def telescoping_check(design: pd.DataFrame) -> pd.DataFrame:
    """Per well with at least two casing strings: do the casing diameters decrease (or stay) with depth?  A well where a deeper string is wider than a shallower one is flagged: a data error, a
    tie-back or an under-reamed section, to be looked at, never corrected."""
    c = design[design["is_casing"] & design["casing_diameter_mm"].notna()].sort_values(["wellbore_id", "casing_depth_md_m"])
    rows = []
    for wid, g in c.groupby("wellbore_id"):
        if len(g) < 2:
            continue
        diam = g["casing_diameter_mm"].to_numpy()
        bad = int((np.diff(diam) > 1.0).sum())
        rows.append({"wellbore_id": wid, "well_name": g["well_name"].iloc[0], "n_strings": len(g), "wider_below": bad, "telescopes": bad == 0})
    return pd.DataFrame(rows)


def well_profile(design: pd.DataFrame, wellbore_id) -> pd.DataFrame:
    """The strings of one well, ready to draw: each string runs from the top to its shoe; the hole section is wider than the casing."""
    return design[(design["wellbore_id"] == wellbore_id) & design["is_casing"]].sort_values("casing_depth_md_m")


def test_coverage(design: pd.DataFrame) -> pd.DataFrame:
    """Share of strings with a leak-off or integrity test at the shoe, by string type, and the share that are LOT, FIT or extended."""
    c = design[design["is_casing"]]
    g = c.groupby("casing_type").agg(strings=("wellbore_id", "size"), tested=("has_test", "sum"))
    g["share_tested"] = g["tested"] / g["strings"]
    t = c[c["has_test"]].groupby(["casing_type", "test_type"]).size().unstack(fill_value=0)
    return g.join(t).reindex([x for x in ORDER if x in g.index])
