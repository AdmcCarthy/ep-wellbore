"""Well design (strings, shoes, telescoping), the age-of-measurement tests, and the effort tracker (docs/well_construction.md, docs/effort_tracker.md)."""
from __future__ import annotations


import numpy as np
import pandas as pd

from ep_wellbore import tool_eras as te
from ep_wellbore import well_design as wd


def _lot():
    return pd.DataFrame({"wellbore_id": [1, 1, 1, 1, 2, 2], "well_name": "W", "casing_type": ["CONDUCTOR", "SURF.COND.", "SURF.COND.", "INTERM.", "SURF.COND.", "INTERM."],
                         "casing_diameter_mm": [762.0, 508.0, 508.0, 339.7, 508.0, 600.0], "hole_diameter_mm": [914.4, 660.4, 660.4, 444.5, 660.4, 700.0],
                         "casing_depth_md_m": [200.0, 900.0, 900.0, 2500.0, 800.0, 2000.0], "hole_depth_md_m": [205.0, 908.0, 908.0, 2512.0, 806.0, 1990.0],
                         "lot_emw_sg": [np.nan, 1.6, 1.62, 1.9, np.nan, 2.0], "test_type": [None, "LOT", "LOT", "FIT", None, "LOT"]})


def test_inch_labels_use_eighths():
    assert wd.inch_label(339.7) == "13 3/8" and wd.inch_label(244.5) == "9 5/8" and wd.inch_label(762.0) == "30" and wd.inch_label(None) == ""


def test_the_design_table_counts_a_string_once_and_finds_the_open_hole_and_the_widening_well():
    d = wd.design_table(_lot())
    w1 = d[d["wellbore_id"] == 1]
    assert len(w1) == 3 and list(w1["string_no"]) == [1, 2, 3]                                   # two tests at one shoe are one string
    assert abs(w1.loc[w1["casing_type"] == "SURF.COND.", "open_hole_below_shoe_m"].iloc[0] - 8.0) < 1e-9 and bool(w1.loc[w1["casing_type"] == "SURF.COND.", "has_test"].iloc[0])
    chk = wd.telescoping_check(d).set_index("wellbore_id")
    assert bool(chk.loc[1, "telescopes"]) and not bool(chk.loc[2, "telescopes"]) and int(chk.loc[2, "wider_below"]) == 1      # well 2: the deeper string is wider
    assert d.loc[(d["wellbore_id"] == 2) & (d["casing_type"] == "INTERM."), "negative_open_hole"].iloc[0]
    tc = wd.test_coverage(d)
    assert tc.loc["SURF.COND.", "tested"] == 1 and tc.loc["CONDUCTOR", "tested"] == 0


def test_era_windows_and_resolution_indicator():
    assert te.window_label(1987) == "1985–89" and te.window_label(np.nan) == ""
    r = te.is_multiple(pd.Series([1.60, 1.62, 1.65, np.nan]), 0.05)
    assert list(r.dropna()) == [1.0, 0.0, 1.0] and np.isnan(r.iloc[3])


def _era_frame(slope: float, n_wells: int = 200, per_well: int = 3, seed: int = 1):
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        y = int(rng.integers(1970, 2020))
        for _ in range(per_well):
            rows.append({"wellbore_id": w, "year": y, "window": te.window_label(y), "value": max(0.0, 10 + slope * (y - 1995) + rng.normal(0, 3))})
    return pd.DataFrame(rows)




def test_bootstrap_is_over_wells_so_many_records_in_few_wells_do_not_narrow_the_interval():
    few = _era_frame(-0.05, n_wells=40, per_well=40, seed=3)
    many = _era_frame(-0.05, n_wells=1600, per_well=1, seed=3)
    a, b = te.trend_test({"frame": few, "expected": "falls"}, 0.25), te.trend_test({"frame": many, "expected": "falls"}, 0.25)
    assert (a["ci_high"] - a["ci_low"]) > (b["ci_high"] - b["ci_low"])



