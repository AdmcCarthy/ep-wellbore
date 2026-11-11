"""Mud type as a categorical variable by depth zone, and the mud weight against the leak-off value at the casing shoes.  Pure functions (docs/knowledge_architecture.md section 6).

The Sodir mud type is free text (water based, WATERBASED, Glydril, Innovert, XP-07 ...).  A class is assigned only from an explicit mapping in config/mud_types.yaml: by words that state the base
fluid, by words whose usual meaning is ASSUMED (spud, KCl, polymer), else the entry stays a brand name with no stated base fluid.  Depths are measured depth (the mud table has no vertical depth).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

BASIS_STATES, BASIS_ASSUMED, BASIS_BRAND, BASIS_NONE = "states the base fluid", "assumed from the name", "brand name only", "no usable entry"


def _compile(words: list[str]):
    return re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(w.lower()) for w in words) + r")(?![a-z0-9])") if words else None


def classify_mud_type(raw: Any, cfg: dict[str, Any]) -> tuple[str, str]:
    """(class key, basis) of one mud-type text.  Class keys: the configured classes, `brand_unclassified`, `no_entry`."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)) or pd.isna(raw):
        return "no_entry", BASIS_NONE
    low = str(raw).strip().lower()
    unk = _compile(cfg.get("unknown_words", []) or [])
    if not low or (unk is not None and unk.search(low)):
        return "no_entry", BASIS_NONE
    for c in cfg["classes"]:
        rx = _compile(c.get("states_base_fluid", []) or [])
        if rx and rx.search(low):
            return c["class"], BASIS_STATES
    for c in cfg["classes"]:
        rx = _compile(c.get("assumed_from_name", []) or [])
        if rx and rx.search(low):
            return c["class"], BASIS_ASSUMED
    return "brand_unclassified", BASIS_BRAND


def class_labels(cfg: dict[str, Any]) -> dict[str, str]:
    d = {c["class"]: c["label"] for c in cfg["classes"]}
    d["brand_unclassified"], d["no_entry"] = cfg["brand_label"], cfg["unknown_label"]
    return d


def classify_frame(mud: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add `mud_class`, `mud_class_basis` to a mud table (one classification per distinct text)."""
    out = mud.copy()
    cache = {t: classify_mud_type(t, cfg) for t in out["mud_type"].astype("string").unique()}
    pairs = out["mud_type"].astype("string").map(cache)
    out["mud_class"] = pairs.map(lambda p: p[0])
    out["mud_class_basis"] = pairs.map(lambda p: p[1])
    return out


def build_mud_zones(mud: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """One row per run of the same mud class along the measured depth of a wellbore: top, base (the top of the next zone, else the last record), records, mud weight range."""
    m = classify_frame(mud, cfg)
    m = m[m["md_m"].notna() & m["wellbore_id"].notna()].copy()
    m["wellbore_id"], m["md_m"], m["mud_weight_sg"] = m["wellbore_id"].astype("int64"), m["md_m"].astype(float), m["mud_weight_sg"].astype(float)
    m["well_name"], m["mud_class"], m["mud_class_basis"] = m["well_name"].astype(str), m["mud_class"].astype(str), m["mud_class_basis"].astype(str)
    m = m.sort_values(["wellbore_id", "md_m"]).reset_index(drop=True)
    wid, cls = m["wellbore_id"].to_numpy(), m["mud_class"].to_numpy()
    new_run = np.r_[True, (wid[1:] != wid[:-1]) | (cls[1:] != cls[:-1])]
    m["zone_no"] = pd.Series(new_run.astype(int)).groupby(wid).cumsum().to_numpy()
    g = m.groupby(["wellbore_id", "zone_no"], sort=True)
    z = g.agg(well_name=("well_name", "first"), mud_class=("mud_class", "first"), md_top_m=("md_m", "min"), md_last_record_m=("md_m", "max"), n_records=("md_m", "size"),
              mud_weight_min_sg=("mud_weight_sg", "min"), mud_weight_max_sg=("mud_weight_sg", "max"), mud_weight_mean_sg=("mud_weight_sg", "mean"),
              class_basis=("mud_class_basis", lambda s: BASIS_STATES if (s == BASIS_STATES).all() else BASIS_ASSUMED if (s != BASIS_BRAND).all() and (s != BASIS_NONE).all() else s.iloc[0])).reset_index()
    nxt = z.groupby("wellbore_id")["md_top_m"].shift(-1)
    z["md_bottom_m"] = nxt.fillna(z["md_last_record_m"])
    z["depth_reference"] = "MD RT"
    return z.drop(columns=["md_last_record_m"])


def mud_vs_leakoff(mud: pd.DataFrame, lot: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """At each casing shoe with a leak-off value: the nearest mud record at or above the shoe within `join_window_m`; margin = leak-off equivalent density - mud weight (both on the rotary-table
    basis; the mud weight is the SURFACE density).  The shoe depth and the mud depth are measured depth."""
    m = classify_frame(mud, cfg)
    m = m[m["md_m"].notna() & m["mud_weight_sg"].notna()].sort_values("md_m")
    L = lot[lot["lot_emw_sg"].notna() & lot["casing_depth_md_m"].notna()].copy()
    L["casing_depth_md_m"] = L["casing_depth_md_m"].astype(float)
    L = L.sort_values("casing_depth_md_m")
    win = float(cfg["join_window_m"])
    j = pd.merge_asof(L, m[["wellbore_id", "md_m", "mud_weight_sg", "mud_class", "mud_class_basis"]].astype({"md_m": float}), left_on="casing_depth_md_m", right_on="md_m",
                      by="wellbore_id", direction="backward", tolerance=win)
    j["lot_minus_mud_sg"] = j["lot_emw_sg"].astype(float) - j["mud_weight_sg"].astype(float)
    cols = ["wellbore_id", "well_name", "casing_type", "casing_depth_md_m", "test_type", "lot_emw_sg", "md_m", "mud_weight_sg", "mud_class", "mud_class_basis", "lot_minus_mud_sg"]
    return j[cols].rename(columns={"md_m": "mud_record_md_m"}).reset_index(drop=True)


def depth_class_table(mud: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Share of mud records per measured-depth bin and class (records, not wells)."""
    m = classify_frame(mud, cfg)
    b = float(cfg["depth_bin_m"])
    m = m[m["md_m"].notna() & (m["md_m"] >= 0)]
    m["md_bin_m"] = (m["md_m"].astype(float) // b * b).astype(float)
    t = m.groupby(["md_bin_m", "mud_class"]).size().rename("records").reset_index()
    t["share"] = t["records"] / t.groupby("md_bin_m")["records"].transform("sum")
    return t
