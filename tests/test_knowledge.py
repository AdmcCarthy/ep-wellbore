"""Knowledge layer: vocabulary, packs, conflicts, import review, plain-language notes, mud classes, back-stripping, cross-checks (docs/knowledge_architecture.md)."""
from __future__ import annotations


import pandas as pd
import yaml

from ep_wellbore import mud_zones as mz
from ep_physics_base.config import load_yaml








def _tmp_pack(tmp_path, items, pid="core", version="2.0.0"):
    f = tmp_path / pid
    f.mkdir()
    (f / "pack.yaml").write_text(yaml.safe_dump({"id": pid, "version": version, "schema_version": 1, "title": "t", "priority": 20, "files": ["items.yaml"]}), encoding="utf-8")
    (f / "items.yaml").write_text(yaml.safe_dump({"items": items}), encoding="utf-8")
    return f










def test_mud_names_are_classified_only_when_they_state_the_base_fluid():
    cfg = load_yaml("config/mud_types.yaml")
    f = lambda s: mz.classify_mud_type(s, cfg)
    assert f("WATER BASED") == ("water_based", mz.BASIS_STATES) and f("seawater")[0] == "water_based" and f("OBM-Low ECD")[0] == "oil_based"
    assert f("Innovert") == ("oil_based", mz.BASIS_ASSUMED) and f("Synthetic")[0] == "synthetic_based" and f("Spud Mud") == ("water_based", mz.BASIS_ASSUMED)
    assert f("Glydril") == ("brand_unclassified", mz.BASIS_BRAND) and f("DUMMY")[0] == "no_entry" and f(None)[0] == "no_entry" and f("  ")[0] == "no_entry"


def test_mud_zones_are_runs_of_one_class_along_the_depth():
    cfg = load_yaml("config/mud_types.yaml")
    mud = pd.DataFrame({"wellbore_id": [1, 1, 1, 1, 2], "well_name": "A", "md_m": [100.0, 500.0, 1500.0, 2500.0, 300.0], "mud_weight_sg": [1.1, 1.3, 1.5, 1.7, 1.2],
                        "mud_type": ["Spud Mud", "WATER BASED", "OBM", "OBM", "Glydril"]})
    z = mz.build_mud_zones(mud, cfg)
    w1 = z[z.wellbore_id == 1].sort_values("zone_no")
    assert list(w1["mud_class"]) == ["water_based", "oil_based"] and list(w1["md_top_m"]) == [100.0, 1500.0] and list(w1["md_bottom_m"]) == [1500.0, 2500.0]
    lot = pd.DataFrame({"wellbore_id": [1], "well_name": "A", "casing_type": "SURF", "casing_depth_md_m": [1600.0], "test_type": "LOT", "lot_emw_sg": [1.9]})
    j = mz.mud_vs_leakoff(mud, lot, cfg)
    assert j["mud_class"].iloc[0] == "oil_based" and abs(j["lot_minus_mud_sg"].iloc[0] - 0.4) < 1e-9






