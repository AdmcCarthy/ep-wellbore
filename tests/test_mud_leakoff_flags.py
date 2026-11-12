"""The mud near a shoe test is evidence beside the value: a FIT is not affected, a water-based leak-off value is likely high, nothing is corrected (src/ep_wellbore/mud_zones.py)."""
from pathlib import Path

import pandas as pd
import yaml

from ep_wellbore import mud_zones as mz

CFG = yaml.safe_load((Path(mz.__file__).parent / "defaults" / "mud_types.yaml").read_text(encoding="utf-8"))


def _mud():
    return pd.DataFrame({"wellbore_id": [1, 1, 2, 3], "md_m": [900.0, 1450.0, 1000.0, 1000.0], "mud_weight_sg": [1.10, 1.30, 1.20, 1.25],
                         "mud_type": ["Water based", "Oil based", "Glydril", None]})


def _lot():
    return pd.DataFrame({"wellbore_id": [1, 1, 2, 3, 4], "well_name": "w", "casing_type": "SHOE", "casing_depth_md_m": [1000.0, 1500.0, 1020.0, 1010.0, 500.0],
                         "lot_emw_sg": [1.6, 1.7, 1.5, 1.4, 1.5], "test_type": ["LOT", "FIT", "LOT", "LOT", "LOT"]})


def test_the_flag_follows_the_type_of_test_and_the_class_of_the_mud_and_the_value_is_not_touched():
    raw = mz.mud_vs_leakoff(_mud(), _lot(), CFG)
    out = raw.set_index(["wellbore_id", "test_type"])[["leakoff_mud_flag"]]
    assert out.loc[(1, "LOT"), "leakoff_mud_flag"] == "water_based_high"
    assert out.loc[(1, "FIT"), "leakoff_mud_flag"] == "not_affected"            # a FIT is not affected by the mud type, even in oil-based mud
    assert out.loc[(2, "LOT"), "leakoff_mud_flag"] == "mud_base_unknown" and out.loc[(3, "LOT"), "leakoff_mud_flag"] == "mud_base_unknown"      # a brand name and an empty entry state no base fluid
    assert out.loc[(4, "LOT"), "leakoff_mud_flag"] == "no_mud_record"
    assert list(raw.sort_values(["wellbore_id", "casing_depth_md_m"])["lot_emw_sg"]) == [1.6, 1.7, 1.5, 1.4, 1.5]      # the equivalent mud density is exactly what the source gave


def test_oil_and_synthetic_mud_give_a_low_direction_and_an_unknown_test_type_is_not_assessed():
    mud_class = pd.Series(["oil_based", "synthetic_based", "water_based"])
    flags, notes = mz.leakoff_mud_flags(pd.Series(["LOT", "xlot", "SPECIAL"]), mud_class, CFG)
    assert list(flags) == ["non_aqueous_low", "non_aqueous_low", "not_assessed"] and notes.str.len().min() > 20


def test_without_the_section_every_row_is_not_assessed_and_missing_values_do_not_raise():
    flags, _ = mz.leakoff_mud_flags(pd.Series([None, "LOT"], dtype="string"), pd.Series(["water_based", None], dtype="string"), {})
    assert list(flags) == ["not_assessed", "not_assessed"]
