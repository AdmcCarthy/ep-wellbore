import numpy as np
import pandas as pd

from ep_wellbore.trajectory import position_at_md
from ep_wellbore.wellhead import choose_rt, classify_wellhead, elevation_verdict, position_sigma, select_trusted_surveys, uncertainty_class


def test_a_survey_that_starts_at_the_seabed_is_a_subsea_slot_and_one_at_sea_level_a_platform_slot():
    t, ev = classify_wellhead(91.0, 91.0, 54.9)
    assert t == "subsea" and "riser" in ev
    assert classify_wellhead(-20.0, 91.0, 25.0)[0] == "platform" and classify_wellhead(0.0, 91.0, 25.0)[0] == "platform"
    assert classify_wellhead(300.0, 91.0, 25.0)[0] == "unknown" and classify_wellhead(np.nan, 91.0, 25.0)[0] == "unknown"


def test_the_elevation_that_agrees_with_the_td_decides():
    assert choose_rt(54.0, 54.9, 3465.1, 3520.0) == (54.9, "source_datum_td_consistent")        # implied 54.9: the source datum, not the Sodir value
    assert choose_rt(54.9, 54.0, 3465.1, 3520.0) == (54.9, "sodir")
    assert choose_rt(54.0, 54.9, 3000.0, np.nan) == (54.0, "sodir") and choose_rt(np.nan, 54.9, 3000.0, np.nan) == (54.9, "source_datum") and choose_rt(np.nan, np.nan, 1.0, 2.0)[1] == "missing"


def test_a_difference_between_the_two_elevations_is_classified_with_the_kb_logic():
    v = lambda s, r, d=np.nan: elevation_verdict(s, r, d, np.nan)
    assert v(54.9, 54.9) == "agree"
    assert v(55.9, 54.9) == "sodir_may_be_kb"                                 # Sodir above the table by about a bushing height: Sodir may report the KB
    assert v(54.0, 54.9, 54.0) == "sodir_equals_source_default"               # lower, and equal to the source's planning default
    assert v(54.0, 54.9) == "sodir_lower_unexplained"                         # a KB sits above the table, so a KB reading does not explain a LOWER Sodir value
    assert v(70.0, 54.9) == "sodir_higher_unexplained" and v(np.nan, 54.9) == "unknown"


def test_sigma_is_the_square_root_of_the_covariance_in_metres_and_the_class_is_plain_words():
    s = position_sigma([100.0], [400.0], [25.0], [0.0], 0.3048)
    assert abs(s["sigma_lateral_m"][0] - 20 * 0.3048) < 1e-9 and abs(s["sigma_vertical_m"][0] - 5 * 0.3048) < 1e-9 and abs(s["sigma_east_m"][0] - 10 * 0.3048) < 1e-9
    limits = {"tight_m": 5.0, "normal_m": 25.0}
    assert uncertainty_class(2.0, 1.0, limits) == "tight" and uncertainty_class(10.0, 1.0, limits) == "normal" and uncertainty_class(40.0, 1.0, limits) == "loose" and uncertainty_class(np.nan, 1.0, limits) == "not stated"


def test_the_most_trusted_source_class_wins_per_wellbore():
    rows = []
    for cls, sid, n in (("real_time_witsml", "W1", 50), ("operator_as_drilled_definitive", "E1", 20), ("operator_survey_runs", "R1", 80)):
        rows += [{"wellbore_id": 1, "survey_id": sid, "survey_source_class": cls, "station_no": i} for i in range(n)]
    rows += [{"wellbore_id": None, "survey_id": "U", "survey_source_class": "regulator_survey", "station_no": 0}]
    best = select_trusted_surveys(pd.DataFrame(rows), {"regulator_survey": 1, "operator_as_drilled_definitive": 2, "operator_survey_runs": 3, "real_time_witsml": 4})
    assert set(best["survey_id"]) == {"E1"} and len(best) == 20                # an unlinked survey is never selected, even if its class ranks first


def test_position_at_md_interpolates_inside_and_never_extrapolates():
    md = np.array([100.0, 200.0, 300.0])
    p = position_at_md(md, {"tvd": np.array([100.0, 190.0, 270.0]), "east": np.array([0.0, 10.0, 30.0])}, [50.0, 150.0, 250.0, 350.0])
    assert p["tvd"].iloc[1] == 145.0 and p["east"].iloc[2] == 20.0 and p["tvd"].isna().tolist() == [True, False, False, True] and p["inside_survey"].tolist() == [False, True, True, False]


