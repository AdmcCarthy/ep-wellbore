import numpy as np

from ep_wellbore.trajectory import compute_tvd_from_survey, minimum_curvature, survey_quality


def test_vertical_well_has_tvd_equal_to_md_and_no_offset():
    md = np.array([0.0, 100.0, 250.0])
    p = minimum_curvature(md, [0, 0, 0], [0, 0, 0])
    assert list(p["tvd"]) == [0.0, 100.0, 250.0] and p["north"].abs().max() == 0 and p["east"].abs().max() == 0


def test_circular_arc_matches_the_analytic_answer():
    r = 1000.0                                           # build from vertical to horizontal along a circle of radius r, azimuth north: length pi r / 2
    inc = np.linspace(0, 90, 91)
    md = np.radians(inc) * r
    p = minimum_curvature(md, inc, np.zeros_like(inc))
    assert abs(p["tvd"].iloc[-1] - r) < 1e-6 and abs(p["north"].iloc[-1] - r) < 1e-6 and abs(p["east"].iloc[-1]) < 1e-9
    assert abs(p["dls"].iloc[-1] - np.degrees(1 / r) * 30) < 1e-6                      # 1 / r radian per metre, per 30 m


def test_one_coarse_step_is_still_exact_on_a_circle_and_azimuth_east_goes_east():
    r = 500.0
    p = minimum_curvature([0.0, np.pi * r / 2], [0, 90], [90, 90])                 # a single interval of the whole quarter circle, azimuth east
    assert abs(p["tvd"].iloc[-1] - r) < 1e-6 and abs(p["east"].iloc[-1] - r) < 1e-6 and abs(p["north"].iloc[-1]) < 1e-9


def test_start_values_and_the_named_function():
    tvd = compute_tvd_from_survey([100.0, 200.0], [0, 0], [0, 0], start_tvd=90.0)
    assert list(tvd) == [90.0, 190.0]


def test_quality_flags_the_usual_traps():
    assert survey_quality([0, 100, 100, 90], [0, 5, 5, 5], [0, 10, 10, 10]) == ["md_not_increasing", "repeated_md"] or "md_not_increasing" in survey_quality([0, 100, 100, 90], [0, 5, 5, 5], [0, 10, 10, 10])
    assert "inclination_out_of_range" in survey_quality([0, 100], [0, 200], [0, 0])
    assert "large_station_gap" in survey_quality([0, 500], [0, 5], [0, 0])
    assert survey_quality([0, 30, 60], [0, 5, 6], [0, 10, 11]) == []
