"""MD to TVD from a survey: the arc between stations, stations as they are, nothing extrapolated, many wellbores at once (src/ep_wellbore/md_to_tvd.py)."""
import time

import numpy as np

from ep_wellbore.md_to_tvd import interpolate_along_md, tvd_at_md
from ep_wellbore.trajectory import minimum_curvature

R = 500.0                                                                    # a build section on a circular arc in one plane: inclination = md / R, TVD = R sin(md / R)


def arc_stations(mds):
    md = np.asarray(mds, dtype=float)
    return md, R * np.sin(md / R), np.degrees(md / R), np.zeros(len(md))


def test_between_two_stations_the_arc_is_followed_not_the_straight_line():
    md, tvd, inc, azi = arc_stations([0.0, 400.0])
    out, _ = tvd_at_md(np.ones(2), md, tvd, inc, azi, [1, 1, 1], [100.0, 200.0, 300.0])
    assert np.allclose(out["tvd_m"], R * np.sin(np.array([100.0, 200.0, 300.0]) / R), atol=1e-9)
    straight = tvd[1] * 200.0 / 400.0
    assert abs(straight - R * np.sin(200.0 / R)) > 14.0                                  # the straight line is 15 m out at the middle of this build
    assert set(out["interp_method"].astype(str)) == {"minimum_curvature"} and (out["bracket_md_m"] == 400.0).all()


def test_the_position_in_the_plane_follows_the_arc_too():
    md, tvd, inc, azi = arc_stations([0.0, 400.0])
    north = R * (1 - np.cos(md / R))
    out, _ = tvd_at_md(np.ones(2), md, tvd, inc, azi, [1], [200.0], station_north=north, station_east=np.zeros(2))
    assert abs(out["north_m"].iloc[0] - R * (1 - np.cos(200.0 / R))) < 1e-9 and abs(out["east_m"].iloc[0]) < 1e-9


def test_the_angles_at_a_depth_are_the_tangent_of_the_arc_and_a_station_gives_its_own():
    md, tvd, inc, azi = arc_stations([0.0, 400.0])
    out, _ = tvd_at_md(np.ones(2), md, tvd, inc, azi, [1, 1], [200.0, 400.0])
    assert abs(out["inclination_deg"].iloc[0] - np.degrees(200.0 / R)) < 1e-9 and abs(out["azimuth_deg"].iloc[0]) < 1e-9
    assert out["inclination_deg"].iloc[1] == inc[1]
    wrap, _ = tvd_at_md([1, 1], [0.0, 100.0], [0.0, 10.0], [10.0, 10.0], [350.0, 10.0], [1], [50.0])
    assert abs(wrap["azimuth_deg"].iloc[0] - 0.0) < 1e-6 or abs(wrap["azimuth_deg"].iloc[0] - 360.0) < 1e-6          # through north, not the long way round


def test_a_station_returns_its_own_values_and_the_result_agrees_with_minimum_curvature_on_a_dense_survey():
    rng = np.random.default_rng(2)
    md = np.cumsum(rng.uniform(20, 40, 60))
    inc = np.clip(np.cumsum(rng.normal(1.0, 0.8, 60)), 0, 80)
    azi = np.cumsum(rng.normal(0, 3, 60)) % 360
    tvd = minimum_curvature(md, inc, azi, start_tvd=1500.0)["tvd"].to_numpy()
    out, _ = tvd_at_md(np.full(60, 7), md, tvd, inc, azi, np.full(60, 7), md)
    assert np.allclose(out["tvd_m"], tvd) and set(out["interp_method"].astype(str)) == {"station"}
    mid = (md[:-1] + md[1:]) / 2
    half, _ = tvd_at_md(np.full(60, 7), md, tvd, inc, azi, np.full(59, 7), mid)
    assert (np.diff(tvd) > 0).all() and ((half["tvd_m"].to_numpy() > tvd[:-1]) & (half["tvd_m"].to_numpy() < tvd[1:])).all()


def test_nothing_is_extrapolated_and_the_distance_outside_is_reported():
    md, tvd, inc, azi = arc_stations([100.0, 300.0])
    out, ev = tvd_at_md(np.ones(2), md, tvd, inc, azi, [1, 1, 1, 2], [50.0, 301.0, 100.0, 150.0])
    assert out["tvd_m"].isna().tolist() == [True, True, False, True] and out["inside_survey"].tolist() == [False, False, True, False]
    assert out["outside_by_m"].iloc[0] == -50.0 and out["outside_by_m"].iloc[1] == 1.0 and np.isnan(out["outside_by_m"].iloc[3])         # the last: a wellbore with no survey
    assert any(e["rule"] == "md_to_tvd_outside_survey" for e in ev)


def test_wellbores_are_independent_even_with_equal_depths_and_a_station_of_another_wellbore_is_never_used():
    md = np.array([0.0, 100.0, 0.0, 100.0])
    tvd = np.array([0.0, 100.0, 0.0, 60.0])
    inc = np.array([0.0, 0.0, 0.0, 60.0])
    azi = np.zeros(4)
    out, _ = tvd_at_md([1, 1, 2, 2], md, tvd, inc, azi, [2, 1, 2], [100.0, 100.0, 50.0])
    assert out["tvd_m"].iloc[0] == 60.0 and out["tvd_m"].iloc[1] == 100.0 and 0.0 < out["tvd_m"].iloc[2] < 60.0


def test_stations_without_angles_are_interpolated_linearly_and_labelled():
    out, ev = tvd_at_md([1, 1], [0.0, 100.0], [0.0, 90.0], [np.nan, np.nan], [np.nan, np.nan], [1], [50.0])
    assert out["tvd_m"].iloc[0] == 45.0 and str(out["interp_method"].iloc[0]) == "linear" and any(e["rule"] == "md_to_tvd_linear_between_stations" for e in ev)


def test_unusable_and_repeated_stations_are_counted_not_used():
    out, ev = tvd_at_md([1, 1, 1, 1], [0.0, 100.0, 100.0, np.nan], [0.0, 100.0, 999.0, 5.0], [0, 0, 0, 0], [0, 0, 0, 0], [1], [100.0])
    assert out["tvd_m"].iloc[0] == 100.0                                                 # the first of two stations at 100 m
    assert {e["rule"] for e in ev} >= {"md_to_tvd_station_unusable", "md_to_tvd_station_repeated"}


def test_a_large_query_over_many_wellbores_is_fast():
    rng = np.random.default_rng(5)
    wells, stations, n = 3000, 80, 1_000_000
    md = np.cumsum(rng.uniform(15, 40, (wells, stations)), axis=1)
    inc = np.clip(np.cumsum(rng.normal(0.8, 0.6, (wells, stations)), axis=1), 0, 75)
    azi = np.cumsum(rng.normal(0, 2, (wells, stations)), axis=1) % 360
    tvd = np.stack([minimum_curvature(md[w], inc[w], azi[w])["tvd"].to_numpy() for w in range(40)])
    sw = np.repeat(np.arange(wells), stations)
    st_tvd = np.vstack([tvd, np.zeros((wells - 40, stations))]).ravel()
    qw = rng.integers(0, wells, n)
    qm = rng.uniform(0, md.max(), n)
    start = time.perf_counter()
    out, _ = tvd_at_md(sw, md.ravel(), st_tvd, inc.ravel(), azi.ravel(), qw, qm)
    seconds = time.perf_counter() - start
    assert seconds < 10.0 and out["inside_survey"].any() and (~out["inside_survey"]).any(), seconds


def test_a_sigma_along_the_hole_is_interpolated_linearly_with_the_same_outside_rule():
    out, _ = interpolate_along_md([1, 1], [0.0, 100.0], {"xy_sigma_m": np.array([2.0, 6.0])}, [1, 1, 2], [25.0, 150.0, 25.0])
    assert out["xy_sigma_m"].iloc[0] == 3.0 and np.isnan(out["xy_sigma_m"].iloc[1]) and np.isnan(out["xy_sigma_m"].iloc[2]) and out["inside_survey"].tolist() == [True, False, False]