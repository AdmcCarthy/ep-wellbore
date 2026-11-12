"""The depth join: boundaries, straddling, overlapping intervals, points, missing data and speed (src/ep_wellbore/depth_link.py)."""
import time

import numpy as np

from ep_wellbore.depth_link import coverage_range, overlap_pairs

W = [1, 1, 1, 2]
TOP = [100.0, 200.0, 300.0, 100.0]
BASE = [200.0, 300.0, 400.0, 500.0]


def pairs(obs_well, ot, ob, **kw):
    out, ev = overlap_pairs(obs_well, ot, ob, W, TOP, BASE, **kw)
    return out, ev


def test_an_interval_that_crosses_two_intervals_is_returned_for_both_with_its_overlap():
    out, _ = pairs([1], [150.0], [250.0])
    assert out["int_pos"].tolist() == [0, 1] and out["overlap_m"].tolist() == [50.0, 50.0] and out["obs_length_m"].tolist() == [100.0, 100.0]


def test_a_point_on_a_boundary_belongs_to_the_interval_below_only():
    out, _ = pairs([1, 1, 1], [200.0, 199.999, 400.0], [200.0, 199.999, 400.0])
    assert out["obs_pos"].tolist() == [0, 1] and out["int_pos"].tolist() == [1, 0]          # 200 -> the interval that starts at 200; 400 is the base of the last: outside


def test_an_interval_that_only_touches_a_boundary_does_not_overlap_the_next_interval():
    out, _ = pairs([1], [100.0], [200.0])
    assert out["int_pos"].tolist() == [0]


def test_wellbores_are_kept_apart_even_with_equal_depths():
    out, _ = pairs([2, 1], [150.0, 150.0], [160.0, 160.0])
    assert sorted(zip(out["obs_pos"], out["int_pos"])) == [(0, 3), (1, 0)]


def test_overlapping_intervals_both_match_and_a_long_early_interval_does_not_hide_later_ones():
    out, _ = overlap_pairs([1], [250.0], [260.0], [1, 1, 1], [0.0, 100.0, 240.0], [1000.0, 200.0, 300.0])
    assert out["int_pos"].tolist() == [0, 2]


def test_missing_base_is_a_point_and_missing_top_inverted_and_unusable_intervals_are_counted_not_placed():
    out, ev = overlap_pairs([1, 1, 1], [150.0, np.nan, 300.0], [np.nan, 160.0, 250.0], [1, 1], [100.0, 200.0], [200.0, 150.0])
    assert out["obs_pos"].tolist() == [0] and out["overlap_m"].tolist() == [0.0]
    assert {e["rule"] for e in ev} == {"depth_link_observation_without_depth", "depth_link_observation_inverted", "depth_link_interval_unusable"}


def test_min_overlap_drops_slivers_of_an_interval_observation():
    out, _ = pairs([1], [150.0], [200.4], min_overlap_m=0.5)
    assert out["int_pos"].tolist() == [0]


def test_coverage_range_gives_the_shallowest_top_and_deepest_base_per_wellbore():
    c = coverage_range(W, TOP, BASE).set_index("well")
    assert c.loc[1, "top"] == 100.0 and c.loc[1, "base"] == 400.0 and c.loc[2, "base"] == 500.0


def test_a_large_query_over_many_wellbores_is_fast_and_agrees_with_the_definition():
    rng = np.random.default_rng(1)
    wells, per = 3000, 12
    iw = np.repeat(np.arange(wells), per)
    edges = np.sort(rng.uniform(0, 4000, (wells, per + 1)), axis=1)
    it, ib = edges[:, :-1].ravel(), edges[:, 1:].ravel()
    n = 500_000
    ow = rng.integers(0, wells, n)
    ot = rng.uniform(0, 4000, n)
    ob = ot + rng.choice([0.0, 0.0, 5.0, 40.0], n)
    start = time.perf_counter()
    out, _ = overlap_pairs(ow, ot, ob, iw, it, ib)
    seconds = time.perf_counter() - start
    assert seconds < 10.0, seconds                                          # the pandas cross join of this size would not finish
    sample = rng.choice(n, 200, replace=False)                              # the definition, checked on a sample by brute force
    for i in sample:
        mine = np.flatnonzero(iw == ow[i])
        if ob[i] > ot[i]:
            hit = mine[(np.minimum(ob[i], ib[mine]) - np.maximum(ot[i], it[mine])) > 0]
        else:
            hit = mine[(it[mine] <= ot[i]) & (ot[i] < ib[mine])]
        assert sorted(out.loc[out["obs_pos"] == i, "int_pos"].tolist()) == sorted(hit.tolist())
