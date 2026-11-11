"""Annular pressure loss, surge and swab, the datum offset and the riser margin (docs/annular_hydraulics.md)."""
from __future__ import annotations


from ep_wellbore import annular_hydraulics as ah
from ep_physics_base import units as pu


def test_bingham_slot_reduces_to_the_newtonian_and_the_yield_limit():
    gap, mu, u = 0.04, 0.03, 0.5
    g = float(ah.bingham_slot_gradient_pa_m(u, gap, mu, 0.0))
    assert abs(g - 12.0 * mu * u / gap ** 2) / g < 1e-9                                  # Newtonian slot: G = 12 mu u / gap^2
    ty = 5.0
    g_y = float(ah.bingham_slot_gradient_pa_m(u, gap, mu, ty))
    assert g_y > g and g_y > ty / (gap / 2.0)                                            # yield stress adds loss, never below the no-flow gradient
    assert float(ah.bingham_slot_gradient_pa_m(0.0, gap, mu, ty)) == 0.0


def test_the_loss_grows_with_flow_and_length_and_falls_with_a_wider_gap():
    kw = dict(density_kg_m3=1500.0, plastic_viscosity_pa_s=0.03, yield_stress_pa=6.0)
    a = ah.annular_pressure_loss(0.03, 0.2159, 0.127, 1000.0, **kw)["loss_bar"]
    b = ah.annular_pressure_loss(0.04, 0.2159, 0.127, 1000.0, **kw)["loss_bar"]
    c = ah.annular_pressure_loss(0.03, 0.2159, 0.127, 2000.0, **kw)["loss_bar"]
    d = ah.annular_pressure_loss(0.03, 0.2445, 0.127, 1000.0, **kw)["loss_bar"]
    assert b > a and abs(c - 2.0 * a) < 1e-9 and d < a


def test_turbulent_screening_is_never_below_the_laminar_loss():
    kw = dict(density_kg_m3=1500.0, plastic_viscosity_pa_s=0.015, yield_stress_pa=2.0)
    r = ah.annular_pressure_loss(0.15, 0.2159, 0.127, 1000.0, **kw)
    assert str(r["regime"]).startswith("turbulent")
    lam = float(ah.bingham_slot_gradient_pa_m(r["velocity_m_s"], 0.0445, 0.015, 2.0)) * 1000.0 / 1e5
    assert float(r["loss_bar"]) >= lam


def test_surge_is_positive_swab_negative_and_equal_in_size():
    kw = dict(density_kg_m3=1500.0, plastic_viscosity_pa_s=0.03, yield_stress_pa=6.0)
    s = ah.surge_swab_pressure_bar(0.5, 0.2159, 0.127, 3000.0, **kw)["pressure_bar"]
    w = ah.surge_swab_pressure_bar(-0.5, 0.2159, 0.127, 3000.0, **kw)["pressure_bar"]
    assert s > 0 > w and abs(s + w) < 1e-12
    assert ah.surge_swab_pressure_bar(0.5, 0.2159, 0.127, 3000.0, clinging_factor=0.4, **kw)["pressure_bar"] > s


def test_ecd_increase_is_the_loss_over_the_rt_depth():
    assert abs(float(ah.ecd_increase_sg(12.0, 3000.0, 25.0)) - 12.0 / (pu.BAR_PER_M_PER_SG * 3025.0)) < 1e-12


def test_datum_offset_fraction_and_the_riser_margin_algebra():
    assert abs(float(ah.datum_offset_fraction(2000.0, 25.0)) - 25.0 / 2025.0) < 1e-12
    z, wd, mud = 3000.0, 350.0, 1.5
    dis, con = float(ah.riser_disconnected_pressure_bar(mud, z, wd)), float(ah.riser_connected_pressure_bar(mud, z, 25.0))
    # the connected column is heavier by the mud in the riser (rotary table to seabed) minus the seawater that replaces it
    assert abs((con - dis) - (mud * pu.BAR_PER_M_PER_SG * (wd + 25.0) - 0.1005 * wd)) < 1e-9
    pp = 380.0
    need = float(ah.mud_sg_for_riser_margin(pp, z, wd))
    assert abs(float(ah.riser_margin_bar(need, z, wd, pp))) < 1e-9
    assert float(ah.riser_margin_bar(need + 0.1, z, wd, pp)) > 0
