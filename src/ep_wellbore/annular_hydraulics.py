"""Annular pressure loss, ECD, surge and swab, the datum and water-depth sensitivities and the riser margin.  Pure functions (docs/annular_hydraulics.md, notebook 25).

A screening model for a reader who does not have the pump, rheology and string data of a real well: a Bingham plastic mud in laminar flow through a concentric annulus approximated by a parallel
slot (valid when the pipe is not small against the hole, d_pipe / d_hole above about 0.3-0.5), with a Blasius screening estimate when the flow is turbulent.  It is NOT a hydraulics simulator:
eccentricity, pipe rotation, cuttings, tool joints, temperature and pressure effects on the mud are left out and are named as such in the results (they lower or raise the loss, see the doc).
All inputs and outputs are SI (Pa, Pa s, m, m3/s); the pressure helpers return bar.  Mud rheology and flow rates are ASSUMPTION inputs (the open data hold none).

Slot flow of a Bingham plastic (plastic viscosity mu, yield stress ty) between plates a gap g apart, half gap h = g / 2, pressure gradient G = dp / dL:
    mean velocity  u = G h^2 / (3 mu) (1 - 1.5 xi + 0.5 xi^3),   xi = ty / (G h)   (a plug of half width xi h flows undisturbed in the middle; xi = 1 is the no-flow limit G = ty / h)
G is found for a given velocity by bisection (the velocity grows monotonically with G).
"""
from __future__ import annotations

import numpy as np

from ep_physics_base import units as pu

PA_PER_BAR = 1.0e5
RE_LAMINAR_LIMIT = 2100.0      # ASSUMPTION: the usual screening limit for the end of laminar flow; the Hedstrom-number criterion of a Bingham fluid is higher (see the doc)


def annulus_area_m2(hole_d_m, pipe_d_m):
    return np.pi / 4.0 * (np.asarray(hole_d_m, dtype=float) ** 2 - np.asarray(pipe_d_m, dtype=float) ** 2)


def mean_velocity_m_s(flow_m3_s, hole_d_m, pipe_d_m):
    return np.asarray(flow_m3_s, dtype=float) / annulus_area_m2(hole_d_m, pipe_d_m)


def _slot_velocity(g_pa_m, h, mu, ty):
    """Mean velocity of the Bingham slot flow for the pressure gradient `g_pa_m` (0 below the yield limit ty / h)."""
    xi = np.clip(ty / (g_pa_m * h), 0.0, 1.0)
    return g_pa_m * h * h / (3.0 * mu) * (1.0 - 1.5 * xi + 0.5 * xi ** 3)


def bingham_slot_gradient_pa_m(velocity_m_s, gap_m, plastic_viscosity_pa_s, yield_stress_pa):
    """Pressure gradient [Pa/m] that drives a Bingham plastic at the mean `velocity_m_s` through a slot of full gap `gap_m` (laminar)."""
    u, gap = np.broadcast_arrays(np.asarray(velocity_m_s, dtype=float), np.asarray(gap_m, dtype=float))
    h, mu, ty = gap / 2.0, float(plastic_viscosity_pa_s), float(yield_stress_pa)
    lo = np.full(u.shape, ty / np.maximum(h, 1e-12))                         # no flow at the yield gradient
    hi = lo + 12.0 * mu * np.maximum(u, 0.0) / np.maximum(h, 1e-12) ** 2 + 1.0   # Newtonian estimate plus the yield gradient is above the root
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        too_slow = _slot_velocity(mid, h, mu, ty) < u
        lo = np.where(too_slow, mid, lo)
        hi = np.where(too_slow, hi, mid)
    out = 0.5 * (lo + hi)
    return np.where(u > 0, out, 0.0)


def effective_viscosity_pa_s(velocity_m_s, gap_m, plastic_viscosity_pa_s, yield_stress_pa):
    """Bingham effective (apparent) viscosity mu + ty h_d / (6 u) of a slot of hydraulic diameter 2 gap (used only for the Reynolds number)."""
    u = np.maximum(np.asarray(velocity_m_s, dtype=float), 1e-9)
    return float(plastic_viscosity_pa_s) + float(yield_stress_pa) * 2.0 * np.asarray(gap_m, dtype=float) / (6.0 * u)


def annular_pressure_loss(flow_m3_s, hole_d_m, pipe_d_m, length_m, density_kg_m3, plastic_viscosity_pa_s, yield_stress_pa) -> dict:
    """Frictional pressure loss [bar] of one annular section and the facts that qualify it.

    Laminar: the Bingham slot solution.  Turbulent (screening Reynolds number above `RE_LAMINAR_LIMIT`): the larger of the laminar value and a Blasius estimate (Fanning f = 0.0791 Re^-0.25,
    dp/L = 2 f rho u^2 / d_h, with the plastic viscosity), so the result is never below the laminar one.  `regime` says which was used.  A negative or zero flow gives zero loss.
    """
    u = mean_velocity_m_s(flow_m3_s, hole_d_m, pipe_d_m)
    gap = (np.asarray(hole_d_m, dtype=float) - np.asarray(pipe_d_m, dtype=float)) / 2.0
    dh = 2.0 * gap
    lam = bingham_slot_gradient_pa_m(u, gap, plastic_viscosity_pa_s, yield_stress_pa)
    mu_eff = effective_viscosity_pa_s(u, gap, plastic_viscosity_pa_s, yield_stress_pa)
    re = np.asarray(density_kg_m3, dtype=float) * u * dh / mu_eff
    with np.errstate(divide="ignore", invalid="ignore"):
        re_p = np.asarray(density_kg_m3, dtype=float) * u * dh / float(plastic_viscosity_pa_s)
        fanning = 0.0791 * np.power(np.maximum(re_p, 1.0), -0.25)
        turb = 2.0 * fanning * np.asarray(density_kg_m3, dtype=float) * u ** 2 / dh
    grad = np.where(re > RE_LAMINAR_LIMIT, np.maximum(lam, turb), lam)
    return {"loss_bar": grad * np.asarray(length_m, dtype=float) / PA_PER_BAR, "velocity_m_s": u, "reynolds": re,
            "regime": np.where(u <= 0, "no flow", np.where(re > RE_LAMINAR_LIMIT, "turbulent (screening)", "laminar"))}


def ecd_increase_sg(loss_bar, z_msl_m, rt_elev_m):
    """The ECD minus the ESD [sg, RT basis] that an annular loss of `loss_bar` gives at the depth of interest."""
    return pu.sg_from_pressure(loss_bar, pu.tvd_rt_from_msl(z_msl_m, rt_elev_m))


def surge_swab_pressure_bar(pipe_speed_m_s, hole_d_m, pipe_d_m, length_m, density_kg_m3, plastic_viscosity_pa_s, yield_stress_pa, closed_end: bool = True, clinging_factor: float = 0.0) -> dict:
    """Surge (running in, +) or swab (pulling out, -) pressure at the bottom of an open or closed-ended string moving at `pipe_speed_m_s` (positive in, negative out).

    Closed end: the fluid displaced by the pipe volume flows up the annulus at u = v d_p^2 / (d_h^2 - d_p^2) (open end: the same with the pipe bore taking its share, not modelled: use closed as the
    upper estimate for a float or plugged bit).  `clinging_factor` adds the part of the pipe speed that the mud drags along the pipe wall (the Couette part); 0 is a lower estimate and the literature
    value is not entered from memory (ASSUMPTION: the user supplies it).  The sign follows the pipe movement.
    """
    v = np.asarray(pipe_speed_m_s, dtype=float)
    dh, dp = np.asarray(hole_d_m, dtype=float), np.asarray(pipe_d_m, dtype=float)
    u = np.abs(v) * dp ** 2 / (dh ** 2 - dp ** 2) + float(clinging_factor) * np.abs(v)
    q = u * annulus_area_m2(dh, dp)
    r = annular_pressure_loss(q, dh, dp, length_m, density_kg_m3, plastic_viscosity_pa_s, yield_stress_pa)
    r["pressure_bar"] = np.sign(v) * r["loss_bar"]
    r["closed_end"] = closed_end
    return r


def datum_offset_fraction(z_msl_m, rt_elev_m):
    """rt / (z + rt): the relative difference of an equivalent density between the sea-level and the rotary-table basis."""
    z = np.asarray(z_msl_m, dtype=float)
    return np.asarray(rt_elev_m, dtype=float) / (z + np.asarray(rt_elev_m, dtype=float))


def seawater_pressure_bar(depth_m, seawater_gradient_bar_per_m: float = 0.1005):
    return np.asarray(depth_m, dtype=float) * seawater_gradient_bar_per_m


def riser_connected_pressure_bar(mud_sg, z_msl_m, rt_elev_m):
    """Static pressure at the depth below sea level with the riser connected: the mud column stands from the rotary table."""
    return pu.mud_pressure_bar(mud_sg, z_msl_m, rt_elev_m)


def riser_disconnected_pressure_bar(mud_sg, z_msl_m, water_depth_m, seawater_gradient_bar_per_m: float = 0.1005):
    """Static pressure at the same depth after the riser is removed: seawater from sea level to the seabed plus the mud column below the seabed."""
    z = np.asarray(z_msl_m, dtype=float)
    return seawater_pressure_bar(water_depth_m, seawater_gradient_bar_per_m) + np.asarray(mud_sg, dtype=float) * pu.BAR_PER_M_PER_SG * (z - float(water_depth_m))


def riser_margin_bar(mud_sg, z_msl_m, water_depth_m, pore_pressure_bar, seawater_gradient_bar_per_m: float = 0.1005):
    """Overbalance that is left when the riser is removed: the disconnected static pressure minus the pore pressure (negative = no riser margin; the BOP has to hold the well)."""
    return riser_disconnected_pressure_bar(mud_sg, z_msl_m, water_depth_m, seawater_gradient_bar_per_m) - np.asarray(pore_pressure_bar, dtype=float)


def mud_sg_for_riser_margin(pore_pressure_bar, z_msl_m, water_depth_m, margin_bar: float = 0.0, seawater_gradient_bar_per_m: float = 0.1005):
    """The mud weight [sg] that keeps `margin_bar` of overbalance after a disconnect: (p_pore + margin - p_seawater(wd)) / (g (z - wd))."""
    z = np.asarray(z_msl_m, dtype=float)
    return (np.asarray(pore_pressure_bar, dtype=float) + margin_bar - seawater_pressure_bar(water_depth_m, seawater_gradient_bar_per_m)) / (pu.BAR_PER_M_PER_SG * (z - float(water_depth_m)))
