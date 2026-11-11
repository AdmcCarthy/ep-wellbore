# Annular pressure, pressure loss, the datum and the water depth

Code: `annular_hydraulics` (pure functions), tests `tests/test_annular_hydraulics.py`, notebook `25_annular_pressure_datum` (a primer for geoscientists), knowledge `knowledge/annular_hydraulics.yaml`.

## 1. Terms

| Term | Meaning |
| :--- | :--- |
| Static pressure, ESD | mud pressure with the pumps off; ESD is the uniform density that gives it |
| Annular pressure loss (APL) | friction pressure of mud flowing up the annulus [bar] |
| ECD | `ESD + (APL + back pressure) / (g z_RT)` (API RP 59, IADC Lexicon; ABS includes back pressure, API STD 65-2 does not) |
| Surge / swab | pressure rise running in / drop pulling out |
| Riser margin | overbalance left when the riser is disconnected |

## 2. The model (screening, not a simulator)

* Bingham plastic mud (plastic viscosity, yield stress), laminar flow in a concentric annulus approximated as a parallel slot: `u = G h^2 / (3 mu) (1 - 1.5 xi + 0.5 xi^3)`, `xi = tau_y / (G h)`, `h` half the radial gap; `G` by bisection. It reduces to `G = 12 mu u / gap^2` without yield stress (tested).
* Turbulence flag: screening Reynolds number above 2100 (ASSUMPTION; a Hedstrom criterion gives a higher limit for yield-stress fluids); then the larger of the laminar value and a Blasius estimate.
* Surge / swab: closed-end displacement velocity `v d_p^2 / (d_h^2 - d_p^2)` plus a user-supplied clinging factor (default 0 = lower estimate; no literature value is entered).
* Left out: eccentricity (lowers the loss), rotation, cuttings, tool joints, temperature and pressure effects on the mud. All rheology, flow-rate and geometry inputs are ASSUMPTIONS (the open data hold none; the Sodir `wellbore_mud` table has a surface mud weight, viscosity and yield point per depth, not a rheology model).

## 3. Does the rotary table height (and the rig type) matter?

`EMW_RT = p / (g (z + rt))` against `EMW_MSL = p / (g z)`: the relative difference is `rt / (z + rt)`.

* Header `rt_elev_m` (9318 wellbores, all with basis `KB`): 5 % / 50 % / 95 % percentiles 3.3 / 32 / 78 m, maximum 85 m. Fixed platforms (Gullfaks, Statfjord) stand at 78-85 m.
* For a 1.5 sg mud at 2500 m the offset is 0.002 sg (RT 3 m), 0.019 (32 m), 0.034 (58 m), 0.049 sg (84 m): comparable with a 0.05 sg window margin only for a tall platform and a shallow section, and zero if the datum is stated and converted with the header value (what the analysis notebooks-25 do).
* The rig type is not in the open data; the RT per facility name is in the notebook. The KB / RT difference of the header is not given (an ASSUMPTION of a few metres: 0.003 sg for 5 m at 2500 m) and is not resolved. **Conclusion: stating the datum matters, the rig type and KB / RT difference do not.**

## 4. Does the water depth matter, and from where?

Modelled with the compaction trend of `config/pore_pressure.yaml` (ASSUMPTION) and the Zhang bounds: the fracture gradient 600 m below the seabed is 1.48 sg with no water, 1.42 (100 m), 1.39 (150 m), 1.30 (400 m), 1.20 sg (1000 m). Wells: median water depth 134 m, 95 % above 381 m only about 3 %. **Conclusion: within 70-150 m the effect is a few hundredths of a sg, which can matter against a narrow window; from a few hundred metres it dominates; compare leak-off and fracture gradients only at similar water depth and equal depth below the seabed.** A density log would change the absolute level, not the trend.

## 5. Riser margin

Connected: `p = MW g (z + rt)`. Disconnected: `p = rho_w g wd + MW g (z - wd)`. Margin = disconnected pressure minus pore pressure; the mud for zero margin is `(p_pore - rho_w g wd) / (g (z - wd))`. In the illustrative case in the notebook (pore pressure 1.15 sg, 1500 m below the seabed) it needs 1.16 sg in 100 m and 1.32 sg in 2000 m of water and reaches the modelled fracture gradient at about 2000 m (all inputs assumed).

## 6. Not recorded because not read

Rheology models in detail (Herschel-Bulkley), eccentricity and rotation factors, lost-circulation classes, ballooning, managed pressure drilling, pressure-while-drilling practice: listed in `docs/literature_to_acquire.md`.
