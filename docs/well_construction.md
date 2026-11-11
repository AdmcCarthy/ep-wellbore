# Well construction, cement, barriers and the age of tools

Code: `src/processing/well_design.py`, `src/processing/tool_eras.py`; notebooks 27 (`27_well_design_cement_barriers`) and 28 (`28_tool_age_uncertainty`); knowledge `knowledge/well_construction.yaml`; marts `well_design`, `well_design_check`, `tool_era_tests`, `tool_era_trends`.

## 1. What was read, and what it says (details in `knowledge/well_construction.yaml`)

| Source (read) | Used for |
| :--- | :--- |
| Wikipedia, *Casing (borehole)* (summary; the page asks for more citations) | the string types, decreasing diameters, the role of cement, how setting depths are chosen |
| Havtil study report *Use cemented shoe track as a well barrier* (Sept 2025, text extracted and read: sections 1, 2, 5) | the shoe track, float equipment, the Macondo and Visund incidents, the requirements of NORSOK D-010:2021, OEUK and API 65-2 as quoted, the conclusion that the verification of cemented shoe tracks as barriers is a gap |
| NTNU wiki *Integrity assessment of oil and gas wells* (summary) | the primary / secondary barrier definition, the 2006 PSA figure (18 % of wells with weakened integrity) |
| drillingmanual.com *Cement bond log* (summary) | what a bond log shows and cannot show |

Not read (requests in `knowledge/requests.yaml`): NORSOK D-010:2021 itself, the primary report of the 2006 survey, the API / ISO standards, the papers on cement-bond reliability, the tool-history papers (formation testers, survey tools, leak-off protocols). Statements about them in notebook 28 come from search results and are labelled so; no number from them is used in a test.

## 2. What the open data hold and do not hold

Held: per casing string the type, casing and hole diameter, shoe depth, the hole depth of the section and the test at the shoe (LOT, FIT, XLOT). Not held: cement type, volume, top of cement, bond logs, barrier schematics, integrity test results, annulus pressures, and the tool or procedure of any measurement. The leak-off test at a shoe is **indirect** evidence of the cement and rock at the shoe; the design figures say "strings and shoes", never "cement quality". Assumption: the hole depth is the depth of the section when the string was run, so `hole depth - shoe depth` is the open hole below the shoe (a few records are negative or over two kilometres: kept and flagged).

## 3. The age-of-measurement hypothesis

Hypothesis: older tools and practices give more uncertain measurements, but not for every data type. Because no tool field is held, the **drilling year** is a proxy for tool generation and practice together (and for the kind of wells drilled). Indicators per data type: residuals (leak-off value from the modelled band, test temperature from the bottom-hole value), resolution (share of values at a coarse step), quality flag (share of pressure tests rated high confidence). Each is tested as a rank correlation with the year, bootstrap over wells, a minimum effect of 0.10 (`negligible` below it), and a decision status from the straight-line change between the 10th and 90th percentile year against a 25 % tolerance (assumptions in `config/tool_eras.yaml`). Results today: the share of high-confidence pressure tests rises with the year (confirmed); leak-off residual, leak-off and shoe-depth reporting resolution are negligible trends; mud-weight resolution and temperature residual are inconclusive. The leak-off test mix changes with time (integrity tests, a lower bound only, become common in recent windows), a change of practice. For future data types the same three steps apply (notebook 28 section 5).

## 4. Limits

No cement or barrier question can be answered from the open data; the model fracture band used for the leak-off residual is itself a placeholder (DQ-110); the era test cannot separate tool, practice and well population.

> **Update 2026-11-08:** the Volve archive holds cement bond and ultrasonic logs (nine runs in four wells); see docs/cased_hole.md and notebook 27 section 8b. The statements above that the data hold no bond log refer to the open Sodir data.
