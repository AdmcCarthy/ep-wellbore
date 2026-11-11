# ep-wellbore: Where the well is, how it was surveyed and what it is made of

Trajectory by minimum curvature, wellhead and slot, survey tool error models, rotary-table and kelly-bushing elevation, annular hydraulics, well design, tool eras and mud classes.

**Status: private, built public-ready.** The library follows the rules of a public one (no host-repository imports, no data, no private paths, an automated publication gate in `scripts/publication_gate.py`) and is published only when that gate passes. Licence: MIT with acknowledgement (`LICENSE`).

## Install

```
uv add git+https://github.com/AdmcCarthy/ep-wellbore
```

Depends on: `ep-physics-base`. Optional extras:

none

## What is in it

| Module | What it does |
| :--- | :--- |
| `trajectory` | Well trajectory mathematics: the minimum-curvature method (AGENTS section 6B: the only method allowed for computing TVD from a survey). |
| `wellhead` | Wellhead, rotary-table and survey-uncertainty logic shared by every trajectory source (pure functions; thresholds come from the adapter or `config/pos |
| `survey_tools` | Survey tools: how the positions of different tool classes and different times compare at the same depth, and what that says about quality (docs/volve_ |
| `elevation` | Which elevation do the depths of a wellbore refer to: the rotary table (RT) or the kelly bushing (KB)?  A monitored judgement that is re-evaluated eve |
| `annular_hydraulics` | Annular pressure loss, ECD, surge and swab, the datum and water-depth sensitivities and the riser margin.  Pure functions (docs/annular_hydraulics.md, |
| `well_design` | Basic well design from the casing and leak-off table: strings, hole sizes, shoe depths, open hole below the shoe, and the telescoping check.  Pure fun |
| `tool_eras` | Does the age of the tool or practice matter for the uncertainty of a measurement?  A hypothesis test per data type, with the evidence kept apart from  |
| `mud_zones` | Mud type as a categorical variable by depth zone, and the mud weight against the leak-off value at the casing shoes.  Pure functions (docs/knowledge_a |

Functions are pure: they take frames and parameters and return `(result, events)` where they change or flag data; nothing reads or writes files except the configuration loader.

## Settings and assumptions

Every setting has a packaged default in `src/ep_wellbore/defaults/`. A project can override a file by providing `config/<same name>` in its working directory; the project file wins. A default is an assumption unless it says otherwise: its level (`placeholder`, `inferred`, `published`, `official`, `exact`) and a one-line basis travel with it, and a citation (author and year) is given where one exists. Libraries never import a knowledge store: a host that has one passes its values in as an overlay.

## Develop

```
uv sync --all-extras
uv run pytest            # 3 test files
uv run ruff check .
uv run python scripts/publication_gate.py
```

Tests use small synthetic data generated on the fly. No network, no real data.
