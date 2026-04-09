# globi-use-cases

Tools and a Streamlit app for **willingness-to-pay (WTP)**, **retrofit adoption**, and **emissions** analysis on [globi](https://github.com/globi)-style building energy outputs (`EnergyAndPeak.pq` parquets).

## What this repo does

1. **Energy & policy impacts** — Compare baseline vs retrofit scenario parquets (per year) to quantify energy and peak differences.
2. **Propensity** — Estimate per-building acceptance probabilities (residential logit with census-backed demographics; commercial NPV threshold).
3. **Uptake** — Apply adoption curves from JSON so buildings adopt over time according to propensity and curve targets.
4. **Emissions** — Combine adopted floor area and fuel mix with editable emissions-factor trajectories.

The main UI walks through upload, configuration, curve selection, emissions editing, and a full pipeline run with charts and tables.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

## Setup

```bash
uv sync
```

For local runs, the Makefile sets `PYTHONPATH=src` so `app` and `use_cases` import correctly.

## Run the analysis app

```bash
make run
```

Equivalent:

```bash
PYTHONPATH=src uv run streamlit run src/app/wtp_app.py
```

Open the URL Streamlit prints (default [http://localhost:8501](http://localhost:8501)).

### App workflow (high level)

1. **Upload data** — Baseline and scenario `EnergyAndPeak.pq` files (one pair per simulated year).
2. **Configure** — Retrofit costs, energy prices, demographics / Monte Carlo settings, and related options in the sidebar flows.
3. **Adoption curves** — Pick or preview scenarios from `data/inputs/adoption_curves.json`.
4. **Emissions** — Inspect or edit trajectories in `data/inputs/emissions_trajectories.json`.
5. **Run & results** — Execute the pipeline and review adoption and emissions outputs.

Bundled sample inputs live under `data/inputs/` so you can experiment without generating new globi runs first.

## Scenario editor (curves & trajectories)

A separate Streamlit UI is available for editing adoption and emissions JSON used by the main app:

```bash
make editor
```

## Split multi-scenario globi parquets

If one parquet stacks many `retrofit.scenario` values (and Monte Carlo IDs), split them into one file per scenario for the main app or for scripting:

```bash
PYTHONPATH=src uv run python src/tools/split_scenarios.py \
  data/inputs/globi_outputs/342501.pq \
  --scenarios Baseline ASHP \
  --out-dir data/inputs/globi_outputs/split
```

Omit `--scenarios` to export every scenario found. Use `--seed` for reproducible Monte Carlo ID sampling.

## Data layout (`data/inputs/`)

| Path | Role |
|------|------|
| `globi_outputs/` | globi `*.pq` outputs (and optional `split/` per-scenario files) |
| `adoption_curves.json` | Named adoption scenarios, modifiers |
| `emissions_trajectories.json` | Fuel / factor time series for emissions calc |
| `retrofit_costs.json` | Capex and incentive inputs for cost / WTP logic |
| `scenario_config.json` | Example year → filename mapping for batch-style configs |
| `climate_opinions.csv` | Census / opinion inputs used by propensity where relevant |

Paths are resolved relative to the repo root inside the app and Docker image.

## Docker

Build and run the main Streamlit app in a container (port 8501 by default):

```bash
make docker-build
make docker-run
```

Override image name or host port:

```bash
make docker-run IMAGE_NAME=globi-use-cases PORT=8501
```

## Project layout

- `src/app/` — Streamlit WTP app (`wtp_app.py`), analysis helpers (`analysis/`)
- `src/use_cases/` — Propensity, uptake, and cost modules used by the pipeline
- `src/tools/` — Utilities (e.g. `split_scenarios.py`) and optional viz WIP (`viz/wip/interface.py`)
