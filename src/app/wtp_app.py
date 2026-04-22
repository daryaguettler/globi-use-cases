"""Willingness-to-Pay & Adoption Analysis — Streamlit Application.

Workflow:
  1. Upload Data        — Baseline + scenario parquet per year (per-year upload, batch pool, or browse `data/inputs/`)
  2. Configure          — Retrofit cost, energy prices, demographics, MC settings
  3. Adoption Curves    — Select / preview adoption curve scenario
  4. Emissions          — Edit emissions factor trajectories
  5. Run & Results      — Execute pipeline, view emissions trajectory outputs

Run with:
    streamlit run src/app/wtp_app.py
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import time
import copy
import zipfile
from functools import reduce
from pathlib import Path

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pydantic import BaseModel, Field

import folium
import geopandas as gpd
from scipy import stats
from streamlit_folium import st_folium

# Add repo root to path so use_cases imports work
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from app.analysis.energy_delta import (
    DEFAULT_ENERGY_PRICES,
    FUEL_LABELS,
    build_policy_impacts,
    load_energy_parquet,
)
from app.analysis.adoption_breakdown import (
    build_adoption_cohort_by_demographics,
    count_adopters_with_positive_incentive,
    expected_incentive_sum_on_adopted_cohort,
    last_year_uptake_row,
    n_adopters_from_yearly_row,
    ranking_displacement_at_equal_n,
    resolve_retrofit_scenario_name,
)
from app.analysis.emissions_calc import compute_emissions_trajectory
from use_cases.apply_propensity import (
    MEAN_ACCEPTANCE_AGGREGATE_DESCRIPTION,
    MEAN_ACCEPTANCE_AGGREGATE_LABEL,
    MEAN_ACCEPTANCE_MEAN_WITHIN_RETROFIT_SCENARIO_DEFINITION,
    PropensityModelEngine,
)
from use_cases.apply_uptake import AdoptionEngine, UptakeResult

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = _REPO_ROOT / "data" / "inputs"
_ADOPTION_CURVES_PATH = _DATA_DIR / "adoption_curves.json"
_EMISSIONS_PATH = _DATA_DIR / "emissions_trajectories.json"
_CONFIGS_DIR = _REPO_ROOT / "data" / "outputs"
_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

_SCENARIOS_DIR = _REPO_ROOT / "outputs"
_SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

_YEARS_RANGE = list(range(2024, 2101))
_PROJECTION_YEARS = list(range(2025, 2101))

# Income tiers for incentive UI — maps display label → list of INCOME_CATEGORIES_K values
# that fall in that tier (midpoints in k$, matching apply_propensity.INCOME_CATEGORIES)
_INCOME_TIERS: list[tuple[str, list[float]]] = [
    ("< $30k  (Very Low Income)",  [5.0, 12.5, 17.5, 22.5, 27.5]),
    ("$30k–$50k  (Low Income)",    [32.5, 37.5, 42.5, 47.5]),
    ("$50k–$80k  (Moderate)",      [55.0, 67.5]),
    ("$80k–$150k  (Middle)",       [87.5, 112.5, 137.5]),
    ("> $150k  (Higher Income)",   [175.0, 225.0]),
]

_WIP_SINGLE_TRACT_MEAN_ACCEPTANCE_HELP = (
    "Mean over Monte Carlo draws of the logit acceptance probability for this tract "
    "(propensity exploration; not adoption uptake)."
)

# markdown for st.expander("Definition & formula") under KPIs and charts
_FORMULA_MD_BUILDINGS_MODELLED = r"""
### Buildings modelled

Count of buildings in the policy impacts table for this run:

$$
N_{\mathrm{bldg}} = \#\{\text{buildings in policy impacts}\}
$$
"""

_FORMULA_MD_MEAN_ACCEPTANCE_AGGREGATE = r"""
### Mean acceptance (avg. over buildings)

Let \(p_i\) be **acceptance probability** for row \(i\) (one row per building–retrofit, etc.).  
\(p_i\) comes **only from the propensity model** (residential: mean of logit draws over propensity MC; commercial: NPV rule).  
Adoption curves and uptake RNG **do not** change \(p_i\).

$$
\bar{p} = \frac{1}{N}\sum_{i=1}^{N} p_i
$$

where \(N\) is the number of propensity rows included in the mean (equal weight per row).
"""

_FORMULA_MD_SCENARIO_SUMMARY_TABLE = r"""
### Scenario summary table

**Final adoption (mean %)** — value of **cumulative adoption %** in the **last projection year**, from the **mean** trajectory of the uptake **MC ensemble** (many `dice_roll` runs).

**Final adoption (P10–P90)** — same year, from the **10th and 90th percentiles** across those uptake runs.

**Cumulative adoption %** at year \(t\):

$$
A(t) = \frac{n_{\mathrm{adopted}}(t)}{N_{\mathrm{bldg}}} \times 100\%
$$

**Emissions reduction vs baseline** (final year \(T\)):

$$
\frac{E_{\mathrm{baseline}}(T) - E_{\mathrm{scenario}}(T)}{E_{\mathrm{baseline}}(T)} \times 100\%
$$
"""

_FORMULA_MD_ADOPTION_TRAJECTORY = r"""
### Cumulative adoption (chart)

The adoption **curve** in config is a **cumulative capacity** \(C(t) \in [0,1]\): target adopted count is \(\lfloor N \cdot C(t) \rfloor\) (see adoption JSON).

Each **MC ensemble** run uses **dice roll** uptake.  If **dice re-entry** is off (``None``), each building draws \(u \sim \mathrm{Uniform}(0,1)\) **once** the first time there is positive adoption need; if \(p_i > u\) it stays in the *willing* pool, otherwise it is out; in later years only ranked allocation fills the curve (no re-roll).  If re-entry is on, each year each **eligible** building draws again, with cooldown **dice re-entry years** after a non-adoption.  Filling is always by highest \(p_i\) first up to the curve.  Optional attrition noise can reduce counts.

The plotted **mean** line is the **mean** of \(A(t)\) across runs; the band is typically **P10–P90** across runs.

The reserved **full adoption** scenario is different: it does **not** use propensity or the dice-roll MC. It assumes every building is on the retrofit from the first projection year (100% cumulative immediately), with no P10–P90 band — a theoretical ceiling for comparison.
"""

_FORMULA_MD_ENERGY_TRAJECTORY = r"""
### Stock energy (GWh/yr)

Total annual energy is a **mix** of baseline kWh for buildings not yet adopted and scenario (retrofit) kWh for adopted buildings, using the **mean** (and P10/P90) adoption fraction from uptake for each year.

Conceptually:

$$
E_{\mathrm{stock}}(t) = \sum_{\text{buildings } i}
\Bigl[ \alpha_i(t)\, E_{\mathrm{scenario},i}(t) + \bigl(1-\alpha_i(t)\bigr)\, E_{\mathrm{baseline},i}(t) \Bigr]
$$

with \(\alpha_i(t)\) the adopted fraction implied by cumulative adoption at \(t\) (from the ensemble mean path for the **mean** line).
"""

_FORMULA_MD_EMISSIONS_TRAJECTORY = r"""
### Emissions (tCO₂/yr)

For each year, building kWh by fuel is converted with **time-varying** emissions factors (kg CO₂/kWh), then summed. The same **adopted vs baseline** split as in the energy calculation applies:

$$
\mathrm{CO}_2(t) = \sum_i \sum_f kWh_{i,f}(t)\cdot \phi_f(t)
$$

where \(kWh_{i,f}\) follows baseline or retrofit depending on whether building \(i\) has adopted by \(t\), and \(\phi_f(t)\) is the factor for fuel \(f\). The chart shows **mean** and **P10–P90** when available.
"""

_FORMULA_MD_PROPENSITY_HISTOGRAM = r"""
### Propensity distribution

Histogram of **acceptance probability** \(p_i\) across buildings (same \(p_i\) as in the mean acceptance KPI). Residential values are **means over propensity MC draws** per building; commercial are **0 or 1** from NPV.
"""

_FORMULA_MD_ENERGY_SAVINGS_SCATTER = r"""
### Energy savings vs propensity

Each point is one building: **x** = annual energy cost savings (\$/yr), **y** = acceptance probability \(p_i\). Shows how financial savings relate to modelled willingness to accept.
"""

_FORMULA_MD_PORTFOLIO_RETROFIT_AND_INCENTIVE = r"""
### Total gross retrofit cost (portfolio)

Sum of **gross_upfront_usd** over all propensity rows in the with-incentive run. Each row is one building (or building–retrofit) using the same upfront cost field as the WTP model: \texttt{net\_cost.AllCustomers} or \texttt{cost.Total} (see propensity code). **Incentive pricing does not change** this — it is the gross **deal** cost before the income draw subtracts a subsidy in the logit.

$$
C_{\mathrm{gross}} = \sum_i \text{gross\_upfront\_usd}_i
$$

### Total expected incentives (portfolio)

**Residential:** For each building, given census income draws, the incentive in USD in each draw is the tier’s configured flat amount. The column **expected_incentive_usd** is the mean of those draws. **Commercial** rows are 0.

$$
S_{\mathrm{exp}} = \sum_i \text{expected\_incentive\_usd}_i
$$

This is an **unweighted portfolio expectation** of subsidy outlay (not divided by adoption uptake; not conditional on a household accepting the deal). If every row in the table were a full retrofit at the assigned tier’s incentive, the implied average budget for incentives would scale with this sum. Optional attrition: compare to adoption-weighted outlay in downstream analysis if needed.
"""

_FORMULA_MD_ADOPTION_DEMO_BREAKDOWN = r"""
### Adoption cohort by income / education (with incentive)

**Who adopts?** The stock energy and emissions model ranks buildings by `acceptance_probability` and marks the first \(n\) rows as *adopted* for that year’s cumulative adoption %, where \(n = \mathrm{round}((A/100) \cdot N)\), \(A\) the mean cumulative % and \(N\) the number of policy rows.

**Income and education bars:** Each building has **tract** multinomials (`income_probs`, `education_probs` from census). For a **stable, reproducible** label per building, we draw a single **deterministic** category index from that row’s using a fixed RNG seed derived from `building.id` (not the propensity MC). Counts in the bar chart are for the **with-incentive** final cohort of adopters (same \(n\) as above, ranked by with-incentive propensity). Commercial rows without tract lists may show **Unknown**.

### Incentive / ranking comparison (same n)

We compare the top \(n\) building ids by with-incentive propensity vs. the top \(n\) by no-incentive propensity for \(n = \min(n_{\text{no inc}}, n_{\text{inc}})\) from the two runs’ final cumulative %. **Replaced in rank** = \(|S_{\text{inc}} \setminus S_{\text{base}}|\) (same as \(|S_{\text{base}} \setminus S_{\text{inc}}|\) when \(|S|=n\) and ids are unique).

**Expected incentives on the adopted cohort** = sum of `expected_incentive_usd` (mean draw per building from the with-incentive propensity run) over building ids in the with-incentive top-\(n_{\text{inc}}\) set.
"""

_FORMULA_MD_SAVED_SCENARIO_NAME = r"""
### Scenario label

The folder / display name chosen when saving this result bundle.
"""

_FORMULA_MD_SAVED_AT = r"""
### Saved

ISO timestamp written when **Save Results** ran (date shown is the calendar prefix).
"""

_FORMULA_MD_WIP_SINGLE_TRACT_ACCEPTANCE = r"""
### Mean / median acceptance (tract, sim draws)

For the selected tract, the model draws demographics \(S\) times (e.g. 10,000), evaluates the logit probability each draw, then reports the **mean** and **median** of those simulated probabilities. This is a **WTP explorer** shortcut; the main run uses the full propensity engine on your building table.
"""

_FORMULA_MD_WIP_TRACT_DRAWS_STD = r"""
### Std deviation (sim draws)

Sample standard deviation of the \(S\) simulated acceptance probabilities for this tract:

$$
s = \sqrt{\frac{1}{S-1}\sum_{k=1}^{S}(p_k - \bar{p})^2}
$$
"""

_FORMULA_MD_WIP_STATEWIDE_MEAN = r"""
### Mean (across tracts)

Let \(\bar{p}_t\) be the tract-level mean of sim draws for tract \(t\). Displayed value:

$$
\frac{1}{T}\sum_{t=1}^{T} \bar{p}_t
$$
"""

_FORMULA_MD_WIP_STATEWIDE_MEDIAN = r"""
### Median (across tracts)

Median of \(\{\bar{p}_1,\ldots,\bar{p}_T\}\).
"""

_FORMULA_MD_WIP_STATEWIDE_MIN = r"""
### Min (across tracts)

\(\displaystyle \min_t \bar{p}_t\).
"""

_FORMULA_MD_WIP_STATEWIDE_MAX = r"""
### Max (across tracts)

\(\displaystyle \max_t \bar{p}_t\).
"""

_FORMULA_MD_WIP_TRACT_ROW_COUNT = r"""
### Census tracts

Number of census tracts (rows) in the loaded demographic table for this state.
"""

_FORMULA_MD_WIP_COUNTY_COUNT = r"""
### Counties

Number of distinct county labels in that table.
"""


class IncentiveConfig(BaseModel):
    """Income-stratified incentive amounts (USD) subtracted from retrofit cost per MC sample."""

    very_low_income: float = Field(0.0, ge=0.0, description="Incentive for < $30k households (USD)")
    low_income: float = Field(0.0, ge=0.0, description="Incentive for $30k–$50k households (USD)")
    moderate_income: float = Field(0.0, ge=0.0, description="Incentive for $50k–$80k households (USD)")
    middle_income: float = Field(0.0, ge=0.0, description="Incentive for $80k–$150k households (USD)")
    higher_income: float = Field(0.0, ge=0.0, description="Incentive for > $150k households (USD)")

    def to_income_map(self) -> dict[float, float]:
        """Map each INCOME_CATEGORIES value to its incentive amount in USD."""
        tier_amounts = [
            self.very_low_income,
            self.low_income,
            self.moderate_income,
            self.middle_income,
            self.higher_income,
        ]
        result: dict[float, float] = {}
        for (_, income_vals), amount in zip(_INCOME_TIERS, tier_amounts):
            for iv in income_vals:
                result[iv] = amount
        return result


def _build_income_incentive_map() -> dict[float, float]:
    """Return {income_k_usd: incentive_usd} from the current IncentiveConfig in session_state."""
    cfg: IncentiveConfig = st.session_state.get("incentive_config", IncentiveConfig())
    return cfg.to_income_map()

st.set_page_config(
    page_title="WTP & Adoption Analysis | globi",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_STATE_DEFAULTS: dict = {
    # {year: {"baseline": bytes, "scenario": bytes}} — one pair per simulated year
    "year_files": {},
    # batch upload pool: filename -> bytes (used by "Upload many, pick per year")
    "uploaded_pq_library": {},
    "selected_years": [2025],
    "n_years": 1,
    "scenario_name": "Retrofit",
    "policy_impacts": None,       # keyed to the reference (earliest) year
    "propensity_result": None,
    # {adoption_scenario_name: {"uptake": UptakeResult, "emissions": pd.DataFrame}}
    "scenario_results": {},
    "run_complete": False,
    "dice_reentry_never": True,
    "wip_dice_reentry_never": True,
}
for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

_EXAMPLE_BASELINE = _DATA_DIR / "example" / "Baseline" / "EnergyAndPeak.pq"
_EXAMPLE_SCENARIO = _DATA_DIR / "example" / "Retrofit" / "EnergyAndPeak.pq"
_EXAMPLE_YEAR = 2025

if (
    not st.session_state["year_files"]
    and _EXAMPLE_BASELINE.exists()
    and _EXAMPLE_SCENARIO.exists()
):
    st.session_state["year_files"] = {
        _EXAMPLE_YEAR: {
            "baseline": _EXAMPLE_BASELINE.read_bytes(),
            "scenario": _EXAMPLE_SCENARIO.read_bytes(),
        }
    }
    st.session_state["selected_years"] = [_EXAMPLE_YEAR]
    st.session_state["n_years"] = 1
    st.session_state["scenario_name"] = "Retrofit"
    st.session_state["_example_preloaded"] = True

@st.cache_data(show_spinner=False)
def _load_adoption_curves() -> dict:
    if _ADOPTION_CURVES_PATH.exists():
        return json.loads(_ADOPTION_CURVES_PATH.read_text())
    return {"scenarios": {}, "retrofit_specific_modifiers": {}, "income_bracket_modifiers": {}}


@st.cache_data(show_spinner=False)
def _load_emissions_json() -> dict:
    if _EMISSIONS_PATH.exists():
        return json.loads(_EMISSIONS_PATH.read_text())
    return {"description": "", "fuels": {}}


def _save_adoption_curves(data: dict) -> None:
    _ADOPTION_CURVES_PATH.write_text(json.dumps(data, indent=4))
    st.cache_data.clear()


def _save_emissions_json(data: dict) -> None:
    _EMISSIONS_PATH.write_text(json.dumps(data, indent=4))
    st.cache_data.clear()


# ── Config management ──────────────────────────────────────────────────────────

def _list_saved_configs() -> list[str]:
    """Return sorted list of config names found in data/outputs/."""
    return sorted(
        p.name for p in _CONFIGS_DIR.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    )


def _save_config(name: str) -> None:
    """Persist current session state config + JSON files to data/outputs/{name}/config.json."""
    import datetime

    incentive_cfg: IncentiveConfig = st.session_state.get("incentive_config", IncentiveConfig())
    config = {
        "name": name,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "adoption_curves": _load_adoption_curves(),
        "emissions_trajectories": _load_emissions_json(),
        "energy_prices": st.session_state.get("energy_prices", dict(DEFAULT_ENERGY_PRICES)),
        "cost_per_sqm": float(st.session_state.get("cost_per_sqm", 150.0)),
        "incentives_enabled": bool(st.session_state.get("incentives_enabled", False)),
        "incentive_config": incentive_cfg.model_dump(),
    }
    config_dir = _CONFIGS_DIR / name
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config, indent=4))


def _load_config(name: str) -> bool:
    """Load a saved config into session state and restore JSON input files. Returns True on success."""
    config_path = _CONFIGS_DIR / name / "config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())

    # Restore adoption curves + emissions to disk (so cached loaders pick them up)
    if "adoption_curves" in config:
        _ADOPTION_CURVES_PATH.write_text(json.dumps(config["adoption_curves"], indent=4))
    if "emissions_trajectories" in config:
        _EMISSIONS_PATH.write_text(json.dumps(config["emissions_trajectories"], indent=4))
    st.cache_data.clear()

    # Restore session state for energy prices, retrofit cost, incentives
    if "energy_prices" in config:
        st.session_state["energy_prices"] = config["energy_prices"]
    if "cost_per_sqm" in config:
        st.session_state["cost_per_sqm"] = float(config["cost_per_sqm"])
    if "incentives_enabled" in config:
        st.session_state["incentives_enabled"] = bool(config["incentives_enabled"])
    if "incentive_config" in config:
        st.session_state["incentive_config"] = IncentiveConfig(**config["incentive_config"])

    return True


# ── Scenario results saving/loading ───────────────────────────────────────────

def _list_saved_scenarios() -> list[str]:
    """Return sorted list of scenario names found in outputs/."""
    if not _SCENARIOS_DIR.exists():
        return []
    return sorted(
        p.name for p in _SCENARIOS_DIR.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )


def _save_scenario_results(name: str) -> None:
    """Save current run results to outputs/{name}/."""
    import datetime

    scenario_dir = _SCENARIOS_DIR / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenario_results: dict = st.session_state.get("scenario_results", {})
    policy_impacts: pd.DataFrame | None = st.session_state.get("policy_impacts")
    propensity_result = st.session_state.get("propensity_result")
    simulated_years = st.session_state.get("simulated_years", [])

    # Metadata
    metadata = {
        "scenario_name": name,
        "run_name": st.session_state.get("scenario_name", "Retrofit"),
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_buildings": len(policy_impacts) if policy_impacts is not None else 0,
        "simulated_years": simulated_years,
        "mean_acceptance_probability": (
            float(propensity_result.mean_acceptance_probability)
            if propensity_result is not None else None
        ),
        "mean_acceptance_probability_definition": MEAN_ACCEPTANCE_AGGREGATE_DESCRIPTION,
        "n_adoption_scenarios": len(scenario_results),
        "adoption_scenario_names": list(scenario_results.keys()),
    }
    (scenario_dir / "metadata.json").write_text(json.dumps(metadata, indent=4))

    # Config snapshot
    incentive_cfg: IncentiveConfig = st.session_state.get("incentive_config", IncentiveConfig())
    config = {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "adoption_curves": _load_adoption_curves(),
        "emissions_trajectories": _load_emissions_json(),
        "energy_prices": st.session_state.get("energy_prices", dict(DEFAULT_ENERGY_PRICES)),
        "cost_per_sqm": float(st.session_state.get("cost_per_sqm", 150.0)),
        "incentives_enabled": bool(st.session_state.get("incentives_enabled", False)),
        "incentive_config": incentive_cfg.model_dump(),
    }
    (scenario_dir / "config.json").write_text(json.dumps(config, indent=4))

    # Emissions trajectories CSV
    em_frames = []
    for scen_name, res in scenario_results.items():
        em = res["emissions"]
        if not em.empty:
            em_frames.append(em.assign(adoption_scenario=scen_name))
    if em_frames:
        pd.concat(em_frames).to_csv(scenario_dir / "emissions_trajectories.csv", index=False)

    # Adoption summaries CSV (include adoption_scenario column)
    adoption_frames = []
    for scen_name, res in scenario_results.items():
        ys = res["uptake"].yearly_summary
        if not ys.empty:
            adoption_frames.append(ys.assign(adoption_scenario=scen_name))
    if adoption_frames:
        pd.concat(adoption_frames).to_csv(scenario_dir / "adoption_summaries.csv", index=False)

    # Policy impacts CSV
    if policy_impacts is not None:
        policy_impacts.to_csv(scenario_dir / "policy_impacts.csv", index=False)


def _load_scenario_data(name: str) -> dict | None:
    """Load saved scenario data from outputs/{name}/."""
    scenario_dir = _SCENARIOS_DIR / name
    if not scenario_dir.exists():
        return None

    data: dict = {}

    metadata_path = scenario_dir / "metadata.json"
    if metadata_path.exists():
        data["metadata"] = json.loads(metadata_path.read_text())

    emissions_path = scenario_dir / "emissions_trajectories.csv"
    if emissions_path.exists():
        data["emissions"] = pd.read_csv(emissions_path)

    adoption_path = scenario_dir / "adoption_summaries.csv"
    if adoption_path.exists():
        data["adoption"] = pd.read_csv(adoption_path)

    policy_path = scenario_dir / "policy_impacts.csv"
    if policy_path.exists():
        data["policy_impacts"] = pd.read_csv(policy_path)

    return data


def _render_config_manager() -> None:
    """Sidebar panel for saving and loading named configurations."""
    st.sidebar.title("Configurations")
    st.sidebar.caption(
        "Save or load a named configuration — including adoption curves, emissions "
        "trajectories, energy prices, and retrofit & incentive costs."
    )

    saved = _list_saved_configs()

    # ── Load existing config ───────────────────────────────────────────────────
    if saved:
        st.sidebar.markdown("**Load saved config**")
        selected = st.sidebar.selectbox(
            "Select config", options=saved, key="cfg_mgr_select",
            label_visibility="collapsed",
        )
        if st.sidebar.button("Load", key="cfg_mgr_load", use_container_width=True):
            if _load_config(selected):
                st.sidebar.success(f"Loaded '{selected}'")
                st.rerun()
            else:
                st.sidebar.error("Could not load config.")
    else:
        st.sidebar.info("No saved configs yet.")

    st.sidebar.divider()

    # ── Save current config ────────────────────────────────────────────────────
    st.sidebar.markdown("**Save current config**")
    new_name = st.sidebar.text_input(
        "Config name", placeholder="e.g. high_incentives_2030",
        key="cfg_mgr_new_name", label_visibility="collapsed",
    )
    save_disabled = not new_name.strip()
    if st.sidebar.button(
        "Save", key="cfg_mgr_save",
        disabled=save_disabled,
        use_container_width=True,
    ):
        clean_name = new_name.strip().replace(" ", "_")
        _save_config(clean_name)
        st.sidebar.success(f"Saved as '{clean_name}'")
        st.rerun()


_YEAR_IN_NAME_RE = re.compile(r"\b(20[2-9]\d{2}|21\d{2})\b")


def _energy_parquet_year_and_role(filename: str) -> tuple[int | None, str | None]:
    """Infer simulated year and baseline vs scenario role from common naming patterns."""
    m = _YEAR_IN_NAME_RE.search(filename)
    yr = int(m.group(1)) if m else None
    ln = filename.lower()
    if "baseline" in ln:
        return yr, "baseline"
    if any(k in ln for k in ("retrofit", "deep", "scenario", "scen")):
        return yr, "scenario"
    return yr, None


def _sync_uploaded_pq_library_from_files(uploaded_files) -> None:
    """Merge multi-file uploader results into session uploaded_pq_library (read once per change)."""
    if not uploaded_files:
        return
    sig = tuple((getattr(f, "name", ""), getattr(f, "size", None)) for f in uploaded_files)
    if sig == st.session_state.get("_batch_upload_sig"):
        return
    st.session_state["_batch_upload_sig"] = sig
    lib: dict[str, bytes] = st.session_state.setdefault("uploaded_pq_library", {})
    names_seen: dict[str, int] = {}
    for f in uploaded_files:
        name = f.name or "unknown"
        names_seen[name] = names_seen.get(name, 0) + 1
        lib[name] = f.read()
    dups = [n for n, c in names_seen.items() if c > 1]
    if dups:
        st.session_state["_batch_upload_duplicate_names"] = dups
    else:
        st.session_state.pop("_batch_upload_duplicate_names", None)


def _auto_assign_library_to_years(
    selected_years: list[int],
    library: dict[str, bytes],
    year_files: dict,
) -> dict:
    """Map library files to year_files using filename year + baseline/scenario hints."""
    by_year: dict[int, dict[str, str]] = {}
    for name in library:
        yr, role = _energy_parquet_year_and_role(name)
        if yr is None or role is None:
            continue
        by_year.setdefault(yr, {})[role] = name
    out = {**year_files}
    for yr in selected_years:
        m = by_year.get(yr)
        if not m:
            continue
        entry = dict(out.get(yr, {}))
        if "baseline" in m:
            bn = m["baseline"]
            entry["baseline"] = library[bn]
            entry["baseline_label"] = bn
        if "scenario" in m:
            sn = m["scenario"]
            entry["scenario"] = library[sn]
            entry["scenario_label"] = sn
        out[yr] = entry
    return out


def _preview_parquet(file_bytes: bytes) -> dict:
    """Return quick stats from an uploaded EnergyAndPeak.pq."""
    try:
        df = pd.read_parquet(io.BytesIO(file_bytes))
        idx = df.index.to_frame(index=False) if isinstance(df.index, pd.MultiIndex) else pd.DataFrame()
        n_buildings = len(df)
        area_col = "feature.geometry.energy_model_conditioned_area"
        total_area = None
        if area_col in idx.columns:
            areas = pd.to_numeric(idx[area_col], errors="coerce").dropna()
            total_area = float(areas.sum()) if not areas.empty else None
        scenario_col = next(
            (c for c in ["retrofit.scenario", "feature.semantic.Scenario"] if c in idx.columns),
            None,
        )
        scenarios = sorted(idx[scenario_col].dropna().unique().tolist()) if scenario_col else []
        return {"n_buildings": n_buildings, "total_area_m2": total_area, "scenarios": scenarios}
    except Exception as exc:
        return {"error": str(exc)}


def _curve_colors() -> list[str]:
    return ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def _render_upload_tab() -> None:
    st.markdown("## Step 1 — Upload Energy Data")
    st.markdown(
        "Upload a **Baseline** and **Scenario** `EnergyAndPeak.pq` for each simulated year. "
        "You can have results for one year or many — each year's simulated savings anchors the "
        "timeseries at that point. The earliest year is used as the reference for per-building "
        "WTP scoring; adoption and emissions are projected forward analytically."
    )

    if st.session_state.get("_example_preloaded"):
        st.info(
            "**Example data pre-loaded** — `EnergyAndPeak.pq` from "
            "`data/inputs/example/Baseline/` and `data/inputs/example/Retrofit/` are "
            "ready to use. Jump straight to **5 · Run & Results** to see the full "
            "pipeline, or upload your own files below to replace them."
        )

    hdr_col1, hdr_col2, _ = st.columns([2, 1, 3])
    with hdr_col1:
        scenario_name = st.text_input(
            "Retrofit scenario name",
            value=st.session_state["scenario_name"],
            placeholder="e.g. ASHP_2030",
            key="upload_scenario_name",
            help="This label will identify your scenario throughout the analysis.",
        )
        st.session_state["scenario_name"] = scenario_name.strip() or "Retrofit"
    with hdr_col2:
        n_years = st.number_input(
            "Number of years", min_value=1, max_value=10,
            value=st.session_state["n_years"], step=1,
            key="upload_n_years",
        )
        st.session_state["n_years"] = int(n_years)

    # ── Year selectors in a compact grid (up to 5 per row) ────────────────────
    st.divider()
    existing_years = st.session_state.get("selected_years", [2025])
    selected_years: list[int] = []
    year_cols = st.columns(min(int(n_years), 5))
    for i in range(int(n_years)):
        default_yr = existing_years[i] if i < len(existing_years) else 2025 + i * 5
        with year_cols[i % 5]:
            yr = st.selectbox(
                f"Year {i + 1}", options=_YEARS_RANGE,
                index=_YEARS_RANGE.index(default_yr) if default_yr in _YEARS_RANGE else i + 1,
                key=f"upload_yr_{i}",
            )
            selected_years.append(int(yr))

    if len(selected_years) != len(set(selected_years)):
        st.warning("Duplicate years — each year should appear only once.")
    st.session_state["selected_years"] = selected_years

    # ── Input mode toggle ─────────────────────────────────────────────────────
    st.divider()
    _um = st.session_state.get("upload_mode")
    if _um in ("Browse directory", "Browse outputs folder"):
        st.session_state["upload_mode"] = "Browse data inputs"
    if _um == "Upload files":
        st.session_state["upload_mode"] = "Upload files (per year)"
    upload_mode = st.radio(
        "File input method",
        [
            "Upload files (per year)",
            "Upload many, pick per year",
            "Browse data inputs",
        ],
        horizontal=True,
        key="upload_mode",
        help=(
            "**Per year** — one baseline + one scenario uploader per simulated year. "
            "**Upload many** — add many `.pq` / `.parquet` files at once (select all in a folder "
            "or drag multiple files; whole-folder drag depends on the browser), then choose each "
            "file from a dropdown per year. **Browse data inputs** — files already under "
            "`data/inputs/` (e.g. Docker mount of `./data`)."
        ),
    )

    year_files: dict = st.session_state.get("year_files", {})

    # ── browse data/inputs/ (host ./data mounted at /code/data in docker) ────
    if upload_mode == "Browse data inputs":
        dir_p = _DATA_DIR.resolve()
        st.caption(
            f"Parquet files under **`data/inputs/`** — `{dir_p}`. "
            "Subfolders are included."
        )
        pq_files = sorted(
            list(dir_p.glob("**/*.pq")) + list(dir_p.glob("**/*.parquet"))
        )
        if not pq_files:
            st.warning(
                "No `.pq` or `.parquet` files in `data/inputs/`. "
                "Add files there or use **Upload files**."
            )
        else:
            rel_paths = [str(f.relative_to(dir_p)) for f in pq_files]
            file_options = ["— select —"] + rel_paths
            st.caption(f"Found **{len(pq_files)}** parquet file(s).")
            st.divider()

            for yr in selected_years:
                with st.expander(f"Year {yr}", expanded=True):
                    col_b, col_s = st.columns(2)
                    entry = year_files.get(yr, {})

                    with col_b:
                        st.markdown("**Baseline**")
                        base_sel = st.selectbox(
                            f"Baseline {yr}", options=file_options,
                            key=f"dir_base_{yr}", label_visibility="collapsed",
                        )
                        if base_sel != "— select —":
                            data = (dir_p / base_sel).read_bytes()
                            entry = {**entry, "baseline": data}
                            st.session_state["_example_preloaded"] = False
                            info = _preview_parquet(data)
                            if "error" in info:
                                st.error(info["error"])
                            else:
                                _show_file_kpis(info)
                        elif entry.get("baseline") is not None:
                            st.caption("✓ Previously selected")

                    with col_s:
                        st.markdown(f"**{st.session_state['scenario_name']}**")
                        scen_sel = st.selectbox(
                            f"Scenario {yr}", options=file_options,
                            key=f"dir_scen_{yr}", label_visibility="collapsed",
                        )
                        if scen_sel != "— select —":
                            data = (dir_p / scen_sel).read_bytes()
                            entry = {**entry, "scenario": data}
                            st.session_state["_example_preloaded"] = False
                            info = _preview_parquet(data)
                            if "error" in info:
                                st.error(info["error"])
                            else:
                                _show_file_kpis(info)
                        elif entry.get("scenario") is not None:
                            st.caption("✓ Previously selected")

                    year_files[yr] = entry

    # ── batch upload: many files, assign baseline/scenario per year from pool ─
    elif upload_mode == "Upload many, pick per year":
        st.markdown(
            "Drop **multiple** parquet files or use the file picker and choose **many** "
            "(e.g. open your folder and select all). Each filename should be unique. "
            "Then pick baseline and scenario files for each year below, or use **Auto-assign "
            "from filenames** when names include the year and `baseline` / `retrofit` / `deep` / "
            "`scenario`."
        )
        batch_files = st.file_uploader(
            "Parquet files (multiple)",
            type=["pq", "parquet"],
            accept_multiple_files=True,
            key="batch_pq_upload",
            label_visibility="collapsed",
        )
        _sync_uploaded_pq_library_from_files(batch_files)
        lib: dict[str, bytes] = st.session_state.get("uploaded_pq_library") or {}
        if st.session_state.get("_batch_upload_duplicate_names"):
            st.warning(
                "Duplicate filenames in this batch — only the last of each name is kept. "
                "Rename files so each is unique before uploading."
            )
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("Clear file pool", key="batch_clear_pool"):
                st.session_state["uploaded_pq_library"] = {}
                st.session_state.pop("_batch_upload_sig", None)
                st.session_state.pop("_batch_upload_duplicate_names", None)
                st.rerun()
        with c2:
            if st.button("Auto-assign from filenames", key="batch_auto_assign"):
                merged = _auto_assign_library_to_years(
                    selected_years, lib, st.session_state.get("year_files", {})
                )
                st.session_state["year_files"] = merged
                st.session_state["_example_preloaded"] = False
                for _yr in selected_years:
                    ey = merged.get(_yr, {})
                    st.session_state[f"batch_base_{_yr}"] = ey.get("baseline_label") or "— select —"
                    st.session_state[f"batch_scen_{_yr}"] = ey.get("scenario_label") or "— select —"
                st.rerun()
        with c3:
            st.caption(
                f"**{len(lib)}** file(s) in pool — used for dropdowns below. "
                "Clearing the uploader does not empty the pool; use **Clear file pool**."
            )

        file_options = ["— select —"] + sorted(lib.keys())
        for _yr in selected_years:
            _bk, _sk = f"batch_base_{_yr}", f"batch_scen_{_yr}"
            if _bk in st.session_state and st.session_state[_bk] not in file_options:
                st.session_state[_bk] = "— select —"
            if _sk in st.session_state and st.session_state[_sk] not in file_options:
                st.session_state[_sk] = "— select —"
        if not lib:
            st.info("Upload at least one `.pq` or `.parquet` file to populate the pool.")
        for yr in selected_years:
            with st.expander(f"Year {yr}", expanded=True):
                col_b, col_s = st.columns(2)
                entry = dict(year_files.get(yr, {}))
                if (obl := entry.get("baseline_label")) and obl not in lib:
                    entry.pop("baseline", None)
                    entry.pop("baseline_label", None)
                if (osl := entry.get("scenario_label")) and osl not in lib:
                    entry.pop("scenario", None)
                    entry.pop("scenario_label", None)

                with col_b:
                    st.markdown("**Baseline**")
                    base_sel = st.selectbox(
                        f"Baseline {yr}", options=file_options,
                        key=f"batch_base_{yr}", label_visibility="collapsed",
                    )
                    if base_sel == "— select —":
                        entry.pop("baseline", None)
                        entry.pop("baseline_label", None)
                    elif base_sel in lib:
                        data = lib[base_sel]
                        entry["baseline"] = data
                        entry["baseline_label"] = base_sel
                        st.session_state["_example_preloaded"] = False
                        info = _preview_parquet(data)
                        if "error" in info:
                            st.error(info["error"])
                        else:
                            _show_file_kpis(info)

                with col_s:
                    st.markdown(f"**{st.session_state['scenario_name']}**")
                    scen_sel = st.selectbox(
                        f"Scenario {yr}", options=file_options,
                        key=f"batch_scen_{yr}", label_visibility="collapsed",
                    )
                    if scen_sel == "— select —":
                        entry.pop("scenario", None)
                        entry.pop("scenario_label", None)
                    elif scen_sel in lib:
                        data = lib[scen_sel]
                        entry["scenario"] = data
                        entry["scenario_label"] = scen_sel
                        st.session_state["_example_preloaded"] = False
                        info = _preview_parquet(data)
                        if "error" in info:
                            st.error(info["error"])
                        else:
                            _show_file_kpis(info)

                year_files[yr] = entry

    # ── Upload files mode (original) ──────────────────────────────────────────
    else:
        for yr in selected_years:
            with st.expander(f"Year {yr}", expanded=True):
                col_b, col_s = st.columns(2)
                entry = year_files.get(yr, {})

                with col_b:
                    st.markdown("**Baseline**")
                    base_file = st.file_uploader(
                        f"Baseline {yr}", type=["pq", "parquet"],
                        key=f"base_file_{yr}", label_visibility="collapsed",
                    )
                    if base_file is not None:
                        data = base_file.read()
                        entry = {**entry, "baseline": data}
                        st.session_state["_example_preloaded"] = False
                        info = _preview_parquet(data)
                        if "error" in info:
                            st.error(info["error"])
                        else:
                            _show_file_kpis(info)
                    elif entry.get("baseline") is not None:
                        st.caption("✓ Previously uploaded")

                with col_s:
                    st.markdown(f"**{st.session_state['scenario_name']}**")
                    scen_file = st.file_uploader(
                        f"Scenario {yr}", type=["pq", "parquet"],
                        key=f"scen_file_{yr}", label_visibility="collapsed",
                    )
                    if scen_file is not None:
                        data = scen_file.read()
                        entry = {**entry, "scenario": data}
                        st.session_state["_example_preloaded"] = False
                        info = _preview_parquet(data)
                        if "error" in info:
                            st.error(info["error"])
                        else:
                            _show_file_kpis(info)
                    elif entry.get("scenario") is not None:
                        st.caption("✓ Previously uploaded")

                year_files[yr] = entry

    st.session_state["year_files"] = year_files

    # ── Validation summary ─────────────────────────────────────────────────────
    st.divider()
    complete = [y for y in selected_years if year_files.get(y, {}).get("baseline") and year_files.get(y, {}).get("scenario")]
    if complete:
        st.success(
            f"✓ Both files uploaded for {len(complete)} year(s): "
            + ", ".join(str(y) for y in sorted(complete))
            + (" — proceed to Configure." if len(complete) == len(selected_years) else "")
        )
    else:
        st.info("Upload Baseline and Scenario files for at least one year to continue.")


def _show_file_kpis(info: dict) -> None:
    k1, k2, k3 = st.columns(3)
    area = info.get("total_area_m2")
    scens = info.get("scenarios", [])
    with k1:
        st.metric("Buildings", info.get("n_buildings", "—"))
    with k2:
        st.metric("Total area", f"{area:,.0f} m²" if area else "—")
    with k3:
        st.metric("Scenario(s) in file", ", ".join(scens) if scens else "—")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def _render_config_tab() -> None:
    st.markdown("## Step 2 — Configure Retrofit & WTP Model")

    # ── Retrofit cost ──────────────────────────────────────────────────────────
    st.markdown("### Retrofit Cost")
    cost_col1, cost_col2 = st.columns(2)
    with cost_col1:
        cost_per_sqm = st.number_input(
            "Retrofit cost ($/m² conditioned area)",
            min_value=0.0, value=float(st.session_state.get("cost_per_sqm", 150.0)),
            step=5.0, format="%.2f",
            key="cfg_cost_sqm",
            help="Gross capital cost before any incentives.",
        )
        st.session_state["cost_per_sqm"] = cost_per_sqm
    with cost_col2:
        st.markdown("")
        st.info(
            "Cost is applied uniformly across all buildings. To model variable "
            "costs, update the cost/m² and re-run for different scenarios."
        )

    st.divider()

    # ── Incentives ─────────────────────────────────────────────────────────────
    st.markdown("### Incentives")
    incentives_enabled = st.toggle(
        "Enable income-based incentives",
        value=st.session_state.get("incentives_enabled", False),
        key="cfg_incentives_enabled",
        help="When enabled, the pipeline runs twice — once without and once with the incentive applied — and results are shown side by side.",
    )
    st.session_state["incentives_enabled"] = incentives_enabled

    if incentives_enabled:
        st.caption(
            "Set a flat incentive amount (USD) per income tier. Each MC sample draws a household income "
            "from the census distribution; the matching tier's incentive is subtracted from the retrofit "
            "cost for that sample only. The net cost is clamped at $0."
        )
        existing: IncentiveConfig = st.session_state.get("incentive_config", IncentiveConfig())
        _TIER_FIELDS = [
            ("< $30k  (Very Low Income)", "very_low_income"),
            ("$30k–$50k  (Low Income)", "low_income"),
            ("$50k–$80k  (Moderate)", "moderate_income"),
            ("$80k–$150k  (Middle)", "middle_income"),
            ("> $150k  (Higher Income)", "higher_income"),
        ]
        tier_cols = st.columns(len(_TIER_FIELDS))
        new_values: dict[str, float] = {}
        for col, (label, field) in zip(tier_cols, _TIER_FIELDS):
            with col:
                new_values[field] = st.number_input(
                    label,
                    min_value=0.0,
                    value=float(getattr(existing, field)),
                    step=500.0, format="%.0f",
                    key=f"inc_tier_{field}",
                )
        st.session_state["incentive_config"] = IncentiveConfig(**new_values)

    st.divider()

    st.markdown("### Energy Prices")
    st.caption("Used to convert kWh to annual energy cost (USD) for the WTP model.")
    prices: dict[str, float] = st.session_state.get("energy_prices", dict(DEFAULT_ENERGY_PRICES))
    price_cols = st.columns(4)
    new_prices: dict[str, float] = {}
    default_units = {
        "Electricity": "$/kWh",
        "Natural Gas": "$/kWh equiv.",
        "Fuel Oil": "$/kWh equiv.",
        "Propane": "$/kWh equiv.",
    }
    for idx, fuel in enumerate(FUEL_LABELS):
        with price_cols[idx]:
            new_prices[fuel] = st.number_input(
                f"{fuel} ({default_units[fuel]})",
                min_value=0.0,
                value=float(prices.get(fuel, DEFAULT_ENERGY_PRICES[fuel])),
                step=0.001, format="%.4f",
                key=f"cfg_price_{fuel}",
            )
    st.session_state["energy_prices"] = new_prices

    st.divider()

    # ── Geographic context ─────────────────────────────────────────────────────
    st.markdown("### Geographic Context & Demographics")
    is_us = st.toggle(
        "Buildings are in the United States",
        value=st.session_state.get("is_us", True),
        key="cfg_is_us",
    )
    st.session_state["is_us"] = is_us

    if is_us:
        st.markdown(
            "Without a census file, the app loads **bundled** ACS tracts from "
            "`data/inputs/census_tract_acs/states/{state_fips}.parquet` for each state that "
            "appears after geocoding (MA, AZ, and WA are shipped). Other states use the Census "
            "API once per tract; geocoder results are still saved under `outputs/census_cache/`. "
            "Upload a **tract-level** "
            "census file — CSV or Parquet — with income, education, and household columns per tract "
            "(same schema as the engine's ``census_data_path`` input) "
            "to skip those calls entirely and use precomputed distributions."
        )
        census_file = st.file_uploader(
            "Census tract file (optional — skips geocoder + ACS when set)",
            type=["csv", "parquet", "pq"], key="cfg_census_csv",
        )
        if census_file is not None:
            st.session_state["census_csv_bytes"] = census_file.read()
            fn = (census_file.name or "").lower()
            if fn.endswith(".parquet"):
                st.session_state["census_upload_suffix"] = ".parquet"
            elif fn.endswith(".pq"):
                st.session_state["census_upload_suffix"] = ".pq"
            else:
                st.session_state["census_upload_suffix"] = ".csv"
            st.success("Census file uploaded.")
        else:
            st.session_state.setdefault("census_csv_bytes", None)
            st.session_state.setdefault("census_upload_suffix", ".csv")

        api_key = st.text_input(
            "Census API key (optional)",
            value=st.session_state.get("census_api_key", ""),
            type="password", key="cfg_census_api_key",
            help="Free key from api.census.gov/data/key_signup.html — improves reliability.",
        )
        st.session_state["census_api_key"] = api_key
    else:
        st.markdown("Enter approximate income and education distributions as priors.")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            inc_mean = st.slider(
                "Mean household income (k USD)", 10, 300,
                int(st.session_state.get("prior_income_mean", 60)), 5,
                key="cfg_inc_mean",
            )
            inc_std = st.slider(
                "Std dev household income (k USD)", 5, 150,
                int(st.session_state.get("prior_income_std", 30)), 5,
                key="cfg_inc_std",
            )
        with p_col2:
            pct_hs = st.slider("% High school / some college", 0, 100,
                               int(st.session_state.get("prior_pct_hs", 35)), key="cfg_pct_hs")
            pct_assoc = st.slider("% Associates degree", 0, 100,
                                  int(st.session_state.get("prior_pct_assoc", 15)), key="cfg_pct_assoc")
            pct_bach = st.slider("% Bachelors degree", 0, 100,
                                 int(st.session_state.get("prior_pct_bach", 35)), key="cfg_pct_bach")
            pct_grad = st.slider("% Graduate degree", 0, 100,
                                 int(st.session_state.get("prior_pct_grad", 15)), key="cfg_pct_grad")
        for k, v in [("prior_income_mean", inc_mean), ("prior_income_std", inc_std),
                     ("prior_pct_hs", pct_hs), ("prior_pct_assoc", pct_assoc),
                     ("prior_pct_bach", pct_bach), ("prior_pct_grad", pct_grad)]:
            st.session_state[k] = v

        # Sanity-check percentages
        total_pct = pct_hs + pct_assoc + pct_bach + pct_grad
        if abs(total_pct - 100) > 5:
            st.warning(f"Education percentages sum to {total_pct}% — they will be normalised.")

    st.divider()

    # ── Monte Carlo settings ───────────────────────────────────────────────────
    st.markdown("### Monte Carlo / Ensemble Settings")
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    with mc_col1:
        n_mc = st.number_input(
            "MC samples per building",
            min_value=10, max_value=1000,
            value=int(st.session_state.get("n_mc_samples", 100)),
            step=10, key="cfg_n_mc",
            help="Number of demographic draws per building for the propensity model.",
        )
    with mc_col2:
        n_ensemble = st.number_input(
            "Ensemble runs (uptake MC)",
            min_value=10, max_value=500,
            value=int(st.session_state.get("n_ensemble_runs", 100)),
            step=10, key="cfg_n_ensemble",
            help="Number of dice-roll runs for the mc_ensemble adoption method.",
        )
    with mc_col3:
        seed = st.number_input(
            "Random seed",
            min_value=0, max_value=9999,
            value=int(st.session_state.get("random_seed", 42)),
            step=1, key="cfg_seed",
        )
    st.caption("Dice roll re-entry (after a year with no adoption for a building):")
    dr_col1, dr_col2 = st.columns(2)
    with dr_col1:
        dice_no_reentry = st.checkbox(
            "No re-entry (without replacement on failure)",
            value=bool(st.session_state.get("dice_reentry_never", True)),
            key="cfg_dice_reentry_never",
            help="If a building could adopt but does not, it never enters the draw again.",
        )
    with dr_col2:
        dice_reentry_n = st.number_input(
            "Years until re-eligible",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("dice_reentry_years", 1)),
            step=1,
            key="cfg_dice_reentry_years",
            disabled=dice_no_reentry,
            help="Default 1 = same as legacy (can try again next year).",
        )
    for k, v in [("n_mc_samples", n_mc), ("n_ensemble_runs", n_ensemble), ("random_seed", seed)]:
        st.session_state[k] = int(v)
    st.session_state["dice_reentry_never"] = dice_no_reentry
    st.session_state["dice_reentry_years"] = int(dice_reentry_n)

    st.divider()

    # ── Census tract confirmation ──────────────────────────────────────────────
    st.markdown("### Census Tract Confirmation")
    if not is_us:
        st.info("Census tract lookup is only available for US buildings.")
    else:
        _render_census_tract_lookup()


def _render_census_tract_lookup() -> None:
    year_files = st.session_state.get("year_files", {})
    selected_years = st.session_state.get("selected_years", [])
    ref_years = [y for y in sorted(selected_years) if year_files.get(y, {}).get("baseline")]

    if not ref_years:
        st.info("Upload a baseline file in Step 1 to look up the census tract.")
        return

    ref_yr = ref_years[0]
    if st.button("Look up census tract from building coordinates", key="cfg_lookup_tract"):
        with st.spinner("Calling Census Geocoder API…"):
            try:
                from app.analysis.census_lookup import geocode_point
                import geopandas as gpd
                from shapely import wkt as shapely_wkt

                base_flat = load_energy_parquet(year_files[ref_yr]["baseline"])

                # Try rotated_rectangle WKT (EPSG:3857) first
                rect_col = next(
                    (c for c in ["rotated_rectangle", "GLOBI_ROTATED_RECTANGLE",
                                 "feature.geometry.rotated_rectangle"]
                     if c in base_flat.columns),
                    None,
                )

                building_lats: list[float] = []
                building_lons: list[float] = []

                if rect_col is not None:
                    import base64
                    from shapely import wkb as shapely_wkb

                    def _load_geometry(s: str):
                        """Parse WKT string or base64-encoded WKB (P3 format)."""
                        try:
                            return shapely_wkt.loads(s)
                        except Exception:
                            pass
                        try:
                            return shapely_wkb.loads(base64.b64decode(s))
                        except Exception:
                            return None

                    wkt_series = base_flat[rect_col].dropna().astype(str)
                    geoms = wkt_series.map(_load_geometry).dropna()
                    gs = gpd.GeoSeries(geoms, crs="EPSG:3857").to_crs("EPSG:4326")
                    centroids = gs.centroid
                    building_lons = centroids.x.tolist()
                    building_lats = centroids.y.tolist()
                else:
                    # Fall back to direct lat/lon columns
                    lat_col = next(
                        (c for c in ["lat", "feature.location.lat"] if c in base_flat.columns),
                        None,
                    )
                    lon_col = next(
                        (c for c in ["lon", "feature.location.lon"] if c in base_flat.columns),
                        None,
                    )
                    if lat_col and lon_col:
                        building_lats = pd.to_numeric(base_flat[lat_col], errors="coerce").dropna().tolist()
                        building_lons = pd.to_numeric(base_flat[lon_col], errors="coerce").dropna().tolist()

                if building_lats and building_lons:
                    centroid_lat = float(sum(building_lats) / len(building_lats))
                    centroid_lon = float(sum(building_lons) / len(building_lons))
                    tract_info = geocode_point(centroid_lat, centroid_lon)
                    st.session_state["census_tract_info"] = {
                        "tract": tract_info,
                        "centroid_lat": centroid_lat,
                        "centroid_lon": centroid_lon,
                        "building_lats": building_lats,
                        "building_lons": building_lons,
                        "n_buildings": len(base_flat),
                    }
                else:
                    st.warning(
                        "No rotated_rectangle or lat/lon columns found in the uploaded baseline file."
                    )
            except Exception as exc:
                st.error(f"Census tract lookup failed: {exc}")

    tract_data = st.session_state.get("census_tract_info")
    if tract_data and tract_data.get("tract"):
        t = tract_data["tract"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Census Tract GEOID", t.geoid or "—")
        col2.metric("State FIPS", t.state or "—")
        col3.metric("County FIPS", t.county or "—")
        st.caption(
            f"Centroid of {tract_data['n_buildings']} buildings: "
            f"({tract_data['centroid_lat']:.4f}°N, {tract_data['centroid_lon']:.4f}°W)"
        )
        m = folium.Map(
            location=[tract_data["centroid_lat"], tract_data["centroid_lon"]],
            zoom_start=12,
        )
        # Plot individual building centroids when available
        bldg_lats = tract_data.get("building_lats", [])
        bldg_lons = tract_data.get("building_lons", [])
        _MAX_MARKERS = 500
        for blat, blon in zip(bldg_lats[:_MAX_MARKERS], bldg_lons[:_MAX_MARKERS]):
            folium.CircleMarker(
                [blat, blon],
                radius=4,
                color="#2563EB",
                fill=True,
                fill_opacity=0.6,
                tooltip="Building",
            ).add_to(m)
        # Always add a distinct centroid marker
        folium.Marker(
            [tract_data["centroid_lat"], tract_data["centroid_lon"]],
            tooltip=f"Centroid — Tract {t.geoid}",
            icon=folium.Icon(color="red", icon="home"),
        ).add_to(m)
        st_folium(m, height=350, use_container_width=True)
    elif tract_data and tract_data.get("tract") is None:
        st.warning("Census Geocoder returned no result for the building centroid coordinates.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADOPTION CURVES
# ══════════════════════════════════════════════════════════════════════════════

# ── Reserved (locked) adoption scenarios ──────────────────────────────────────
# These are always present and cannot be edited or removed.
_RESERVED_SCENARIOS: dict[str, dict] = {
    "no_adoption": {
        "description": "No adoption — 0% throughout (baseline reference)",
        "curve_type": "flat",
        "max_adoption": 0.0,
        "annual_attrition": 0.0,
        "locked": True,
        "values": {str(y): 0.0 for y in range(2024, 2101)},
    },
    "full_adoption": {
        "description": (
            "Theoretical max — 100% of buildings retrofitted in the first projection year; "
            "no propensity / uptake MC (ceiling for comparison only)"
        ),
        "curve_type": "flat",
        "max_adoption": 1.0,
        "annual_attrition": 0.0,
        "locked": True,
        "values": {str(y): 1.0 for y in range(2024, 2101)},
    },
}


def _ensure_reserved_scenarios(raw: dict) -> tuple[dict, bool]:
    """Inject no_adoption / full_adoption if missing. Returns (raw, was_changed)."""
    changed = False
    scenarios = raw.setdefault("scenarios", {})
    for name, template in _RESERVED_SCENARIOS.items():
        if name not in scenarios:
            scenarios[name] = copy.deepcopy(template)
            changed = True
    return raw, changed


def _ac_compute_curve(
    shape: str,
    start: float,
    end: float,
    n: int,
    sigmoid_midpoint: float = 0.5,
    sigmoid_steepness: float = 10.0,
) -> np.ndarray:
    """Compute n adoption-rate values from start→end using the chosen shape.

    sigmoid_midpoint  — 0-to-1 fraction of the timeline where the curve inflects.
    sigmoid_steepness — controls how sharply the curve transitions (higher = steeper).
    """
    xs = np.linspace(0, 1, n)
    if shape == "Linear":
        return start + (end - start) * xs
    if shape == "Sigmoid":
        sig = 1 / (1 + np.exp(-sigmoid_steepness * (xs - sigmoid_midpoint)))
        sig = (sig - sig[0]) / (sig[-1] - sig[0] + 1e-9)
        return start + (end - start) * sig
    # Exponential
    exp_raw = np.expm1(xs * 3)
    exp_raw = exp_raw / (exp_raw[-1] + 1e-9)
    return start + (end - start) * exp_raw


def _ac_scenario_editor(
    prefix: str,
    curve_years: list[int],
    defaults: dict,
) -> dict:
    """Render the parameter widgets for one adoption scenario.

    prefix   — unique key prefix to avoid widget-key collisions.
    defaults — existing scenario dict used to pre-fill values.
    Returns a dict of the current widget values (not yet saved).
    """
    shape_opts = ["Linear", "Sigmoid", "Exponential"]
    saved_shape = defaults.get("curve_type", "linear").capitalize()

    col1, col2, col3 = st.columns(3)
    with col1:
        shape = st.selectbox(
            "Curve shape", shape_opts,
            index=shape_opts.index(saved_shape) if saved_shape in shape_opts else 0,
            key=f"{prefix}_shape",
            help=(
                "**Linear** — adoption grows at a steady, constant rate each year.\n\n"
                "**Sigmoid** — slow start, rapid acceleration in the middle years, then "
                "flattening as saturation approaches. Good for most realistic programs.\n\n"
                "**Exponential** — slow start that accelerates throughout, never plateauing "
                "until it hits the max cap. Best for disruptive/rapid-uptake scenarios."
            ),
        )
    with col2:
        start_val = st.number_input(
            "Start adoption (2024)", min_value=0.0, max_value=1.0,
            value=float(defaults.get("values", {}).get("2024", 0.0)),
            step=0.01, format="%.2f", key=f"{prefix}_start",
            help="Fraction of buildings adopted at the beginning of the projection (0 = none, 1 = all).",
        )
    with col3:
        saved_end = float(
            defaults.get("values", {}).get("2100")
            or defaults.get("values", {}).get("2050")
            or defaults.get("max_adoption", 0.85)
        )
        end_val = st.number_input(
            "End adoption (2100)", min_value=0.0, max_value=1.0,
            value=saved_end,
            step=0.01, format="%.2f", key=f"{prefix}_end",
            help="Target fraction adopted by the end of the projection horizon.",
        )

    # Sigmoid-specific controls
    sigmoid_mid = defaults.get("sigmoid_midpoint", 0.5)
    sigmoid_steep = defaults.get("sigmoid_steepness", 10.0)
    if shape == "Sigmoid":
        s_col1, s_col2 = st.columns(2)
        proj_span = curve_years[-1] - curve_years[0]
        saved_infl_year = int(curve_years[0] + sigmoid_mid * proj_span)
        with s_col1:
            infl_year = st.slider(
                "Inflection year",
                min_value=curve_years[0], max_value=curve_years[-1],
                value=saved_infl_year,
                key=f"{prefix}_infl",
                help=(
                    "The year when adoption growth is fastest. "
                    "Before this year growth is slow; after it, growth decelerates toward the cap."
                ),
            )
            sigmoid_mid = (infl_year - curve_years[0]) / proj_span
        with s_col2:
            sigmoid_steep = st.slider(
                "Steepness", min_value=2.0, max_value=20.0,
                value=float(sigmoid_steep),
                step=0.5,
                key=f"{prefix}_steep",
                help=(
                    "How sharply the curve transitions around the inflection year. "
                    "Low (~2–4): gradual S. High (~15–20): near step-change."
                ),
            )

    st.divider()
    cap_col, attr_col, desc_col = st.columns(3)
    with cap_col:
        max_adoption = st.number_input(
            "Maximum adoption cap", min_value=0.0, max_value=1.0,
            value=float(defaults.get("max_adoption", 0.85)),
            step=0.01, format="%.2f", key=f"{prefix}_max",
            help=(
                "Hard ceiling on adoption regardless of curve shape. "
                "Set to 1.0 to allow 100% adoption, or lower to reflect "
                "structural limits (e.g. renters, exempt buildings)."
            ),
        )
    with attr_col:
        annual_attrition = st.number_input(
            "Last-minute dropout rate", min_value=0.0, max_value=1.0,
            value=float(defaults.get("annual_attrition", 0.02)),
            step=0.005, format="%.3f", key=f"{prefix}_attr",
            help=(
                "A noise term on each year's adoption. Each year, a random fraction "
                "between **0 and this value** of buildings that were about to adopt "
                "back out at the last minute — for example due to financing falling "
                "through or late-stage cold feet. **Once a building adopts it stays "
                "adopted** (this is not a reversion rate). "
                "Set to **0.00** for a noise-free deterministic curve; "
                "**0.05** adds up to 5 % annual last-minute dropout uncertainty, "
                "which widens the P10–P90 band in the MC ensemble."
            ),
        )
    with desc_col:
        description = st.text_input(
            "Description (optional)", value=defaults.get("description", ""),
            key=f"{prefix}_desc",
        )

    return {
        "shape": shape,
        "start_val": start_val,
        "end_val": end_val,
        "sigmoid_midpoint": sigmoid_mid,
        "sigmoid_steepness": sigmoid_steep,
        "max_adoption": max_adoption,
        "annual_attrition": annual_attrition,
        "description": description,
    }


def _ac_build_scenario_dict(params: dict, curve_years: list[int]) -> dict:
    """Build a scenario data dict from editor widget output."""
    preview_ys = np.clip(
        _ac_compute_curve(
            params["shape"],
            params["start_val"],
            params["end_val"],
            len(curve_years),
            sigmoid_midpoint=params["sigmoid_midpoint"],
            sigmoid_steepness=params["sigmoid_steepness"],
        ),
        0.0,
        params["max_adoption"],
    )
    return {
        "description": params["description"],
        "curve_type": params["shape"].lower(),
        "max_adoption": params["max_adoption"],
        "annual_attrition": params["annual_attrition"],
        "sigmoid_midpoint": params["sigmoid_midpoint"],
        "sigmoid_steepness": params["sigmoid_steepness"],
        "values": {str(y): round(float(v), 4) for y, v in zip(curve_years, preview_ys)},
    }


def _render_adoption_tab() -> None:
    st.markdown("## Step 3 — Adoption Curves")
    st.caption(
        "Define how retrofit adoption evolves over time. "
        "**No adoption** and **Full adoption** are always included as reference bounds. "
        "Add your own scenarios below."
    )

    raw = _load_adoption_curves()
    # Ensure reserved scenarios are present; save if they were missing
    raw, reserved_added = _ensure_reserved_scenarios(raw)
    if reserved_added:
        _save_adoption_curves(raw)

    scenarios: dict = raw.get("scenarios", {})
    curve_years = list(range(2024, 2101))
    colors = _curve_colors()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Scenario list
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Scenarios")

    user_scenarios = {k: v for k, v in scenarios.items() if not v.get("locked")}
    locked_scenarios = {k: v for k, v in scenarios.items() if v.get("locked")}

    # Locked reference scenarios (no remove button, no editor)
    for sname, sdata in locked_scenarios.items():
        lk_col, desc_col = st.columns([2, 6])
        with lk_col:
            st.markdown(f"🔒 **{sname}**")
        with desc_col:
            st.caption(sdata.get("description", ""))

    if user_scenarios:
        st.markdown("")  # spacing
    for sname in list(user_scenarios.keys()):
        row_col, btn_col = st.columns([7, 1], vertical_alignment="center")
        with row_col:
            desc = scenarios[sname].get("description", "")
            st.markdown(f"**{sname}**" + (f" — _{desc}_" if desc else ""))
        with btn_col:
            if st.button("Remove", key=f"rm_{sname}"):
                del raw["scenarios"][sname]
                _save_adoption_curves(raw)
                if st.session_state.get("adoption_scenario") == sname:
                    remaining = [k for k in raw.get("scenarios", {}) if not raw["scenarios"][k].get("locked")]
                    st.session_state["adoption_scenario"] = remaining[0] if remaining else None
                st.rerun()

    if not user_scenarios:
        st.info("No custom scenarios yet — create one below.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Edit existing scenario
    # ══════════════════════════════════════════════════════════════════════════
    if user_scenarios:
        st.markdown("### Edit Scenario")
        st.caption("Adjust the curve shape — the preview updates live. Click **Save** to persist.")

        scenario_names = list(user_scenarios.keys())
        default_sel = st.session_state.get("adoption_scenario")
        sel_idx = scenario_names.index(default_sel) if default_sel in scenario_names else 0
        selected = st.selectbox(
            "Scenario to edit", options=scenario_names,
            index=sel_idx, key="ac_selected",
        )
        st.session_state["adoption_scenario"] = selected

        scen_data = copy.deepcopy(scenarios[selected])
        params = _ac_scenario_editor("edit", curve_years, scen_data)
        updated = _ac_build_scenario_dict(params, curve_years)

        # Live preview chart (all saved curves as context + current preview)
        fig = go.Figure()
        for i, (name, s) in enumerate(scenarios.items()):
            if name == selected:
                continue
            saved_ys = [float(s.get("values", {}).get(str(y), 0.0)) for y in curve_years]
            is_locked = s.get("locked", False)
            fig.add_trace(go.Scatter(
                x=curve_years, y=[v * 100 for v in saved_ys],
                name=name,
                line=dict(
                    color=colors[i % len(colors)],
                    width=1.5,
                    dash="dot" if is_locked else "dash",
                ),
                opacity=0.35 if is_locked else 0.55,
            ))
        preview_ys = [float(v) for v in updated["values"].values()]
        fig.add_trace(go.Scatter(
            x=curve_years, y=[v * 100 for v in preview_ys],
            name=f"{selected} (preview)",
            line=dict(color="#2563eb", width=3),
        ))
        fig.update_layout(
            yaxis=dict(title="Adoption (%)", range=[0, 105]),
            xaxis_title="Year", legend_title="Scenario",
            height=320, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="ac_edit_preview")

        save_col, reset_col, _ = st.columns([1, 1, 4], vertical_alignment="center")
        with save_col:
            if st.button("Save", type="primary", key="ac_save"):
                raw["scenarios"][selected] = updated
                _save_adoption_curves(raw)
                st.success(f"Saved '{selected}'.")
                st.rerun()
        with reset_col:
            if st.button("Reset to saved", key="ac_reset"):
                st.rerun()

        st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Create new scenario
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("＋ Add new scenario", expanded=not bool(user_scenarios)):
        st.caption(
            "Define the curve parameters below, then click **Create**. "
            "The scenario is saved immediately — no extra steps."
        )
        new_name = st.text_input(
            "Scenario name", placeholder="e.g. fast_electrification",
            key="ac_new_name",
            help="Use letters, numbers, and underscores. Must be unique.",
        )
        # Sensible defaults for a new scenario
        new_defaults = {
            "curve_type": "sigmoid", "max_adoption": 0.85,
            "annual_attrition": 0.02, "description": "",
            "sigmoid_midpoint": 0.4, "sigmoid_steepness": 10.0,
            "values": {},
        }
        new_params = _ac_scenario_editor("new", curve_years, new_defaults)
        new_scen = _ac_build_scenario_dict(new_params, curve_years)

        # Mini preview inside the expander
        new_ys = [float(v) for v in new_scen["values"].values()]
        fig_new = go.Figure(go.Scatter(
            x=curve_years, y=[v * 100 for v in new_ys],
            name="Preview", line=dict(color="#16a34a", width=2.5),
        ))
        fig_new.update_layout(
            yaxis=dict(title="Adoption (%)", range=[0, 105]),
            xaxis_title="Year", height=240, margin=dict(t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_new, use_container_width=True, key="ac_new_preview")

        if st.button("Create scenario", type="primary", key="ac_create_btn"):
            n = new_name.strip().replace(" ", "_")
            if not n:
                st.error("Enter a scenario name.")
            elif n in scenarios:
                st.error(f"'{n}' already exists — choose a different name.")
            else:
                raw["scenarios"][n] = new_scen
                _save_adoption_curves(raw)
                st.session_state["adoption_scenario"] = n
                st.success(f"Created '{n}'.")
                st.rerun()

    st.divider()
    _ac_render_projection_period()


def _ac_render_projection_period() -> None:
    st.markdown("### Projection Period")
    py_col1, py_col2 = st.columns(2)
    with py_col1:
        start_year = st.number_input(
            "Start year", 2024, 2099,
            int(st.session_state.get("proj_start_year", 2025)), 1,
            key="cfg_start_year",
        )
    with py_col2:
        end_year = st.number_input(
            "End year", 2025, 2100,
            int(st.session_state.get("proj_end_year", 2050)), 1,
            key="cfg_end_year",
        )
    if end_year <= start_year:
        st.error("End year must be after start year.")
    st.session_state["proj_start_year"] = int(start_year)
    st.session_state["proj_end_year"] = int(end_year)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EMISSIONS TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════════

def _render_emissions_tab() -> None:
    st.markdown("## Step 4 — Emissions Factor Trajectories")
    st.markdown(
        "Define how grid/fuel emissions intensity (kg CO2/kWh) evolves over time. "
        "Changes are saved to "
        f"`{_EMISSIONS_PATH.relative_to(_REPO_ROOT)}`."
    )

    emissions_data = _load_emissions_json()
    fuels_cfg: dict = emissions_data.get("fuels", {})
    plot_years = list(range(2024, 2101))

    # Per-fuel editors
    for fuel in FUEL_LABELS:
        with st.expander(f"{fuel}", expanded=False):
            fuel_cfg = fuels_cfg.get(fuel, {"values": {}, "unit": "kg CO2/kWh"})
            values = fuel_cfg.get("values", {})

            # Quick set: start + end values + shape
            qe_col1, qe_col2, qe_col3 = st.columns(3)
            current_start = float(values.get("2024", 0.3))
            current_end = float(values.get("2100", values.get("2050", 0.1)))
            with qe_col1:
                v_start = st.number_input(f"{fuel} — 2024 value (kg CO2/kWh)",
                                          0.0, 5.0, current_start, 0.01,
                                          key=f"em_{fuel}_start")
                v_end = st.number_input(f"{fuel} — 2100 value (kg CO2/kWh)",
                                        0.0, 5.0, current_end, 0.01,
                                        key=f"em_{fuel}_end")
            with qe_col2:
                shape = st.selectbox("Shape", ["Linear", "S-curve (electricity grid)"],
                                     key=f"em_{fuel}_shape")
            with qe_col3:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button(f"Apply — {fuel}", key=f"em_{fuel}_apply"):
                    xs = np.linspace(0, 1, len(plot_years))
                    if shape == "Linear":
                        ys = v_start + (v_end - v_start) * xs
                    else:
                        sig = 1 / (1 + np.exp(-8 * (xs - 0.5)))
                        sig = (sig - sig[0]) / (sig[-1] - sig[0] + 1e-9)
                        ys = v_start + (v_end - v_start) * sig
                    fuel_cfg["values"] = {str(y): round(float(v), 4) for y, v in zip(plot_years, ys)}
                    fuel_cfg["unit"] = "kg CO2/kWh"
                    fuels_cfg[fuel] = fuel_cfg
                    emissions_data["fuels"] = fuels_cfg
                    _save_emissions_json(emissions_data)
                    st.success(f"Saved {fuel} trajectory.")
                    st.rerun()

            # Mini chart
            if values:
                yvals = [float(values.get(str(y), np.nan)) for y in plot_years]
                fig = go.Figure(go.Scatter(x=plot_years, y=yvals, mode="lines",
                                           line=dict(color="#2563eb", width=2)))
                fig.update_layout(height=200, margin=dict(t=10, b=10),
                                  yaxis_title="kg CO2/kWh", xaxis_title="Year")
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RUN & RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def _render_results_tab() -> None:
    st.markdown("## Step 5 — Run Analysis & View Results")

    year_files: dict = st.session_state.get("year_files", {})
    selected_years: list[int] = st.session_state.get("selected_years", [])
    complete_years = sorted(
        y for y in selected_years
        if year_files.get(y, {}).get("baseline") and year_files.get(y, {}).get("scenario")
    )

    if not complete_years:
        st.warning("Upload Baseline and Scenario files for at least one year in Step 1 before running.")
        return

    raw_curves = _load_adoption_curves()
    n_adoption_scenarios = len(raw_curves.get("scenarios", {}))

    col_run, col_info = st.columns([1, 3], vertical_alignment="center")
    with col_run:
        run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=True)
    with col_info:
        st.caption(
            f"Simulated years: **{', '.join(str(y) for y in complete_years)}**  |  "
            f"Retrofit scenario: **{st.session_state['scenario_name']}**  |  "
            f"Adoption scenarios: **{n_adoption_scenarios}**  |  "
            f"MC ensemble runs: {st.session_state.get('n_ensemble_runs', 100)}"
        )

    if run_clicked:
        _run_pipeline(complete_years)

    if st.session_state.get("run_complete"):
        _render_result_charts()


def _run_pipeline(simulated_years: list[int]) -> None:
    progress = st.progress(0, text="Starting…")
    year_files: dict = st.session_state["year_files"]

    # ── 1. Build policy_impacts (ref year) and per-year energy data ───────────
    progress.progress(5, text=f"Computing energy savings across {len(simulated_years)} year(s)…")
    year_energy_data: dict = {}
    policy_impacts = None

    for yr_idx, yr in enumerate(simulated_years):
        try:
            entry = year_files[yr]
            base_flat = load_energy_parquet(entry["baseline"])
            scen_flat = load_energy_parquet(entry["scenario"])
            pi = build_policy_impacts(
                baseline_df=base_flat,
                scenario_df=scen_flat,
                scenario_name=st.session_state["scenario_name"],
                cost_per_sqm=float(st.session_state.get("cost_per_sqm", 150.0)),
                energy_prices=st.session_state.get("energy_prices"),
            )
            kwh_cols = [c for c in pi.columns if "_kwh_" in c]
            year_energy_data[yr] = pi.set_index("building.id")[kwh_cols]
            if yr_idx == 0:
                policy_impacts = pi
        except Exception as exc:
            st.error(f"Energy delta failed for {yr}: {exc}")
            if yr_idx == 0:
                return

    st.session_state["policy_impacts"] = policy_impacts
    st.session_state["year_energy_data"] = year_energy_data
    st.session_state["simulated_years"] = simulated_years
    progress.progress(25, text=f"Energy savings computed for {len(policy_impacts)} buildings across {len(year_energy_data)} year(s).")

    # ── 2. Census enrichment (US only) ───────────────────────────────────────
    if st.session_state.get("is_us", True) and st.session_state.get("census_csv_bytes") is None:
        if "lat" in policy_impacts.columns and "lon" in policy_impacts.columns:
            progress.progress(30, text="Looking up census demographics (may take a moment)…")
            from app.analysis.census_lookup import enrich_with_census
            policy_impacts = enrich_with_census(
                policy_impacts,
                api_key=st.session_state.get("census_api_key") or None,
                lat_col="lat", lon_col="lon",
                max_buildings=min(500, len(policy_impacts)),
            )
            st.session_state["policy_impacts"] = policy_impacts

    # ── 3. Propensity model (base — no incentive) ─────────────────────────────
    progress.progress(40, text="Running WTP propensity model (base)…")
    census_path = None
    try:
        if st.session_state.get("census_csv_bytes"):
            import tempfile, os
            suf = st.session_state.get("census_upload_suffix", ".csv")
            tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
            tmp.write(st.session_state["census_csv_bytes"])
            tmp.close()
            census_path = tmp.name

        propensity_engine = PropensityModelEngine(
            policy_impacts_df=policy_impacts,
            census_data_path=census_path,
            n_monte_carlo_samples=int(st.session_state.get("n_mc_samples", 100)),
            random_seed=int(st.session_state.get("random_seed", 42)),
        )
        if not st.session_state.get("is_us", True):
            _apply_non_us_priors(propensity_engine)
        else:
            from app.analysis.census_lookup import tract_distributions_from_enriched

            td = tract_distributions_from_enriched(policy_impacts, propensity_engine.county_fips_map)
            if td:
                propensity_engine.tract_distributions.update(td)
        propensity_result = propensity_engine.calculate_all_probabilities()
        st.session_state["propensity_result"] = propensity_result
    except Exception as exc:
        st.error(f"Propensity model failed: {exc}")
        return
    finally:
        if census_path:
            import os
            os.unlink(census_path)

    # Join scores into policy_impacts for propensity-ranked emissions
    policy_impacts = _join_propensity_scores(policy_impacts, propensity_result)
    st.session_state["policy_impacts"] = policy_impacts
    progress.progress(55, text="Base propensity complete.")

    # ── 3b. Propensity model (with incentive) — optional ──────────────────────
    incentives_enabled = st.session_state.get("incentives_enabled", False)
    propensity_result_inc = None
    policy_impacts_inc = None

    if incentives_enabled:
        progress.progress(56, text="Running propensity model with incentives…")
        try:
            incentive_map = _build_income_incentive_map()
            propensity_engine_inc = PropensityModelEngine(
                policy_impacts_df=st.session_state["policy_impacts"].drop(
                    columns=["acceptance_probability"], errors="ignore"
                ),
                census_data_path=None,
                n_monte_carlo_samples=int(st.session_state.get("n_mc_samples", 100)),
                random_seed=int(st.session_state.get("random_seed", 42)),
                incentive_by_income=incentive_map,
            )
            if not st.session_state.get("is_us", True):
                _apply_non_us_priors(propensity_engine_inc)
            # Re-use tract distributions from base engine to avoid redundant census calls
            propensity_engine_inc.tract_distributions = propensity_engine.tract_distributions
            propensity_engine_inc.county_concern = propensity_engine.county_concern
            propensity_result_inc = propensity_engine_inc.calculate_all_probabilities()
            policy_impacts_inc = _join_propensity_scores(
                st.session_state["policy_impacts"].drop(columns=["acceptance_probability"], errors="ignore"),
                propensity_result_inc,
            )
            st.session_state["propensity_result_incentive"] = propensity_result_inc
            st.session_state["policy_impacts_incentive"] = policy_impacts_inc
        except Exception as exc:
            st.warning(f"Incentive propensity model failed: {exc}")
            incentives_enabled = False

    # ── 4 & 5. Uptake + emissions ─────────────────────────────────────────────
    start_yr = int(st.session_state.get("proj_start_year", 2025))
    end_yr = int(st.session_state.get("proj_end_year", 2050))
    projection_years = list(range(start_yr, end_yr + 1))
    emissions_json = _load_emissions_json()
    raw_curves = _load_adoption_curves()
    all_adoption_scenarios = list(raw_curves.get("scenarios", {}).keys())
    if not all_adoption_scenarios:
        st.error("No adoption curve scenarios found. Define them in Step 3.")
        return

    progress.progress(60, text=f"Projecting {len(all_adoption_scenarios)} adoption scenario(s)…")
    scenario_results = _run_adoption_emissions_loop(
        propensity_result=propensity_result,
        policy_impacts=policy_impacts,
        all_adoption_scenarios=all_adoption_scenarios,
        start_yr=start_yr, end_yr=end_yr,
        projection_years=projection_years,
        emissions_json=emissions_json,
        progress_range=(60, 80),
        progress_bar=progress,
    )
    if not scenario_results:
        st.error("All adoption scenarios failed — check inputs.")
        return
    st.session_state["scenario_results"] = scenario_results

    if incentives_enabled and propensity_result_inc is not None:
        progress.progress(80, text="Projecting incentive adoption scenarios…")
        scenario_results_inc = _run_adoption_emissions_loop(
            propensity_result=propensity_result_inc,
            policy_impacts=policy_impacts_inc,
            all_adoption_scenarios=all_adoption_scenarios,
            start_yr=start_yr, end_yr=end_yr,
            projection_years=projection_years,
            emissions_json=emissions_json,
            progress_range=(80, 98),
            progress_bar=progress,
        )
        st.session_state["scenario_results_incentive"] = scenario_results_inc
    else:
        st.session_state["scenario_results_incentive"] = None

    progress.progress(100, text="Done.")
    st.session_state["run_complete"] = True
    n_inc = " + incentive variant" if incentives_enabled else ""
    st.success(f"Analysis complete — {len(scenario_results)} adoption scenario(s){n_inc} projected.")
    st.rerun()


def _join_propensity_scores(policy_impacts: pd.DataFrame, propensity_result) -> pd.DataFrame:
    """Merge acceptance_probability from propensity result into policy_impacts."""
    if "building.id" in propensity_result.data.columns and "building.id" in policy_impacts.columns:
        scores = (
            propensity_result.data[["building.id", "acceptance_probability"]]
            .drop_duplicates("building.id")
        )
        return policy_impacts.merge(scores, on="building.id", how="left")
    return policy_impacts


def _theoretical_full_uptake_result(
    propensity_df: pd.DataFrame,
    start_yr: int,
    end_yr: int,
) -> UptakeResult:
    """100% of buildings adopted from the first projection year — no propensity-weighted uptake.

    used only for the reserved scenario name full_adoption, not for user curves that hit 100%.
    """
    years = list(range(start_yr, end_yr + 1))
    rfs = (
        propensity_df["retrofit.scenario"].unique().tolist()
        if "retrofit.scenario" in propensity_df.columns
        else ["default"]
    )
    out_df = propensity_df.copy()
    out_df["passes_threshold"] = True

    yearly_rows: list[dict] = []
    for rf in rfs:
        sub = (
            out_df[out_df["retrofit.scenario"] == rf]
            if "retrofit.scenario" in out_df.columns
            else out_df
        )
        n_b = len(sub)
        prev = 0.0
        for y in years:
            cum = 100.0
            yearly_pct = max(0.0, cum - prev)
            yearly_rows.append({
                "year": y,
                "retrofit.scenario": rf,
                "method": "theoretical_full",
                "adoption_scenario": "full_adoption",
                "n_buildings": n_b,
                "cumulative_adoption_pct": cum,
                "yearly_adoption_pct": yearly_pct,
                "n_adopting": int(round(yearly_pct / 100.0 * n_b)) if n_b else 0,
                "curve_target_pct": 100.0,
                "cumulative_adoption_pct_p10": 100.0,
                "cumulative_adoption_pct_p90": 100.0,
            })
            prev = cum

    by_scen: dict[str, dict] = {}
    for rf in rfs:
        sub = (
            out_df[out_df["retrofit.scenario"] == rf]
            if "retrofit.scenario" in out_df.columns
            else out_df
        )
        mean_p = (
            float(sub["acceptance_probability"].mean())
            if "acceptance_probability" in sub.columns and len(sub) else 0.0
        )
        by_scen[rf] = {
            "n_total": len(sub),
            "n_passing_threshold": len(sub),
            "pass_rate": 1.0 if len(sub) else 0.0,
            "mean_acceptance_probability": mean_p,
        }
    scenario_summary: dict = {
        "method": "theoretical_full",
        "adoption_scenario": "full_adoption",
        "theoretical_maximum_no_propensity": True,
        "acceptance_threshold": 0.0,
        "floor_threshold": 0.0,
        "dice_reentry_years": None,
        "time_horizon": {"start_year": start_yr, "end_year": end_yr},
        "mean_acceptance_probability_definition": MEAN_ACCEPTANCE_MEAN_WITHIN_RETROFIT_SCENARIO_DEFINITION,
        "by_scenario": by_scen,
    }
    return UptakeResult(
        data=out_df,
        yearly_summary=pd.DataFrame(yearly_rows),
        scenario_summary=scenario_summary,
        method="theoretical_full",
        adoption_scenario_name="full_adoption",
    )


def _run_adoption_emissions_loop(
    propensity_result,
    policy_impacts: pd.DataFrame,
    all_adoption_scenarios: list[str],
    start_yr: int,
    end_yr: int,
    projection_years: list[int],
    emissions_json: dict,
    progress_range: tuple[int, int],
    progress_bar,
) -> dict:
    """Run uptake + emissions for every adoption scenario. Returns scenario_results dict."""
    scenario_results: dict = {}
    n_scen = len(all_adoption_scenarios)
    p_lo, p_hi = progress_range

    for i, adoption_scenario_name in enumerate(all_adoption_scenarios):
        pct = p_lo + int((p_hi - p_lo) * i / n_scen)
        progress_bar.progress(pct, text=f"Projecting '{adoption_scenario_name}' ({i + 1}/{n_scen})…")
        try:
            # only the locked reserved id "full_adoption" bypasses the engine.
            # user-defined curves that reach 100% still use AdoptionEngine + propensity.
            if adoption_scenario_name == "full_adoption":
                uptake_result = _theoretical_full_uptake_result(
                    propensity_result.data, start_yr, end_yr
                )
            else:
                dr_years = st.session_state.get("dice_reentry_years", 1)
                dice_reentry: int | None
                if st.session_state.get("dice_reentry_never", True):
                    dice_reentry = None
                else:
                    dice_reentry = int(dr_years)
                uptake_engine = AdoptionEngine(
                    propensity_df=propensity_result.data,
                    adoption_rates_path=_ADOPTION_CURVES_PATH,
                    adoption_scenario=adoption_scenario_name,
                    method="mc_ensemble",
                    start_year=start_yr,
                    end_year=end_yr,
                    n_ensemble_runs=int(st.session_state.get("n_ensemble_runs", 100)),
                    random_seed=int(st.session_state.get("random_seed", 42)),
                    dice_reentry_years=dice_reentry,
                )
                uptake_result = uptake_engine.calculate_uptake()
        except Exception as exc:
            st.warning(f"Adoption scenario '{adoption_scenario_name}' failed: {exc}")
            continue

        try:
            emissions_df = compute_emissions_trajectory(
                policy_impacts=policy_impacts,
                yearly_summary=uptake_result.yearly_summary,
                emissions_factors_json=emissions_json,
                years=projection_years,
                year_energy=st.session_state.get("year_energy_data") or None,
            )
        except Exception as exc:
            st.warning(f"Emissions for '{adoption_scenario_name}' failed: {exc}")
            emissions_df = pd.DataFrame()

        scenario_results[adoption_scenario_name] = {
            "uptake": uptake_result,
            "emissions": emissions_df,
        }

    return scenario_results


def _apply_non_us_priors(engine: PropensityModelEngine) -> None:
    """Override census tract distributions with user-specified priors."""
    from app.analysis.census_lookup import (
        build_prior_distributions,
        INCOME_CATEGORIES_K,
        EDUCATION_CATEGORIES,
    )
    prior = build_prior_distributions(
        income_mean_k=float(st.session_state.get("prior_income_mean", 60)),
        income_std_k=float(st.session_state.get("prior_income_std", 30)),
        pct_hs=float(st.session_state.get("prior_pct_hs", 35)),
        pct_associates=float(st.session_state.get("prior_pct_assoc", 15)),
        pct_bachelors=float(st.session_state.get("prior_pct_bach", 35)),
        pct_graduate=float(st.session_state.get("prior_pct_grad", 15)),
    )
    # Inject as a single synthetic tract used for all buildings
    engine.tract_distributions = {
        "_global": {
            "education": np.array(prior["education_probs"]),
            "household_size": np.ones(7) / 7,
            "income": np.array(prior["income_probs"]),
            "county": "_global",
        }
    }
    # Override the sample function to always use this distribution
    import types

    def _sample_overridden(self, county, tract_id=None, n_samples=1):
        dist = self.tract_distributions["_global"]
        return {
            "education": self.rng.choice(EDUCATION_CATEGORIES, size=n_samples, p=dist["education"]),
            "household_size": self.rng.choice(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], size=n_samples, p=dist["household_size"]
            ),
            "income": self.rng.choice(INCOME_CATEGORIES_K, size=n_samples, p=dist["income"]),
        }

    engine._sample_demographics = types.MethodType(_sample_overridden, engine)


# ── Palette helpers ────────────────────────────────────────────────────────────

# (line_color, band_rgba) pairs — one per adoption scenario
_SCENARIO_PALETTE = [
    ("#2563eb", "rgba(37,99,235,0.20)"),
    ("#16a34a", "rgba(22,163,74,0.20)"),
    ("#9333ea", "rgba(147,51,234,0.20)"),
    ("#ea580c", "rgba(234,88,12,0.20)"),
    ("#0891b2", "rgba(8,145,178,0.20)"),
    ("#65a30d", "rgba(101,163,13,0.20)"),
    ("#db2777", "rgba(219,39,119,0.20)"),
    ("#854d0e", "rgba(133,77,14,0.20)"),
]


def _add_scenario_traces(
    fig: go.Figure,
    years: "list[int]",
    mean_pct: "pd.Series",
    p10_pct: "pd.Series | None",
    p90_pct: "pd.Series | None",
    name: str,
    line_color: str,
    band_rgba: str,
    y_scale: float = 1.0,
) -> None:
    """Add a mean line + P10/P90 band for one scenario to *fig*."""
    has_band = p10_pct is not None and p90_pct is not None
    if has_band:
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=(p90_pct * y_scale).tolist() + (p10_pct * y_scale).iloc[::-1].tolist(),
            fill="toself", fillcolor=band_rgba,
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{name} P10–P90",
            showlegend=True,
            legendgroup=name,
            hovertemplate="%{y:.2f}<extra>P10–P90</extra>",
        ))
    fig.add_trace(go.Scatter(
        x=years, y=(mean_pct * y_scale).tolist(),
        mode="lines", name=f"{name} (mean)" if has_band else name,
        line=dict(color=line_color, width=2.5),
        legendgroup=name,
        hovertemplate="%{y:.2f}<extra>" + name + " mean</extra>",
    ))


# ── Figure builders ────────────────────────────────────────────────────────────

def _build_adoption_fig(scenario_results: dict, simulated_years: list[int]) -> go.Figure:
    fig = go.Figure()
    for i, (name, res) in enumerate(scenario_results.items()):
        ys = res["uptake"].yearly_summary.sort_values("year")
        if ys.empty:
            continue
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        years = ys["year"].tolist()
        mean_s = ys["cumulative_adoption_pct"]
        p10_s = ys.get("cumulative_adoption_pct_p10") if "cumulative_adoption_pct_p10" in ys.columns else None
        p90_s = ys.get("cumulative_adoption_pct_p90") if "cumulative_adoption_pct_p90" in ys.columns else None
        _add_scenario_traces(fig, years, mean_s, p10_s, p90_s, name, line_color, band_rgba)
    for yr in simulated_years:
        fig.add_vline(x=yr, line_dash="dot", line_color="#94a3b8", line_width=1,
                      annotation_text=str(yr), annotation_position="top",
                      annotation_font_size=11, annotation_font_color="#64748b")
    fig.update_layout(
        yaxis=dict(title="Cumulative adoption (%)", range=[0, 105]),
        xaxis_title="Year", height=360,
        margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.22),
    )
    return fig


def _build_emissions_fig(scenario_results: dict, first_em: pd.DataFrame, simulated_years: list[int]) -> go.Figure:
    fig = go.Figure()
    if not first_em.empty and "baseline_emissions_t" in first_em.columns:
        fig.add_trace(go.Scatter(
            x=first_em["year"].tolist(), y=first_em["baseline_emissions_t"].tolist(),
            mode="lines", name="Baseline (0% adoption)",
            line=dict(color="#dc2626", width=2.5, dash="dot"),
        ))
    for i, (name, res) in enumerate(scenario_results.items()):
        em = res["emissions"]
        if em.empty:
            continue
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        yrs = em["year"].tolist()
        mean_vals = em["scenario_mean_t"].tolist()
        lo_vals = em["scenario_p10_t"].tolist() if "scenario_p10_t" in em.columns else [v * 0.90 for v in mean_vals]
        hi_vals = em["scenario_p90_t"].tolist() if "scenario_p90_t" in em.columns else [v * 1.10 for v in mean_vals]
        fig.add_trace(go.Scatter(x=yrs, y=lo_vals, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 showlegend=False, hoverinfo="skip", legendgroup=name))
        fig.add_trace(go.Scatter(x=yrs, y=hi_vals, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 fill="tonexty", fillcolor=band_rgba,
                                 showlegend=False, hoverinfo="skip", legendgroup=name))
        fig.add_trace(go.Scatter(x=yrs, y=mean_vals, mode="lines", name=name,
                                 line=dict(color=line_color, width=2.5), legendgroup=name))
    for yr in simulated_years:
        fig.add_vline(x=yr, line_dash="dot", line_color="#94a3b8", line_width=1,
                      annotation_text=str(yr), annotation_position="top",
                      annotation_font_size=11, annotation_font_color="#64748b")
    fig.update_layout(
        yaxis_title="tCO₂/yr", xaxis_title="Year", height=360,
        margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.22),
    )
    return fig


def _build_energy_usage_fig(
    scenario_results: dict, first_em: pd.DataFrame, simulated_years: list[int],
) -> go.Figure:
    fig = go.Figure()
    if not first_em.empty and "baseline_kwh_GWh" in first_em.columns:
        fig.add_trace(go.Scatter(
            x=first_em["year"].tolist(), y=first_em["baseline_kwh_GWh"].tolist(),
            mode="lines", name="Baseline (0% adoption)",
            line=dict(color="#dc2626", width=2.5, dash="dot"),
        ))
    for i, (name, res) in enumerate(scenario_results.items()):
        em = res["emissions"]
        if em.empty or "scenario_kwh_mean_GWh" not in em.columns:
            continue
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        mean_kwh = em["scenario_kwh_mean_GWh"]
        p10_kwh = em["scenario_kwh_p10_GWh"] if "scenario_kwh_p10_GWh" in em.columns else None
        p90_kwh = em["scenario_kwh_p90_GWh"] if "scenario_kwh_p90_GWh" in em.columns else None
        _add_scenario_traces(
            fig, em["year"].tolist(), mean_kwh,
            pd.Series(p10_kwh.values, index=mean_kwh.index) if p10_kwh is not None else None,
            pd.Series(p90_kwh.values, index=mean_kwh.index) if p90_kwh is not None else None,
            name, line_color, band_rgba,
        )
    for yr in simulated_years:
        fig.add_vline(x=yr, line_dash="dot", line_color="#94a3b8", line_width=1,
                      annotation_text=str(yr), annotation_position="top",
                      annotation_font_size=11, annotation_font_color="#64748b")
    fig.update_layout(yaxis_title="GWh/yr", xaxis_title="Year",
                      height=400, margin=dict(t=20, b=20),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _first_em(scenario_results: dict) -> pd.DataFrame:
    return next(
        (v["emissions"] for v in scenario_results.values() if not v["emissions"].empty),
        pd.DataFrame(),
    )


def _sum_portfolio_retrofit_and_incentive(inc_data: pd.DataFrame) -> tuple[float | None, float | None]:
    """(total_gross_upfront_usd, total_expected_incentive_usd) or (None, None) if not available."""
    if inc_data.empty or "gross_upfront_usd" not in inc_data.columns:
        return None, None
    g = float(inc_data["gross_upfront_usd"].fillna(0.0).sum())
    ecol = inc_data.get("expected_incentive_usd")
    s = float(ecol.fillna(0.0).sum()) if ecol is not None else 0.0
    return g, s


def _render_adoption_incentive_breakdown(
    policy_base: pd.DataFrame,
    policy_inc: pd.DataFrame,
    prop_base,
    prop_inc,
    scenario_results: dict,
    scenario_results_inc: dict,
    retrofit_name: str,
) -> None:
    """Cohort by income/education; ranking displacement and incentive on adopted."""
    if (
        "acceptance_probability" not in policy_base.columns
        or "acceptance_probability" not in policy_inc.columns
    ):
        st.info("Adoption breakdown needs **acceptance_probability** on policy impacts (re-run analysis).")
        return
    keys_i = [k for k in scenario_results_inc if scenario_results.get(k)]
    if not keys_i:
        return
    ad_name = keys_i[0]
    yb = scenario_results[ad_name]["uptake"].yearly_summary
    yi = scenario_results_inc[ad_name]["uptake"].yearly_summary
    # session scenario_name can differ from what was written to policy/ propensity
    r_label = resolve_retrofit_scenario_name(policy_inc, retrofit_name)
    row_b = last_year_uptake_row(yb, ad_name, r_label)
    row_i = last_year_uptake_row(yi, ad_name, r_label)
    n_b = n_adopters_from_yearly_row(row_b) if row_b is not None else 0
    n_i = n_adopters_from_yearly_row(row_i) if row_i is not None else 0
    cum_b = float(row_b.get("cumulative_adoption_pct", 0.0) or 0.0) if row_b is not None else 0.0
    cum_i = float(row_i.get("cumulative_adoption_pct", 0.0) or 0.0) if row_i is not None else 0.0
    n_w_incentive = count_adopters_with_positive_incentive(
        policy_inc, "acceptance_probability", n_i, prop_inc.data, min_usd=0.0
    )
    n_eq = min(n_b, n_i)

    st.markdown("### Adoption cohort & incentive lift")
    st.caption(
        f"**Retrofit label used:** “{r_label}”  |  **Adoption curve:** `{ad_name}`. "
        "If you renamed the retrofit after the run, we match from policy data. "
        "Counts = mean uptake path ×**n_buildings** in that run (same as emissions). "
        "“Use incentives” = adopters in the retrofit cohort with **expected_incentive_usd** > 0 "
        "(residential only get non-zero in the model; commercial counts as 0 here)."
    )
    with st.expander("Definition & formula — adoption cohort & incentive lift", expanded=False):
        st.markdown(_FORMULA_MD_ADOPTION_DEMO_BREAKDOWN)

    mcols = st.columns(2)
    if row_b is None and row_i is None:
        st.warning(
            f"No **yearly_summary** row matched (adoption `{ad_name}`, retrofit `{r_label}`). "
            "If you renamed the retrofit scenario in the app, re-run the analysis. "
        )
    mcols[0].metric(
        "Buildings that retrofit (with incentive run, final year, mean path)",
        f"{n_i:,}" if row_i is not None else "—",
        help=(
            f"round(cumulative % × n_buildings) in uptake; here ≈{cum_i:.1f}% and "
            f"n_bld={int(row_i.get('n_buildings', 0) or 0)}"
        )
        if row_i is not None
        else "no matching row",
    )
    mcols[1].metric(
        "Of those, with a positive program incentive (expected, mean draw)",
        f"{n_w_incentive:,}" if row_i is not None else "—",
        help="Among top n_i by with-incentive propensity; count expected_incentive_usd > 0 on propensity data",
    )
    mcols2 = st.columns(3)
    mcols2[0].metric(
        "Buildings that retrofit (no incentive run)",
        f"{n_b:,}" if row_b is not None else "—",
        help=f"≈{cum_b:.1f}% cumulative" if row_b is not None else "no matching row",
    )
    mcols2[1].metric("Extra adopters (with − without)", f"{n_i - n_b:+,}", help="same adoption curve, different propensity")
    if n_eq > 0 and n_b and n_i:
        only_i, _only_b, _ovl = ranking_displacement_at_equal_n(
            policy_inc["building.id"],
            policy_base["acceptance_probability"],
            policy_inc["acceptance_probability"],
            n_eq,
        )
        mcols2[2].metric(
            f"Re-ranked in top {n_eq} (same n, both runs)",
            f"{only_i}",
            help="How many of the n slots go to different buildings when ranking by with- vs no-incentive propensity",
        )
    else:
        mcols2[2].metric("Re-ranked in top n", "—")

    cohort = build_adoption_cohort_by_demographics(
        policy_inc, "acceptance_probability", n_i
    )
    inc_cohort_usd = expected_incentive_sum_on_adopted_cohort(
        policy_inc, "acceptance_probability", n_i, prop_inc.data
    )
    tot_s = _sum_portfolio_retrofit_and_incentive(prop_inc.data)[1]
    if inc_cohort_usd is not None and tot_s is not None and tot_s > 0:
        st.caption(
            f"**Expected incentives on the adopted cohort (with incentive):** ${inc_cohort_usd:,.0f} "
            f"({100.0 * inc_cohort_usd / tot_s:.1f}% of the portfolio-expected total incentive pool)."
        )
    elif inc_cohort_usd is not None:
        st.caption(
            f"**Expected incentives on the adopted cohort (with incentive):** ${inc_cohort_usd:,.0f}."
        )

    in_counts = cohort["income_counts"]
    ed_counts = cohort["education_counts"]
    c1, c2 = st.columns(2)
    with c1:
        if in_counts:
            s_inc = pd.Series(in_counts).sort_index()
            fig = go.Figure(
                go.Bar(
                    x=s_inc.index,
                    y=s_inc.values,
                    marker_color="#16a34a",
                    name="Buildings",
                )
            )
            fig.update_layout(
                title="Adopted cohort by income (tract-based, n=%d)" % n_i,
                xaxis_title="",
                yaxis_title="Buildings",
                height=400,
                margin=dict(t=40, b=80),
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True, key="adopt_demo_income")
        else:
            st.caption("No income breakdown (missing tract `income_probs` on policy impacts).")
    with c2:
        if ed_counts:
            s_ed = pd.Series(ed_counts).sort_index()
            fig2 = go.Figure(
                go.Bar(
                    x=s_ed.index,
                    y=s_ed.values,
                    marker_color="#2563eb",
                )
            )
            fig2.update_layout(
                title="Adopted cohort by education (tract-based, n=%d)" % n_i,
                xaxis_title="",
                yaxis_title="Buildings",
                height=400,
                margin=dict(t=40, b=80),
            )
            fig2.update_xaxis(tickangle=35)
            st.plotly_chart(fig2, use_container_width=True, key="adopt_demo_edu")
        else:
            st.caption("No education breakdown (missing `education_probs`).")


# ── Results charts ─────────────────────────────────────────────────────────────

def _render_result_charts() -> None:
    scenario_results: dict = st.session_state["scenario_results"]
    scenario_results_inc: dict | None = st.session_state.get("scenario_results_incentive")
    propensity_result = st.session_state["propensity_result"]
    propensity_result_inc = st.session_state.get("propensity_result_incentive")
    policy_impacts: pd.DataFrame = st.session_state["policy_impacts"]
    retrofit_name = st.session_state["scenario_name"]
    simulated_years: list[int] = st.session_state.get("simulated_years", [])
    incentives_enabled = bool(scenario_results_inc)

    em_base = _first_em(scenario_results)
    em_inc = _first_em(scenario_results_inc) if scenario_results_inc else pd.DataFrame()

    st.divider()

    # ── KPI row ────────────────────────────────────────────────────────────────
    n_buildings = len(policy_impacts)
    mean_prop = float(propensity_result.mean_acceptance_probability)
    kpi_cols = st.columns(3 if incentives_enabled else 2)
    with kpi_cols[0]:
        st.metric("Buildings modelled", f"{n_buildings:,}")
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_BUILDINGS_MODELLED)
    with kpi_cols[1]:
        st.metric(
            f"{MEAN_ACCEPTANCE_AGGREGATE_LABEL} (no incentive)",
            f"{mean_prop:.1%}",
            help=MEAN_ACCEPTANCE_AGGREGATE_DESCRIPTION,
        )
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_MEAN_ACCEPTANCE_AGGREGATE)
    if incentives_enabled and propensity_result_inc:
        mean_prop_inc = float(propensity_result_inc.mean_acceptance_probability)
        with kpi_cols[2]:
            st.metric(
                f"{MEAN_ACCEPTANCE_AGGREGATE_LABEL} (with incentive)",
                f"{mean_prop_inc:.1%}",
                delta=f"{mean_prop_inc - mean_prop:+.1%}",
                help=MEAN_ACCEPTANCE_AGGREGATE_DESCRIPTION,
            )
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_MEAN_ACCEPTANCE_AGGREGATE)
        tot_g, tot_s = _sum_portfolio_retrofit_and_incentive(propensity_result_inc.data)
        if tot_g is not None:
            kpi_cost = st.columns(2)
            with kpi_cost[0]:
                st.metric(
                    "Total gross retrofit cost (portfolio)",
                    f"${tot_g:,.0f}",
                    help="sum of per-row upfront deal cost (USD); not net of incentives. see expander for formula.",
                )
            with kpi_cost[1]:
                st.metric(
                    "Total expected incentives (portfolio)",
                    f"${tot_s:,.0f}",
                    help="sum of mean draw incentive per building (USD); not adoption-weighted. see expander.",
                )
            with st.expander("Definition & formula — portfolio cost & incentives", expanded=False):
                st.markdown(_FORMULA_MD_PORTFOLIO_RETROFIT_AND_INCENTIVE)

    # ── Summary table (base only) ──────────────────────────────────────────────
    summary_rows = []
    for name, res in scenario_results.items():
        ys = res["uptake"].yearly_summary
        em = res["emissions"]
        if ys.empty:
            continue
        last = ys.sort_values("year").iloc[-1]
        final_adopt = float(last.get("cumulative_adoption_pct", 0))
        if not em.empty and "baseline_emissions_t" in em.columns:
            baseline_end = float(em.iloc[-1]["baseline_emissions_t"])
            scen_end = float(em.iloc[-1]["scenario_mean_t"])
            savings = (baseline_end - scen_end) / baseline_end * 100 if baseline_end > 0 else 0.0
        else:
            savings = float("nan")
        summary_rows.append({
            "Adoption scenario": name,
            "Final adoption (mean %)": f"{final_adopt:.1f}%",
            "Final adoption (P10–P90)": (
                f"{float(last.get('cumulative_adoption_pct_p10', final_adopt)):.1f}% – "
                f"{float(last.get('cumulative_adoption_pct_p90', final_adopt)):.1f}%"
            ),
            "Emissions reduction vs baseline": f"{savings:.1f}%" if not pd.isna(savings) else "—",
        })
    if summary_rows:
        with st.expander("Definition & formula — scenario summary columns", expanded=False):
            st.markdown(_FORMULA_MD_SCENARIO_SUMMARY_TABLE)
        st.dataframe(pd.DataFrame(summary_rows).set_index("Adoption scenario"),
                     use_container_width=True)

    if (
        incentives_enabled
        and st.session_state.get("policy_impacts_incentive") is not None
        and propensity_result_inc
        and scenario_results_inc
    ):
        st.divider()
        _render_adoption_incentive_breakdown(
            policy_base=policy_impacts,
            policy_inc=st.session_state["policy_impacts_incentive"],
            prop_base=propensity_result,
            prop_inc=propensity_result_inc,
            scenario_results=scenario_results,
            scenario_results_inc=scenario_results_inc,
            retrofit_name=retrofit_name,
        )

    st.divider()

    # ── Adoption trajectories ─────────────────────────────────────────────────
    st.markdown(f"### Adoption Trajectories — {retrofit_name}")
    with st.expander("Definition & formula — cumulative adoption", expanded=False):
        st.markdown(_FORMULA_MD_ADOPTION_TRAJECTORY)
    if incentives_enabled and scenario_results_inc:
        col_l, col_r = st.columns(2)
        with col_l:
            st.caption("No incentive")
            st.plotly_chart(_build_adoption_fig(scenario_results, simulated_years),
                            use_container_width=True, key="adoption_base")
        with col_r:
            st.caption("With incentive")
            st.plotly_chart(_build_adoption_fig(scenario_results_inc, simulated_years),
                            use_container_width=True, key="adoption_inc")
    else:
        st.plotly_chart(_build_adoption_fig(scenario_results, simulated_years),
                        use_container_width=True, key="adoption_base")

    # ── Energy usage trajectories (full width — same baseline for both) ────────
    st.divider()
    st.markdown("### Energy Usage Trajectories (GWh/yr)")
    st.caption(
        "Total building-stock energy each year: adopted buildings contribute retrofit-scenario kWh, "
        "all others contribute baseline kWh. As adoption grows the total decreases toward the full-retrofit level."
    )
    with st.expander("Definition & formula — stock energy", expanded=False):
        st.markdown(_FORMULA_MD_ENERGY_TRAJECTORY)
    if incentives_enabled and scenario_results_inc:
        col_el, col_er = st.columns(2)
        with col_el:
            st.caption("No incentive")
            st.plotly_chart(
                _build_energy_usage_fig(scenario_results, em_base, simulated_years),
                use_container_width=True, key="energy_usage_base",
            )
        with col_er:
            st.caption("With incentive")
            st.plotly_chart(
                _build_energy_usage_fig(scenario_results_inc, em_inc, simulated_years),
                use_container_width=True, key="energy_usage_inc",
            )
    else:
        st.plotly_chart(
            _build_energy_usage_fig(scenario_results, em_base, simulated_years),
            use_container_width=True, key="energy_usage",
        )

    # ── Emissions trajectories ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Emissions Trajectories (metric tonnes CO₂/yr)")
    st.caption(
        "kWh from the energy chart × per-fuel emissions factors (which decline with grid decarbonisation). "
        "Shaded band = MC P10–P90."
    )
    with st.expander("Definition & formula — emissions", expanded=False):
        st.markdown(_FORMULA_MD_EMISSIONS_TRAJECTORY)
    if incentives_enabled and scenario_results_inc:
        col_l, col_r = st.columns(2)
        with col_l:
            st.caption("No incentive")
            st.plotly_chart(_build_emissions_fig(scenario_results, em_base, simulated_years),
                            use_container_width=True, key="emissions_base")
        with col_r:
            st.caption("With incentive")
            st.plotly_chart(_build_emissions_fig(scenario_results_inc, em_inc, simulated_years),
                            use_container_width=True, key="emissions_inc")
    else:
        st.plotly_chart(_build_emissions_fig(scenario_results, em_base, simulated_years),
                        use_container_width=True, key="emissions_base")

    # ── Propensity distribution ───────────────────────────────────────────────
    st.divider()
    st.markdown("### Propensity Distribution")
    with st.expander("Definition & formula — propensity histogram", expanded=False):
        st.markdown(_FORMULA_MD_PROPENSITY_HISTOGRAM)

    def _propensity_hist(result, color: str, label: str) -> go.Figure:
        probs = result.data["acceptance_probability"].dropna()
        fig = go.Figure(go.Histogram(x=probs, nbinsx=40, marker_color=color, opacity=0.75, name=label))
        fig.update_layout(xaxis_title="Acceptance probability", yaxis_title="Count",
                          height=280, margin=dict(t=10, b=10))
        return fig

    if incentives_enabled and propensity_result_inc:
        col_l, col_r = st.columns(2)
        with col_l:
            st.caption("No incentive")
            st.plotly_chart(_propensity_hist(propensity_result, "#2563eb", "No incentive"),
                            use_container_width=True, key="propensity_base")
        with col_r:
            st.caption("With incentive")
            st.plotly_chart(_propensity_hist(propensity_result_inc, "#16a34a", "With incentive"),
                            use_container_width=True, key="propensity_inc")
    else:
        col_p, col_e = st.columns(2)
        with col_p:
            st.plotly_chart(_propensity_hist(propensity_result, "#2563eb", "Propensity"),
                            use_container_width=True, key="propensity_base")
        with col_e:
            st.markdown("### Energy Savings vs. Propensity")
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_ENERGY_SAVINGS_SCATTER)
            plot_df = policy_impacts if "acceptance_probability" in policy_impacts.columns else (
                policy_impacts.join(
                    propensity_result.data.set_index("building.id")[["acceptance_probability"]],
                    on="building.id", how="left",
                )
            )
            if "acceptance_probability" in plot_df.columns and "energy_cost.annual_savings" in plot_df.columns:
                fig_scatter = go.Figure(go.Scatter(
                    x=plot_df["energy_cost.annual_savings"],
                    y=plot_df["acceptance_probability"],
                    mode="markers", marker=dict(color="#9333ea", size=4, opacity=0.5),
                ))
                fig_scatter.update_layout(
                    xaxis_title="Annual energy savings ($/yr)",
                    yaxis_title="Acceptance probability",
                    height=280, margin=dict(t=10, b=10),
                )
                st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_savings")

    # ── Download ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Download Results")

    # Combined emissions CSV: all scenarios stacked with an adoption_scenario column
    em_frames = []
    for name, res in scenario_results.items():
        em = res["emissions"]
        if not em.empty:
            em_frames.append(em.assign(adoption_scenario=name))
    combined_em = pd.concat(em_frames) if em_frames else pd.DataFrame()

    adoption_frames = []
    for name, res in scenario_results.items():
        ys = res["uptake"].yearly_summary
        if not ys.empty:
            adoption_frames.append(ys)
    combined_adoption = pd.concat(adoption_frames) if adoption_frames else pd.DataFrame()

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        if not combined_em.empty:
            st.download_button(
                "Emissions trajectories (CSV)",
                combined_em.to_csv(index=False).encode(),
                "emissions_trajectories.csv", "text/csv",
                key="dl_emissions",
            )
    with dl2:
        if not combined_adoption.empty:
            st.download_button(
                "Adoption summaries (CSV)",
                combined_adoption.to_csv(index=False).encode(),
                "adoption_summaries.csv", "text/csv",
                key="dl_adoption",
            )
    with dl3:
        st.download_button(
            "Policy impacts (CSV)",
            policy_impacts.to_csv(index=False).encode(),
            "policy_impacts.csv", "text/csv",
            key="dl_policy",
        )

    # ── Save scenario ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Save Scenario")
    st.caption(
        "Store this run's outputs — config, emissions trajectories, adoption summaries, and "
        "policy impacts — to `outputs/{name}/` for later review in the **Visualize** tab."
    )
    save_col1, save_col2, _ = st.columns([2, 1, 3])
    with save_col1:
        save_name = st.text_input(
            "Scenario name",
            placeholder="e.g. high_incentives_2030",
            key="save_scenario_name",
            label_visibility="collapsed",
        )
    with save_col2:
        if st.button(
            "Save Results",
            disabled=not save_name.strip(),
            use_container_width=True,
            key="btn_save_scenario",
        ):
            clean = save_name.strip().replace(" ", "_")
            _save_scenario_results(clean)
            st.success(f"Saved to `outputs/{clean}/`")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — WIP EXPLORER  (National Retrofit Deal Acceptance Simulator)
# ══════════════════════════════════════════════════════════════════════════════

_CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "e865d3af152108cb504df27535d196f586c21729")

_WIP_COEFFICIENTS = {
    "intercept": 0.0,
    "Year built": -0.051,
    "Education": 0.110,
    "bedrooms": 0.272,
    "residents": 0.087,
    "Income": 0.046,
    "Concern": 0.271,
    "Upfront cost": -0.058,
    "Neighbor": -0.029,
    "Energy cost": -0.012,
}

_US_STATES = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
    "District of Columbia": "11", "Florida": "12", "Georgia": "13", "Hawaii": "15",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19", "Kansas": "20",
    "Kentucky": "21", "Louisiana": "22", "Maine": "23", "Maryland": "24",
    "Massachusetts": "25", "Michigan": "26", "Minnesota": "27", "Mississippi": "28",
    "Missouri": "29", "Montana": "30", "Nebraska": "31", "Nevada": "32",
    "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35", "New York": "36",
    "North Carolina": "37", "North Dakota": "38", "Ohio": "39", "Oklahoma": "40",
    "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44", "South Carolina": "45",
    "South Dakota": "46", "Tennessee": "47", "Texas": "48", "Utah": "49",
    "Vermont": "50", "Virginia": "51", "Washington": "53", "West Virginia": "54",
    "Wisconsin": "55", "Wyoming": "56",
}

_STATE_MAP_CONFIG = {
    "Alabama": ([32.8, -86.8], 7), "Alaska": ([64.2, -153.4], 4),
    "Arizona": ([34.3, -111.1], 7), "Arkansas": ([34.8, -92.2], 7),
    "California": ([36.8, -119.4], 6), "Colorado": ([39.0, -105.5], 7),
    "Connecticut": ([41.6, -72.7], 9), "Delaware": ([38.9, -75.5], 9),
    "District of Columbia": ([38.9, -77.0], 12), "Florida": ([27.7, -81.7], 7),
    "Georgia": ([32.2, -83.4], 7), "Hawaii": ([20.8, -156.3], 7),
    "Idaho": ([44.2, -114.5], 6), "Illinois": ([40.0, -89.2], 7),
    "Indiana": ([40.3, -86.1], 7), "Iowa": ([41.9, -93.1], 7),
    "Kansas": ([38.5, -98.3], 7), "Kentucky": ([37.7, -84.9], 7),
    "Louisiana": ([31.2, -91.8], 7), "Maine": ([44.7, -69.4], 7),
    "Maryland": ([39.1, -76.8], 8), "Massachusetts": ([42.4, -71.4], 8),
    "Michigan": ([44.2, -85.4], 7), "Minnesota": ([46.4, -93.1], 6),
    "Mississippi": ([32.7, -89.7], 7), "Missouri": ([38.5, -92.5], 7),
    "Montana": ([47.0, -110.5], 6), "Nebraska": ([41.5, -99.9], 7),
    "Nevada": ([38.8, -116.4], 6), "New Hampshire": ([43.7, -71.6], 8),
    "New Jersey": ([40.1, -74.5], 8), "New Mexico": ([34.5, -105.9], 7),
    "New York": ([42.9, -75.5], 7), "North Carolina": ([35.6, -79.8], 7),
    "North Dakota": ([47.5, -100.5], 7), "Ohio": ([40.4, -82.8], 7),
    "Oklahoma": ([35.6, -96.9], 7), "Oregon": ([44.6, -122.1], 7),
    "Pennsylvania": ([40.9, -77.8], 7), "Rhode Island": ([41.7, -71.5], 10),
    "South Carolina": ([33.8, -80.9], 7), "South Dakota": ([44.4, -100.2], 7),
    "Tennessee": ([35.9, -86.4], 7), "Texas": ([31.1, -97.6], 6),
    "Utah": ([39.3, -111.1], 7), "Vermont": ([44.0, -72.7], 8),
    "Virginia": ([37.8, -79.5], 7), "Washington": ([47.4, -120.7], 7),
    "West Virginia": ([38.9, -80.5], 7), "Wisconsin": ([44.3, -89.6], 7),
    "Wyoming": ([43.0, -107.6], 7),
}

_MA_CENSUS_CSV = _REPO_ROOT / "data" / "census_data" / "massachusetts_census_data.csv"
_MA_COUNTY_FIPS = {
    1: "Barnstable", 3: "Berkshire", 5: "Bristol", 7: "Dukes", 9: "Essex",
    11: "Franklin", 13: "Hampden", 15: "Hampshire", 17: "Middlesex",
    19: "Nantucket", 21: "Norfolk", 23: "Plymouth", 25: "Suffolk", 27: "Worcester",
}

_ACS_VARIABLE_CHUNKS = {
    "income": {
        "endpoint": "acs/acs5",
        "vars": {
            "B19001_001E": "income_total_households",
            "B19001_002E": "income_less_than_10k",
            "B19001_003E": "income_10k_to_14999",
            "B19001_004E": "income_15k_to_19999",
            "B19001_005E": "income_20k_to_24999",
            "B19001_006E": "income_25k_to_29999",
            "B19001_007E": "income_30k_to_34999",
            "B19001_008E": "income_35k_to_39999",
            "B19001_009E": "income_40k_to_44999",
            "B19001_010E": "income_45k_to_49999",
            "B19001_011E": "income_50k_to_59999",
            "B19001_012E": "income_60k_to_74999",
            "B19001_013E": "income_75k_to_99999",
            "B19001_014E": "income_100k_to_124999",
            "B19001_015E": "income_125k_to_149999",
            "B19001_016E": "income_150k_to_199999",
            "B19001_017E": "income_200k_or_more",
        },
    },
    "education": {
        "endpoint": "acs/acs5",
        "vars": {
            "B15003_001E": "education_total_pop_25_over",
            "B15003_002E": "education_no_schooling",
            "B15003_017E": "education_high_school_grad",
            "B15003_018E": "education_ged",
            "B15003_021E": "education_associates_degree",
            "B15003_022E": "education_bachelors_degree",
            "B15003_023E": "education_masters_degree",
            "B15003_024E": "education_professional_school_degree",
            "B15003_025E": "education_doctorate_degree",
        },
    },
    "age": {
        "endpoint": "acs/acs5/profile",
        "vars": {
            "DP05_0001E": "age_total_population",
            "DP05_0005E": "age_under_5",
            "DP05_0006E": "age_5_to_9",
            "DP05_0007E": "age_10_to_14",
            "DP05_0008E": "age_15_to_19",
            "DP05_0009E": "age_20_to_24",
            "DP05_0010E": "age_25_to_34",
            "DP05_0011E": "age_35_to_44",
            "DP05_0012E": "age_45_to_54",
            "DP05_0013E": "age_55_to_59",
            "DP05_0014E": "age_60_to_64",
            "DP05_0015E": "age_65_to_74",
            "DP05_0016E": "age_75_to_84",
            "DP05_0017E": "age_85_and_over",
            "DP05_0018E": "median_age",
        },
    },
    "household_size": {
        "endpoint": "acs/acs5",
        "vars": {
            "B11016_001E": "household_total",
            "B11016_002E": "household_1_person",
            "B11016_003E": "household_2_person",
            "B11016_004E": "household_3_person",
            "B11016_005E": "household_4_person",
            "B11016_006E": "household_5_person",
            "B11016_007E": "household_6_person",
            "B11016_008E": "household_7_or_more_person",
        },
    },
}


def _wip_census_get(url: str, params: dict, *, read_timeout: int = 300) -> requests.Response:
    # large tract:* state pulls often exceed 60s; retry on timeouts and brief upstream errors
    last_exc: BaseException | None = None
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=(30, read_timeout))
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(min(2.0 * (2**attempt), 30.0))
            continue
        if resp.status_code in (502, 503, 504) and attempt < 3:
            time.sleep(min(2.0 * (2**attempt), 30.0))
            continue
        return resp
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("census request failed with no response")


@st.cache_data(show_spinner=False, ttl=86400)
def _wip_load_ma_census(file_path: str) -> pd.DataFrame | None:
    import os
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)

    def _to_int_county(code):
        try:
            return int(str(code).split(".")[0])
        except Exception:
            return None

    df["county_name"] = df["county"].apply(lambda c: _MA_COUNTY_FIPS.get(_to_int_county(c)))
    df["tract_str"] = df["tract"].astype(str).str.split(".").str[0].str.strip().str.zfill(6)
    df["display_label"] = "Tract " + df["tract_str"] + ", " + df["county_name"] + " County"
    return df.set_index("display_label")


@st.cache_data(show_spinner=False, ttl=86400)
def _wip_fetch_census(state_fips: str) -> pd.DataFrame | None:
    dataframes = []
    for chunk_name, chunk in _ACS_VARIABLE_CHUNKS.items():
        base_url = f"https://api.census.gov/data/2023/{chunk['endpoint']}"
        params = {
            "get": ",".join(chunk["vars"].keys()),
            "for": "tract:*",
            "in": f"state:{state_fips}",
            "key": _CENSUS_API_KEY,
        }
        try:
            resp = _wip_census_get(base_url, params)
        except requests.RequestException as exc:
            st.error(
                f"Census API request failed after retries: {exc}. "
                "Try again in a moment, pick a smaller state, or set CENSUS_API_KEY for higher rate limits."
            )
            return None
        except Exception as exc:
            st.error(f"Census API request failed: {exc}")
            return None
        if resp.status_code != 200:
            st.error(f"Census API error ({chunk_name}): HTTP {resp.status_code}")
            return None
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0]).rename(columns=chunk["vars"])
        dataframes.append(df)
    merged = reduce(
        lambda l, r: pd.merge(l, r, on=["state", "county", "tract"], how="outer"),
        dataframes,
    )
    for col in merged.columns:
        if col not in ("state", "county", "tract"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


@st.cache_data(show_spinner=False, ttl=86400)
def _wip_fetch_county_names(state_fips: str) -> dict[str, str]:
    try:
        resp = _wip_census_get(
            "https://api.census.gov/data/2020/dec/pl",
            {"get": "NAME", "for": "county:*", "in": f"state:{state_fips}", "key": _CENSUS_API_KEY},
            read_timeout=120,
        )
        if resp.status_code != 200:
            return {}
        result = {}
        for row in resp.json()[1:]:
            name, county_fips = row[0], row[2]
            short = name.split(" County")[0].split(" Parish")[0].split(" Borough")[0]
            result[county_fips.zfill(3)] = short
        return result
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=86400)
def _wip_fetch_shapefile(state_fips: str):
    url = f"https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_{state_fips}_tract.zip"
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            return None
        tmpdir = f"/tmp/tiger_{state_fips}"
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall(tmpdir)
            shp_files = [f for f in z.namelist() if f.endswith(".shp")]
        if not shp_files:
            return None
        return gpd.read_file(f"{tmpdir}/{shp_files[0]}")
    except Exception:
        return None


def _wip_build_display_df(raw_df: pd.DataFrame, county_names: dict[str, str]) -> pd.DataFrame:
    df = raw_df.copy()

    def county_label(c):
        try:
            return county_names.get(str(int(str(c).split(".")[0])).zfill(3), f"County {c}")
        except Exception:
            return str(c)

    df["county_name"] = df["county"].apply(county_label)
    df["tract_str"] = df["tract"].astype(str).str.split(".").str[0].str.strip().str.zfill(6)
    df["display_label"] = "Tract " + df["tract_str"] + ", " + df["county_name"] + " County"
    return df.set_index("display_label")


def _wip_weight(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) and v >= 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _wip_run_simulation(tract_data, building_age, upfront_cost_range, energy_cost_range, concern, neighbor, n_runs=10000) -> np.ndarray:
    edu_cats = [1, 2, 3, 4]
    edu_counts = [
        _wip_weight(tract_data[["education_high_school_grad", "education_ged"]].sum()),
        _wip_weight(tract_data["education_associates_degree"]),
        _wip_weight(tract_data["education_bachelors_degree"]),
        _wip_weight(tract_data[["education_masters_degree", "education_professional_school_degree", "education_doctorate_degree"]].sum()),
    ]
    hh_cats = [1, 2, 3, 4, 5, 6, 7]
    hh_counts = np.array([_wip_weight(tract_data[c]) for c in (
        "household_1_person", "household_2_person", "household_3_person", "household_4_person",
        "household_5_person", "household_6_person", "household_7_or_more_person",
    )], dtype=float)
    inc_cats = [5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 55, 67.5, 87.5, 112.5, 137.5, 175, 250]
    inc_counts = np.array([_wip_weight(tract_data[c]) for c in (
        "income_less_than_10k", "income_10k_to_14999", "income_15k_to_19999", "income_20k_to_24999",
        "income_25k_to_29999", "income_30k_to_34999", "income_35k_to_39999", "income_40k_to_44999",
        "income_45k_to_49999", "income_50k_to_59999", "income_60k_to_74999", "income_75k_to_99999",
        "income_100k_to_124999", "income_125k_to_149999", "income_150k_to_199999", "income_200k_or_more",
    )], dtype=float)

    C = _WIP_COEFFICIENTS
    probs = []
    for _ in range(n_runs):
        edu = random.choices(edu_cats, weights=edu_counts, k=1)[0] if sum(edu_counts) > 0 else 1
        hh = random.choices(hh_cats, weights=hh_counts, k=1)[0] if float(np.sum(hh_counts)) > 0 else 2
        inc = random.choices(inc_cats, weights=inc_counts, k=1)[0] if float(np.sum(inc_counts)) > 0 else 55
        cost = random.uniform(*upfront_cost_range)
        energy = random.uniform(*energy_cost_range)
        Z = (C["intercept"] + C["Year built"] * building_age + C["Education"] * edu
             + C["bedrooms"] * hh + C["residents"] * hh + C["Income"] * inc
             + C["Concern"] * concern + C["Upfront cost"] * cost
             + C["Neighbor"] * neighbor + C["Energy cost"] * energy)
        probs.append(1 / (1 + np.exp(-Z)))
    return np.array(probs)


def _wip_run_all_tracts(df, building_age, upfront_cost_range, energy_cost_range, concern, neighbor, n_runs=1000):
    results = {}
    bar = st.progress(0)
    status = st.empty()
    total = len(df.index)
    for i, label in enumerate(df.index):
        status.text(f"Processing {label}… ({i + 1}/{total})")
        try:
            probs = _wip_run_simulation(df.loc[label], building_age, upfront_cost_range, energy_cost_range, concern, neighbor, n_runs)
            results[label] = float(np.mean(probs))
        except Exception:
            results[label] = 0.5
        bar.progress((i + 1) / total)
    status.text("Simulation complete!")
    return results


def _wip_status_quo(t, max_adoption=0.85):
    return min(max_adoption, (max_adoption / (2050 - 2024)) * (t - 2024))


def _wip_s_curve(t, max_adoption=0.95):
    tn = (t - 2024) / (2050 - 2024)
    return max_adoption / (1 + np.exp(-4 * (tn - 0.4)))


def _wip_regulatory(t, max_adoption=0.98):
    return min(max_adoption, 0.05 * np.exp(0.15 * 1.5 * (t - 2024)))


def _wip_curve_dict(func, years):
    return {int(y): float(func(y)) for y in years}


def _wip_fill_curve(ranked, adoption_by_year, adopted_set, years_list, curve, n_total):
    queue = [b for b in ranked if b not in adopted_set]
    ptr, adopted_so_far = 0, len(adopted_set)
    for year in years_list:
        need = max(0, int(n_total * curve.get(year, 0.0)) - adopted_so_far)
        added = 0
        while added < need and ptr < len(queue):
            bid = queue[ptr]; ptr += 1
            if bid in adopted_set:
                continue
            adoption_by_year[bid] = year
            adopted_set.add(bid)
            adopted_so_far += 1
            added += 1
    return adoption_by_year, adopted_set


def _wip_integrate_threshold(prop, curve, years, threshold):
    n = len(prop)
    if n == 0:
        return {}
    willing = sorted([t for t, p in prop.items() if p >= threshold], key=lambda t: prop[t], reverse=True)
    r, _ = _wip_fill_curve(willing, {}, set(), years, curve, n)
    return r


def _wip_integrate_ranked(prop, curve, years, floor):
    n = len(prop)
    if n == 0:
        return {}
    eligible = sorted([t for t, p in prop.items() if p >= floor], key=lambda t: prop[t], reverse=True)
    r, _ = _wip_fill_curve(eligible, {}, set(), years, curve, n)
    return r


def _wip_integrate_dice(
    prop,
    curve,
    years,
    rng,
    floor=0.0,
    dice_reentry_years: int | None = 1,
):
    n = len(prop)
    if n == 0:
        return {}
    candidates = {k: float(v) for k, v in prop.items() if v >= floor}
    all_bids = list(candidates.keys())
    adopted_set, adoption_by_year, adopted_so_far = set(), {}, 0
    excluded = set()
    next_eligible = {}
    reentry = None if dice_reentry_years is None else max(1, int(dice_reentry_years))
    for year in years:
        need = max(0, int(n * curve.get(year, 0.0)) - adopted_so_far)
        if need == 0:
            continue
        eligible = [
            b
            for b in all_bids
            if b not in adopted_set
            and b not in excluded
            and year >= next_eligible.get(b, 0)
        ]
        if not eligible:
            break
        draws = rng.uniform(0, 1, len(eligible))
        hits = sorted(
            [b for b, u in zip(eligible, draws) if candidates[b] > u],
            key=lambda b: candidates[b],
            reverse=True,
        )
        chosen = set(hits[:need])
        for bid in hits[:need]:
            adoption_by_year[bid] = year
            adopted_set.add(bid)
            adopted_so_far += 1
        for bid in eligible:
            if bid in chosen:
                continue
            if reentry is None:
                excluded.add(bid)
            else:
                next_eligible[bid] = year + reentry
    return adoption_by_year


def _wip_integrate_ensemble(
    prop,
    curve,
    years,
    n_runs,
    floor,
    seed,
    ci_low=10.0,
    ci_high=90.0,
    dice_reentry_years: int | None = 1,
):
    rng0 = np.random.default_rng(seed)
    seeds = rng0.integers(0, 2**32, size=n_runs)
    assignments, rows = [], []
    for s in seeds:
        a = _wip_integrate_dice(
            prop,
            curve,
            years,
            np.random.default_rng(int(s)),
            floor=floor,
            dice_reentry_years=dice_reentry_years,
        )
        assignments.append(a)
        rows.append(_wip_cumulative_pct(a, years, len(prop)).values)
    arr = np.stack(rows)
    return (pd.Series(arr.mean(0), index=years),
            pd.Series(np.percentile(arr, ci_low, 0), index=years),
            pd.Series(np.percentile(arr, ci_high, 0), index=years),
            assignments)


def _wip_cumulative_pct(adoption_by_year, years, n_total):
    counts = {y: 0 for y in years}
    for y in adoption_by_year.values():
        if y in counts:
            counts[y] += 1
    running, out = 0, []
    for y in years:
        running += counts.get(y, 0)
        out.append(running / n_total * 100 if n_total else 0.0)
    return pd.Series(out, index=years)


def _wip_choropleth(census_df, tract_results, map_data, state_name):
    center, zoom = _STATE_MAP_CONFIG.get(state_name, ([39.5, -98.4], 4))
    m = folium.Map(location=center, zoom_start=zoom, tiles="Cartodb Positron")
    if map_data is None or not tract_results:
        return m
    min_p, max_p = min(tract_results.values()), max(tract_results.values())

    def get_color(p):
        n = (p - min_p) / (max_p - min_p) if max_p > min_p else 0.5
        return matplotlib.colors.rgb2hex(matplotlib.cm.get_cmap("viridis")(n))

    def style_fn(feature):
        props = feature["properties"]
        tract_id = props.get("TRACTCE", props.get("TRACTCE20", ""))
        county_fp = props.get("COUNTYFP", props.get("COUNTYFP20", ""))
        label = next(
            (lbl for lbl in tract_results
             if lbl.startswith(f"Tract {tract_id},")
             and "county" in census_df.columns
             and county_fp in str(census_df.loc[lbl, "county"]).zfill(3)),
            None,
        )
        p = tract_results.get(label, 0) if label else 0
        feature["properties"]["probability"] = f"{p:.1%}"
        return {"fillColor": get_color(p), "color": "#000", "weight": 0.8, "fillOpacity": 0.7}

    tract_field = "TRACTCE20" if "TRACTCE20" in map_data.columns else "TRACTCE"
    folium.GeoJson(
        map_data, name="Deal Acceptance",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[tract_field, "probability"],
            aliases=["Tract:", "Acceptance:"],
            style="background-color:white;color:#333;font-size:12px;padding:8px",
        ),
    ).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:50px;left:50px;width:210px;height:95px;'
        f'background:white;border:2px solid grey;z-index:9999;font-size:13px;padding:10px">'
        f'<b>Deal Acceptance Probability</b><br>'
        f'<span style="color:#440154">&#9632;</span> Low ({min_p:.1%})<br>'
        f'<span style="color:#31688e">&#9632;</span> Medium<br>'
        f'<span style="color:#fde725">&#9632;</span> High ({max_p:.1%})</div>'
    ))
    return m


def _render_wip_explorer_tab() -> None:
    st.markdown("## National Retrofit Deal Acceptance Simulator")
    st.markdown(
        "Select any US state to run the Monte Carlo logit simulation across its census tracts. "
        "**Massachusetts** uses a bundled local CSV; other states use the Census ACS 5-year API (2023)."
    )

    # ── Parameters panel ──────────────────────────────────────────────────────
    with st.expander("Scenario & behavioural parameters", expanded=True):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.markdown("**State & retrofit scenario**")
            selected_state = st.selectbox(
                "State", options=list(_US_STATES.keys()),
                index=list(_US_STATES.keys()).index("Massachusetts"),
                key="wip_state",
            )
            building_age = st.number_input(
                "Building age (years since built)", min_value=0, max_value=200, value=50, step=5,
                key="wip_age",
            )
            cost_min, cost_max = st.select_slider(
                "Upfront cost range (k$)",
                options=list(np.arange(0, 101, 5)),
                value=(10, 40),
                key="wip_cost",
            )
            energy_min, energy_max = st.select_slider(
                "Perceived annual energy savings range (hundreds $)",
                options=[round(x, 1) for x in np.arange(0, 11, 0.5)],
                value=(0.0, 10.0),
                key="wip_energy",
            )
        with p_col2:
            st.markdown("**Behavioural** _(not in census data)_")
            concern_level = st.slider("Environmental concern (1=Low, 5=High)", 1, 5, 3, key="wip_concern")
            neighbor_effect = st.slider("Neighbour adoption influence (1=Low, 5=High)", 1, 5, 3, key="wip_neighbor")
        with p_col3:
            st.markdown("**Run options**")
            st.caption("Pick a single census tract below, or run across the entire state.")

    state_fips = _US_STATES[selected_state]

    # ── Load census data ───────────────────────────────────────────────────────
    if selected_state == "Massachusetts" and _MA_CENSUS_CSV.is_file():
        with st.spinner("Loading Massachusetts census tracts…"):
            df = _wip_load_ma_census(str(_MA_CENSUS_CSV))
        if df is None:
            st.error(f"Could not read {_MA_CENSUS_CSV}")
            return
    else:
        with st.spinner(f"Loading census data for {selected_state} from ACS API…"):
            raw_df = _wip_fetch_census(state_fips)
        if raw_df is None:
            st.error("Failed to load census data. Check network connection.")
            return
        with st.spinner("Loading county names…"):
            county_names = _wip_fetch_county_names(state_fips)
        df = _wip_build_display_df(raw_df, county_names)

    n_tracts = len(df)
    n_counties = df["county_name"].nunique() if "county_name" in df.columns else "?"
    m1, m2 = st.columns(2)
    with m1:
        st.metric("Census Tracts", f"{n_tracts:,}")
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_WIP_TRACT_ROW_COUNT)
    with m2:
        st.metric("Counties", f"{n_counties}")
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_WIP_COUNTY_COUNT)

    # ── Single-tract selector + run buttons ───────────────────────────────────
    sorted_tracts = sorted(df.index.unique())
    st.divider()
    tc1, tc2, tc3 = st.columns([3, 1, 1])
    with tc1:
        selected_tract = st.selectbox("Census tract for single-tract simulation:", sorted_tracts, key="wip_tract")
    with tc2:
        run_single = st.button("Run single tract (10k draws)", type="primary", key="wip_run_single")
    with tc3:
        run_all = st.button("Run all tracts statewide (1k draws each)", type="secondary", key="wip_run_all")

    # ── Single-tract result ────────────────────────────────────────────────────
    if run_single:
        tract_data = df.loc[selected_tract]
        with st.spinner(f"Running 10,000 simulations for {selected_tract}…"):
            probs = _wip_run_simulation(
                tract_data, building_age, (cost_min, cost_max),
                (energy_min, energy_max), concern_level, neighbor_effect,
            )
        mean_p = np.mean(probs)
        st.markdown(f"#### Results for {selected_tract}")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric(
                "Mean acceptance (tract, sim draws)",
                f"{mean_p:.2%}",
                help=_WIP_SINGLE_TRACT_MEAN_ACCEPTANCE_HELP,
            )
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_SINGLE_TRACT_ACCEPTANCE)
        with r2:
            st.metric(
                "Median acceptance (tract, sim draws)",
                f"{np.median(probs):.2%}",
                help=_WIP_SINGLE_TRACT_MEAN_ACCEPTANCE_HELP,
            )
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_SINGLE_TRACT_ACCEPTANCE)
        with r3:
            st.metric("Std deviation (sim draws)", f"{np.std(probs):.3f}")
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_TRACT_DRAWS_STD)

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.hist(probs, bins=50, density=True, color="skyblue", edgecolor="black")
        ax1.axvline(mean_p, color="red", linestyle="--", label=f"Mean: {mean_p:.2%}")
        ax1.set_title("Distribution of Deal Acceptance Probabilities")
        ax1.set_xlabel("Probability of accepting deal")
        ax1.set_ylabel("Density")
        ax1.legend()
        st.pyplot(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(np.sort(probs), np.arange(len(probs)) / len(probs))
        ax2.set_title("CDF of Deal Acceptance Probabilities")
        ax2.set_xlabel("Probability")
        ax2.set_ylabel("Cumulative probability")
        ax2.grid(True, linestyle=":")
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Statewide result ───────────────────────────────────────────────────────
    if run_all:
        with st.spinner(f"Downloading {selected_state} shapefile…"):
            map_data = _wip_fetch_shapefile(state_fips)

        st.markdown(f"#### Statewide Analysis — {selected_state}")
        tract_results = _wip_run_all_tracts(
            df, building_age, (cost_min, cost_max),
            (energy_min, energy_max), concern_level, neighbor_effect,
        )
        st.session_state["wip_tract_results"] = tract_results
        st.session_state["wip_tract_state"] = selected_state
        st.session_state["wip_sim_params"] = {
            "building_age": building_age, "cost_range": (cost_min, cost_max),
            "energy_range": (energy_min, energy_max),
            "concern_level": concern_level, "neighbor_effect": neighbor_effect,
        }

        all_p = list(tract_results.values())
        min_p, max_p = min(all_p), max(all_p)
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Mean", f"{np.mean(all_p):.2%}")
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_STATEWIDE_MEAN)
        with s2:
            st.metric("Median", f"{np.median(all_p):.2%}")
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_STATEWIDE_MEDIAN)
        with s3:
            st.metric("Min", f"{min_p:.2%}")
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_STATEWIDE_MIN)
        with s4:
            st.metric("Max", f"{max_p:.2%}")
            with st.expander("Definition & formula", expanded=False):
                st.markdown(_FORMULA_MD_WIP_STATEWIDE_MAX)

        fig_d, ax_d = plt.subplots(figsize=(12, 5))
        ax_d.hist(all_p, bins=30, density=True, color="lightblue", edgecolor="navy", alpha=0.7)
        ax_d.axvline(np.mean(all_p), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(all_p):.2%}")
        ax_d.axvline(np.median(all_p), color="orange", linestyle="--", linewidth=2, label=f"Median: {np.median(all_p):.2%}")
        kde = stats.gaussian_kde(all_p)
        xr = np.linspace(min_p, max_p, 200)
        ax_d.plot(xr, kde(xr), "r-", linewidth=2, label="KDE")
        ax_d.set_title(f"Distribution — {selected_state} Census Tracts", fontsize=14, fontweight="bold")
        ax_d.set_xlabel("Probability of accepting deal")
        ax_d.set_ylabel("Density")
        ax_d.legend()
        ax_d.grid(True, alpha=0.3)
        ax_d.text(0.02, 0.98,
                  f"Std: {np.std(all_p):.3f}\n25th: {np.percentile(all_p, 25):.2%}\n75th: {np.percentile(all_p, 75):.2%}",
                  transform=ax_d.transAxes, va="top",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_d)
        plt.close(fig_d)

        # Choropleth map
        st.markdown("**Choropleth Map**")
        if map_data is not None:
            try:
                cmap_fig = _wip_choropleth(df, tract_results, map_data, selected_state)
                st_folium(cmap_fig, width=900, height=600, returned_objects=[])
            except Exception as exc:
                st.warning(f"Map error: {exc}")
        else:
            st.info("Shapefile could not be loaded — choropleth unavailable.")

        # Top / bottom tracts
        res_s = pd.Series(
            tract_results,
            name="Mean acceptance (tract avg., propensity sim draws)",
        )
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Top 10 tracts**")
            st.dataframe(res_s.sort_values(ascending=False).head(10).map("{:.2%}".format))
        with t2:
            st.markdown("**Bottom 10 tracts**")
            st.dataframe(res_s.sort_values(ascending=True).head(10).map("{:.2%}".format))

        export_df = pd.DataFrame({
            "tract": list(tract_results.keys()),
            "mean_acceptance_tract_avg_propensity_sim": list(tract_results.values()),
        })
        st.download_button(
            "Download results as CSV",
            export_df.to_csv(index=False),
            f"{selected_state.lower().replace(' ', '_')}_acceptance.csv", "text/csv",
        )

    # ── Adoption rate analysis ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Adoption Rate Analysis")
    st.caption("Requires statewide results — click 'Run all tracts statewide' above first.")

    years_list = list(range(2024, 2051))
    years_np = np.arange(2024, 2051)

    adopt_col1, adopt_col2 = st.columns([2, 1])
    with adopt_col1:
        adoption_scenario = st.selectbox(
            "Adoption scenario",
            ["Status Quo", "Disruptive Technology (S-Curve)", "Regulatory (Rapid Uptake)"],
            key="wip_adopt_scenario",
        )
        integration_method = st.selectbox(
            "Propensity integration method",
            ["set_threshold", "dice_roll", "ranked_distribution", "mc_ensemble"],
            format_func=lambda x: {
                "set_threshold": "M1: set threshold (willing pool + rank fill)",
                "dice_roll": "M2: dice roll (stochastic candidates + rank fill)",
                "ranked_distribution": "M3: ranked distribution (soft floor + deterministic)",
                "mc_ensemble": "M4: MC ensemble (dice roll repeated; mean + P10–P90 band)",
            }[x],
            key="wip_adopt_method",
        )
    with adopt_col2:
        threshold_pct = st.slider("M1 threshold (%)", 50, 95, 75, 5,
                                   disabled=(integration_method != "set_threshold"), key="wip_thresh")
        floor_pct = st.slider("M2/M3/M4 floor (%)", 0, 40, 10,
                               disabled=(integration_method not in ("dice_roll", "ranked_distribution", "mc_ensemble")),
                               key="wip_floor")
        n_ensemble = st.slider("M4 ensemble runs", 20, 300, 100, 10,
                                disabled=(integration_method != "mc_ensemble"), key="wip_ens")
        wip_dice_never = st.checkbox(
            "M2/M4: no re-entry after failed year",
            value=bool(st.session_state.get("wip_dice_reentry_never", True)),
            key="wip_dice_reentry_never",
            disabled=(integration_method not in ("dice_roll", "mc_ensemble")),
        )
        wip_dice_n = st.slider(
            "M2/M4: years to re-entry",
            1, 20, 1, 1,
            key="wip_dice_reentry_years",
            disabled=wip_dice_never or (integration_method not in ("dice_roll", "mc_ensemble")),
            help="After a year where a building could adopt but does not, wait this many years before the next draw.",
        )

    # Adoption curve reference chart
    fig_ac, ax_ac = plt.subplots(figsize=(10, 4))
    ax_ac.plot(years_np, [_wip_status_quo(y) for y in years_np], color="#440154", lw=2, label="Status Quo", marker="o", ms=3)
    ax_ac.plot(years_np, [_wip_s_curve(y) for y in years_np], color="#31688e", lw=2, label="S-Curve", marker="s", ms=3)
    ax_ac.plot(years_np, [_wip_regulatory(y) for y in years_np], color="#35b779", lw=2, label="Regulatory", marker="^", ms=3)
    ax_ac.set_title("Adoption scenarios (2024–2050)")
    ax_ac.set_ylim(0, 1)
    ax_ac.set_xlabel("Year")
    ax_ac.set_ylabel("Adoption rate")
    ax_ac.legend()
    ax_ac.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_ac)
    plt.close(fig_ac)

    if st.button("Calculate adoption rates by census tract", type="primary", key="wip_calc_adopt"):
        if "wip_tract_results" not in st.session_state:
            st.error("Run **all tracts (statewide)** first to generate propensity scores.")
        elif st.session_state.get("wip_tract_state") != selected_state:
            st.error("Stored results are for a different state — re-run statewide for this state.")
        else:
            tract_prop = {k: float(v) for k, v in st.session_state["wip_tract_results"].items()}
            n_tot = len(tract_prop)
            func_map = {
                "Status Quo": _wip_status_quo,
                "Disruptive Technology (S-Curve)": _wip_s_curve,
                "Regulatory (Rapid Uptake)": _wip_regulatory,
            }
            curve = _wip_curve_dict(func_map[adoption_scenario], years_list)
            thr, fl = threshold_pct / 100, floor_pct / 100

            with st.spinner("Running propensity integration…"):
                ensemble_lo = ensemble_hi = None
                wip_reentry: int | None
                if wip_dice_never:
                    wip_reentry = None
                else:
                    wip_reentry = int(wip_dice_n)
                st.session_state["wip_dice_reentry_never"] = wip_dice_never
                st.session_state["wip_dice_reentry_years"] = int(wip_dice_n)
                if integration_method == "set_threshold":
                    aby = _wip_integrate_threshold(tract_prop, curve, years_list, thr)
                    aggregate_pct = _wip_cumulative_pct(aby, years_list, n_tot)
                elif integration_method == "dice_roll":
                    aby = _wip_integrate_dice(
                        tract_prop,
                        curve,
                        years_list,
                        np.random.default_rng(42),
                        floor=fl,
                        dice_reentry_years=wip_reentry,
                    )
                    aggregate_pct = _wip_cumulative_pct(aby, years_list, n_tot)
                elif integration_method == "ranked_distribution":
                    aby = _wip_integrate_ranked(tract_prop, curve, years_list, fl)
                    aggregate_pct = _wip_cumulative_pct(aby, years_list, n_tot)
                else:
                    aggregate_pct, ensemble_lo, ensemble_hi, _ = _wip_integrate_ensemble(
                        tract_prop,
                        curve,
                        years_list,
                        n_ensemble,
                        fl,
                        seed=0,
                        dice_reentry_years=wip_reentry,
                    )

            fig_ar, ax_ar = plt.subplots(figsize=(12, 5))
            ax_ar.plot(years_list, aggregate_pct, color="#2563eb", lw=2.5, label="Cumulative adoption %")
            if ensemble_lo is not None:
                ax_ar.fill_between(years_list, ensemble_lo, ensemble_hi, alpha=0.2, color="#2563eb", label="P10–P90")
            ax_ar.set_title(
                f"Cumulative Adoption — {selected_state} | {adoption_scenario} | {integration_method}",
                fontsize=13, fontweight="bold",
            )
            ax_ar.set_xlabel("Year")
            ax_ar.set_ylabel("% of tracts adopted")
            ax_ar.set_ylim(0, 100)
            ax_ar.legend()
            ax_ar.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_ar)
            plt.close(fig_ar)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — VISUALIZE SAVED SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

# Alternate palette used for Scenario B in comparison overlay
_SCENARIO_PALETTE_B = [
    ("#9333ea", "rgba(147,51,234,0.20)"),
    ("#ea580c", "rgba(234,88,12,0.20)"),
    ("#0891b2", "rgba(8,145,178,0.20)"),
    ("#65a30d", "rgba(101,163,13,0.20)"),
    ("#db2777", "rgba(219,39,119,0.20)"),
    ("#854d0e", "rgba(133,77,14,0.20)"),
]


def _build_adoption_fig_from_df(adoption_df: pd.DataFrame, label_prefix: str = "") -> go.Figure:
    """Build an adoption trajectory figure from a flat saved-CSV DataFrame."""
    fig = go.Figure()
    if "adoption_scenario" not in adoption_df.columns:
        return fig
    for i, scen in enumerate(adoption_df["adoption_scenario"].unique()):
        sub = adoption_df[adoption_df["adoption_scenario"] == scen].sort_values("year")
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        name = f"{label_prefix}{scen}" if label_prefix else scen
        years = sub["year"].tolist()
        mean_s = sub["cumulative_adoption_pct"]
        p10_s = sub["cumulative_adoption_pct_p10"] if "cumulative_adoption_pct_p10" in sub.columns else None
        p90_s = sub["cumulative_adoption_pct_p90"] if "cumulative_adoption_pct_p90" in sub.columns else None
        _add_scenario_traces(fig, years, mean_s, p10_s, p90_s, name, line_color, band_rgba)
    fig.update_layout(
        yaxis=dict(title="Cumulative adoption (%)", range=[0, 105]),
        xaxis_title="Year", height=360,
        margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.22),
    )
    return fig


def _build_emissions_fig_from_df(emissions_df: pd.DataFrame, label_prefix: str = "") -> go.Figure:
    """Build an emissions trajectory figure from a flat saved-CSV DataFrame."""
    fig = go.Figure()
    if emissions_df.empty or "adoption_scenario" not in emissions_df.columns:
        return fig

    # Baseline is the same across adoption scenarios — draw it once
    first_sub = emissions_df[
        emissions_df["adoption_scenario"] == emissions_df["adoption_scenario"].iloc[0]
    ].sort_values("year")
    if "baseline_emissions_t" in first_sub.columns:
        baseline_label = f"{label_prefix}Baseline (0% adoption)" if label_prefix else "Baseline (0% adoption)"
        fig.add_trace(go.Scatter(
            x=first_sub["year"].tolist(),
            y=first_sub["baseline_emissions_t"].tolist(),
            mode="lines", name=baseline_label,
            line=dict(color="#dc2626", width=2.5, dash="dot"),
        ))

    for i, scen in enumerate(emissions_df["adoption_scenario"].unique()):
        sub = emissions_df[emissions_df["adoption_scenario"] == scen].sort_values("year")
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        name = f"{label_prefix}{scen}" if label_prefix else scen
        yrs = sub["year"].tolist()
        mean_vals = sub["scenario_mean_t"].tolist()
        lo_vals = sub["scenario_p10_t"].tolist() if "scenario_p10_t" in sub.columns else [v * 0.90 for v in mean_vals]
        hi_vals = sub["scenario_p90_t"].tolist() if "scenario_p90_t" in sub.columns else [v * 1.10 for v in mean_vals]
        fig.add_trace(go.Scatter(x=yrs, y=lo_vals, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 showlegend=False, hoverinfo="skip", legendgroup=name))
        fig.add_trace(go.Scatter(x=yrs, y=hi_vals, mode="lines",
                                 line=dict(color="rgba(0,0,0,0)"),
                                 fill="tonexty", fillcolor=band_rgba,
                                 showlegend=False, hoverinfo="skip", legendgroup=name))
        fig.add_trace(go.Scatter(x=yrs, y=mean_vals, mode="lines", name=name,
                                 line=dict(color=line_color, width=2.5), legendgroup=name))
    fig.update_layout(
        yaxis_title="tCO₂/yr", xaxis_title="Year", height=360,
        margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.22),
    )
    return fig


def _render_scenario_kpis(data: dict, label: str) -> None:
    """Display summary KPI metrics for a saved scenario."""
    meta = data.get("metadata", {})
    cols = st.columns(4)
    with cols[0]:
        st.metric("Scenario", label)
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_SAVED_SCENARIO_NAME)
    n_bldg = meta.get("n_buildings")
    with cols[1]:
        st.metric("Buildings", f"{n_bldg:,}" if n_bldg else "—")
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_BUILDINGS_MODELLED)
    mean_acc = meta.get("mean_acceptance_probability")
    mean_acc_def = meta.get("mean_acceptance_probability_definition")
    with cols[2]:
        st.metric(
            MEAN_ACCEPTANCE_AGGREGATE_LABEL,
            f"{mean_acc:.1%}" if mean_acc is not None else "—",
            help=mean_acc_def or MEAN_ACCEPTANCE_AGGREGATE_DESCRIPTION,
        )
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_MEAN_ACCEPTANCE_AGGREGATE)
    saved_at = meta.get("saved_at", "—")
    with cols[3]:
        st.metric("Saved", saved_at[:10] if saved_at != "—" else "—")
        with st.expander("Definition & formula", expanded=False):
            st.markdown(_FORMULA_MD_SAVED_AT)


def _render_visualize_tab() -> None:
    st.markdown("## Visualize Saved Scenarios")
    st.caption(
        "Select a previously saved scenario to review its results, "
        "or compare two scenarios side by side."
    )

    saved = _list_saved_scenarios()
    if not saved:
        st.info(
            "No saved scenarios yet. Run an analysis in **5 · Run & Results** "
            "and click **Save Results** to store outputs here."
        )
        return

    view_mode = st.radio(
        "View mode",
        ["Single scenario", "Side-by-side comparison"],
        horizontal=True,
        key="viz_mode",
    )

    # ── Single scenario ────────────────────────────────────────────────────────
    if view_mode == "Single scenario":
        selected = st.selectbox("Select scenario", saved, key="viz_single_select")
        if not selected:
            return
        data = _load_scenario_data(selected)
        if not data:
            st.error("Could not load scenario data.")
            return

        _render_scenario_kpis(data, selected)
        st.divider()

        adoption_df = data.get("adoption")
        emissions_df = data.get("emissions")

        if adoption_df is not None and not adoption_df.empty:
            st.markdown("### Adoption Trajectories")
            st.plotly_chart(
                _build_adoption_fig_from_df(adoption_df),
                use_container_width=True, key="viz_adoption",
            )

        if emissions_df is not None and not emissions_df.empty:
            st.divider()
            st.markdown("### Emissions Trajectories (tCO₂/yr)")
            st.plotly_chart(
                _build_emissions_fig_from_df(emissions_df),
                use_container_width=True, key="viz_emissions",
            )

        policy_df = data.get("policy_impacts")
        if policy_df is not None and not policy_df.empty:
            st.divider()
            st.markdown("### Policy Impacts (per building)")
            st.dataframe(policy_df.head(500), use_container_width=True)

    # ── Side-by-side comparison ────────────────────────────────────────────────
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            sel_a = st.selectbox("Scenario A", saved, key="viz_compare_a")
        with col_b:
            sel_b = st.selectbox(
                "Scenario B",
                saved,
                index=min(1, len(saved) - 1),
                key="viz_compare_b",
            )

        if not sel_a or not sel_b:
            return

        data_a = _load_scenario_data(sel_a)
        data_b = _load_scenario_data(sel_b)
        if not data_a or not data_b:
            st.error("Could not load one or both scenarios.")
            return

        # Metadata rows
        st.divider()
        mc_a, mc_b = st.columns(2)
        with mc_a:
            st.markdown(f"**{sel_a}**")
            _render_scenario_kpis(data_a, sel_a)
        with mc_b:
            st.markdown(f"**{sel_b}**")
            _render_scenario_kpis(data_b, sel_b)

        # Adoption side by side
        adop_a = data_a.get("adoption")
        adop_b = data_b.get("adoption")
        if adop_a is not None and adop_b is not None:
            st.divider()
            st.markdown("### Adoption Trajectories")
            ac1, ac2 = st.columns(2)
            with ac1:
                st.caption(sel_a)
                st.plotly_chart(
                    _build_adoption_fig_from_df(adop_a),
                    use_container_width=True, key="viz_cmp_adopt_a",
                )
            with ac2:
                st.caption(sel_b)
                st.plotly_chart(
                    _build_adoption_fig_from_df(adop_b),
                    use_container_width=True, key="viz_cmp_adopt_b",
                )

        # Emissions side by side
        em_a = data_a.get("emissions")
        em_b = data_b.get("emissions")
        if em_a is not None and em_b is not None:
            st.divider()
            st.markdown("### Emissions Trajectories (tCO₂/yr)")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.caption(sel_a)
                st.plotly_chart(
                    _build_emissions_fig_from_df(em_a),
                    use_container_width=True, key="viz_cmp_em_a",
                )
            with ec2:
                st.caption(sel_b)
                st.plotly_chart(
                    _build_emissions_fig_from_df(em_b),
                    use_container_width=True, key="viz_cmp_em_b",
                )

            # Overlay both scenarios on a single chart
            st.divider()
            st.markdown("### Overlay Comparison — Emissions")
            st.caption(
                f"Both scenarios overlaid for direct comparison. "
                f"Solid lines = **{sel_a}**, dashed lines = **{sel_b}**."
            )
            fig_ov = go.Figure()

            first_sub_a = em_a[em_a["adoption_scenario"] == em_a["adoption_scenario"].iloc[0]].sort_values("year")
            if "baseline_emissions_t" in first_sub_a.columns:
                fig_ov.add_trace(go.Scatter(
                    x=first_sub_a["year"].tolist(),
                    y=first_sub_a["baseline_emissions_t"].tolist(),
                    mode="lines", name="Baseline",
                    line=dict(color="#dc2626", width=2.5, dash="dot"),
                ))

            for i, scen in enumerate(em_a["adoption_scenario"].unique()):
                sub = em_a[em_a["adoption_scenario"] == scen].sort_values("year")
                lc, br = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
                name = f"{sel_a} · {scen}"
                yrs = sub["year"].tolist()
                mean_vals = sub["scenario_mean_t"].tolist()
                lo_v = sub["scenario_p10_t"].tolist() if "scenario_p10_t" in sub.columns else [v * 0.90 for v in mean_vals]
                hi_v = sub["scenario_p90_t"].tolist() if "scenario_p90_t" in sub.columns else [v * 1.10 for v in mean_vals]
                fig_ov.add_trace(go.Scatter(x=yrs, y=lo_v, mode="lines",
                                            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                                            hoverinfo="skip", legendgroup=name))
                fig_ov.add_trace(go.Scatter(x=yrs, y=hi_v, mode="lines",
                                            line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                                            fillcolor=br, showlegend=False, hoverinfo="skip", legendgroup=name))
                fig_ov.add_trace(go.Scatter(x=yrs, y=mean_vals, mode="lines", name=name,
                                            line=dict(color=lc, width=2.5), legendgroup=name))

            for i, scen in enumerate(em_b["adoption_scenario"].unique()):
                sub = em_b[em_b["adoption_scenario"] == scen].sort_values("year")
                lc, br = _SCENARIO_PALETTE_B[i % len(_SCENARIO_PALETTE_B)]
                name = f"{sel_b} · {scen}"
                yrs = sub["year"].tolist()
                mean_vals = sub["scenario_mean_t"].tolist()
                lo_v = sub["scenario_p10_t"].tolist() if "scenario_p10_t" in sub.columns else [v * 0.90 for v in mean_vals]
                hi_v = sub["scenario_p90_t"].tolist() if "scenario_p90_t" in sub.columns else [v * 1.10 for v in mean_vals]
                fig_ov.add_trace(go.Scatter(x=yrs, y=lo_v, mode="lines",
                                            line=dict(color="rgba(0,0,0,0)"), showlegend=False,
                                            hoverinfo="skip", legendgroup=name))
                fig_ov.add_trace(go.Scatter(x=yrs, y=hi_v, mode="lines",
                                            line=dict(color="rgba(0,0,0,0)"), fill="tonexty",
                                            fillcolor=br, showlegend=False, hoverinfo="skip", legendgroup=name))
                fig_ov.add_trace(go.Scatter(x=yrs, y=mean_vals, mode="lines", name=name,
                                            line=dict(color=lc, width=2.5, dash="dash"), legendgroup=name))

            fig_ov.update_layout(
                yaxis_title="tCO₂/yr", xaxis_title="Year", height=420,
                margin=dict(t=20, b=20), legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_ov, use_container_width=True, key="viz_overlay_em")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _render_config_manager()

    st.title("Willingness-to-Pay & Adoption Analysis")
    st.markdown(
        "Analyse building retrofit adoption and emissions trajectories using "
        "energy simulation output from [globi](https://github.com/globi)."
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "1 · Upload Data",
        "2 · Configure",
        "3 · Adoption Curves",
        "4 · Emissions",
        "5 · Run & Results",
        "Visualize",
        "WIP Explorer",
    ])

    with tab1:
        _render_upload_tab()
    with tab2:
        _render_config_tab()
    with tab3:
        _render_adoption_tab()
    with tab4:
        _render_emissions_tab()
    with tab5:
        _render_results_tab()
    with tab6:
        _render_visualize_tab()
    with tab7:
        _render_wip_explorer_tab()


if __name__ == "__main__":
    main()
