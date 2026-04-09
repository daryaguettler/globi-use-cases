"""Willingness-to-Pay & Adoption Analysis — Streamlit Application.

Workflow:
  1. Upload Data        — Baseline + Scenario EnergyAndPeak.pq files (one per year)
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
import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add repo root to path so use_cases imports work
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from app.analysis.energy_delta import (
    DEFAULT_ENERGY_PRICES,
    FUEL_LABELS,
    build_policy_impacts,
    load_energy_parquet,
)
from app.analysis.emissions_calc import compute_emissions_trajectory
from use_cases.apply_propensity import PropensityModelEngine
from use_cases.apply_uptake import AdoptionEngine

# ── Paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = _REPO_ROOT / "data" / "inputs"
_ADOPTION_CURVES_PATH = _DATA_DIR / "adoption_curves.json"
_EMISSIONS_PATH = _DATA_DIR / "emissions_trajectories.json"

_YEARS_RANGE = list(range(2024, 2061))
_PROJECTION_YEARS = list(range(2025, 2051))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WTP & Adoption Analysis | globi",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ─────────────────────────────────────────────────────
_STATE_DEFAULTS: dict = {
    # {year: {"baseline": bytes, "scenario": bytes}} — one pair per simulated year
    "year_files": {},
    "selected_years": [2025],
    "n_years": 1,
    "scenario_name": "Retrofit",
    "policy_impacts": None,       # keyed to the reference (earliest) year
    "propensity_result": None,
    # {adoption_scenario_name: {"uptake": UptakeResult, "emissions": pd.DataFrame}}
    "scenario_results": {},
    "run_complete": False,
}
for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Example data preload ───────────────────────────────────────────────────────
_EXAMPLE_BASELINE = _DATA_DIR / "globi_outputs" / "split" / "Baseline_EnergyAndPeak.pq"
_EXAMPLE_SCENARIO = _DATA_DIR / "globi_outputs" / "split" / "ASHP_EnergyAndPeak.pq"
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
    st.session_state["scenario_name"] = "ASHP"
    st.session_state["_example_preloaded"] = True


# ── JSON loaders ───────────────────────────────────────────────────────────────

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


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _render_upload_tab() -> None:
    st.markdown("## Step 1 — Upload Energy Data")
    st.markdown(
        "Upload a **Baseline** and **Scenario** `EnergyAndPeak.pq` for each simulated year. "
        "You can have results for one year or many — each year's simulated savings anchors the "
        "timeseries at that point. The earliest year is used as the reference for per-building "
        "WTP scoring; adoption and emissions are projected forward analytically."
    )

    # ── Example data notice ────────────────────────────────────────────────────
    if st.session_state.get("_example_preloaded"):
        st.info(
            "**Example data pre-loaded** — `Baseline_EnergyAndPeak.pq` and "
            "`ASHP_EnergyAndPeak.pq` from `data/inputs/globi_outputs/split/` are "
            "ready to use. Jump straight to **5 · Run & Results** to see the full "
            "pipeline, or upload your own files below to replace them.",
            icon="ℹ️",
        )

    # ── Scenario name + number of years ───────────────────────────────────────
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

    # ── Per-year file uploaders ────────────────────────────────────────────────
    st.divider()
    year_files: dict = st.session_state.get("year_files", {})

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
                        _show_file_kpis(info, "Baseline")
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
                        _show_file_kpis(info, st.session_state["scenario_name"])
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


def _show_file_kpis(info: dict, label: str) -> None:
    k1, k2, k3 = st.columns(3)
    k1.metric("Buildings", info.get("n_buildings", "—"))
    area = info.get("total_area_m2")
    k2.metric("Total area", f"{area:,.0f} m²" if area else "—")
    scens = info.get("scenarios", [])
    k3.metric("Scenario(s) in file", ", ".join(scens) if scens else "—")


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

    # ── Energy prices ──────────────────────────────────────────────────────────
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
            "Census tract demographics will be looked up automatically from building "
            "lat/lon coordinates in the uploaded files.  Optionally upload a pre-built "
            "census CSV (same format as `data/inputs/climate_opinions.csv`) to skip "
            "API calls."
        )
        census_file = st.file_uploader(
            "Census CSV (optional — leave blank for API lookup)",
            type=["csv"], key="cfg_census_csv",
        )
        if census_file is not None:
            st.session_state["census_csv_bytes"] = census_file.read()
            st.success("Census CSV uploaded.")
        else:
            st.session_state.setdefault("census_csv_bytes", None)

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
    for k, v in [("n_mc_samples", n_mc), ("n_ensemble_runs", n_ensemble), ("random_seed", seed)]:
        st.session_state[k] = int(v)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ADOPTION CURVES
# ══════════════════════════════════════════════════════════════════════════════

def _ac_compute_curve(shape: str, start: float, end: float, n: int) -> np.ndarray:
    xs = np.linspace(0, 1, n)
    if shape == "Linear":
        return start + (end - start) * xs
    if shape == "Sigmoid":
        sig = 1 / (1 + np.exp(-10 * (xs - 0.5)))
        sig = (sig - sig[0]) / (sig[-1] - sig[0] + 1e-9)
        return start + (end - start) * sig
    # Exponential
    exp_raw = np.expm1(xs * 3)
    exp_raw = exp_raw / (exp_raw[-1] + 1e-9)
    return start + (end - start) * exp_raw


def _render_adoption_tab() -> None:
    st.markdown("## Step 3 — Adoption Curves")

    raw = _load_adoption_curves()
    scenarios: dict = raw.get("scenarios", {})
    curve_years = list(range(2024, 2051))
    colors = _curve_colors()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Manage scenarios (add / remove)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Manage Scenarios")
    st.caption("Add new scenarios or remove existing ones before parametrizing.")

    # List existing scenarios with remove buttons
    if scenarios:
        for sname in list(scenarios.keys()):
            row_col, btn_col = st.columns([6, 1], vertical_alignment="center")
            with row_col:
                desc = scenarios[sname].get("description", "")
                st.markdown(f"**{sname}**" + (f" — _{desc}_" if desc else ""))
            with btn_col:
                remove_disabled = len(scenarios) <= 1
                if st.button(
                    "Remove", key=f"rm_{sname}",
                    disabled=remove_disabled,
                    help="Cannot remove the last scenario." if remove_disabled else None,
                ):
                    del raw["scenarios"][sname]
                    _save_adoption_curves(raw)
                    # Switch active selection away from deleted scenario
                    if st.session_state.get("adoption_scenario") == sname:
                        remaining = [k for k in raw.get("scenarios", {}) if k != sname]
                        st.session_state["adoption_scenario"] = remaining[0] if remaining else None
                    st.rerun()
    else:
        st.info("No scenarios yet — add one below to get started.")

    # Add new scenario row
    add_col1, add_col2, _ = st.columns([3, 1, 4], vertical_alignment="bottom")
    with add_col1:
        new_name = st.text_input(
            "New scenario name", placeholder="e.g. fast_2030", key="ac_new_name",
        )
    with add_col2:
        if st.button("Add scenario", key="ac_add_btn", type="secondary"):
            n = new_name.strip()
            if not n:
                st.error("Enter a name.")
            elif n in scenarios:
                st.error(f"'{n}' already exists.")
            else:
                template = {
                    "description": "", "curve_type": "linear",
                    "max_adoption": 0.85, "annual_attrition": 0.05,
                    "values": {str(y): 0.0 for y in curve_years},
                }
                raw["scenarios"][n] = template
                _save_adoption_curves(raw)
                st.session_state["adoption_scenario"] = n
                st.rerun()

    if not scenarios:
        st.divider()
        _ac_render_projection_period()
        return

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Parametrize selected scenario
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Parametrize")
    st.caption(
        "Select a scenario and adjust its shape. Changes preview live — click "
        "**Save curve** to persist."
    )

    scenario_names = list(scenarios.keys())
    default_sel = st.session_state.get("adoption_scenario")
    sel_idx = scenario_names.index(default_sel) if default_sel in scenario_names else 0
    selected = st.selectbox(
        "Scenario to edit", options=scenario_names,
        index=sel_idx, key="ac_selected",
    )
    st.session_state["adoption_scenario"] = selected

    scen_data = copy.deepcopy(scenarios[selected])

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        shape_opts = ["Linear", "Sigmoid", "Exponential"]
        saved_shape = scen_data.get("curve_type", "linear").capitalize()
        qf_shape = st.selectbox(
            "Curve shape", shape_opts,
            index=shape_opts.index(saved_shape) if saved_shape in shape_opts else 0,
            key="qf_shape",
            help="Linear: constant rate. Sigmoid: slow start, fast middle, slow end. Exponential: accelerating.",
        )
    with p_col2:
        qf_start = st.number_input(
            "Start adoption (2024)", min_value=0.0, max_value=1.0,
            value=float(scen_data.get("values", {}).get("2024", 0.0)),
            step=0.01, format="%.2f", key="qf_start",
        )
    with p_col3:
        qf_end = st.number_input(
            "End adoption (2050)", min_value=0.0, max_value=1.0,
            value=float(scen_data.get("values", {}).get("2050", scen_data.get("max_adoption", 0.85))),
            step=0.01, format="%.2f", key="qf_end",
        )

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        new_max = st.number_input(
            "Max adoption (cap)", min_value=0.0, max_value=1.0,
            value=float(scen_data.get("max_adoption", 0.85)),
            step=0.01, format="%.2f", key="ac_max",
            help="Hard ceiling — the curve is capped at this fraction regardless of shape.",
        )
    with m_col2:
        new_attr = st.number_input(
            "Annual attrition", min_value=0.0, max_value=1.0,
            value=float(scen_data.get("annual_attrition", 0.05)),
            step=0.01, format="%.2f", key="ac_attr",
            help="Fraction of adopted buildings that revert each year.",
        )
    with m_col3:
        new_desc = st.text_input(
            "Description", value=scen_data.get("description", ""), key="ac_desc",
        )

    # ── Live preview chart ─────────────────────────────────────────────────────
    preview_ys = np.clip(_ac_compute_curve(qf_shape, qf_start, qf_end, len(curve_years)), 0.0, new_max)

    fig = go.Figure()
    for i, (name, s) in enumerate(scenarios.items()):
        if name == selected:
            continue
        saved_ys = [float(s.get("values", {}).get(str(y), 0.0)) for y in curve_years]
        fig.add_trace(go.Scatter(
            x=curve_years, y=[v * 100 for v in saved_ys],
            name=name,
            line=dict(color=colors[i % len(colors)], width=1.5, dash="dot"),
            opacity=0.4,
        ))
    fig.add_trace(go.Scatter(
        x=curve_years, y=[v * 100 for v in preview_ys],
        name=f"{selected} (preview)",
        line=dict(color="#2563eb", width=3),
    ))
    fig.update_layout(
        yaxis=dict(title="Adoption (%)", range=[0, 105]),
        xaxis_title="Year",
        legend_title="Scenario",
        height=340, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Save / reset ───────────────────────────────────────────────────────────
    save_col, reset_col, _ = st.columns([1, 1, 4], vertical_alignment="center")
    with save_col:
        if st.button("Save curve", type="primary", key="ac_save"):
            scen_data["values"] = {str(y): round(float(v), 4) for y, v in zip(curve_years, preview_ys)}
            scen_data.update({
                "description": new_desc,
                "max_adoption": new_max,
                "annual_attrition": new_attr,
                "curve_type": qf_shape.lower(),
            })
            raw["scenarios"][selected] = scen_data
            _save_adoption_curves(raw)
            st.success(f"Saved '{selected}'.")
            st.rerun()
    with reset_col:
        if st.button("Reset to saved", key="ac_reset"):
            st.rerun()

    st.divider()
    _ac_render_projection_period()


def _ac_render_projection_period() -> None:
    st.markdown("### Projection Period")
    py_col1, py_col2 = st.columns(2)
    with py_col1:
        start_year = st.number_input(
            "Start year", 2024, 2060,
            int(st.session_state.get("proj_start_year", 2025)), 1,
            key="cfg_start_year",
        )
    with py_col2:
        end_year = st.number_input(
            "End year", 2025, 2070,
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
    plot_years = list(range(2024, 2051))

    # Per-fuel editors
    for fuel in FUEL_LABELS:
        with st.expander(f"{fuel}", expanded=False):
            fuel_cfg = fuels_cfg.get(fuel, {"values": {}, "unit": "kg CO2/kWh"})
            values = fuel_cfg.get("values", {})

            # Quick set: start + end values + shape
            qe_col1, qe_col2, qe_col3 = st.columns(3)
            current_start = float(values.get("2024", 0.3))
            current_end = float(values.get("2050", 0.1))
            with qe_col1:
                v_start = st.number_input(f"{fuel} — 2024 value (kg CO2/kWh)",
                                          0.0, 5.0, current_start, 0.01,
                                          key=f"em_{fuel}_start")
                v_end = st.number_input(f"{fuel} — 2050 value (kg CO2/kWh)",
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
    ref_year = simulated_years[0]  # earliest year — reference for per-building WTP

    # ── 1. Build policy_impacts from the reference (earliest) simulated year ──
    progress.progress(5, text=f"Computing energy savings (reference year {ref_year})…")
    try:
        entry = year_files[ref_year]
        base_flat = load_energy_parquet(entry["baseline"])
        scen_flat = load_energy_parquet(entry["scenario"])
        policy_impacts = build_policy_impacts(
            baseline_df=base_flat,
            scenario_df=scen_flat,
            scenario_name=st.session_state["scenario_name"],
            cost_per_sqm=float(st.session_state.get("cost_per_sqm", 150.0)),
            energy_prices=st.session_state.get("energy_prices"),
        )
    except Exception as exc:
        st.error(f"Energy delta failed for {ref_year}: {exc}")
        return

    st.session_state["policy_impacts"] = policy_impacts
    st.session_state["simulated_years"] = simulated_years
    progress.progress(25, text=f"Energy savings computed for {len(policy_impacts)} buildings.")

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

    # ── 3. Propensity model ───────────────────────────────────────────────────
    progress.progress(40, text="Running WTP propensity model (MC ensemble)…")
    try:
        census_path = None
        if st.session_state.get("census_csv_bytes"):
            import tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
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
            # Apply user-specified priors by injecting synthetic distributions
            _apply_non_us_priors(propensity_engine)

        propensity_result = propensity_engine.calculate_all_probabilities()
        st.session_state["propensity_result"] = propensity_result
        if census_path:
            os.unlink(census_path)
    except Exception as exc:
        st.error(f"Propensity model failed: {exc}")
        return
    progress.progress(65, text="Propensity model complete.")

    # Join acceptance_probability into policy_impacts so the emissions calculation
    # can rank buildings by propensity and correctly propagate MC uncertainty.
    if "building.id" in propensity_result.data.columns and "building.id" in policy_impacts.columns:
        prop_scores = propensity_result.data[["building.id", "acceptance_probability"]].drop_duplicates("building.id")
        policy_impacts = policy_impacts.merge(prop_scores, on="building.id", how="left")
        st.session_state["policy_impacts"] = policy_impacts

    # ── 4 & 5. Uptake + emissions for every adoption scenario ─────────────────
    start_yr = int(st.session_state.get("proj_start_year", 2025))
    end_yr = int(st.session_state.get("proj_end_year", 2050))
    projection_years = list(range(start_yr, end_yr + 1))
    emissions_json = _load_emissions_json()

    raw_curves = _load_adoption_curves()
    all_adoption_scenarios = list(raw_curves.get("scenarios", {}).keys())
    if not all_adoption_scenarios:
        st.error("No adoption curve scenarios found. Define them in Step 3.")
        return

    scenario_results: dict = {}
    n_scen = len(all_adoption_scenarios)
    for i, adoption_scenario_name in enumerate(all_adoption_scenarios):
        pct_start = 70 + int(28 * i / n_scen)
        pct_end = 70 + int(28 * (i + 1) / n_scen)
        progress.progress(pct_start, text=f"Projecting '{adoption_scenario_name}' ({i + 1}/{n_scen})…")
        try:
            uptake_engine = AdoptionEngine(
                propensity_df=propensity_result.data,
                adoption_rates_path=_ADOPTION_CURVES_PATH,
                adoption_scenario=adoption_scenario_name,
                method="mc_ensemble",
                start_year=start_yr,
                end_year=end_yr,
                n_ensemble_runs=int(st.session_state.get("n_ensemble_runs", 100)),
                random_seed=int(st.session_state.get("random_seed", 42)),
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
            )
        except Exception as exc:
            st.warning(f"Emissions for '{adoption_scenario_name}' failed: {exc}")
            emissions_df = pd.DataFrame()

        scenario_results[adoption_scenario_name] = {
            "uptake": uptake_result,
            "emissions": emissions_df,
        }
        progress.progress(pct_end)

    if not scenario_results:
        st.error("All adoption scenarios failed — check inputs.")
        return

    st.session_state["scenario_results"] = scenario_results
    progress.progress(100, text="Done.")
    st.session_state["run_complete"] = True
    st.success(f"Analysis complete — {len(scenario_results)} adoption scenario(s) projected.")
    st.rerun()


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
    ("#2563eb", "rgba(37,99,235,0.12)"),
    ("#16a34a", "rgba(22,163,74,0.12)"),
    ("#9333ea", "rgba(147,51,234,0.12)"),
    ("#ea580c", "rgba(234,88,12,0.12)"),
    ("#0891b2", "rgba(8,145,178,0.12)"),
    ("#65a30d", "rgba(101,163,13,0.12)"),
    ("#db2777", "rgba(219,39,119,0.12)"),
    ("#854d0e", "rgba(133,77,14,0.12)"),
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
    if p10_pct is not None and p90_pct is not None:
        fig.add_trace(go.Scatter(
            x=years + years[::-1],
            y=(p90_pct * y_scale).tolist() + (p10_pct * y_scale).iloc[::-1].tolist(),
            fill="toself", fillcolor=band_rgba,
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=years, y=(mean_pct * y_scale).tolist(),
        mode="lines", name=name,
        line=dict(color=line_color, width=2.5),
    ))


# ── Results charts ─────────────────────────────────────────────────────────────

def _render_result_charts() -> None:
    scenario_results: dict = st.session_state["scenario_results"]
    propensity_result = st.session_state["propensity_result"]
    policy_impacts: pd.DataFrame = st.session_state["policy_impacts"]
    retrofit_name = st.session_state["scenario_name"]
    simulated_years: list[int] = st.session_state.get("simulated_years", [])

    # Grab baseline from first available emissions df (same across scenarios)
    first_em: pd.DataFrame = next(
        (v["emissions"] for v in scenario_results.values() if not v["emissions"].empty),
        pd.DataFrame(),
    )

    st.divider()

    # ── KPI row ────────────────────────────────────────────────────────────────
    n_buildings = len(policy_impacts)
    mean_prop = float(propensity_result.mean_acceptance_probability)

    k1, k2 = st.columns(2)
    k1.metric("Buildings modelled", f"{n_buildings:,}")
    k2.metric("Mean acceptance probability", f"{mean_prop:.1%}")

    # Per-scenario summary table
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
        st.dataframe(
            pd.DataFrame(summary_rows).set_index("Adoption scenario"),
            use_container_width=True,
        )

    st.divider()

    # ── Adoption trajectories — all scenarios with bands ──────────────────────
    st.markdown(f"### Adoption Trajectories — {retrofit_name}")
    fig_adopt = go.Figure()
    for i, (name, res) in enumerate(scenario_results.items()):
        ys = res["uptake"].yearly_summary.sort_values("year")
        if ys.empty:
            continue
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        years = ys["year"].tolist()
        mean_s = ys["cumulative_adoption_pct"]
        p10_s = ys["cumulative_adoption_pct_p10"] if "cumulative_adoption_pct_p10" in ys.columns else None
        p90_s = ys["cumulative_adoption_pct_p90"] if "cumulative_adoption_pct_p90" in ys.columns else None
        _add_scenario_traces(fig_adopt, years, mean_s, p10_s, p90_s, name, line_color, band_rgba)
    for yr in simulated_years:
        fig_adopt.add_vline(
            x=yr, line_dash="dot", line_color="#94a3b8", line_width=1,
            annotation_text=str(yr), annotation_position="top",
            annotation_font_size=11, annotation_font_color="#64748b",
        )
    fig_adopt.update_layout(
        yaxis=dict(title="Cumulative adoption (%)", range=[0, 105]),
        xaxis_title="Year",
        height=380, margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_adopt, use_container_width=True)

    # ── Emissions trajectories — all scenarios with bands + baseline ───────────
    st.divider()
    st.markdown("### Emissions Trajectories (metric tonnes CO₂/yr)")
    fig_em = go.Figure()
    if not first_em.empty:
        fig_em.add_trace(go.Scatter(
            x=first_em["year"].tolist(), y=first_em["baseline_emissions_t"].tolist(),
            mode="lines", name="Baseline (no retrofit)",
            line=dict(color="#dc2626", width=2, dash="dash"),
        ))
    for i, (name, res) in enumerate(scenario_results.items()):
        em = res["emissions"]
        if em.empty:
            continue
        line_color, band_rgba = _SCENARIO_PALETTE[i % len(_SCENARIO_PALETTE)]
        years = em["year"].tolist()
        p10 = em["scenario_p10_t"] if "scenario_p10_t" in em.columns else None
        p90 = em["scenario_p90_t"] if "scenario_p90_t" in em.columns else None
        _add_scenario_traces(fig_em, years, em["scenario_mean_t"], p10, p90, name, line_color, band_rgba)
    for yr in simulated_years:
        fig_em.add_vline(
            x=yr, line_dash="dot", line_color="#94a3b8", line_width=1,
            annotation_text=str(yr), annotation_position="top",
            annotation_font_size=11, annotation_font_color="#64748b",
        )
    fig_em.update_layout(
        yaxis_title="tCO₂/yr", xaxis_title="Year",
        height=380, margin=dict(t=20, b=20),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_em, use_container_width=True)

    # ── Propensity distribution + energy savings scatter ──────────────────────
    st.divider()
    col_p, col_e = st.columns(2)

    with col_p:
        st.markdown("### Propensity Distribution")
        probs = propensity_result.data["acceptance_probability"].dropna()
        if not probs.empty:
            fig_hist = go.Figure(go.Histogram(x=probs, nbinsx=40, marker_color="#2563eb", opacity=0.75))
            fig_hist.update_layout(
                xaxis_title="Acceptance probability", yaxis_title="Count",
                height=300, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_e:
        st.markdown("### Energy Savings vs. Propensity")
        if "acceptance_probability" not in policy_impacts.columns:
            plot_df = policy_impacts.join(
                propensity_result.data.set_index("building.id")[["acceptance_probability"]],
                on="building.id", how="left",
            )
        else:
            plot_df = policy_impacts
        if "acceptance_probability" in plot_df.columns and "energy_cost.annual_savings" in plot_df.columns:
            fig_scatter = go.Figure(go.Scatter(
                x=plot_df["energy_cost.annual_savings"],
                y=plot_df["acceptance_probability"],
                mode="markers",
                marker=dict(color="#9333ea", size=4, opacity=0.5),
            ))
            fig_scatter.update_layout(
                xaxis_title="Annual energy savings ($/yr)",
                yaxis_title="Acceptance probability",
                height=300, margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("🏗️ Willingness-to-Pay & Adoption Analysis")
    st.markdown(
        "Analyse building retrofit adoption and emissions trajectories using "
        "energy simulation output from [globi](https://github.com/globi)."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1 · Upload Data",
        "2 · Configure",
        "3 · Adoption Curves",
        "4 · Emissions",
        "5 · Run & Results",
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


if __name__ == "__main__":
    main()
