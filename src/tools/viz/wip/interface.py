"""Streamlit editor for adoption curves, retrofit costs, and emissions trajectories.

Lets users interactively edit:
  1. Adoption curve scenarios in data/use_case_inputs/adoption_curves.json
  2. Retrofit cost per m² per scenario (saved into the same JSON)
  3. Emissions factor trajectories by fuel type over time
     (saved to data/use_case_inputs/emissions_trajectories.json)

Run with::

    uv run streamlit run scripts/scenario_editor.py
"""

from __future__ import annotations

import io
import json
import copy
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_ADOPTION_CURVES_PATH = _REPO_ROOT / "data" / "use_case_inputs" / "adoption_curves.json"
_EMISSIONS_PATH = _REPO_ROOT / "data" / "use_case_inputs" / "emissions_trajectories.json"

# ── Constants ─────────────────────────────────────────────────────────────────
_FUEL_LABELS = ("Electricity", "Natural Gas", "Fuel Oil", "Propane")
_DEFAULT_EMISSIONS_2025 = {"Electricity": 0.40, "Natural Gas": 0.20, "Fuel Oil": 0.27, "Propane": 0.23}
_DEFAULT_EMISSIONS_2050 = {"Electricity": 0.10, "Natural Gas": 0.18, "Fuel Oil": 0.25, "Propane": 0.21}
_YEARS = list(range(2024, 2051))

_CURVE_COLORS = [
    "#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c",
    "#0891b2", "#65a30d", "#db2777",
]


# ── JSON I/O ──────────────────────────────────────────────────────────────────

def _load_adoption_curves() -> dict:
    if _ADOPTION_CURVES_PATH.exists():
        with open(_ADOPTION_CURVES_PATH) as f:
            return json.load(f)
    return {"scenarios": {}, "retrofit_specific_modifiers": {}, "income_bracket_modifiers": {}}


def _save_adoption_curves(data: dict) -> None:
    with open(_ADOPTION_CURVES_PATH, "w") as f:
        json.dump(data, f, indent=4)


def _load_emissions() -> dict:
    if _EMISSIONS_PATH.exists():
        with open(_EMISSIONS_PATH) as f:
            return json.load(f)
    # Build defaults: linear interpolation 2025→2050 for each fuel
    scenarios: dict = {}
    for fuel in _FUEL_LABELS:
        start = _DEFAULT_EMISSIONS_2025[fuel]
        end = _DEFAULT_EMISSIONS_2050[fuel]
        values = {
            str(y): round(start + (end - start) * (y - 2024) / (2050 - 2024), 4)
            for y in _YEARS
        }
        scenarios[fuel] = {"values": values, "unit": "kg CO2/kWh"}
    return {"description": "Emissions factor trajectories by fuel type", "fuels": scenarios}


def _save_emissions(data: dict) -> None:
    with open(_EMISSIONS_PATH, "w") as f:
        json.dump(data, f, indent=4)


# ── Tab 1: Adoption Curves ────────────────────────────────────────────────────

def _ac_scenario_selector(scenarios: dict, raw: dict) -> str | None:
    """Render scenario dropdown + add-new form. Returns selected name or None."""
    scenario_names = list(scenarios.keys())
    if not scenario_names:
        st.warning("No scenarios found in adoption_curves.json.")

    col_sel, col_new = st.columns([3, 2])
    with col_sel:
        selected = st.selectbox(
            "Scenario to edit",
            options=scenario_names or [],
            key="ac_selected_scenario",
        ) if scenario_names else None

    with col_new:
        new_name = st.text_input("New scenario name", key="ac_new_name", placeholder="e.g. accelerated_2028")
        if st.button("Add scenario", key="ac_add_btn"):
            n = new_name.strip()
            if not n:
                st.error("Enter a name first.")
            elif n in scenarios:
                st.error(f"'{n}' already exists.")
            else:
                template = copy.deepcopy(next(iter(scenarios.values()))) if scenarios else {
                    "description": "", "curve_type": "linear",
                    "max_adoption": 0.85, "annual_attrition": 0.05,
                    "values": {str(y): 0.0 for y in _YEARS},
                }
                template["description"] = f"Custom scenario: {n}"
                scenarios[n] = template
                raw["scenarios"] = scenarios
                _save_adoption_curves(raw)
                st.success(f"Added '{n}'. Select it from the dropdown.")
                st.rerun()

    return selected


def _ac_metadata_form(scen_data: dict) -> tuple[str, float, float, str]:
    """Render metadata expander. Returns (description, max_adoption, attrition, curve_type)."""
    with st.expander("Scenario metadata", expanded=False):
        new_desc = st.text_input("Description", value=scen_data.get("description", ""), key="ac_desc")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            new_max = st.number_input("Max adoption", 0.0, 1.0, float(scen_data.get("max_adoption", 0.85)), 0.01, key="ac_max")
        with col_m2:
            new_attr = st.number_input("Annual attrition", 0.0, 1.0, float(scen_data.get("annual_attrition", 0.05)), 0.01, key="ac_attr")
        with col_m3:
            _CTYPES = ["linear", "sigmoid", "exponential", "custom"]
            ctype_val = scen_data.get("curve_type", "linear")
            new_ctype = st.selectbox(
                "Curve type", _CTYPES,
                index=_CTYPES.index(ctype_val) if ctype_val in _CTYPES else 0,
                key="ac_ctype",
            )
    return new_desc, new_max, new_attr, new_ctype


def _ac_curve_shape_fill(selected: str, scen_data: dict, raw: dict) -> None:
    """Render quick-fill expander; saves to disk and reruns on apply."""
    with st.expander("Set curve shape", expanded=True):
        qf_col1, qf_col2, qf_col3 = st.columns(3)
        with qf_col1:
            qf_start = st.number_input("Start value (2024)", 0.0, 1.0, 0.0, 0.01, key="qf_start")
            qf_end = st.number_input("End value (2050)", 0.0, 1.0, 0.85, 0.01, key="qf_end")
        with qf_col2:
            qf_shape = st.selectbox("Shape", ["Linear", "Sigmoid", "Exponential"], key="qf_shape")
        with qf_col3:
            st.markdown("")
            st.markdown("")
            apply_fill = st.button("Apply & save", key="qf_apply", type="primary")

        if apply_fill:
            xs = np.linspace(0, 1, len(_YEARS))
            if qf_shape == "Linear":
                ys = qf_start + (qf_end - qf_start) * xs
            elif qf_shape == "Sigmoid":
                sig = 1 / (1 + np.exp(-10 * (xs - 0.5)))
                sig = (sig - sig[0]) / (sig[-1] - sig[0])
                ys = qf_start + (qf_end - qf_start) * sig
            else:
                exp_raw = np.expm1(xs * 3)
                exp_raw = exp_raw / exp_raw[-1]
                ys = qf_start + (qf_end - qf_start) * exp_raw
            scen_data["values"] = {str(y): round(float(v), 4) for y, v in zip(_YEARS, ys, strict=True)}
            raw["scenarios"][selected] = scen_data
            _save_adoption_curves(raw)
            st.success(f"Saved '{selected}'.")
            st.rerun()


def _ac_retrofit_modifiers(raw: dict) -> None:
    """Render and save the retrofit-specific multipliers table."""
    st.divider()
    st.markdown("### Retrofit-specific modifiers")
    st.markdown("Multipliers applied on top of the base adoption curve per retrofit type. Values > 1 mean faster adoption; < 1 means slower.")

    modifiers: dict = raw.get("retrofit_specific_modifiers", {})
    mod_rows = [
        {"Retrofit type": k, "Multiplier": float(v.get("adoption_multiplier", 1.0)), "Description": v.get("description", "")}
        for k, v in modifiers.items()
    ]
    mod_df = pd.DataFrame(mod_rows) if mod_rows else pd.DataFrame(columns=["Retrofit type", "Multiplier", "Description"])

    edited_mods = st.data_editor(
        mod_df, use_container_width=True, num_rows="dynamic", key="ac_modifiers",
        column_config={"Multiplier": st.column_config.NumberColumn(min_value=0.0, step=0.05, format="%.2f")},
    )
    if st.button("Save modifiers", key="ac_save_mods"):
        new_mods = {
            str(row["Retrofit type"]).strip(): {
                "adoption_multiplier": float(row["Multiplier"]),
                "description": str(row.get("Description", "")),
            }
            for _, row in edited_mods.iterrows()
            if str(row["Retrofit type"]).strip()
        }
        raw["retrofit_specific_modifiers"] = new_mods
        _save_adoption_curves(raw)
        st.success("Saved retrofit-specific modifiers.")


def _render_adoption_curves_tab() -> None:
    st.markdown("### Adoption Curves")
    st.markdown(
        "Edit adoption rate trajectories (0-1 fraction of buildings that adopt by each year) "
        "for each scenario. Changes are saved back to `adoption_curves.json`."
    )

    raw = _load_adoption_curves()
    scenarios: dict = raw.get("scenarios", {})
    scenario_names = list(scenarios.keys())

    selected = _ac_scenario_selector(scenarios, raw)
    if selected is None:
        return

    scen_data = copy.deepcopy(scenarios[selected])
    new_desc, new_max, new_attr, new_ctype = _ac_metadata_form(scen_data)
    _ac_curve_shape_fill(selected, scen_data, raw)

    # ── Chart: all scenarios ──────────────────────────────────────────────────
    chart_df = pd.DataFrame(
        {name: [float(s.get("values", {}).get(str(y), 0.0)) for y in _YEARS] for name, s in scenarios.items()},
        index=_YEARS,
    )
    st.line_chart(chart_df, x_label="Year", y_label="Adoption fraction")

    # ── Save metadata / delete ────────────────────────────────────────────────
    col_save, col_del = st.columns([2, 1])
    with col_save:
        if st.button("Save metadata", type="primary", key="ac_save"):
            scen_data.update({"description": new_desc, "max_adoption": new_max,
                              "annual_attrition": new_attr, "curve_type": new_ctype})
            raw["scenarios"][selected] = scen_data
            _save_adoption_curves(raw)
            st.success(f"Saved '{selected}' to {_ADOPTION_CURVES_PATH.name}.")
    with col_del:
        if len(scenario_names) > 1 and st.button("Delete scenario", key="ac_del"):
            del raw["scenarios"][selected]
            _save_adoption_curves(raw)
            st.warning(f"Deleted '{selected}'.")
            st.rerun()

    _ac_retrofit_modifiers(raw)


# ── Tab 2: Retrofit Cost per m² ───────────────────────────────────────────────

_SCENARIO_CONFIG_PATH = _REPO_ROOT / "data" / "use_case_inputs" / "scenario_config.json"
_YEAR_RANGE = list(range(2024, 2061))


def _load_scenario_config() -> dict:
    if _SCENARIO_CONFIG_PATH.exists():
        with open(_SCENARIO_CONFIG_PATH) as f:
            return json.load(f)
    return {"analysis_years": [], "year_files": {}, "retrofit_cost_per_sqm": 0.0, "scenario_name": ""}


def _save_scenario_config(cfg: dict) -> None:
    with open(_SCENARIO_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=4)


def _preview_energy_file(uploaded_file) -> dict | None:
    """Read an uploaded EnergyAndPeak parquet and return summary stats."""
    try:
        import io
        df = pd.read_parquet(io.BytesIO(uploaded_file.read()))
        uploaded_file.seek(0)

        idx = df.index.to_frame(index=False)
        n_buildings = len(df)

        # Detect scenarios (feature.semantic.Scenario column in index)
        scenarios_found: list[str] = []
        if "feature.semantic.Scenario" in idx.columns:
            scenarios_found = sorted(idx["feature.semantic.Scenario"].dropna().unique().tolist())

        # Conditioned area stats
        area_stats: dict = {}
        if "feature.geometry.energy_model_conditioned_area" in idx.columns:
            areas = idx["feature.geometry.energy_model_conditioned_area"].dropna().astype(float)
            if not areas.empty:
                area_stats = {
                    "total_m2": round(float(areas.sum()), 0),
                    "mean_m2": round(float(areas.mean()), 1),
                    "min_m2": round(float(areas.min()), 1),
                    "max_m2": round(float(areas.max()), 1),
                }

        # Column-level energy summary (annual total across all months)
        energy_kwh: float | None = None
        if hasattr(df.columns, "names") and "Measurement" in df.columns.names:
            try:
                energy_cols = df.loc[:, ("Energy", "Raw")]
                energy_kwh = float(energy_cols.sum().sum())
            except Exception:
                pass

        return {
            "n_buildings": n_buildings,
            "scenarios": scenarios_found,
            "area_stats": area_stats,
            "energy_kwh": energy_kwh,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _render_retrofit_cost_tab() -> None:
    st.markdown("### Retrofit Scenario Setup")
    st.markdown(
        "Select the analysis years, upload one **EnergyAndPeak.pq** file per year, "
        "then set the retrofit cost per m² that applies to your scenario across all years."
    )

    cfg = _load_scenario_config()

    # ── Step 1: analysis years ────────────────────────────────────────────────
    st.markdown("#### Step 1 — Analysis years")
    col_n, _ = st.columns([1, 3])
    with col_n:
        n_years = st.number_input(
            "Number of years",
            min_value=1, max_value=10,
            value=max(1, len(cfg.get("analysis_years", [])) or 4),
            step=1,
            key="rc_n_years",
        )

    # Build year selectors
    existing_years: list[int] = cfg.get("analysis_years", [])
    selected_years: list[int] = []
    year_cols = st.columns(min(int(n_years), 5))
    for i in range(int(n_years)):
        default_year = existing_years[i] if i < len(existing_years) else _YEAR_RANGE[i + 1]
        with year_cols[i % 5]:
            yr = st.selectbox(
                f"Year {i + 1}",
                options=_YEAR_RANGE,
                index=_YEAR_RANGE.index(default_year) if default_year in _YEAR_RANGE else i + 1,
                key=f"rc_year_{i}",
            )
            selected_years.append(int(yr))

    # Warn on duplicates
    if len(selected_years) != len(set(selected_years)):
        st.warning("Duplicate years selected — each year should appear only once.")

    st.divider()

    # ── Step 2: file upload per year ──────────────────────────────────────────
    st.markdown("#### Step 2 — Upload simulation files")
    st.caption(
        "Upload one `EnergyAndPeak.pq` (or `.parquet`) file per year. "
        "Files are read in-memory for preview only — use Save Config to record the mapping."
    )

    uploaded: dict[int, Any] = {}
    previews: dict[int, dict] = {}

    for yr in sorted(set(selected_years)):
        with st.expander(f"Year {yr}", expanded=True):
            f = st.file_uploader(
                f"EnergyAndPeak file for {yr}",
                type=["pq", "parquet"],
                key=f"rc_file_{yr}",
                label_visibility="collapsed",
            )
            if f is not None:
                uploaded[yr] = f
                with st.spinner(f"Reading {yr}…"):
                    info = _preview_energy_file(f)
                previews[yr] = info or {}

                if "error" in previews[yr]:
                    st.error(f"Could not read file: {previews[yr]['error']}")
                else:
                    p = previews[yr]
                    kpi_cols = st.columns(4)
                    with kpi_cols[0]:
                        st.metric("Buildings", p.get("n_buildings", "—"))
                    with kpi_cols[1]:
                        area = p.get("area_stats", {})
                        st.metric("Total floor area", f"{area.get('total_m2', 0):,.0f} m²" if area else "—")
                    with kpi_cols[2]:
                        scens = p.get("scenarios", [])
                        st.metric("Scenarios in file", len(scens))
                    with kpi_cols[3]:
                        kwh = p.get("energy_kwh")
                        st.metric("Total energy (raw)", f"{kwh / 1e6:,.1f} GWh" if kwh else "—")

                    if p.get("scenarios"):
                        st.caption(f"Scenarios detected: {', '.join(p['scenarios'])}")
            else:
                prev_file = cfg.get("year_files", {}).get(str(yr))
                if prev_file:
                    st.caption(f"Previously configured: `{prev_file}`")

    st.divider()

    # ── Step 3: retrofit cost per m² ─────────────────────────────────────────
    st.markdown("#### Step 3 — Retrofit cost (applies to all years)")

    col_name, col_cost = st.columns(2)
    with col_name:
        scenario_name = st.text_input(
            "Scenario name",
            value=cfg.get("scenario_name", ""),
            placeholder="e.g. ASHP_retrofit",
            key="rc_scenario_name",
            help="Label for this retrofit scenario — used to identify the cost in outputs.",
        )
    with col_cost:
        cost_per_sqm = st.number_input(
            "Retrofit cost ($/m² of conditioned floor area)",
            min_value=0.0,
            value=float(cfg.get("retrofit_cost_per_sqm", 0.0)),
            step=1.0,
            format="%.2f",
            key="rc_cost_sqm",
        )

    # Preview cost × area for uploaded files
    if previews:
        preview_rows = []
        for yr in sorted(previews):
            p = previews[yr]
            if "error" in p:
                continue
            total_area = p.get("area_stats", {}).get("total_m2", None)
            total_cost = (total_area * cost_per_sqm) if total_area and cost_per_sqm else None
            preview_rows.append({
                "Year": yr,
                "Buildings": p.get("n_buildings", "—"),
                "Total floor area (m²)": f"{total_area:,.0f}" if total_area else "—",
                "Est. total capital cost ($)": f"${total_cost:,.0f}" if total_cost else "—",
            })
        if preview_rows:
            st.markdown("**Estimated capital cost by year**")
            st.dataframe(pd.DataFrame(preview_rows).set_index("Year"), use_container_width=True)

    st.divider()

    # ── Save config ───────────────────────────────────────────────────────────
    if st.button("Save scenario config", type="primary", key="rc_save"):
        year_files: dict[str, str] = {}
        for yr in sorted(set(selected_years)):
            if yr in uploaded:
                f_obj = uploaded[yr]
                year_files[str(yr)] = f_obj.name if hasattr(f_obj, "name") else ""
            else:
                year_files[str(yr)] = cfg.get("year_files", {}).get(str(yr), "")
        new_cfg: dict = {
            "analysis_years": selected_years,
            "year_files": year_files,
            "retrofit_cost_per_sqm": float(cost_per_sqm),
            "scenario_name": (scenario_name or "").strip(),
        }
        _save_scenario_config(new_cfg)
        st.success(f"Saved to {_SCENARIO_CONFIG_PATH.name}.")


# ── Tab 3: Emissions Trajectories ─────────────────────────────────────────────

def _em_ensure_fuels(fuels_data: dict) -> None:
    """Add default linear trajectory for any missing fuel."""
    for fuel in _FUEL_LABELS:
        if fuel not in fuels_data:
            start = _DEFAULT_EMISSIONS_2025[fuel]
            end = _DEFAULT_EMISSIONS_2050[fuel]
            fuels_data[fuel] = {
                "values": {
                    str(y): round(start + (end - start) * (y - 2024) / (2050 - 2024), 4)
                    for y in _YEARS
                },
                "unit": "kg CO2/kWh",
            }


def _em_quickfill(fuels_data: dict, raw: dict) -> None:
    """Render quick-fill expander; saves to disk and reruns on apply."""
    with st.expander("Set trajectory shape", expanded=True):
        qf_cols = st.columns([2, 1, 1, 1, 2])
        with qf_cols[0]:
            qf_fuels = st.multiselect("Fuels to update", options=list(_FUEL_LABELS),
                                      default=["Electricity"], key="em_qf_fuels")
        with qf_cols[1]:
            qf_start_em = st.number_input("Start (2024, kg CO2/kWh)", 0.0, 2.0, 0.40, 0.01, key="em_qf_start")
        with qf_cols[2]:
            qf_end_em = st.number_input("End (2050, kg CO2/kWh)", 0.0, 2.0, 0.10, 0.01, key="em_qf_end")
        with qf_cols[3]:
            qf_shape_em = st.selectbox("Shape", ["Linear", "Sigmoid", "Exponential"], key="em_qf_shape")
        with qf_cols[4]:
            st.markdown("")
            st.markdown("")
            if st.button("Apply & save", key="em_qf_apply", type="primary"):
                xs = np.linspace(0, 1, len(_YEARS))
                if qf_shape_em == "Linear":
                    ys = qf_start_em + (qf_end_em - qf_start_em) * xs
                elif qf_shape_em == "Sigmoid":
                    sig = 1 / (1 + np.exp(-10 * (xs - 0.5)))
                    sig = (sig - sig[0]) / (sig[-1] - sig[0])
                    ys = qf_start_em + (qf_end_em - qf_start_em) * sig
                else:
                    exp_raw = np.expm1(xs * 3)
                    exp_raw = exp_raw / exp_raw[-1]
                    ys = qf_start_em + (qf_end_em - qf_start_em) * exp_raw
                for fuel in qf_fuels:
                    fuels_data[fuel]["values"] = {
                        str(y): round(float(v), 4) for y, v in zip(_YEARS, ys, strict=True)
                    }
                raw["fuels"] = fuels_data
                _save_emissions(raw)
                st.success(f"Applied {qf_shape_em.lower()} fill to: {', '.join(qf_fuels)}.")
                st.rerun()


def _render_emissions_tab() -> None:
    st.markdown("### Emissions Trajectories")
    st.markdown(
        "Define how the emissions intensity (kg CO2/kWh) of each fuel changes over time. "
        "This captures grid decarbonisation and fuel-switching effects. "
        "Results are saved to `emissions_trajectories.json`."
    )

    raw = _load_emissions()
    fuels_data: dict = raw.get("fuels", {})
    _em_ensure_fuels(fuels_data)
    _em_quickfill(fuels_data, raw)

    # ── Chart: all fuels ──────────────────────────────────────────────────────
    em_chart_df = pd.DataFrame(
        {
            fuel: [float(fuels_data[fuel]["values"].get(str(y), 0.0)) for y in _YEARS]
            for fuel in _FUEL_LABELS
        },
        index=_YEARS,
    )
    st.line_chart(em_chart_df, x_label="Year", y_label="kg CO\u2082/kWh")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Summary — emissions intensity at key years**")
    summary_rows = []
    for fuel in _FUEL_LABELS:
        fvals = fuels_data[fuel]["values"]
        row = {"Fuel": fuel}
        for sy in [2025, 2030, 2035, 2040, 2045, 2050]:
            row[str(sy)] = round(float(fvals.get(str(sy), 0.0)), 4)
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows).set_index("Fuel"), use_container_width=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    export_rows = []
    for fuel in _FUEL_LABELS:
        fvals = fuels_data[fuel]["values"]
        for y in _YEARS:
            export_rows.append({"fuel": fuel, "year": y, "kg_co2_per_kwh": float(fvals.get(str(y), 0.0))})
    export_df = pd.DataFrame(export_rows)
    st.download_button(
        "Download as CSV",
        data=export_df.to_csv(index=False),
        file_name="emissions_trajectories.csv",
        mime="text/csv",
        key="em_download",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ── Tab 4: Run Models ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

_CENSUS_API_KEY = "e865d3af152108cb504df27535d196f586c21729"

_US_STATES: dict[str, str] = {
    "Alabama": "01", "Alaska": "02", "Arizona": "04", "Arkansas": "05",
    "California": "06", "Colorado": "08", "Connecticut": "09", "Delaware": "10",
    "District of Columbia": "11", "Florida": "12", "Georgia": "13", "Hawaii": "15",
    "Idaho": "16", "Illinois": "17", "Indiana": "18", "Iowa": "19",
    "Kansas": "20", "Kentucky": "21", "Louisiana": "22", "Maine": "23",
    "Maryland": "24", "Massachusetts": "25", "Michigan": "26", "Minnesota": "27",
    "Mississippi": "28", "Missouri": "29", "Montana": "30", "Nebraska": "31",
    "Nevada": "32", "New Hampshire": "33", "New Jersey": "34", "New Mexico": "35",
    "New York": "36", "North Carolina": "37", "North Dakota": "38", "Ohio": "39",
    "Oklahoma": "40", "Oregon": "41", "Pennsylvania": "42", "Rhode Island": "44",
    "South Carolina": "45", "South Dakota": "46", "Tennessee": "47", "Texas": "48",
    "Utah": "49", "Vermont": "50", "Virginia": "51", "Washington": "53",
    "West Virginia": "54", "Wisconsin": "55", "Wyoming": "56",
}

_PROPENSITY_COEFFICIENTS: dict[str, float] = {
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

_ACS_CHUNKS: dict[str, dict] = {
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
            "B15003_017E": "education_high_school_grad",
            "B15003_018E": "education_ged",
            "B15003_021E": "education_associates_degree",
            "B15003_022E": "education_bachelors_degree",
            "B15003_023E": "education_masters_degree",
            "B15003_024E": "education_professional_school_degree",
            "B15003_025E": "education_doctorate_degree",
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

_INCOME_BRACKETS_K = [5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 55, 67.5, 87.5, 112.5, 137.5, 175, 250]
_INCOME_COLS = [
    "income_less_than_10k", "income_10k_to_14999", "income_15k_to_19999",
    "income_20k_to_24999", "income_25k_to_29999", "income_30k_to_34999",
    "income_35k_to_39999", "income_40k_to_44999", "income_45k_to_49999",
    "income_50k_to_59999", "income_60k_to_74999", "income_75k_to_99999",
    "income_100k_to_124999", "income_125k_to_149999", "income_150k_to_199999",
    "income_200k_or_more",
]
_EDUCATION_CATS = [1, 2, 3, 4]
_EDUCATION_COLS = [
    ["education_high_school_grad", "education_ged"],
    ["education_associates_degree"],
    ["education_bachelors_degree"],
    ["education_masters_degree", "education_professional_school_degree", "education_doctorate_degree"],
]
_HH_SIZE_CATS = [1, 2, 3, 4, 5, 6, 7]
_HH_SIZE_COLS = [
    "household_1_person", "household_2_person", "household_3_person",
    "household_4_person", "household_5_person", "household_6_person",
    "household_7_or_more_person",
]


# ── Census API helpers ────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_census_acs(state_fips: str) -> pd.DataFrame | None:
    """Fetch ACS 5-year tract-level data for a state and merge all variable chunks."""
    frames = []
    for chunk in _ACS_CHUNKS.values():
        url = f"https://api.census.gov/data/2023/{chunk['endpoint']}"
        params = {
            "get": ",".join(chunk["vars"].keys()),
            "for": "tract:*",
            "in": f"state:{state_fips}",
            "key": _CENSUS_API_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            st.error(f"Census API request failed: {exc}")
            return None
        if resp.status_code != 200:
            st.error(f"Census API error (HTTP {resp.status_code}): {resp.text[:300]}")
            return None
        data = resp.json()
        df = pd.DataFrame(data[1:], columns=data[0]).rename(columns=chunk["vars"])
        frames.append(df)

    merged = reduce(
        lambda left, right: pd.merge(left, right, on=["state", "county", "tract"], how="outer"),
        frames,
    )
    for col in merged.columns:
        if col not in ("state", "county", "tract"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged


@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_county_names(state_fips: str) -> dict[str, str]:
    """Return {county_fips_3digit: county_name} for a state."""
    try:
        resp = requests.get(
            "https://api.census.gov/data/2020/dec/pl",
            params={"get": "NAME", "for": "county:*", "in": f"state:{state_fips}", "key": _CENSUS_API_KEY},
            timeout=30,
        )
    except requests.RequestException:
        return {}
    if resp.status_code != 200:
        return {}
    result = {}
    for row in resp.json()[1:]:
        name, county_fips = row[0], row[2]
        county_short = name.split(" County")[0].split(" Parish")[0].split(" Borough")[0]
        result[county_fips.zfill(3)] = county_short
    return result


def _build_display_df(raw_df: pd.DataFrame, county_names: dict[str, str]) -> pd.DataFrame:
    df = raw_df.copy()
    df["county_name"] = df["county"].apply(
        lambda c: county_names.get(str(int(str(c).split(".")[0])).zfill(3), f"County {c}")
    )
    df["tract_str"] = df["tract"].astype(str).str.split(".").str[0].str.strip().str.zfill(6)
    df["display_label"] = "Tract " + df["tract_str"] + ", " + df["county_name"] + " County"
    return df.set_index("display_label")


# ── Per-tract Monte Carlo propensity ──────────────────────────────────────────

def _simulate_tract(
    row: pd.Series,
    building_age: float,
    upfront_range: tuple[float, float],
    energy_range: tuple[float, float],
    concern: float,
    neighbor: float,
    n_runs: int,
    coeffs: dict[str, float],
) -> float:
    """Return mean acceptance probability for one census tract via Monte Carlo logit."""
    edu_weights = [max(0, sum(row.get(c, 0) for c in cols)) for cols in _EDUCATION_COLS]
    hh_weights = [max(0, row.get(c, 0)) for c in _HH_SIZE_COLS]
    inc_weights = [max(0, row.get(c, 0)) for c in _INCOME_COLS]

    rng = np.random.default_rng()
    edu_w = np.array(edu_weights, dtype=float)
    hh_w = np.array(hh_weights, dtype=float)
    inc_w = np.array(inc_weights, dtype=float)
    edu_w = edu_w / edu_w.sum() if edu_w.sum() > 0 else None
    hh_w = hh_w / hh_w.sum() if hh_w.sum() > 0 else None
    inc_w = inc_w / inc_w.sum() if inc_w.sum() > 0 else None

    probs = []
    for _ in range(n_runs):
        edu = rng.choice(_EDUCATION_CATS, p=edu_w)
        hh = rng.choice(_HH_SIZE_CATS, p=hh_w)
        inc = rng.choice(_INCOME_BRACKETS_K, p=inc_w)
        upfront = rng.uniform(*upfront_range)
        energy = rng.uniform(*energy_range)
        z = (
            coeffs["intercept"]
            + coeffs["Year built"] * building_age
            + coeffs["Education"] * edu
            + coeffs["bedrooms"] * hh
            + coeffs["residents"] * hh
            + coeffs["Income"] * inc
            + coeffs["Concern"] * concern
            + coeffs["Upfront cost"] * upfront
            + coeffs["Neighbor"] * neighbor
            + coeffs["Energy cost"] * energy
        )
        probs.append(1 / (1 + np.exp(-z)))
    return float(np.mean(probs))


def _run_propensity_for_state(
    census_df: pd.DataFrame,
    building_age: float,
    upfront_range: tuple[float, float],
    energy_range: tuple[float, float],
    concern: float,
    neighbor: float,
    n_runs: int,
    coeffs: dict[str, float],
) -> dict[str, float]:
    """Run Monte Carlo propensity simulation for every tract; return {label: mean_prob}."""
    results: dict[str, float] = {}
    total = len(census_df)
    bar = st.progress(0, text="Running propensity simulation…")
    for i, (label, row) in enumerate(census_df.iterrows()):
        try:
            results[str(label)] = _simulate_tract(
                row, building_age, upfront_range, energy_range, concern, neighbor, n_runs, coeffs
            )
        except Exception:
            results[str(label)] = 0.5
        bar.progress((i + 1) / total, text=f"Tract {i + 1} / {total}")
    bar.empty()
    return results


# ── Adoption integration ──────────────────────────────────────────────────────

def _adopt_set_threshold(
    tract_props: dict[str, float],
    curve: dict[int, float],
    years: list[int],
    threshold: float,
) -> pd.Series:
    """Set-threshold adoption: willing tracts fill curve quota in propensity order."""
    n = len(tract_props)
    if n == 0:
        return pd.Series(0.0, index=years)
    ranked = sorted(
        [t for t, p in tract_props.items() if p >= threshold],
        key=lambda t: tract_props[t],
        reverse=True,
    )
    adoption_by_tract: dict[str, int] = {}
    adopted: set[str] = set()
    queue = [t for t in ranked if t not in adopted]
    ptr = 0
    adopted_so_far = 0
    for year in years:
        target = int(n * curve.get(year, 0.0))
        need = max(0, target - adopted_so_far)
        added = 0
        while added < need and ptr < len(queue):
            t = queue[ptr]
            ptr += 1
            if t in adopted:
                continue
            adoption_by_tract[t] = year
            adopted.add(t)
            adopted_so_far += 1
            added += 1

    counts = pd.Series(0, index=years, dtype=int)
    for yr in adoption_by_tract.values():
        if yr in counts.index:
            counts[yr] += 1
    return counts.cumsum() / n * 100


def _run_adoption_for_all_scenarios(
    tract_props: dict[str, float],
    adoption_scenarios: dict[str, dict],
    years: list[int],
    threshold: float,
) -> dict[str, pd.Series]:
    """Return {scenario_name: Series(year → cumulative_adoption_pct)} for every scenario."""
    results: dict[str, pd.Series] = {}
    for name, sdata in adoption_scenarios.items():
        curve = {int(y): min(float(v), 1.0) for y, v in sdata.get("values", {}).items()}
        results[name] = _adopt_set_threshold(tract_props, curve, years, threshold)
    return results


# ── Emissions trajectory ──────────────────────────────────────────────────────

def _compute_emissions_trajectories(
    adoption_results: dict[str, pd.Series],
    n_buildings: int,
    baseline_kwh_per_building: float,
    energy_reduction_pct: float,
    fuel_fractions: dict[str, float],
    emissions_data: dict,
    years: list[int],
) -> pd.DataFrame:
    """Compute annual portfolio CO₂ (tonnes) for each adoption scenario + baseline.

    For each year:
        adopted_frac = cum_adoption_pct / 100
        total_kwh(fuel) = n_buildings * baseline_kwh * fuel_fraction * [1 - adopted_frac * reduction]
        CO2(fuel, year) = total_kwh * emission_factor(fuel, year)
    """
    fuels_data = emissions_data.get("fuels", {})

    def ef(fuel: str, year: int) -> float:
        return float(fuels_data.get(fuel, {}).get("values", {}).get(str(year), 0.0))

    rows: list[dict] = []

    for year in years:
        row: dict = {"year": year}
        # Baseline (no adoption)
        baseline_co2 = sum(
            n_buildings * baseline_kwh_per_building * frac * ef(fuel, year)
            for fuel, frac in fuel_fractions.items()
        ) / 1000  # kg → tonnes
        row["Baseline"] = baseline_co2

        for name, series in adoption_results.items():
            adopted_frac = min(series.get(year, 0.0) / 100.0, 1.0)
            reduction = adopted_frac * energy_reduction_pct / 100.0
            scenario_co2 = sum(
                n_buildings * baseline_kwh_per_building * frac * (1 - reduction) * ef(fuel, year)
                for fuel, frac in fuel_fractions.items()
            ) / 1000
            row[name] = scenario_co2
        rows.append(row)

    return pd.DataFrame(rows).set_index("year")


# ── Tab 4 render ──────────────────────────────────────────────────────────────

def _render_run_models_tab() -> None:  # noqa: C901
    st.markdown("### Run Models")
    st.markdown(
        "Fetch census data for a US state, run the Monte Carlo propensity model "
        "across census tracts, integrate with adoption curve scenarios, "
        "and produce year-by-year emissions trajectories."
    )

    # ── Step 1: state selection + census fetch ────────────────────────────────
    st.markdown("#### Step 1 — Select state")
    col_state, col_btn = st.columns([2, 1])
    with col_state:
        state_name = st.selectbox(
            "State",
            options=sorted(_US_STATES.keys()),
            index=sorted(_US_STATES.keys()).index("Massachusetts"),
            key="run_state",
        )
    with col_btn:
        st.markdown("")
        fetch_clicked = st.button("Fetch census data", key="run_fetch", type="primary")

    if fetch_clicked:
        fips = _US_STATES[state_name]
        with st.spinner(f"Fetching ACS data for {state_name}…"):
            raw_df = _fetch_census_acs(fips)
            county_names = _fetch_county_names(fips)
        if raw_df is not None:
            display_df = _build_display_df(raw_df, county_names)
            st.session_state["rm_census_df"] = display_df
            st.session_state["rm_state_name"] = state_name
            st.success(f"Loaded {len(display_df):,} census tracts in {state_name}.")
        else:
            st.error("Failed to fetch census data — check your internet connection.")

    census_df: pd.DataFrame | None = st.session_state.get("rm_census_df")
    loaded_state: str = st.session_state.get("rm_state_name", "")

    if census_df is None:
        st.info("Select a state and click **Fetch census data** to continue.")
        return

    st.caption(f"{len(census_df):,} tracts loaded for **{loaded_state}**.")

    # ── Step 2: model parameters ──────────────────────────────────────────────
    st.divider()
    st.markdown("#### Step 2 — Propensity model parameters")

    cfg = _load_scenario_config()
    cost_sqm = float(cfg.get("retrofit_cost_per_sqm", 100.0))
    # Convert $/m² to rough $k total (assuming ~100 m² average unit)
    default_upfront_mid = round(cost_sqm * 100 / 1000, 1)

    with st.expander("Logit model coefficients", expanded=False):
        coeffs = {}
        coeff_cols = st.columns(5)
        for i, (k, default_v) in enumerate(_PROPENSITY_COEFFICIENTS.items()):
            with coeff_cols[i % 5]:
                coeffs[k] = st.number_input(k, value=default_v, format="%.3f", key=f"rm_coeff_{k}")

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        n_mc = st.number_input("MC draws per tract", 100, 2000, 300, 100, key="rm_mc",
                               help="More draws = smoother distribution but slower.")
    with col_b:
        building_age = st.number_input("Building age (years)", 0, 120, 40, 5, key="rm_age")
    with col_c:
        concern = st.slider("Climate concern (1-5)", 1.0, 5.0, 3.3, 0.1, key="rm_concern")
    with col_d:
        neighbor = st.slider("Neighbor effect (1-5)", 1.0, 5.0, 3.0, 0.1, key="rm_neighbor")

    col_e, col_f, col_g = st.columns(3)
    with col_e:
        up_min = st.number_input("Upfront cost min ($k)", 0.0, 500.0,
                                 max(0.0, default_upfront_mid - 5), 1.0, key="rm_up_min")
        up_max = st.number_input("Upfront cost max ($k)", 0.0, 500.0,
                                 max(1.0, default_upfront_mid + 10), 1.0, key="rm_up_max")
    with col_f:
        en_min = st.number_input("Annual energy cost min ($k)", 0.0, 50.0, 2.0, 0.5, key="rm_en_min")
        en_max = st.number_input("Annual energy cost max ($k)", 0.0, 100.0, 5.0, 0.5, key="rm_en_max")
    with col_g:
        threshold = st.slider("Acceptance threshold", 0.1, 0.9, 0.5, 0.05, key="rm_threshold",
                              help="Buildings/tracts at or above this propensity are 'willing' to adopt.")

    # ── Step 3: emissions / portfolio inputs ──────────────────────────────────
    st.divider()
    st.markdown("#### Step 3 — Portfolio energy parameters")
    st.caption(
        "Used to convert adoption trajectories into CO₂ emissions. "
        "If you have uploaded simulation files in Tab 2, enter the totals from those files here."
    )

    col_h, col_i, col_j = st.columns(3)
    with col_h:
        n_buildings = st.number_input("Number of buildings", 10, 1_000_000, 1000, 100, key="rm_nbldg")
        baseline_kwh = st.number_input("Baseline energy per building (kWh/yr)", 1000, 500_000, 15_000, 1000, key="rm_kwh")
    with col_i:
        energy_reduction = st.number_input(
            "Energy reduction from retrofit (%)", 0.0, 100.0, 30.0, 5.0, key="rm_ered",
            help="% reduction in total energy use after retrofit adoption."
        )
    with col_j:
        st.markdown("**Fuel mix (must sum to 1.0)**")
        frac_elec = st.number_input("Electricity", 0.0, 1.0, 0.6, 0.05, format="%.2f", key="rm_f_elec")
        frac_gas = st.number_input("Natural Gas", 0.0, 1.0, 0.3, 0.05, format="%.2f", key="rm_f_gas")
        frac_oil = st.number_input("Fuel Oil", 0.0, 1.0, 0.05, 0.05, format="%.2f", key="rm_f_oil")
        frac_prop = st.number_input("Propane", 0.0, 1.0, 0.05, 0.05, format="%.2f", key="rm_f_prop")
        total_frac = frac_elec + frac_gas + frac_oil + frac_prop
        if abs(total_frac - 1.0) > 0.01:
            st.warning(f"Fuel fractions sum to {total_frac:.2f} — should be 1.0.")

    fuel_fractions = {
        "Electricity": frac_elec,
        "Natural Gas": frac_gas,
        "Fuel Oil": frac_oil,
        "Propane": frac_prop,
    }

    # ── Run button ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Step 4 — Run")

    raw_adoption = _load_adoption_curves()
    adoption_scenarios = raw_adoption.get("scenarios", {})
    if not adoption_scenarios:
        st.warning("No adoption scenarios defined. Add some in the **Adoption Curves** tab first.")
        return

    selected_scenarios = st.multiselect(
        "Adoption scenarios to include",
        options=list(adoption_scenarios.keys()),
        default=list(adoption_scenarios.keys()),
        key="rm_scenarios",
    )

    if st.button("Run propensity + adoption models", type="primary", key="rm_run"):
        filtered_scenarios = {k: adoption_scenarios[k] for k in selected_scenarios}

        # 1. Propensity per tract
        st.markdown("**Running propensity simulation…**")
        tract_props = _run_propensity_for_state(
            census_df,
            building_age=float(building_age),
            upfront_range=(float(up_min), float(up_max)),
            energy_range=(float(en_min), float(en_max)),
            concern=float(concern),
            neighbor=float(neighbor),
            n_runs=int(n_mc),
            coeffs=coeffs if "intercept" in coeffs else _PROPENSITY_COEFFICIENTS,
        )

        # 2. Adoption integration
        adoption_results = _run_adoption_for_all_scenarios(
            tract_props, filtered_scenarios, _YEARS, float(threshold)
        )

        # 3. Emissions trajectories
        emissions_raw = _load_emissions()
        emissions_df = _compute_emissions_trajectories(
            adoption_results,
            n_buildings=int(n_buildings),
            baseline_kwh_per_building=float(baseline_kwh),
            energy_reduction_pct=float(energy_reduction),
            fuel_fractions=fuel_fractions,
            emissions_data=emissions_raw,
            years=_YEARS,
        )

        st.session_state["rm_tract_props"] = tract_props
        st.session_state["rm_adoption_results"] = adoption_results
        st.session_state["rm_emissions_df"] = emissions_df

    # ── Results ───────────────────────────────────────────────────────────────
    if "rm_tract_props" not in st.session_state:
        return

    tract_props: dict[str, float] = st.session_state["rm_tract_props"]
    adoption_results: dict[str, pd.Series] = st.session_state["rm_adoption_results"]
    emissions_df: pd.DataFrame = st.session_state["rm_emissions_df"]

    st.divider()
    st.markdown("### Results")

    # ── Propensity distribution ───────────────────────────────────────────────
    st.markdown("#### Propensity distribution across tracts")
    prop_vals = list(tract_props.values())
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("Tracts simulated", f"{len(prop_vals):,}")
    with col_kpi2:
        st.metric("Mean propensity", f"{np.mean(prop_vals):.1%}")
    with col_kpi3:
        st.metric(f"Above threshold ({threshold:.0%})", f"{sum(p >= threshold for p in prop_vals):,}")
    with col_kpi4:
        st.metric("Median propensity", f"{float(np.median(prop_vals)):.1%}")

    counts, edges = np.histogram(prop_vals, bins=40)
    fig_hist = go.Figure(go.Bar(
        x=((edges[:-1] + edges[1:]) / 2).tolist(),
        y=counts.tolist(),
        marker_color="#2563eb",
        opacity=0.8,
    ))
    fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#dc2626",
                       annotation_text=f"threshold={threshold:.0%}", annotation_position="top right")
    fig_hist.update_layout(
        xaxis_title="Mean acceptance probability",
        yaxis_title="Number of tracts",
        height=260,
        margin={"l": 50, "r": 20, "t": 20, "b": 40},
        bargap=0,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Adoption trajectories ─────────────────────────────────────────────────
    st.markdown("#### Adoption trajectories by scenario")
    fig_adopt = go.Figure()
    for i, (name, series) in enumerate(adoption_results.items()):
        fig_adopt.add_trace(go.Scatter(
            x=series.index.tolist(),
            y=series.tolist(),
            mode="lines",
            name=name,
            line={"color": _CURVE_COLORS[i % len(_CURVE_COLORS)], "width": 2},
        ))
    fig_adopt.update_layout(
        xaxis_title="Year",
        yaxis_title="Cumulative adoption (%)",
        yaxis={"range": [0, 105]},
        height=300,
        margin={"l": 50, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "y": -0.3},
    )
    st.plotly_chart(fig_adopt, use_container_width=True)

    # ── Emissions trajectories ────────────────────────────────────────────────
    st.markdown("#### Portfolio emissions trajectories (tonnes CO₂/yr)")
    fig_em = go.Figure()
    for i, col in enumerate(emissions_df.columns):
        dash = "solid" if col != "Baseline" else "dot"
        fig_em.add_trace(go.Scatter(
            x=emissions_df.index.tolist(),
            y=emissions_df[col].tolist(),
            mode="lines",
            name=col,
            line={"color": _CURVE_COLORS[i % len(_CURVE_COLORS)], "width": 2, "dash": dash},
        ))
    fig_em.update_layout(
        xaxis_title="Year",
        yaxis_title="tonnes CO₂/yr",
        height=340,
        margin={"l": 60, "r": 20, "t": 20, "b": 40},
        legend={"orientation": "h", "y": -0.3},
    )
    st.plotly_chart(fig_em, use_container_width=True)

    # Savings vs baseline table
    st.markdown("**Emissions savings vs baseline at key years**")
    baseline = emissions_df["Baseline"]
    summary_rows = []
    for yr in [2025, 2030, 2035, 2040, 2050]:
        if yr not in emissions_df.index:
            continue
        row: dict = {"Year": yr, "Baseline (t CO₂)": f"{baseline.loc[yr]:,.0f}"}
        for name in adoption_results:
            val = emissions_df.loc[yr, name]
            saved = baseline.loc[yr] - val
            row[f"{name} (t CO₂)"] = f"{val:,.0f}"
            row[f"{name} savings"] = f"{saved:,.0f} ({saved / baseline.loc[yr]:.0%})"
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows).set_index("Year"), use_container_width=True)

    # Download
    st.download_button(
        "Download emissions trajectories (CSV)",
        data=emissions_df.reset_index().to_csv(index=False),
        file_name="emissions_trajectories_modelled.csv",
        mime="text/csv",
        key="rm_dl_em",
    )


# ── App entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Scenario Editor",
        page_icon=":material/edit:",
        layout="wide",
    )
    st.title("Scenario Editor")
    st.caption(
        f"Editing files in `{_REPO_ROOT.name}/data/use_case_inputs/`. "
        "Changes are written to disk immediately on Save."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Adoption Curves",
        "Retrofit Cost ($/m²)",
        "Emissions Trajectories",
        "Run Models",
    ])

    with tab1:
        _render_adoption_curves_tab()

    with tab2:
        _render_retrofit_cost_tab()

    with tab3:
        _render_emissions_tab()

    with tab4:
        _render_run_models_tab()


if __name__ == "__main__":
    main()
