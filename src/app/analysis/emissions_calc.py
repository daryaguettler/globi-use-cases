"""Compute building-stock emissions trajectories under adoption scenarios.

Given:
  - Per-building annual kWh by fuel (baseline and scenario) from energy_delta.py
  - Adoption projections (mean, P10, P90) from AdoptionEngine (apply_uptake.py)
  - Emissions factor trajectories by fuel type

Produces a time-series DataFrame of total CO2 emissions for the building stock,
accounting for which buildings have adopted the retrofit in each year.

Emissions model
---------------
For each year ``y`` in the projection:
    emissions(y) = Σ_buildings [
        adopted(building, y) × scenario_emissions(building, y)
        + (1 - adopted(building, y)) × baseline_emissions(building, y)
    ]

Where:
    building_emissions(fuel_kwh, y) = Σ_fuels kwh_fuel × factor(fuel, y)

The adoption fraction comes from the ``mc_ensemble`` uptake result, which
provides mean / P10 / P90 trajectories.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FUEL_LABELS = ("Electricity", "Natural Gas", "Fuel Oil", "Propane")

# Default emissions factors (kg CO2/kWh) — these are overridden by user input
_DEFAULT_FACTORS_2024: dict[str, float] = {
    "Electricity": 0.40,
    "Natural Gas": 0.20,
    "Fuel Oil": 0.27,
    "Propane": 0.23,
}
_DEFAULT_FACTORS_2050: dict[str, float] = {
    "Electricity": 0.10,
    "Natural Gas": 0.18,
    "Fuel Oil": 0.25,
    "Propane": 0.21,
}


# ── Emissions factor helpers ───────────────────────────────────────────────────

def interpolate_emissions_factors(
    factors_by_fuel: dict[str, dict[str, float]],
    years: list[int],
) -> pd.DataFrame:
    """Interpolate emissions factors (kg CO2/kWh) for each year.

    Args:
        factors_by_fuel: Mapping of fuel → {year_str: factor_value}.
            Matches the structure in ``emissions_trajectories.json``.
        years: List of projection years.

    Returns:
        DataFrame indexed by year with one column per fuel.
    """
    rows = {}
    for year in years:
        row = {}
        for fuel in FUEL_LABELS:
            fuel_data = factors_by_fuel.get(fuel, {})
            values: dict[str, float] = fuel_data.get("values", fuel_data)  # handle both
            if not values:
                # Linear interpolation using defaults, clamped so years > 2050
                # hold the 2050 value rather than extrapolating to negatives.
                t = max(0.0, min(1.0, (year - 2024) / (2050 - 2024)))
                row[fuel] = (
                    _DEFAULT_FACTORS_2024[fuel]
                    + t * (_DEFAULT_FACTORS_2050[fuel] - _DEFAULT_FACTORS_2024[fuel])
                )
                continue
            year_ints = sorted(int(y) for y in values)
            if year <= year_ints[0]:
                row[fuel] = float(values[str(year_ints[0])])
            elif year >= year_ints[-1]:
                row[fuel] = float(values[str(year_ints[-1])])
            else:
                for i, y in enumerate(year_ints[:-1]):
                    if y <= year < year_ints[i + 1]:
                        t = (year - y) / (year_ints[i + 1] - y)
                        row[fuel] = float(values[str(y)]) + t * (
                            float(values[str(year_ints[i + 1])]) - float(values[str(y)])
                        )
                        break
                else:
                    row[fuel] = float(values[str(year_ints[-1])])
        rows[year] = row
    return pd.DataFrame(rows).T.rename_axis("year")


def compute_building_annual_emissions(
    fuel_kwh: pd.DataFrame,
    emissions_factor_row: pd.Series,
) -> pd.Series:
    """Compute annual CO2 (kg) per building for a single year's emission factors.

    Args:
        fuel_kwh:             DataFrame with columns ``kwh_<Fuel>`` per building.
        emissions_factor_row: Series indexed by fuel label (kg CO2/kWh).

    Returns:
        Series of annual CO2 (kg) per building.
    """
    total = pd.Series(np.zeros(len(fuel_kwh)), index=fuel_kwh.index)
    for fuel in FUEL_LABELS:
        kwh_col = f"kwh_{fuel}"
        if kwh_col in fuel_kwh.columns:
            factor = float(emissions_factor_row.get(fuel, 0.0))
            total += fuel_kwh[kwh_col] * factor
    return total


# ── Main trajectory calculation ────────────────────────────────────────────────

def interpolate_building_energy(
    year_energy: dict[int, pd.DataFrame],
    year: int,
) -> pd.DataFrame:
    """Return per-building kWh DataFrame interpolated (or clamped) to *year*.

    year_energy maps anchor years to DataFrames indexed by building.id with
    columns ``baseline_kwh_<Fuel>`` and ``scenario_kwh_<Fuel>``.
    Years outside the anchor range hold the nearest anchor value constant.
    """
    anchors = sorted(year_energy.keys())
    if year <= anchors[0]:
        return year_energy[anchors[0]]
    if year >= anchors[-1]:
        return year_energy[anchors[-1]]
    for i in range(len(anchors) - 1):
        y0, y1 = anchors[i], anchors[i + 1]
        if y0 <= year <= y1:
            t = (year - y0) / (y1 - y0)
            df0, df1 = year_energy[y0], year_energy[y1]
            # Align df1 to df0's index so we interpolate each building against
            # itself, not an arbitrary peer from a differently-ordered parquet.
            df1_aligned = df1.reindex(df0.index).fillna(df0)
            result = df0.copy()
            for col in df0.columns:
                if col in df1_aligned.columns:
                    result[col] = df0[col].values * (1 - t) + df1_aligned[col].values * t
            return result
    return year_energy[anchors[-1]]


def compute_emissions_trajectory(
    policy_impacts: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    emissions_factors_json: dict,
    years: list[int],
    year_energy: dict[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute total stock emissions and energy trajectory under MC-ensemble adoption.

    Args:
        policy_impacts:         Output of ``build_policy_impacts`` for the reference
                                year.  Used for propensity ranking and (if year_energy
                                is None) for static per-building kWh.
        yearly_summary:         ``UptakeResult.yearly_summary`` from AdoptionEngine.
        emissions_factors_json: Dict with key ``"fuels"`` mapping per-fuel trajectories.
        years:                  Projection years.
        year_energy:            Optional dict mapping anchor years → DataFrames indexed
                                by building.id with ``baseline_kwh_<Fuel>`` /
                                ``scenario_kwh_<Fuel>`` columns.  When provided, energy
                                is linearly interpolated between anchor years so each
                                projection year uses year-appropriate kWh values.

    Returns:
        DataFrame with columns:
            year, baseline_emissions_t, scenario_mean_t, scenario_p10_t,
            scenario_p90_t, emissions_savings_mean_t,
            baseline_kwh_GWh, scenario_kwh_mean_GWh,
            scenario_kwh_p10_GWh, scenario_kwh_p90_GWh
    """
    fuels_data = emissions_factors_json.get("fuels", emissions_factors_json)
    ef_df = interpolate_emissions_factors(fuels_data, years)

    # Static per-building kWh (used only when year_energy is None)
    base_fuel_kwh_static = None
    scen_fuel_kwh_static = None
    if year_energy is None:
        base_fuel_kwh_static = _extract_fuel_kwh(policy_impacts, prefix="baseline")
        scen_fuel_kwh_static = _extract_fuel_kwh(policy_impacts, prefix="scenario")

    n_buildings = len(policy_impacts)
    bid_values = (
        policy_impacts["building.id"].values
        if "building.id" in policy_impacts.columns
        else None
    )

    if "acceptance_probability" in policy_impacts.columns:
        propensity_order = (
            policy_impacts["acceptance_probability"]
            .rank(method="first", ascending=False)
            .astype(int)
            .sub(1)
        )
        ranked_idx = propensity_order.argsort().values
        use_ranked = True
    else:
        ranked_idx = None
        use_ranked = False

    rows = []
    for year in years:
        ef_row = ef_df.loc[year] if year in ef_df.index else ef_df.iloc[-1]

        if year_energy is not None:
            e_df = interpolate_building_energy(year_energy, year)
            if bid_values is not None:
                e_df = e_df.reindex(bid_values).fillna(0.0)
            base_fuel_kwh = e_df[
                [c for c in e_df.columns if c.startswith("baseline_kwh_")]
            ].rename(columns=lambda c: c.replace("baseline_kwh_", "kwh_"))
            scen_fuel_kwh = e_df[
                [c for c in e_df.columns if c.startswith("scenario_kwh_")]
            ].rename(columns=lambda c: c.replace("scenario_kwh_", "kwh_"))
        else:
            base_fuel_kwh = base_fuel_kwh_static
            scen_fuel_kwh = scen_fuel_kwh_static

        base_em = compute_building_annual_emissions(base_fuel_kwh, ef_row).values
        scen_em = compute_building_annual_emissions(scen_fuel_kwh, ef_row).values
        base_kwh_per_bldg = base_fuel_kwh.sum(axis=1).values
        scen_kwh_per_bldg = scen_fuel_kwh.sum(axis=1).values

        total_baseline_kg = float(base_em.sum())
        total_baseline_kwh = float(base_kwh_per_bldg.sum())

        year_rows = yearly_summary[yearly_summary["year"] == year]
        if year_rows.empty:
            mean_pct = p10_pct = p90_pct = 0.0
        else:
            r = year_rows.iloc[0]
            mean_pct = float(r.get("cumulative_adoption_pct", 0.0))
            p10_pct = float(r.get("cumulative_adoption_pct_p10", mean_pct))
            p90_pct = float(r.get("cumulative_adoption_pct_p90", mean_pct))

        def _stock_metric(
            adopted_frac: float,
            adopted_vals: np.ndarray,
            baseline_vals: np.ndarray,
        ) -> float:
            n_adopt = max(0, min(round(adopted_frac / 100.0 * n_buildings), n_buildings))
            if use_ranked:
                adopted_pos = ranked_idx[:n_adopt]
                rest_pos = ranked_idx[n_adopt:]
                return float(adopted_vals[adopted_pos].sum() + baseline_vals[rest_pos].sum())
            avg_base = float(baseline_vals.mean()) if n_buildings > 0 else 0.0
            avg_adopted = float(adopted_vals.mean()) if n_buildings > 0 else 0.0
            return n_adopt * avg_adopted + (n_buildings - n_adopt) * avg_base

        rows.append({
            "year": year,
            "baseline_emissions_t": total_baseline_kg / 1000.0,
            "scenario_mean_t": _stock_metric(mean_pct, scen_em, base_em) / 1000.0,
            "scenario_p10_t": _stock_metric(p10_pct, scen_em, base_em) / 1000.0,
            "scenario_p90_t": _stock_metric(p90_pct, scen_em, base_em) / 1000.0,
            "baseline_kwh_GWh": total_baseline_kwh / 1e6,
            "scenario_kwh_mean_GWh": _stock_metric(mean_pct, scen_kwh_per_bldg, base_kwh_per_bldg) / 1e6,
            "scenario_kwh_p10_GWh": _stock_metric(p10_pct, scen_kwh_per_bldg, base_kwh_per_bldg) / 1e6,
            "scenario_kwh_p90_GWh": _stock_metric(p90_pct, scen_kwh_per_bldg, base_kwh_per_bldg) / 1e6,
            "adoption_pct_mean": mean_pct,
            "adoption_pct_p10": p10_pct,
            "adoption_pct_p90": p90_pct,
        })

    df = pd.DataFrame(rows)
    df["emissions_savings_mean_t"] = df["baseline_emissions_t"] - df["scenario_mean_t"]
    return df


def _extract_fuel_kwh(
    policy_impacts: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """Extract ``{prefix}_kwh_<Fuel>`` columns and rename to ``kwh_<Fuel>``."""
    cols = {f"{prefix}_kwh_{fuel}": f"kwh_{fuel}" for fuel in FUEL_LABELS}
    available = {k: v for k, v in cols.items() if k in policy_impacts.columns}
    if not available:
        logger.warning(f"No '{prefix}_kwh_*' columns found in policy_impacts.")
        return pd.DataFrame(
            {f"kwh_{fuel}": np.zeros(len(policy_impacts)) for fuel in FUEL_LABELS}
        )
    return policy_impacts[list(available.keys())].rename(columns=available)
