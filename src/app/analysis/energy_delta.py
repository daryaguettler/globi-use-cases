"""Parse globi EnergyAndPeak.pq files and compute per-building energy savings.

Each EnergyAndPeak.pq is a single-scenario, single-year parquet with:
  - MultiIndex rows encoding building metadata (scenario, semantic features,
    geometry, location, etc.)
  - Columns like ``("Energy", "Heating")``, ``("Energy", "Cooling")``, etc.
    (energy intensity, kWh/m²)

Energy is attributed to fuel type via ``feature.semantic.Heating`` /
``feature.semantic.DHW`` semantic columns in the index.

The output ``policy_impacts`` DataFrame is compatible with
``PropensityModelEngine`` (apply_propensity.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ── Unit conversions ───────────────────────────────────────────────────────────
KWH_PER_THERM = 29.3001
KWH_PER_GALLON_OIL = 10.35
KWH_PER_GALLON_PROPANE = 7.53

# ── Fuel system classification (mirrors apply_costs.py) ───────────────────────
NG_HEATING_SYSTEMS = {"NaturalGasHeating", "NaturalGasCondensingHeating"}
OIL_HEATING_SYSTEMS = {"OilHeating"}
ELECTRIC_HEATING_SYSTEMS = {"ElectricResistance", "ASHPHeating", "GSHPHeating"}
NG_DHW_SYSTEMS = {"NaturalGasDHW", "NaturalGasHeatingDHWCombo"}
ELECTRIC_DHW_SYSTEMS = {"ElectricResistanceDHW", "HPWH"}
PROPANE_HEATING_SYSTEMS = {"PropaneHeating"}
PROPANE_DHW_SYSTEMS = {"PropaneDHW"}

# End-use columns present in the parquet (energy intensity kWh/m²)
_END_USES = ["Lighting", "Equipment", "Cooling", "Heating", "Domestic Hot Water"]

# ── New-schema utility fuel mapping ────────────────────────────────────────────
# Maps the column name used in Energy.Utilities.* to the canonical FUEL_LABELS key.
_UTILITY_FUEL_MAP: dict[str, str] = {
    "Electricity": "Electricity",
    "NaturalGas": "Natural Gas",
    "FuelOil": "Fuel Oil",
    "Propane": "Propane",
}


class EnergyPrices(BaseModel):
    """Per-fuel energy prices (USD/kWh equivalent)."""

    electricity: float = Field(0.17, ge=0.0, description="$/kWh")
    natural_gas: float = Field(0.043, ge=0.0, description="$/kWh eq (~$1.26/therm)")
    fuel_oil: float = Field(0.038, ge=0.0, description="$/kWh eq (~$4.00/gallon)")
    propane: float = Field(0.052, ge=0.0, description="$/kWh eq (~$3.80/gallon)")

    _LABEL_TO_FIELD: ClassVar[dict[str, str]] = {
        "Electricity": "electricity",
        "Natural Gas": "natural_gas",
        "Fuel Oil": "fuel_oil",
        "Propane": "propane",
    }

    def get(self, fuel: str, default: float = 0.0) -> float:
        field = self._LABEL_TO_FIELD.get(fuel)
        return getattr(self, field, default) if field else default

    def to_dict(self) -> dict[str, float]:
        return {label: getattr(self, field) for label, field in self._LABEL_TO_FIELD.items()}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "EnergyPrices":
        return cls(
            electricity=d.get("Electricity", 0.17),
            natural_gas=d.get("Natural Gas", 0.043),
            fuel_oil=d.get("Fuel Oil", 0.038),
            propane=d.get("Propane", 0.052),
        )


class PolicyImpactsConfig(BaseModel):
    """Configuration for building policy impacts computation."""

    scenario_name: str
    cost_per_sqm: float = Field(150.0, ge=0.0, description="Gross retrofit cost per m² conditioned area")
    energy_prices: EnergyPrices = Field(default_factory=EnergyPrices)


# Kept as a plain dict for backward-compat import sites; use EnergyPrices() for new code
DEFAULT_ENERGY_PRICES: dict[str, float] = EnergyPrices().to_dict()

FUEL_LABELS = tuple(DEFAULT_ENERGY_PRICES.keys())


# ── Parquet loading ────────────────────────────────────────────────────────────

def load_energy_parquet(source: str | Path | bytes) -> pd.DataFrame:
    """Load a globi EnergyAndPeak.pq and return a flat DataFrame.

    The MultiIndex is expanded into regular columns with their original dotted
    names (e.g. ``feature.semantic.Heating``).  MultiIndex columns are joined
    with ``"."`` (e.g. ``Energy.Heating``).
    """
    if isinstance(source, (bytes, bytearray)):
        import io
        df = pd.read_parquet(io.BytesIO(source))
    else:
        df = pd.read_parquet(source)

    # Flatten index
    if isinstance(df.index, pd.MultiIndex):
        meta = df.index.to_frame(index=False)
        meta.columns = [str(c) for c in meta.columns]
    else:
        meta = pd.DataFrame({"_idx": range(len(df))})

    # Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        flat_cols = [".".join(str(lv) for lv in col).strip(".") for col in df.columns]
    else:
        flat_cols = [str(c) for c in df.columns]

    data = df.copy()
    data.columns = flat_cols
    data = data.reset_index(drop=True)
    meta = meta.reset_index(drop=True)

    result = pd.concat([meta, data], axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result


# ── Feature extraction ─────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """Return the first matching column, or None."""
    for c in candidates:
        if c in df.columns:
            return df[c]
    return None


def _get_area(df: pd.DataFrame) -> pd.Series:
    s = _col(df, "feature.geometry.energy_model_conditioned_area", "conditioned_area", "area_m2")
    if s is not None:
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    logger.warning("No conditioned area column found; defaulting to 0 m².")
    return pd.Series(np.zeros(len(df)), index=df.index)


def _get_building_id(df: pd.DataFrame) -> pd.Series:
    s = _col(df, "building.id", "building_id", "feature.id", "id")
    if s is not None:
        return s.astype(str)
    return pd.Series([str(i) for i in range(len(df))], index=df.index)


# ── Energy attribution ─────────────────────────────────────────────────────────

def _has_utilities_schema(df: pd.DataFrame) -> bool:
    """Detect the 4-level MultiIndex column schema (Energy.Utilities.{fuel}.{month})."""
    return any(
        any(c.startswith(f"Energy.Utilities.{f}.") for c in df.columns)
        for f in _UTILITY_FUEL_MAP
    )


def compute_fuel_kwh(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-building annual kWh by fuel type.

    Supports two parquet schemas:

    **New schema** (4-level MultiIndex columns, e.g. ``Energy.Utilities.NaturalGas.1``):
        Fuel breakdown is read directly from ``Energy.Utilities.*`` monthly columns.
        No heating-system semantic field is needed.

    **Legacy schema** (flattened 2-level columns, e.g. ``Energy.Heating``):
        Total end-use energy is read and attributed to fuel via
        ``feature.semantic.Heating`` / ``feature.semantic.DHW`` columns.

    Returns a DataFrame with columns:
        ``kwh_Electricity``, ``kwh_Natural Gas``, ``kwh_Fuel Oil``, ``kwh_Propane``
    """
    area = _get_area(df)

    # ── New schema: Energy.Utilities.{fuel}.{month} ────────────────────────────
    if _has_utilities_schema(df):
        result: dict[str, np.ndarray] = {}
        for raw_name, label in _UTILITY_FUEL_MAP.items():
            month_cols = [c for c in df.columns if c.startswith(f"Energy.Utilities.{raw_name}.")]
            if month_cols:
                intensity = (
                    df[month_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .sum(axis=1)
                )
            else:
                intensity = pd.Series(np.zeros(len(df)), index=df.index)
            result[f"kwh_{label}"] = (intensity * area).values
        return pd.DataFrame(result, index=df.index)

    # ── Legacy schema: Energy.{end_use} + feature.semantic.Heating/DHW ─────────
    def _intensity(end_use: str) -> pd.Series:
        s = _col(df, f"Energy.{end_use}", f"energy.{end_use.lower()}")
        if s is not None:
            return pd.to_numeric(s, errors="coerce").fillna(0.0) * area
        return pd.Series(np.zeros(len(df)), index=df.index)

    lighting_kwh = _intensity("Lighting")
    equip_kwh = _intensity("Equipment")
    cooling_kwh = _intensity("Cooling")
    heating_kwh = _intensity("Heating")
    dhw_kwh = _intensity("Domestic Hot Water")

    # Baseline electricity: lighting + equipment + cooling
    elec = lighting_kwh + equip_kwh + cooling_kwh
    gas = pd.Series(np.zeros(len(df)), index=df.index)
    oil = pd.Series(np.zeros(len(df)), index=df.index)
    propane = pd.Series(np.zeros(len(df)), index=df.index)

    # Attribute heating to fuel
    heat_sys = _col(df, "feature.semantic.Heating", "Heating")
    if heat_sys is not None:
        is_elec_heat = heat_sys.isin(ELECTRIC_HEATING_SYSTEMS)
        is_gas_heat = heat_sys.isin(NG_HEATING_SYSTEMS)
        is_oil_heat = heat_sys.isin(OIL_HEATING_SYSTEMS)
        is_prop_heat = heat_sys.isin(PROPANE_HEATING_SYSTEMS)
        elec += heating_kwh.where(is_elec_heat, 0.0)
        gas += heating_kwh.where(is_gas_heat, 0.0)
        oil += heating_kwh.where(is_oil_heat, 0.0)
        propane += heating_kwh.where(is_prop_heat, 0.0)
        known = is_elec_heat | is_gas_heat | is_oil_heat | is_prop_heat
        elec += heating_kwh.where(~known, 0.0)
    else:
        elec += heating_kwh

    # Attribute DHW to fuel
    dhw_sys = _col(df, "feature.semantic.DHW", "DHW")
    if dhw_sys is not None:
        is_elec_dhw = dhw_sys.isin(ELECTRIC_DHW_SYSTEMS)
        is_gas_dhw = dhw_sys.isin(NG_DHW_SYSTEMS)
        is_prop_dhw = dhw_sys.isin(PROPANE_DHW_SYSTEMS)
        elec += dhw_kwh.where(is_elec_dhw, 0.0)
        gas += dhw_kwh.where(is_gas_dhw, 0.0)
        propane += dhw_kwh.where(is_prop_dhw, 0.0)
        known_dhw = is_elec_dhw | is_gas_dhw | is_prop_dhw
        elec += dhw_kwh.where(~known_dhw, 0.0)
    else:
        elec += dhw_kwh

    return pd.DataFrame(
        {
            "kwh_Electricity": elec.values,
            "kwh_Natural Gas": gas.values,
            "kwh_Fuel Oil": oil.values,
            "kwh_Propane": propane.values,
        },
        index=df.index,
    )


def compute_annual_energy_cost(
    fuel_kwh: pd.DataFrame,
    prices: EnergyPrices | dict[str, float],
) -> pd.Series:
    """Return total annual energy cost per building (USD)."""
    p = prices if isinstance(prices, EnergyPrices) else EnergyPrices.from_dict(prices)
    total = pd.Series(np.zeros(len(fuel_kwh)), index=fuel_kwh.index)
    for fuel in FUEL_LABELS:
        col = f"kwh_{fuel}"
        if col in fuel_kwh.columns:
            total += fuel_kwh[col] * p.get(fuel, 0.0)
    return total


# ── Building ID alignment ──────────────────────────────────────────────────────

def _align_on_id(
    base_df: pd.DataFrame,
    scen_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join both DataFrames on building ID, return aligned copies."""
    base_ids = _get_building_id(base_df)
    scen_ids = _get_building_id(scen_df)

    base_df = base_df.copy()
    scen_df = scen_df.copy()
    base_df["_bid"] = base_ids.values
    scen_df["_bid"] = scen_ids.values

    base_df = base_df.set_index("_bid")
    scen_df = scen_df.set_index("_bid")

    common = base_df.index.intersection(scen_df.index)
    if len(common) < len(base_df):
        logger.warning(
            f"{len(base_df) - len(common)} buildings in baseline not found in "
            "scenario; they will be excluded."
        )
    return base_df.loc[common], scen_df.loc[common]


# ── Public API ─────────────────────────────────────────────────────────────────

def build_policy_impacts(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    scenario_name: str,
    cost_per_sqm: float,
    energy_prices: EnergyPrices | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Produce a ``policy_impacts`` DataFrame compatible with PropensityModelEngine.

    Args:
        baseline_df:   Flat DataFrame from ``load_energy_parquet`` for baseline.
        scenario_df:   Flat DataFrame from ``load_energy_parquet`` for the scenario.
        scenario_name: User-provided label for the retrofit scenario.
        cost_per_sqm:  Gross retrofit cost per m² of conditioned floor area.
        energy_prices: Optional per-fuel USD/kWh overrides as ``EnergyPrices`` or dict.

    Returns:
        Flat DataFrame (one row per building) ready for PropensityModelEngine.
    """
    if isinstance(energy_prices, EnergyPrices):
        prices = energy_prices
    elif isinstance(energy_prices, dict):
        prices = EnergyPrices.from_dict({**DEFAULT_ENERGY_PRICES, **energy_prices})
    else:
        prices = EnergyPrices()

    base_aligned, scen_aligned = _align_on_id(baseline_df, scenario_df)
    n = len(base_aligned)
    logger.info(f"Matched {n} buildings for scenario '{scenario_name}'.")

    base_kwh = compute_fuel_kwh(base_aligned)
    scen_kwh = compute_fuel_kwh(scen_aligned)

    base_cost = compute_annual_energy_cost(base_kwh, prices)
    scen_cost = compute_annual_energy_cost(scen_kwh, prices)
    energy_savings = (base_cost - scen_cost).clip(lower=0.0)

    area = _get_area(base_aligned)
    retrofit_cost = area * cost_per_sqm

    out = pd.DataFrame(index=base_aligned.index)
    out["building.id"] = base_aligned.index.astype(str)
    out["retrofit.scenario"] = scenario_name

    # Costs compatible with PropensityModelEngine
    out["cost.Total"] = retrofit_cost.values
    out["net_cost.AllCustomers"] = retrofit_cost.values
    out["adjusted_net_cost.AllCustomers"] = retrofit_cost.values

    # Energy financials
    out["energy_cost.annual_savings"] = energy_savings.values
    out["energy_cost.Total"] = scen_cost.values
    out["energy_cost.baseline_Total"] = base_cost.values

    # Per-fuel kWh (needed for emissions calculations downstream)
    for fuel in FUEL_LABELS:
        out[f"baseline_kwh_{fuel}"] = base_kwh[f"kwh_{fuel}"].values
        out[f"scenario_kwh_{fuel}"] = scen_kwh[f"kwh_{fuel}"].values

    # Passthrough building metadata for PropensityModelEngine
    _copy_metadata(base_aligned, out)

    return out.reset_index(drop=True)


_METADATA_MAP: dict[str, str] = {
    "feature.semantic.Typology": "feature.semantic.Typology",
    "feature.semantic.Age_bracket": "feature.semantic.Age_bracket",
    "feature.semantic.Income": "feature.semantic.Income",
    "feature.geometry.energy_model_conditioned_area": "area_m2",
    # New-schema geometry: rotated rectangle WKT (EPSG:3857) used for geocoding
    "rotated_rectangle": "rotated_rectangle",
    "GLOBI_ROTATED_RECTANGLE": "rotated_rectangle",
    # Legacy-schema location fields
    "feature.location.lat": "lat",
    "feature.location.lon": "lon",
    "feature.location.county": "building.county",
    "building.county": "building.county",
    "county": "building.county",
    "feature.location.tract_id": "building.tract_id",
    "building.tract_id": "building.tract_id",
    "GEOID20": "building.tract_id",
}


def _copy_metadata(src: pd.DataFrame, dst: pd.DataFrame) -> None:
    for src_col, dst_col in _METADATA_MAP.items():
        if src_col in src.columns and dst_col not in dst.columns:
            try:
                dst[dst_col] = src[src_col].values
            except Exception:
                pass
