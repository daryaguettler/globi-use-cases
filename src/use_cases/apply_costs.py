"""Retrofit scenario cost and incentive analysis.

Computes per-building capital costs, incentives, net costs, and energy savings
for each retrofit scenario relative to the baseline scenario.

The model output parquet must have a MultiIndex (or columns) including
``retrofit.scenario``. One scenario must be named ``"baseline"`` (or the
first scenario found is used as the reference).

Outputs a flat per-building × scenario DataFrame with columns:
  - ``cost.Total``                   gross capex
  - ``net_cost.<bracket>``           capex minus incentives
  - ``adjusted_net_cost.<bracket>``  same as net_cost (alias kept for compatibility)
  - ``incentive.<bracket>``          total incentive value
  - ``energy_cost.annual_savings``   annual USD saved vs baseline
  - ``energy_cost.Total``            annual energy cost for this scenario
  - ``feature.semantic.*``           pass-through semantic fields
  - ``feature.geometry.*``           pass-through geometry features
  - ``retrofit.scenario``            scenario name

Income brackets are configurable; defaults are ``["IncomeEligible", "AllCustomers"]``.

Example usage::

    analyzer = RetrofitScenarioAnalyzer(
        model_output_path="results/GroupA/baseline/EnergyAndPeak.pq",
        costs_path="inputs/costs/retrofit_costs.json",
        incentives_path="inputs/costs/incentives.json",
    )
    df = analyzer.compute_all_scenario_costs()
    df.to_parquet("results/GroupA/policy_impacts.pq", index=False)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ── Unit conversion constants ─────────────────────────────────────────────────
KWH_PER_THERM = 29.3001
KWH_PER_GALLON_OIL = 10.35
KWH_PER_GALLON_PROPANE = 7.53

# ── Default income brackets ───────────────────────────────────────────────────
DEFAULT_INCOME_BRACKETS: list[str] = ["IncomeEligible", "AllCustomers"]

# ── Semantic field names (controls which columns are compared for cost triggers) ─
SEMANTIC_FIELDS = [
    "Heating",
    "Cooling",
    "DHW",
    "Walls",
    "RoofInsulation",
    "Windows",
    "Weatherization",
    "Lighting",
    "Equipment",
    "Thermostat",
    "Distribution",
    "AtticFloorInsulation",
    "BasementCeilingInsulation",
    "BasementWallsInsulation",
    "OnsiteSolar",
]

# ── Fuel system classification ────────────────────────────────────────────────
NG_HEATING_SYSTEMS = ["NaturalGasHeating", "NaturalGasCondensingHeating"]
OIL_HEATING_SYSTEMS = ["OilHeating"]
ELECTRIC_HEATING_SYSTEMS = ["ElectricResistance", "ASHPHeating", "GSHPHeating"]
NG_DHW_SYSTEMS = ["NaturalGasDHW", "NaturalGasHeatingDHWCombo"]
ELECTRIC_DHW_SYSTEMS = ["ElectricResistanceDHW", "HPWH"]

# ── Equipment sizing constants ────────────────────────────────────────────────
KW_PER_TON = 3.517
CONDENSER_MAX_KW = 17.6
HEAD_COVERAGE_M2 = 46.0
BOREHOLE_FT_PER_TON = 180.0
BOREHOLE_DEPTH_FT = 200.0

# ── Cost distribution percentiles ────────────────────────────────────────────
PERCENTILES = {
    0.05: "p5",
    0.10: "p10",
    0.25: "p25",
    0.50: "p50",
    0.75: "p75",
    0.90: "p90",
    0.95: "p95",
}


def compute_equipment_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Derive HVAC equipment counts from building geometry and heating capacity."""
    out = df.copy()
    capacity_col = "feature.calculated.heating_capacity_kW"
    area_col = "feature.geometry.energy_model_conditioned_area"

    if capacity_col in out.columns:
        capacity_kw = out[capacity_col]
    elif area_col in out.columns:
        capacity_kw = (out[area_col] * 0.05).clip(lower=5.3, upper=31.6)
    else:
        capacity_kw = pd.Series(10.5, index=out.index)

    tons = capacity_kw / KW_PER_TON
    out["NUMBER_OF_CONDENSERS"] = np.ceil(capacity_kw / CONDENSER_MAX_KW).clip(lower=1)
    if area_col in out.columns:
        out["NUMBER_OF_HEADS"] = np.ceil(out[area_col] / HEAD_COVERAGE_M2).clip(lower=1)
    else:
        out["NUMBER_OF_HEADS"] = out["NUMBER_OF_CONDENSERS"]
    out["BOREHOLE_FEET"] = tons * BOREHOLE_FT_PER_TON
    out["NUMBER_OF_BOREHOLES"] = np.ceil(out["BOREHOLE_FEET"] / BOREHOLE_DEPTH_FT).clip(lower=1)
    return out


class CostDistribution(BaseModel):
    """Summary statistics for a cost distribution."""

    mean: float
    std: float
    min: float
    max: float
    p5: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float

    @classmethod
    def from_series(cls, series: pd.Series) -> "CostDistribution":
        return cls(
            mean=float(series.mean()),
            std=float(series.std()),
            min=float(series.min()),
            max=float(series.max()),
            p5=float(series.quantile(0.05)),
            p10=float(series.quantile(0.10)),
            p25=float(series.quantile(0.25)),
            p50=float(series.quantile(0.50)),
            p75=float(series.quantile(0.75)),
            p90=float(series.quantile(0.90)),
            p95=float(series.quantile(0.95)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "p5": self.p5,
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
        }


# ── Quantity factor types ─────────────────────────────────────────────────────


class QuantityFactor(BaseModel):
    """Base class for quantity factors."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compute(
        self,
        features: pd.DataFrame,
        context_df: pd.DataFrame | None = None,
        trigger_column: str | None = None,
    ) -> pd.Series:
        raise NotImplementedError


class FixedQuantity(QuantityFactor):
    amount: float
    error_scale: float | None = None
    description: str = ""
    source: str = ""

    def compute(self, features: pd.DataFrame, context_df: pd.DataFrame | None = None, trigger_column: str | None = None) -> pd.Series:
        if self.error_scale is None or self.error_scale == 0:
            return pd.Series(np.full(len(features), self.amount), index=features.index)
        std_dev = abs(self.amount) * self.error_scale
        random_values = np.random.normal(self.amount, std_dev, len(features))
        clipped = np.maximum(random_values, 0) if self.amount > 0 else np.minimum(random_values, 0)
        return pd.Series(clipped, index=features.index)


class LinearQuantity(QuantityFactor):
    coefficient: float
    indicator_cols: list[str]
    error_scale: float | None = None
    units: str = ""
    per: str = ""
    description: str = ""
    source: str = ""

    def compute(self, features: pd.DataFrame, context_df: pd.DataFrame | None = None, trigger_column: str | None = None) -> pd.Series:
        available = [c for c in self.indicator_cols if c in features.columns]
        if not available:
            logger.warning(f"No indicator columns found for LinearQuantity: {self.indicator_cols}")
            return pd.Series(0, index=features.index)
        base = features[available].product(axis=1)
        result = self.coefficient * base
        if self.error_scale is not None and self.error_scale > 0:
            error_factor = np.random.normal(1.0, self.error_scale, len(features)).clip(min=0)
            result = result * pd.Series(error_factor, index=features.index)
        return result


class PercentQuantity(QuantityFactor):
    percent: float
    limit: float | None = None
    limit_unit: str | None = None
    error_scale: float | None = None
    description: str = ""
    source: str = ""

    def compute(self, features: pd.DataFrame, context_df: pd.DataFrame | None = None, trigger_column: str | None = None) -> pd.Series:
        if context_df is None:
            return pd.Series(0, index=features.index)
        cost_cols = [col for col in context_df.columns if col.startswith("cost.")]
        if not cost_cols:
            return pd.Series(0, index=features.index)
        detailed_cols = [c for c in cost_cols if c.startswith(f"cost.{trigger_column}.")]
        if detailed_cols:
            context_col = detailed_cols[0]
        elif f"cost.{trigger_column}" in cost_cols:
            context_col = f"cost.{trigger_column}"
        else:
            return pd.Series(0, index=features.index)
        base_quantity = context_df[context_col] * self.percent
        if self.error_scale is not None and self.error_scale > 0:
            error_factor = np.random.normal(1.0, self.error_scale, len(features)).clip(min=0)
            base_quantity = base_quantity * pd.Series(error_factor, index=features.index)
        if self.limit is not None:
            base_quantity = base_quantity.clip(upper=self.limit)
        return base_quantity


class RetrofitQuantity(BaseModel):
    """Single retrofit quantity (cost or incentive) for one upgrade transition."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    trigger_column: str
    initial: str | None
    final: str
    quantity_factors: list[QuantityFactor]
    order: list[str]

    def compute(self, features: pd.DataFrame, context_df: pd.DataFrame | None = None, output_key: str = "cost") -> pd.Series:
        if not self.quantity_factors:
            return pd.Series(0, index=features.index, name=f"{output_key}.{self.trigger_column}")
        factor_class_map = {"FixedQuantity": FixedQuantity, "PercentQuantity": PercentQuantity, "LinearQuantity": LinearQuantity}
        total = pd.Series(0.0, index=features.index)
        for factor_type_name in self.order:
            factor_class = factor_class_map.get(factor_type_name)
            if factor_class is None:
                continue
            for factor in [f for f in self.quantity_factors if isinstance(f, factor_class)]:
                if isinstance(factor, PercentQuantity):
                    total += factor.compute(features, context_df, self.trigger_column)
                else:
                    total += factor.compute(features, context_df)
        return total.rename(f"{output_key}.{self.trigger_column}")


class RetrofitQuantities(BaseModel):
    """Collection of retrofit quantity definitions loaded from JSON."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    quantities: list[RetrofitQuantity]
    output_key: str = "cost"
    raise_on_duplicate_trigger: bool = True
    create_metadata: bool = False
    metadata_aggregation: str | None = None

    @classmethod
    def from_json(cls, path: Path) -> "RetrofitQuantities":
        with open(path) as f:
            data = json.load(f)
        quantities = []
        for q in data.get("quantities", []):
            factors: list[QuantityFactor] = []
            for fd in q.get("quantity_factors", []):
                ft = fd.get("type")
                if ft == "FixedQuantity":
                    factors.append(FixedQuantity(amount=fd["amount"], error_scale=fd.get("error_scale"), description=fd.get("description", ""), source=fd.get("source", "")))
                elif ft == "LinearQuantity":
                    factors.append(LinearQuantity(coefficient=fd["coefficient"], indicator_cols=fd["indicator_cols"], error_scale=fd.get("error_scale"), units=fd.get("units", ""), per=fd.get("per", ""), description=fd.get("description", ""), source=fd.get("source", "")))
                elif ft == "PercentQuantity":
                    factors.append(PercentQuantity(percent=fd["percent"], limit=fd.get("limit"), limit_unit=fd.get("limit_unit"), error_scale=fd.get("error_scale"), description=fd.get("description", ""), source=fd.get("source", "")))
            quantities.append(RetrofitQuantity(trigger_column=q["trigger_column"], initial=q.get("initial"), final=q["final"], quantity_factors=factors, order=q.get("order", ["FixedQuantity", "LinearQuantity", "PercentQuantity"])))
        return cls(quantities=quantities, output_key=data.get("output_key", "cost"), raise_on_duplicate_trigger=data.get("raise_on_duplicate_trigger", True), create_metadata=data.get("create_metadata", False), metadata_aggregation=data.get("metadata_aggregation"))

    @property
    def all_trigger_features(self) -> list[str]:
        return list({q.trigger_column for q in self.quantities})

    def compute(self, features: pd.DataFrame, context_df: pd.DataFrame | None = None, selected_quantities: list[RetrofitQuantity] | None = None) -> pd.DataFrame:
        selected = selected_quantities if selected_quantities is not None else self.quantities
        if not selected:
            return pd.DataFrame({f"{self.output_key}.Total": [0] * len(features)})
        parts = [q.compute(features, context_df, self.output_key) for q in selected]
        df = pd.concat(parts, axis=1)
        df[f"{self.output_key}.Total"] = df.sum(axis=1)
        return df


class ScenarioCostResult(BaseModel):
    """Result container for a single scenario cost calculation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_name: str
    n_samples: int
    cost_distribution: CostDistribution
    cost_by_component: dict[str, CostDistribution]
    incentives_by_bracket: dict[str, CostDistribution]
    net_cost_by_bracket: dict[str, CostDistribution]
    payback_by_bracket: dict[str, CostDistribution]
    npv_by_bracket: dict[str, CostDistribution] | None = None
    energy_cost_distribution: CostDistribution | None = None
    annual_energy_savings_distribution: CostDistribution | None = None
    detailed_costs: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "scenario_name": self.scenario_name,
            "n_samples": self.n_samples,
            "total_cost": self.cost_distribution.to_dict(),
            "cost_by_component": {k: v.to_dict() for k, v in self.cost_by_component.items()},
            "incentives_by_bracket": {k: v.to_dict() for k, v in self.incentives_by_bracket.items()},
            "net_cost_by_bracket": {k: v.to_dict() for k, v in self.net_cost_by_bracket.items()},
            "payback_by_bracket": {k: v.to_dict() for k, v in self.payback_by_bracket.items()},
        }
        if self.npv_by_bracket:
            result["npv_by_bracket"] = {k: v.to_dict() for k, v in self.npv_by_bracket.items()}
        if self.energy_cost_distribution:
            result["energy_cost"] = self.energy_cost_distribution.to_dict()
        if self.annual_energy_savings_distribution:
            result["annual_energy_savings"] = self.annual_energy_savings_distribution.to_dict()
        return result


class RetrofitScenarioAnalyzer(BaseModel):
    """Compute retrofit costs, incentives, and energy savings from simulation output.

    Args:
        model_output_path: Path to parquet file with energy simulation results.
            Must have ``retrofit.scenario`` as an index level or column.
        costs_path: Path to retrofit costs JSON (``RetrofitQuantities`` format).
        incentives_path: Path to incentives JSON (same format).
        energy_costs_path: Path to energy costs JSON with fuel prices by year.
        energy_cost_year: Year key to look up in the energy costs JSON.
        income_brackets: Income bracket names used in incentive calculations.
            Defaults to ``["IncomeEligible", "AllCustomers"]``.
        counties: List of county names for which to add ``feature.location.in_county.*``
            indicator columns used by cost LinearQuantity factors.  Pass ``None``
            (default) to skip county indicators — appropriate when costs JSON does
            not reference county columns.
        region: Region string set on ``feature.location.region`` for incentive
            calculations.  Pass ``None`` to skip (default).
        baseline_scenario: Name of the baseline scenario.  Defaults to
            ``"baseline"``; the first scenario found is used as fallback.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input fields
    model_output_path: Path
    costs_path: Path | None = None
    incentives_path: Path | None = None
    energy_costs_path: Path | None = None
    energy_cost_year: str = "2025"
    income_brackets: list[str] = Field(default_factory=lambda: list(DEFAULT_INCOME_BRACKETS))
    counties: list[str] = Field(default_factory=list)
    region: str | None = None
    baseline_scenario: str = "baseline"

    # Computed fields (set in model_post_init)
    cost_config: RetrofitQuantities | None = None
    incentive_config: RetrofitQuantities | None = None
    energy_costs: dict[str, float] = Field(default_factory=dict)
    model_output: pd.DataFrame | None = None

    def model_post_init(self, __context: Any) -> None:
        data_dir = Path(__file__).parent.parent.parent / "inputs" / "costs"
        costs_path = self.costs_path or data_dir / "retrofit_costs.json"
        incentives_path = self.incentives_path or data_dir / "incentives.json"
        energy_costs_path = self.energy_costs_path or data_dir / "energy_costs.json"

        self.cost_config = RetrofitQuantities.from_json(costs_path)
        self.incentive_config = RetrofitQuantities.from_json(incentives_path)
        self.energy_costs = self._load_energy_costs(energy_costs_path)
        self.model_output = pd.read_parquet(self.model_output_path)
        logger.info(f"Loaded model output: {len(self.model_output)} rows, {len(self.model_output.columns)} columns")

    def _load_energy_costs(self, path: Path) -> dict[str, float]:
        if not path.exists():
            logger.warning(f"Energy costs file not found: {path}, using defaults")
            return {"electricity": 0.22, "natural_gas": 0.05, "fuel_oil": 0.10, "propane": 0.08}
        with open(path) as f:
            data = json.load(f)
        y = self.energy_cost_year
        costs: dict[str, float] = {}
        if "electricity" in data:
            costs["electricity"] = float(data["electricity"]["values"].get(y, 0.22))
        if "natural_gas" in data:
            costs["natural_gas"] = float(data["natural_gas"]["values"].get(y, 1.50)) / KWH_PER_THERM
        if "fuel_oil" in data:
            costs["fuel_oil"] = float(data["fuel_oil"]["values"].get(y, 4.00)) / KWH_PER_GALLON_OIL
        if "propane" in data:
            costs["propane"] = float(data["propane"]["values"].get(y, 3.00)) / KWH_PER_GALLON_PROPANE
        return costs

    def get_scenarios(self) -> list[str]:
        if isinstance(self.model_output.index, pd.MultiIndex):
            return self.model_output.index.get_level_values("retrofit.scenario").unique().tolist()
        if "retrofit.scenario" in self.model_output.columns:
            return self.model_output["retrofit.scenario"].unique().tolist()
        return []

    def extract_features_for_scenario(self, scenario: str) -> pd.DataFrame:
        """Extract and reset-index the rows for a given scenario."""
        if isinstance(self.model_output.index, pd.MultiIndex):
            mask = self.model_output.index.get_level_values("retrofit.scenario") == scenario
            return self.model_output.loc[mask].reset_index()
        if "retrofit.scenario" in self.model_output.columns:
            return self.model_output[self.model_output["retrofit.scenario"] == scenario].copy()
        return self.model_output.copy()

    def prepare_cost_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived geometry / system columns needed by cost quantity factors."""
        out = features_df.copy()

        # Heating capacity
        if "feature.calculated.heating_capacity_kW" not in out.columns:
            area_col = "feature.geometry.energy_model_conditioned_area"
            if area_col in out.columns:
                out["feature.calculated.heating_capacity_kW"] = (out[area_col] * 0.05).clip(lower=5.3, upper=31.6)
            else:
                out["feature.calculated.heating_capacity_kW"] = 10.5

        # County indicators
        county_col = next((c for c in ["feature.location.county", "building.county"] if c in out.columns), None)
        for county in self.counties:
            col = f"feature.location.in_county.{county}"
            out[col] = out[county_col] == county if county_col else False

        # Gas / cooling indicators
        heating_col = "feature.semantic.Heating"
        if heating_col in out.columns:
            out["feature.system.has_gas.true"] = out[heating_col].isin(NG_HEATING_SYSTEMS)
            out["feature.system.has_gas.false"] = ~out["feature.system.has_gas.true"]
        else:
            out["feature.system.has_gas.true"] = False
            out["feature.system.has_gas.false"] = True

        cooling_col = "feature.semantic.Cooling"
        cooling_systems = ["ACWindow", "ACCentral", "WindowASHP", "ASHPCooling", "GSHPCooling"]
        if cooling_col in out.columns:
            out["feature.system.has_cooling.true"] = out[cooling_col].isin(cooling_systems)
            out["feature.system.has_cooling.false"] = ~out["feature.system.has_cooling.true"]
        else:
            out["feature.system.has_cooling.true"] = False
            out["feature.system.has_cooling.false"] = True

        # Geometry defaults
        if "feature.geometry.est_fp_ratio" not in out.columns:
            out["feature.geometry.est_fp_ratio"] = 1.0
        if "feature.geometry.est_uniform_linear_scaling_factor" not in out.columns:
            out["feature.geometry.est_uniform_linear_scaling_factor"] = 1.0

        # Derived geometry
        short_col, long_col = "feature.geometry.short_edge", "feature.geometry.long_edge"
        if short_col in out.columns and long_col in out.columns:
            short, long = out[short_col], out[long_col]
            out["feature.geometry.computed.footprint_area"] = short * long
            perimeter = 2 * (short + long)
            out["feature.geometry.computed.perimeter"] = perimeter
            floors = out.get("feature.geometry.num_floors", pd.Series(1, index=out.index))
            f2f = out.get("feature.geometry.f2f_height", pd.Series(3.0, index=out.index))
            if "feature.geometry.num_floors" not in out.columns:
                floors = pd.Series(1, index=out.index)
            if "feature.geometry.f2f_height" not in out.columns:
                f2f = pd.Series(3.0, index=out.index)
            facade = perimeter * f2f * floors
            out["feature.geometry.computed.whole_bldg_facade_area"] = facade
            wwr = out["feature.geometry.wwr"] if "feature.geometry.wwr" in out.columns else pd.Series(0.15, index=out.index)
            out["feature.geometry.computed.window_area"] = facade * wwr
            is_attic = out["feature.geometry.roof_is_attic.num"] if "feature.geometry.roof_is_attic.num" in out.columns else pd.Series(1, index=out.index)
            attic_h = out["feature.geometry.attic_height"] if "feature.geometry.attic_height" in out.columns else short * 0.25
            run = short / 2
            hypotenuse = np.sqrt(run**2 + attic_h**2)
            out["feature.geometry.computed.roof_surface_area"] = (
                2 * hypotenuse * long * is_attic
                + out["feature.geometry.computed.footprint_area"] * (1 - is_attic)
            )

        # Basement / attic numeric indicators
        for space in ["basement", "attic"]:
            for prop in ["exists", "occupied"]:
                src = f"feature.extra_spaces.{space}.{prop}"
                dst = f"{src}.num"
                out[dst] = (out[src] == "Yes").astype(float) if src in out.columns else 0.0
            if space == "basement":
                occ = f"feature.extra_spaces.{space}.occupied"
                out[f"feature.extra_spaces.{space}.not_occupied.num"] = (
                    (out[occ] != "Yes").astype(float) if occ in out.columns else 1.0
                )

        out = compute_equipment_quantities(out)
        return out

    def prepare_incentive_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        out = features_df.copy()
        if self.region is not None:
            out["feature.location.region"] = self.region
        heating_col = "feature.semantic.Heating"
        if heating_col in out.columns:
            out["feature.fuel.natural_gas"] = out[heating_col].isin(NG_HEATING_SYSTEMS)
            out["feature.fuel.oil"] = out[heating_col].isin(OIL_HEATING_SYSTEMS)
        else:
            out["feature.fuel.natural_gas"] = False
            out["feature.fuel.oil"] = False
        out["feature.fuel.electricity"] = True
        for bracket in self.income_brackets:
            out[f"feature.homeowner.in_bracket.{bracket}"] = False
        return out

    def compute_energy_costs(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Compute annual energy costs from simulation output columns."""
        result = pd.DataFrame(index=features_df.index)
        elec = pd.Series(0.0, index=features_df.index)
        gas = pd.Series(0.0, index=features_df.index)
        oil = pd.Series(0.0, index=features_df.index)

        if "site_kwh" in features_df.columns:
            site_kwh = features_df["site_kwh"].fillna(0.0)
            heating_col = "feature.semantic.Heating"
            if heating_col in features_df.columns:
                is_elec = features_df[heating_col].isin(ELECTRIC_HEATING_SYSTEMS)
                is_gas = features_df[heating_col].isin(NG_HEATING_SYSTEMS)
                is_oil = features_df[heating_col].isin(OIL_HEATING_SYSTEMS)
                elec = site_kwh.where(is_elec, 0) + site_kwh.where(is_gas, 0) * 0.3 + site_kwh.where(is_oil, 0) * 0.3
                gas = site_kwh.where(is_gas, 0) * 0.7
                oil = site_kwh.where(is_oil, 0) * 0.7
            else:
                elec = site_kwh
        else:
            area_col = "feature.geometry.energy_model_conditioned_area"
            area_m2 = features_df[area_col] if area_col in features_df.columns else pd.Series(200.0, index=features_df.index)

            def _get_energy(end_use: str) -> pd.Series | None:
                for col in [("Energy", end_use), f"Energy.{end_use}"]:
                    if col in features_df.columns:
                        return pd.Series(np.asarray(features_df[col].values, dtype=float) * np.asarray(area_m2.values, dtype=float), index=features_df.index)
                return None

            lighting = _get_energy("Lighting")
            equipment = _get_energy("Equipment")
            cooling = _get_energy("Cooling")
            heating = _get_energy("Heating")
            dhw = _get_energy("Domestic Hot Water")

            if lighting is not None:
                elec += lighting
            if equipment is not None:
                elec += equipment
            if cooling is not None:
                elec += cooling

            heating_col = "feature.semantic.Heating"
            if heating is not None and heating_col in features_df.columns:
                elec += heating.where(features_df[heating_col].isin(ELECTRIC_HEATING_SYSTEMS), 0)
                gas += heating.where(features_df[heating_col].isin(NG_HEATING_SYSTEMS), 0)
                oil += heating.where(features_df[heating_col].isin(OIL_HEATING_SYSTEMS), 0)

            dhw_col = "feature.semantic.DHW"
            if dhw is not None and dhw_col in features_df.columns:
                elec += dhw.where(features_df[dhw_col].isin(ELECTRIC_DHW_SYSTEMS), 0)
                gas += dhw.where(features_df[dhw_col].isin(NG_DHW_SYSTEMS), 0)

        result["energy_cost.electricity_kwh"] = elec
        result["energy_cost.natural_gas_kwh"] = gas
        result["energy_cost.fuel_oil_kwh"] = oil
        result["energy_cost.electricity"] = elec * self.energy_costs.get("electricity", 0.22)
        result["energy_cost.natural_gas"] = gas * self.energy_costs.get("natural_gas", 0.05)
        result["energy_cost.fuel_oil"] = oil * self.energy_costs.get("fuel_oil", 0.10)
        result["energy_cost.Total"] = result["energy_cost.electricity"] + result["energy_cost.natural_gas"] + result["energy_cost.fuel_oil"]
        return result

    def _compute_costs_sample_by_sample(
        self,
        baseline_features: pd.DataFrame,
        scenario_features: pd.DataFrame,
        cost_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute costs for each sample based on its specific semantic changes."""
        trigger_costs: dict[str, pd.Series] = {t: pd.Series(0.0, index=cost_features.index) for t in self.cost_config.all_trigger_features}
        all_costs = pd.DataFrame(index=cost_features.index)

        quantities_checked = quantities_applied = quantities_missing = quantities_no_match = 0

        for quantity in self.cost_config.quantities:
            quantities_checked += 1
            trigger_col = f"feature.semantic.{quantity.trigger_column}"

            if trigger_col not in baseline_features.columns or trigger_col not in scenario_features.columns:
                quantities_missing += 1
                continue

            baseline_vals = np.asarray(baseline_features[trigger_col].values)
            scenario_vals = np.asarray(scenario_features[trigger_col].values)
            final_match = scenario_vals == quantity.final
            initial_match = (baseline_vals == quantity.initial) if quantity.initial is not None else np.ones(len(baseline_vals), dtype=bool)
            applies_mask = (baseline_vals != scenario_vals) & final_match & initial_match

            if not applies_mask.any():
                quantities_no_match += 1
                continue

            applicable = cost_features.loc[applies_mask].copy()
            context_cols = [c for c in all_costs.columns if c.startswith(f"{self.cost_config.output_key}.")]
            context_for_applicable = all_costs.loc[applies_mask, context_cols] if context_cols else None

            cost_values = quantity.compute(applicable, context_df=context_for_applicable, output_key=self.cost_config.output_key)
            trigger_costs[quantity.trigger_column].loc[applies_mask] = (
                trigger_costs[quantity.trigger_column].loc[applies_mask] + cost_values.values
            )
            all_costs.loc[applies_mask, f"{self.cost_config.output_key}.{quantity.trigger_column}"] = trigger_costs[quantity.trigger_column][applies_mask]
            quantities_applied += 1

        logger.info(f"Costs: {quantities_applied}/{quantities_checked} applied, {quantities_no_match} no match, {quantities_missing} missing cols")

        result = pd.DataFrame(index=cost_features.index)
        for trigger, series in trigger_costs.items():
            if series.abs().sum() > 0:
                result[f"{self.cost_config.output_key}.{trigger}"] = series

        cost_component_cols = [c for c in result.columns if c != f"{self.cost_config.output_key}.Total"]
        result[f"{self.cost_config.output_key}.Total"] = result[cost_component_cols].sum(axis=1) if cost_component_cols else 0.0
        return result

    def compute_all_scenario_costs(self) -> pd.DataFrame:
        """Compute costs for all non-baseline scenarios vs the baseline.

        Returns:
            Flat DataFrame (one row per building × scenario) with cost, incentive,
            net cost, and energy savings columns ready for use by
            ``PropensityModelEngine``.
        """
        scenarios = self.get_scenarios()
        if not scenarios:
            logger.error("No scenarios found in model output")
            return pd.DataFrame()

        # Resolve baseline scenario name
        base_name = self.baseline_scenario
        if base_name not in scenarios:
            for s in scenarios:
                if "baseline" in s.lower():
                    base_name = s
                    break
            else:
                base_name = scenarios[0]
            logger.warning(f"Using '{base_name}' as baseline scenario")

        baseline_df = self.extract_features_for_scenario(base_name)
        baseline_cost_feats = self.prepare_cost_features(baseline_df)
        baseline_energy = self.compute_energy_costs(baseline_cost_feats)

        all_results: list[pd.DataFrame] = []

        for scenario in scenarios:
            if scenario == base_name:
                continue
            logger.info(f"Processing scenario: {scenario}")

            scenario_df = self.extract_features_for_scenario(scenario)
            cost_feats = self.prepare_cost_features(scenario_df)

            cost_df = self._compute_costs_sample_by_sample(baseline_df, scenario_df, cost_feats)
            scenario_energy = self.compute_energy_costs(cost_feats)
            energy_savings = baseline_energy["energy_cost.Total"].values - scenario_energy["energy_cost.Total"].values

            incentive_base = self.prepare_incentive_features(cost_feats)

            row = scenario_df.copy()
            row["retrofit.scenario"] = scenario
            for col in cost_df.columns:
                row[col] = cost_df[col].values
            row["energy_cost.annual_savings"] = energy_savings
            row["energy_cost.Total"] = scenario_energy["energy_cost.Total"].values

            total_cost = cost_df.get(f"{self.cost_config.output_key}.Total", pd.Series(0.0, index=row.index))

            for bracket in self.income_brackets:
                bracket_feats = incentive_base.copy()
                bracket_feats[f"feature.homeowner.in_bracket.{bracket}"] = True
                incentive_df = self.incentive_config.compute(bracket_feats, context_df=cost_df)
                incentive_total = incentive_df.get("incentive.Total", pd.Series(0.0, index=row.index))
                row[f"incentive.{bracket}"] = incentive_total.values
                net = total_cost.values - incentive_total.values
                row[f"net_cost.{bracket}"] = net
                row[f"adjusted_net_cost.{bracket}"] = net

            all_results.append(row)

        if not all_results:
            logger.warning("No non-baseline scenarios found")
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def export_results(self, df: pd.DataFrame, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info(f"Exported {len(df)} rows to {output_path}")
