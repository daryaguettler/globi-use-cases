"""Apply propensity models to retrofit cost data.

Takes per-building retrofit cost output (from ``RetrofitScenarioAnalyzer``) and
computes the probability that each building's occupants will accept a retrofit deal.

Residential buildings
---------------------
Uses a logistic regression model with demographic features sampled from census
tract distributions (Monte Carlo).  For each building-scenario row, ``n_monte_carlo_samples``
demographic draws are made; the resulting probability distribution is summarised
as min / max / mean / sd.

Defaults use Massachusetts-specific model coefficients and county concern levels.
Override via ``model_coefficients``, ``default_county_concern``, and
``county_fips_map`` constructor arguments to adapt to other geographies.

Commercial buildings
--------------------
Purely financial: NPV-based binary threshold (0 % or 100 %).
Propensity is 100 % when NPV ≥ ``npv_threshold_usd``, else 0 %.

Output columns added
--------------------
- ``acceptance_probability``              mean propensity (residential) or NPV binary (commercial)
- ``propensity_min/max/mean/sd``          Monte Carlo statistics (residential only)
- ``acceptance_probability_min/max/mean`` aliases for the MC stats
- ``acceptance_probability_li/moderate/non_lmi`` cohort-level estimates at fixed incomes

Example usage::

    engine = PropensityModelEngine(
        policy_impacts_path="results/GroupA/policy_impacts.pq",
        census_data_path="inputs/census/census_data.csv",
        params_path="params.yaml",
    )
    result = engine.calculate_all_probabilities()
    engine.export_results(result, "results/GroupA/propensity_results.pq")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ── Massachusetts-specific defaults ──────────────────────────────────────────
# These are trained on MA survey data.  Pass overrides to PropensityModelEngine
# to use coefficients appropriate for another geography or survey.

_DEFAULT_MODEL_COEFFICIENTS: dict[str, float] = {
    "intercept": -2.5,
    "Year built": -0.01,
    "Education": 0.3,
    "bedrooms": 0.1,
    "residents": 0.05,
    "Income": 0.008,
    "Concern": 0.4,
    "Upfront cost": -0.15,
    "Neighbor": 0.3,
    "Energy cost": 0.2,
}

_DEFAULT_COUNTY_CONCERN: dict[str, float] = {
    "Barnstable": 3.2,
    "Berkshire": 3.5,
    "Bristol": 3.1,
    "Dukes": 3.8,
    "Essex": 3.3,
    "Franklin": 3.6,
    "Hampden": 3.0,
    "Hampshire": 3.7,
    "Middlesex": 3.4,
    "Nantucket": 3.9,
    "Norfolk": 3.3,
    "Plymouth": 3.2,
    "Suffolk": 3.5,
    "Worcester": 3.2,
}

# MA county FIPS (int) → county name
_DEFAULT_COUNTY_FIPS_MAP: dict[int, str] = {
    1: "Barnstable", 3: "Berkshire", 5: "Bristol", 7: "Dukes",
    9: "Essex", 11: "Franklin", 13: "Hampden", 15: "Hampshire",
    17: "Middlesex", 19: "Nantucket", 21: "Norfolk", 23: "Plymouth",
    25: "Suffolk", 27: "Worcester",
}

# Survey scale categories used by the logistic model
EDUCATION_CATEGORIES: list[float] = [1.0, 2.0, 3.0, 4.0]
HOUSEHOLD_SIZE_CATEGORIES: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
INCOME_CATEGORIES: list[float] = [5.0, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 55.0, 67.5, 87.5, 112.5, 137.5, 175.0, 225.0]

_DEFAULT_COHORT_INCOME_K_USD: dict[str, float] = {"li": 25.0, "moderate": 67.5, "non_lmi": 125.0}


def _merge_cohort_income_levels(propensity_cfg: dict) -> dict[str, float]:
    raw = (propensity_cfg or {}).get("cohort_income_levels_k_usd") or {}
    out = dict(_DEFAULT_COHORT_INCOME_K_USD)
    for k in out:
        if k in raw and raw[k] is not None:
            out[k] = float(raw[k])
    return out


class PropensityResult(BaseModel):
    """Container for propensity calculation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    n_buildings: int
    n_scenarios: int
    mean_acceptance_probability: float


class PropensityModelEngine(BaseModel):
    """Engine for calculating retrofit deal acceptance probabilities.

    Args:
        policy_impacts_path: Path to policy impacts parquet (output of
            ``RetrofitScenarioAnalyzer.compute_all_scenario_costs``).
        census_data_path: Path to census data CSV with tract-level demographic
            distributions.  Optional; if absent, uniform defaults are used.
        climate_opinions_path: Optional path to county-level climate opinions CSV.
        enhanced_building_data_path: Optional path to enhanced building parquet.
        params_path: Optional path to params.yaml for loading configuration.
        policy_impacts_df: Pre-loaded DataFrame alternative to path.
        model_coefficients: Logistic regression coefficients dict.  Defaults to
            ``_DEFAULT_MODEL_COEFFICIENTS`` (Massachusetts-calibrated).
        default_county_concern: Fallback concern level per county name (1-5 scale).
            Defaults to ``_DEFAULT_COUNTY_CONCERN`` (Massachusetts).
        county_fips_map: Mapping from integer FIPS code to county name, used when
            the census CSV stores county as a FIPS integer.  Defaults to
            ``_DEFAULT_COUNTY_FIPS_MAP`` (Massachusetts).
        neighbor_effect: Default neighbour adoption influence (1-5 scale).
        random_seed: RNG seed for reproducibility.
        n_monte_carlo_samples: Monte Carlo draws per residential building row.
        npv_years: Years over which commercial NPV is evaluated.
        wacc: Discount rate for commercial NPV.
        npv_threshold_usd: NPV must exceed this (USD) for 100 % propensity.
        energy_price_growth_rate: Annual energy price growth for commercial NPV.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input paths
    policy_impacts_path: Path | None = None
    census_data_path: Path | None = None
    climate_opinions_path: Path | None = None
    enhanced_building_data_path: Path | None = None
    params_path: Path | None = None

    # Pre-loaded data alternative to path
    policy_impacts_df: pd.DataFrame | None = None

    # Geography / model overrides
    model_coefficients: dict[str, float] | None = None
    default_county_concern: dict[str, float] | None = None
    county_fips_map: dict[int, str] | None = None

    # Behaviour parameters
    neighbor_effect: float | None = None
    random_seed: int | None = None
    n_monte_carlo_samples: int | None = None
    incentive_by_income: dict[float, float] | None = None

    # Commercial model parameters
    npv_years: int | None = None
    wacc: float | None = None
    npv_threshold_usd: float | None = None
    energy_price_growth_rate: float | None = None

    # Legacy aliases
    equipment_lifetime_years: int | None = None
    discount_rate: float | None = None

    # Computed fields (set in model_post_init)
    params: dict[str, Any] = Field(default_factory=dict)
    data: pd.DataFrame | None = None
    census_data: pd.DataFrame | None = None
    county_concern: dict[str, float] = Field(default_factory=dict)
    enhanced_building_data: pd.DataFrame | None = None
    compute_cohort_acceptance: bool = True
    cohort_income_levels_k_usd: dict[str, float] = Field(default_factory=dict)
    rng: Any = None
    tract_distributions: dict[str, dict] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.params = {}
        if self.params_path is not None:
            p = Path(self.params_path)
            if p.exists():
                with open(p) as f:
                    self.params = yaml.safe_load(f) or {}
            else:
                logger.warning(f"Params file not found: {self.params_path}")

        prop_cfg = self.params.get("propensity", {})
        comm_cfg = self.params.get("commercial_propensity", {})

        # Load data
        if self.policy_impacts_df is not None:
            self.data = self.policy_impacts_df.copy()
        elif self.policy_impacts_path is not None:
            self.data = pd.read_parquet(self.policy_impacts_path)
        else:
            msg = "Must provide policy_impacts_path or policy_impacts_df"
            raise ValueError(msg)

        # Optional tract filter from params
        tract_ids_filter = self.params.get("gsep_town_analysis", {}).get("tract_ids")
        if tract_ids_filter is not None:
            tract_ids_filter = [str(t).strip() for t in tract_ids_filter]
            tract_set = set(tract_ids_filter)
            for col in ["tract_id", "building.tract_id", "feature.location.tract_id", "GEOID20", "GEOID"]:
                if col in self.data.columns:
                    before = len(self.data)
                    s = self.data[col].astype(str).str.replace(r"\.0+$", "", regex=True)
                    mask = s.isin(tract_set) | ((s.str.len() >= 6) & s.str[-6:].isin(tract_set))
                    self.data = self.data[mask].copy()
                    logger.info(f"Filtered to tract_ids: {len(self.data)} rows (was {before})")
                    break

        # Resolve model coefficients / concern maps — use overrides or defaults
        self.model_coefficients = self.model_coefficients or _DEFAULT_MODEL_COEFFICIENTS
        self.default_county_concern = self.default_county_concern or _DEFAULT_COUNTY_CONCERN
        self.county_fips_map = self.county_fips_map or _DEFAULT_COUNTY_FIPS_MAP

        # Census / climate data
        self.census_data = self._load_census_data()
        self.county_concern = self._load_climate_opinions()
        self.enhanced_building_data = self._load_enhanced_building_data()

        # Resolve behaviour parameters
        self.neighbor_effect = self.neighbor_effect if self.neighbor_effect is not None else prop_cfg.get("neighbor_effect", 3.0)
        self.random_seed = self.random_seed if self.random_seed is not None else prop_cfg.get("random_seed", 42)
        self.rng = np.random.default_rng(self.random_seed)
        self.n_monte_carlo_samples = self.n_monte_carlo_samples if self.n_monte_carlo_samples is not None else prop_cfg.get("n_monte_carlo_samples", 100)
        self.compute_cohort_acceptance = bool(prop_cfg.get("compute_cohort_acceptance", True))
        self.cohort_income_levels_k_usd = _merge_cohort_income_levels(prop_cfg)

        # Resolve commercial parameters
        self.npv_years = self.npv_years if self.npv_years is not None else (self.equipment_lifetime_years if self.equipment_lifetime_years is not None else comm_cfg.get("npv_years") or comm_cfg.get("equipment_lifetime_years", 15))
        self.wacc = self.wacc if self.wacc is not None else (self.discount_rate if self.discount_rate is not None else comm_cfg.get("wacc") or comm_cfg.get("discount_rate", 0.07))
        self.npv_threshold_usd = self.npv_threshold_usd if self.npv_threshold_usd is not None else comm_cfg.get("npv_threshold_usd", 0.0)
        self.energy_price_growth_rate = self.energy_price_growth_rate if self.energy_price_growth_rate is not None else comm_cfg.get("energy_price_growth_rate", 0.02)

        self._precompute_tract_distributions()
        self._classify_buildings()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_census_data(self) -> pd.DataFrame | None:
        if self.census_data_path is None or not self.census_data_path.exists():
            logger.warning("Census data not found, will use default distributions")
            return None

        df = pd.read_csv(self.census_data_path)

        def _to_int(code: object) -> int | None:
            try:
                return int(str(code).split(".")[0])
            except Exception:
                return None

        def _lookup_county(code: object) -> str | None:
            key = _to_int(code)
            return self.county_fips_map.get(key) if key is not None else None

        df["county_name"] = df["county"].apply(_lookup_county)
        df["tract_str"] = df["tract"].astype(str).str.split(".").str[0].str.strip().str.zfill(6)
        logger.info(f"Loaded census data: {len(df)} tracts")
        return df

    def _load_enhanced_building_data(self) -> pd.DataFrame | None:
        if self.enhanced_building_data_path is None or not self.enhanced_building_data_path.exists():
            return None
        try:
            df = pd.read_parquet(self.enhanced_building_data_path)
            logger.info(f"Loaded enhanced building data: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error loading enhanced building data: {e}")
            return None
        else:
            return df

    def _load_climate_opinions(self) -> dict[str, float]:
        """Load county-level climate concern (1–5 scale) from CSV."""
        if self.climate_opinions_path is not None and self.climate_opinions_path.exists():
            try:
                df = pd.read_csv(self.climate_opinions_path)
                if "GeoType" in df.columns:
                    df = df[df["GeoType"] == "county"]
                    df["county_name"] = df["GeoName"].str.split(",").str[0].str.split().str[0]
                    concern = np.clip(df["worried"].fillna(50.0) / 20.0, 1.0, 5.0)
                    result = dict(zip(df["county_name"], concern, strict=False))
                    logger.info(f"Loaded climate opinions for {len(result)} counties")
                    return result
            except Exception as e:
                logger.warning(f"Error loading climate opinions: {e}")
        return dict(self.default_county_concern)

    # ── Census tract distributions ────────────────────────────────────────────

    def _precompute_tract_distributions(self) -> None:
        self.tract_distributions: dict[str, dict] = {}
        if self.census_data is None:
            return
        for _, row in self.census_data.iterrows():
            tract_key = f"{row['county_name']}_{row['tract_str']}"
            edu_counts = np.array([
                row.get("education_high_school_grad", 0) + row.get("education_ged", 0),
                row.get("education_associates_degree", 0),
                row.get("education_bachelors_degree", 0),
                row.get("education_masters_degree", 0) + row.get("education_professional_school_degree", 0) + row.get("education_doctorate_degree", 0),
            ], dtype=float)
            hh_counts = np.array([row.get(f"household_{i}_person", 0) for i in range(1, 7)] + [row.get("household_7_or_more_person", 0)], dtype=float)
            income_counts = np.array([
                row.get("income_less_than_10k", 0), row.get("income_10k_to_14999", 0), row.get("income_15k_to_19999", 0),
                row.get("income_20k_to_24999", 0), row.get("income_25k_to_29999", 0), row.get("income_30k_to_34999", 0),
                row.get("income_35k_to_39999", 0), row.get("income_40k_to_44999", 0), row.get("income_45k_to_49999", 0),
                row.get("income_50k_to_59999", 0), row.get("income_60k_to_74999", 0), row.get("income_75k_to_99999", 0),
                row.get("income_100k_to_124999", 0), row.get("income_125k_to_149999", 0), row.get("income_150k_to_199999", 0),
                row.get("income_200k_or_more", 0),
            ], dtype=float)
            self.tract_distributions[tract_key] = {
                "education": edu_counts / edu_counts.sum() if edu_counts.sum() > 0 else np.ones(4) / 4,
                "household_size": hh_counts / hh_counts.sum() if hh_counts.sum() > 0 else np.ones(7) / 7,
                "income": income_counts / income_counts.sum() if income_counts.sum() > 0 else np.ones(16) / 16,
                "county": row["county_name"],
            }
        logger.info(f"Precomputed distributions for {len(self.tract_distributions)} tracts")

    def _sample_demographics(
        self,
        county: str,
        tract_id: str | None = None,
        n_samples: int = 1,
        geoid: str | None = None,
    ) -> dict[str, np.ndarray]:
        if geoid:
            g = str(geoid).strip()
            if g in self.tract_distributions:
                dist = self.tract_distributions[g]
                return {
                    "education": self.rng.choice(EDUCATION_CATEGORIES, size=n_samples, p=dist["education"]),
                    "household_size": self.rng.choice(HOUSEHOLD_SIZE_CATEGORIES, size=n_samples, p=dist["household_size"]),
                    "income": self.rng.choice(INCOME_CATEGORIES, size=n_samples, p=dist["income"]),
                }

        tract_key = None
        if tract_id is not None:
            tid = str(tract_id).strip()
            tract_str = tid[-6:].zfill(6) if len(tid) >= 6 else tid.zfill(6)
            expected = f"{county}_{tract_str}"
            if expected in self.tract_distributions:
                tract_key = expected
            else:
                for k, d in self.tract_distributions.items():
                    if k.endswith(f"_{tract_str}") and d["county"] == county:
                        tract_key = k
                        break

        if tract_key is None:
            for k, d in self.tract_distributions.items():
                if d["county"] == county:
                    tract_key = k
                    break

        if tract_key is None:
            return {
                "education": self.rng.choice(EDUCATION_CATEGORIES, size=n_samples),
                "household_size": self.rng.choice(HOUSEHOLD_SIZE_CATEGORIES, size=n_samples),
                "income": self.rng.choice(INCOME_CATEGORIES, size=n_samples),
            }

        dist = self.tract_distributions[tract_key]
        return {
            "education": self.rng.choice(EDUCATION_CATEGORIES, size=n_samples, p=dist["education"]),
            "household_size": self.rng.choice(HOUSEHOLD_SIZE_CATEGORIES, size=n_samples, p=dist["household_size"]),
            "income": self.rng.choice(INCOME_CATEGORIES, size=n_samples, p=dist["income"]),
        }

    # ── Building classification ───────────────────────────────────────────────

    def _classify_buildings(self) -> None:
        residential_typologies = {"SFH", "MFH", "Single Family", "Multi Family", "Residential"}
        typology_col = next((c for c in ["feature.semantic.Typology", "Typology", "building.typology", "typology"] if c in self.data.columns), None)
        if typology_col is None:
            logger.warning("No typology column found, defaulting all buildings to residential")
            self.data["building_type"] = "residential"
            return

        def _classify(t: object) -> Literal["residential", "commercial"]:
            if pd.isna(t):
                return "residential"
            return "residential" if any(r in str(t) for r in residential_typologies) else "commercial"

        self.data["building_type"] = self.data[typology_col].apply(_classify)
        n_res = (self.data["building_type"] == "residential").sum()
        n_com = (self.data["building_type"] == "commercial").sum()
        logger.info(f"Classified: {n_res} residential, {n_com} commercial")

    # ── Probability calculation ───────────────────────────────────────────────

    def _logistic(self, age: float | np.ndarray, edu: float | np.ndarray, hh: float | np.ndarray, inc: float | np.ndarray, concern: float, cost: float | np.ndarray, neighbor: float, savings: float | np.ndarray) -> float | np.ndarray:
        mc = self.model_coefficients
        z = (mc["intercept"] + mc["Year built"] * age + mc["Education"] * edu + mc["bedrooms"] * hh + mc["residents"] * hh + mc["Income"] * inc + mc["Concern"] * concern + mc["Upfront cost"] * cost + mc["Neighbor"] * neighbor + mc["Energy cost"] * savings)
        return 1 / (1 + np.exp(-z))

    def _calculate_commercial_npv(self, capex: float, annual_savings: float, npv_years: int | None = None, wacc: float | None = None, energy_price_growth_rate: float | None = None) -> float:
        lifetime = npv_years or self.npv_years
        rate = wacc or self.wacc
        growth = energy_price_growth_rate or self.energy_price_growth_rate or 0.0
        npv = -capex
        for year in range(1, lifetime + 1):
            npv += annual_savings * ((1 + growth) ** year) / ((1 + rate) ** year)
        return npv

    def _calculate_commercial_probability(self, row: pd.Series) -> float:
        capex = row.get("cost.Total", 0.0)
        if pd.isna(capex) or capex <= 0:
            capex = row.get("adjusted_net_cost.AllCustomers", row.get("net_cost.AllCustomers", 0.0))
        savings = row.get("energy_cost.annual_savings", row.get("energy_savings", 1000.0))
        if pd.isna(savings) or savings <= 0:
            savings = 1000.0
        npv = self._calculate_commercial_npv(float(capex), float(savings))
        return 1.0 if npv >= self.npv_threshold_usd else 0.0

    # ── Main calculation ──────────────────────────────────────────────────────

    def calculate_all_probabilities(self) -> PropensityResult:
        """Calculate acceptance probabilities for all buildings and scenarios."""
        logger.info(f"Calculating probabilities for {len(self.data)} rows...")
        if "building_type" not in self.data.columns:
            self._classify_buildings()

        res_mask = self.data["building_type"] == "residential"
        com_mask = self.data["building_type"] == "commercial"
        logger.info(f"Residential: {res_mask.sum()}, Commercial: {com_mask.sum()}")

        for col in ["acceptance_probability", "propensity_min", "propensity_max", "propensity_mean", "propensity_sd",
                    "acceptance_probability_min", "acceptance_probability_max", "acceptance_probability_mean",
                    "acceptance_probability_li", "acceptance_probability_moderate", "acceptance_probability_non_lmi"]:
            self.data[col] = np.nan if "acceptance_probability" not in col else 0.0

        if res_mask.any():
            self._calculate_residential_probabilities(res_mask)
        if com_mask.any():
            self._calculate_commercial_probabilities(com_mask)

        n_buildings = self.data["building.id"].nunique() if "building.id" in self.data.columns else 0
        n_scenarios = self.data["retrofit.scenario"].nunique() if "retrofit.scenario" in self.data.columns else 0
        mean_prob = float(self.data["acceptance_probability"].mean())
        logger.info(f"Mean acceptance probability: {mean_prob:.2%}")

        return PropensityResult(data=self.data.copy(), n_buildings=n_buildings, n_scenarios=n_scenarios, mean_acceptance_probability=mean_prob)

    def _calculate_residential_probabilities(self, mask: pd.Series) -> None:
        chunk_size = 200_000
        positions = np.where(mask.values)[0]
        for start in range(0, len(positions), chunk_size):
            chunk_pos = positions[start : start + chunk_size]
            chunk_mask = pd.Series(False, index=mask.index)
            chunk_mask.iloc[chunk_pos] = True
            self._residential_chunk(chunk_mask)

    def _residential_chunk(self, mask: pd.Series) -> None:
        sub = self.data.loc[mask]
        n = len(sub)
        n_samples = self.n_monte_carlo_samples

        age_map = {"pre_1975": 60, "btw_1975_2003": 35, "post_2003": 15}

        county_col = next((c for c in ["building.county", "feature.location.county", "county", "county_fips"] if c in sub.columns), None)
        counties = sub[county_col].fillna("Middlesex").astype(str) if county_col else pd.Series(["Middlesex"] * n, index=sub.index)

        tract_col = next(
            (c for c in ["building.tract_id", "feature.location.tract_id", "tract_id", "GEOID20", "GEOID"] if c in sub.columns),
            None,
        )
        tract_ids = sub[tract_col] if tract_col else pd.Series([None] * n, index=sub.index)
        geoid_col = "geoid" if "geoid" in sub.columns else None

        if "feature.semantic.Age_bracket" in sub.columns:
            building_age = sub["feature.semantic.Age_bracket"].map(age_map).fillna(40).values.astype(float)
        else:
            building_age = np.full(n, 40.0)

        if "adjusted_net_cost.AllCustomers" in sub.columns:
            upfront_cost = sub["adjusted_net_cost.AllCustomers"].fillna(0) / 1000
        elif "net_cost.AllCustomers" in sub.columns:
            upfront_cost = sub["net_cost.AllCustomers"].fillna(0) / 1000
        else:
            upfront_cost = sub.get("cost.Total", pd.Series(0, index=sub.index)).fillna(0) / 1000
        upfront_cost = upfront_cost.values.astype(float)

        if "energy_cost.annual_savings" in sub.columns:
            energy_savings = sub["energy_cost.annual_savings"].clip(lower=0).fillna(0) / 100
        else:
            energy_savings = np.full(n, 5.0)
        energy_savings = np.asarray(energy_savings).reshape(-1).astype(float)

        county_vals = counties.values
        tract_vals = np.where(tract_ids.fillna("").astype(str).values == "nan", "", tract_ids.fillna("").astype(str).values)
        geoid_vals = None
        if geoid_col:
            geoid_vals = sub[geoid_col].astype(object).values

        propensity_min = np.empty(n)
        propensity_max = np.empty(n)
        propensity_mean = np.empty(n)
        propensity_sd = np.empty(n)
        cohort_li = np.full(n, np.nan)
        cohort_mod = np.full(n, np.nan)
        cohort_non = np.full(n, np.nan)

        # group by geoid when present and non-empty, else (county, tract)
        group_indices: dict[tuple, list[int]] = {}
        for i, (county, tract) in enumerate(zip(county_vals, tract_vals, strict=False)):
            gid = ""
            if geoid_vals is not None:
                gv = geoid_vals[i]
                if gv is not None and not pd.isna(gv) and str(gv).strip():
                    gid = str(gv).strip()
            if gid:
                k = ("geoid", gid)
            else:
                k = ("tract", str(county), str(tract))
            group_indices.setdefault(k, []).append(i)

        for gkey, indices in group_indices.items():
            idx = np.array(indices)
            m = len(idx)
            if gkey[0] == "geoid":
                demo = self._sample_demographics("", tract_id=None, n_samples=m * n_samples, geoid=gkey[1])
                cname = (self.tract_distributions.get(gkey[1]) or {}).get("county") or ""
            else:
                _, county, tract_str = gkey
                demo = self._sample_demographics(county, tract_id=tract_str or None, n_samples=m * n_samples)
                cname = county
            concern = self.county_concern.get(cname, 3.5)

            edu = demo["education"].reshape(m, n_samples)
            hh = demo["household_size"].reshape(m, n_samples)
            inc = demo["income"].reshape(m, n_samples)
            age = building_age[idx]
            cost = upfront_cost[idx]
            savings = energy_savings[idx]

            if self.incentive_by_income is not None:
                # Map each sampled income value → incentive USD, convert to k$ (cost units)
                incentive_lookup = {v: self.incentive_by_income.get(v, 0.0) for v in INCOME_CATEGORIES}
                incentive_matrix = np.vectorize(incentive_lookup.__getitem__)(inc) / 1000
                cost_input = np.maximum(cost[:, None] - incentive_matrix, 0.0)
            else:
                cost_input = cost[:, None]

            probs: np.ndarray = np.asarray(self._logistic(age[:, None], edu, hh, inc, concern, cost_input, self.neighbor_effect, savings[:, None]))
            propensity_min[idx] = probs.min(axis=1)
            propensity_max[idx] = probs.max(axis=1)
            propensity_mean[idx] = probs.mean(axis=1)
            propensity_sd[idx] = probs.std(axis=1) if n_samples > 1 else 0.0

            if self.compute_cohort_acceptance:
                edu_mean = edu.mean(axis=1)
                hh_mean = hh.mean(axis=1)
                for inc_k, arr_out in ((self.cohort_income_levels_k_usd["li"], cohort_li), (self.cohort_income_levels_k_usd["moderate"], cohort_mod), (self.cohort_income_levels_k_usd["non_lmi"], cohort_non)):
                    arr_out[idx] = self._logistic(age, edu_mean, hh_mean, inc_k, concern, cost, self.neighbor_effect, savings)

        self.data.loc[mask, "acceptance_probability"] = propensity_mean
        self.data.loc[mask, "propensity_min"] = propensity_min
        self.data.loc[mask, "propensity_max"] = propensity_max
        self.data.loc[mask, "propensity_mean"] = propensity_mean
        self.data.loc[mask, "propensity_sd"] = propensity_sd
        self.data.loc[mask, "acceptance_probability_min"] = propensity_min
        self.data.loc[mask, "acceptance_probability_max"] = propensity_max
        self.data.loc[mask, "acceptance_probability_mean"] = propensity_mean
        if self.compute_cohort_acceptance:
            self.data.loc[mask, "acceptance_probability_li"] = cohort_li
            self.data.loc[mask, "acceptance_probability_moderate"] = cohort_mod
            self.data.loc[mask, "acceptance_probability_non_lmi"] = cohort_non

    def _calculate_commercial_probabilities(self, mask: pd.Series) -> None:
        probs = self.data.loc[mask].apply(self._calculate_commercial_probability, axis=1).values
        for col in ["acceptance_probability", "propensity_min", "propensity_max", "propensity_mean", "acceptance_probability_min", "acceptance_probability_max", "acceptance_probability_mean"]:
            self.data.loc[mask, col] = probs
        self.data.loc[mask, "propensity_sd"] = 0.0

    # ── Export ────────────────────────────────────────────────────────────────

    def export_results(self, result: PropensityResult, output_path: str | Path) -> None:
        """Export propensity results to a parquet file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_parquet(output_path, index=False)
        logger.info(f"Results exported to {output_path} ({len(result.data)} rows)")

    def export_summary(self, result: PropensityResult, output_path: str | Path) -> None:
        """Export summary statistics to a JSON file."""
        output_path = Path(output_path)
        scenario_summary: dict = {}
        if "retrofit.scenario" in result.data.columns:
            for scenario in result.data["retrofit.scenario"].unique():
                sd = result.data[result.data["retrofit.scenario"] == scenario]
                scenario_summary[scenario] = {
                    "mean_probability": float(sd["acceptance_probability"].mean()),
                    "std_probability": float(sd["acceptance_probability"].std()),
                    "p10_probability": float(sd["acceptance_probability"].quantile(0.10)),
                    "p50_probability": float(sd["acceptance_probability"].quantile(0.50)),
                    "p90_probability": float(sd["acceptance_probability"].quantile(0.90)),
                    "n_samples": len(sd),
                }
        summary = {
            "n_buildings": result.n_buildings,
            "n_scenarios": result.n_scenarios,
            "total_rows": len(result.data),
            "mean_acceptance_probability": result.mean_acceptance_probability,
            "model_coefficients": self.model_coefficients,
            "neighbor_effect": self.neighbor_effect,
            "n_monte_carlo_samples": self.n_monte_carlo_samples,
            "commercial_npv_years": self.npv_years,
            "commercial_wacc": self.wacc,
            "commercial_npv_threshold_usd": self.npv_threshold_usd,
            "scenario_summary": scenario_summary,
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary exported to {output_path}")
