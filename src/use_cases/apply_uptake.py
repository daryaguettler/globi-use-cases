"""Calculate retrofit uptake based on propensity scores and adoption rate curves.

Takes the propensity results (output of ``PropensityModelEngine``) and an
adoption-curve JSON, then projects which buildings adopt each year using one
of four integration methods.

Methods:
--------
set_threshold (default)
    Hard binary threshold.  Buildings with ``acceptance_probability >= threshold``
    are "willing".  Each year, fill the adoption curve target from willing
    buildings ranked highest-to-lowest propensity (spillover to next year).
    If fewer willing buildings exist than the curve target, adoption stays below
    the curve.

dice_roll
    No hard threshold.  Each year every non-adopted building draws
    ``u ~ Uniform(0, 1)``; if ``acceptance_probability > u`` it becomes a
    candidate.  Among candidates, highest propensity fills first up to the
    curve target.  Remaining candidates roll again next year.

ranked_distribution
    Soft floor threshold.  Buildings below ``floor_threshold`` are excluded;
    above it, all buildings are ranked by propensity descending and fill the
    curve in strict order each year.  Deterministic.

mc_ensemble
    Run ``dice_roll`` N times with different seeds; report mean, P10, and P90
    of the cumulative adoption trajectory.

Adoption curves
---------------
Loaded from a JSON file with this schema::

    {
      "scenarios": {
        "status_quo": {
          "description": "Business as usual",
          "curve_type": "linear",
          "max_adoption": 0.85,
          "annual_attrition": 0.05,
          "values": {"2025": 0.02, "2026": 0.05, ...}
        }
      },
      "retrofit_specific_modifiers": {},
      "income_bracket_modifiers": {}
    }

Example usage::

    engine = AdoptionEngine(
        propensity_results_path="results/GroupA/propensity_results.pq",
        adoption_rates_path="inputs/adoption/adoption_curves.json",
        adoption_scenario="status_quo",
        acceptance_threshold=0.50,
    )
    result = engine.calculate_uptake()
    engine.export_results(result, "results/GroupA/uptake_results.pq")
"""

from __future__ import annotations

import json
import logging
from pydantic import BaseModel, ConfigDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


MethodName = (
    str  # "set_threshold" | "dice_roll" | "ranked_distribution" | "mc_ensemble"
)


class AdoptionScenario(BaseModel):
    """Named adoption rate scenario loaded from JSON config."""

    name: str
    description: str
    curve_type: str
    max_adoption: float
    annual_attrition: float
    values: dict[str, float]

    def get_rate(self, year: int) -> float:
        """Return cumulative adoption fraction for ``year`` (interpolated if needed)."""
        year_str = str(year)
        if year_str in self.values:
            return float(self.values[year_str])
        years = sorted(int(y) for y in self.values)
        if not years:
            return 0.0
        if year <= years[0]:
            return float(self.values[str(years[0])])
        if year >= years[-1]:
            return float(self.values[str(years[-1])])
        for i, y in enumerate(years[:-1]):
            if y <= year < years[i + 1]:
                t = (year - y) / (years[i + 1] - y)
                return float(self.values[str(y)]) + t * (
                    float(self.values[str(years[i + 1])]) - float(self.values[str(y)])
                )
        return self.max_adoption


class UptakeResult(BaseModel):
    """Container for uptake calculation results."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame
    yearly_summary: pd.DataFrame
    scenario_summary: dict[str, Any]
    method: str
    adoption_scenario_name: str


# ── Shared adoption-fill utility ──────────────────────────────────────────────


def _fill_adoption_curve(
    ranked_candidates: list[str],
    adoption_by_year: dict[str, int],
    adopted_set: set[str],
    years: list[int],
    curve: dict[int, float],
    n_buildings: int,
) -> tuple[dict[str, int], set[str]]:
    """Fill adoption curve quotas from an ordered candidate list.

    Args:
        ranked_candidates: Building IDs ordered most- to least-preferred.
        adoption_by_year: Output mapping ``{building_id: adoption_year}``
            (mutated in place).
        adopted_set: Already-adopted building IDs (mutated in place).
        years: Calendar years to iterate over.
        curve: ``{year: cumulative_fraction}`` mapping.
        n_buildings: Total number of buildings in the pool (denominator).

    Returns:
        Updated ``(adoption_by_year, adopted_set)``.
    """
    queue = [b for b in ranked_candidates if b not in adopted_set]
    adopted_so_far = len(adopted_set)
    ptr = 0
    for year in years:
        target_n = int(n_buildings * curve.get(year, 0.0))
        need = max(0, target_n - adopted_so_far)
        added = 0
        while added < need and ptr < len(queue):
            bid = queue[ptr]
            ptr += 1
            if bid in adopted_set:
                continue
            adoption_by_year[bid] = year
            adopted_set.add(bid)
            adopted_so_far += 1
            added += 1
    return adoption_by_year, adopted_set


# ── Method implementations ────────────────────────────────────────────────────


def method_set_threshold(
    df: pd.DataFrame,
    scenario: str,
    curve: dict[int, float],
    years: list[int],
    threshold: float = 0.80,
    building_id_col: str = "building_id",
    propensity_col: str = "acceptance_probability",
    scenario_col: str = "retrofit.scenario",
) -> pd.Series:
    """Hard binary threshold method.

    Returns a Series indexed by year with cumulative adoption as a percentage.
    """
    sub = df[df[scenario_col] == scenario]
    n = len(sub)
    if n == 0:
        return pd.Series(0.0, index=years)
    willing = sub[sub[propensity_col] >= threshold].sort_values(
        propensity_col, ascending=False
    )
    ranked = willing[building_id_col].tolist()
    adoption_by_year: dict[str, int] = {}
    adopted_set: set[str] = set()
    adoption_by_year, adopted_set = _fill_adoption_curve(
        ranked, adoption_by_year, adopted_set, years, curve, n
    )
    counts = pd.Series(0, index=years, dtype=int)
    for yr in adoption_by_year.values():
        if yr in counts.index:
            counts[yr] += 1
    return counts.cumsum() / n * 100


def method_dice_roll(
    df: pd.DataFrame,
    scenario: str,
    curve: dict[int, float],
    years: list[int],
    rng: np.random.Generator | None = None,
    floor_threshold: float = 0.0,
    building_id_col: str = "building_id",
    propensity_col: str = "acceptance_probability",
    scenario_col: str = "retrofit.scenario",
) -> pd.Series:
    """Bernoulli dice-roll method.

    Each year, non-adopted buildings draw u~U(0,1).  Those with
    propensity > u become candidates; highest propensity fills first.

    Returns a Series indexed by year with cumulative adoption as a percentage.
    """
    resolved_rng: np.random.Generator = (
        rng if rng is not None else np.random.default_rng(42)
    )
    sub = df[df[scenario_col] == scenario]
    n = len(sub)
    if n == 0:
        return pd.Series(0.0, index=years)
    if floor_threshold > 0:
        sub = sub[sub[propensity_col] >= floor_threshold]
    propensity = sub.set_index(building_id_col)[propensity_col].to_dict()
    all_bids = list(propensity)
    adopted_set: set[str] = set()
    year_counts: dict[int, int] = dict.fromkeys(years, 0)
    adopted_so_far = 0
    for year in years:
        target_n = int(n * curve.get(year, 0.0))
        need = max(0, target_n - adopted_so_far)
        if need == 0:
            continue
        eligible = [b for b in all_bids if b not in adopted_set]
        if not eligible:
            break
        draws = resolved_rng.uniform(0, 1, len(eligible))
        candidates = sorted(
            [b for b, u in zip(eligible, draws, strict=False) if propensity[b] > u],
            key=lambda b: propensity[b],
            reverse=True,
        )
        for bid in candidates[:need]:
            adopted_set.add(bid)
            year_counts[year] += 1
            adopted_so_far += 1
    return pd.Series(year_counts).cumsum() / n * 100


def method_ranked_distribution(
    df: pd.DataFrame,
    scenario: str,
    curve: dict[int, float],
    years: list[int],
    floor_threshold: float = 0.10,
    building_id_col: str = "building_id",
    propensity_col: str = "acceptance_probability",
    scenario_col: str = "retrofit.scenario",
) -> pd.Series:
    """Soft-floor ranked distribution method.  Deterministic.

    Buildings below ``floor_threshold`` are excluded; the rest fill the curve
    in strict propensity-descending order.

    Returns a Series indexed by year with cumulative adoption as a percentage.
    """
    sub = df[df[scenario_col] == scenario]
    n = len(sub)
    if n == 0:
        return pd.Series(0.0, index=years)
    eligible = sub[sub[propensity_col] >= floor_threshold].sort_values(
        propensity_col, ascending=False
    )
    ranked = eligible[building_id_col].tolist()
    adoption_by_year: dict[str, int] = {}
    adopted_set: set[str] = set()
    adoption_by_year, adopted_set = _fill_adoption_curve(
        ranked, adoption_by_year, adopted_set, years, curve, n
    )
    counts = pd.Series(0, index=years, dtype=int)
    for yr in adoption_by_year.values():
        if yr in counts.index:
            counts[yr] += 1
    return counts.cumsum() / n * 100


def method_mc_ensemble(
    df: pd.DataFrame,
    scenario: str,
    curve: dict[int, float],
    years: list[int],
    n_runs: int = 100,
    floor_threshold: float = 0.0,
    seed: int = 0,
    ci_low: float = 10.0,
    ci_high: float = 90.0,
    building_id_col: str = "building_id",
    propensity_col: str = "acceptance_probability",
    scenario_col: str = "retrofit.scenario",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Monte Carlo ensemble: run dice_roll N times and return (mean, P_low, P_high).

    Args:
        df: Propensity DataFrame.
        scenario: Retrofit scenario name to filter to.
        curve: Adoption curve ``{year: fraction}``.
        years: Calendar years to iterate over.
        n_runs: Number of dice-roll repetitions.
        floor_threshold: Minimum propensity below which buildings are excluded.
        seed: Base seed for child RNGs.
        ci_low: Lower percentile for the confidence band (default 10).
        ci_high: Upper percentile for the confidence band (default 90).
        building_id_col: Column name for building identifiers.
        propensity_col: Column name for the propensity score.
        scenario_col: Column name for the retrofit scenario label.

    Returns:
        Three Series (mean, p_low, p_high) each indexed by year, values in %.
    """
    base_rng = np.random.default_rng(seed)
    child_seeds = base_rng.integers(0, 2**32, size=n_runs)
    results = [
        method_dice_roll(
            df,
            scenario,
            curve,
            years,
            rng=np.random.default_rng(int(s)),
            floor_threshold=floor_threshold,
            building_id_col=building_id_col,
            propensity_col=propensity_col,
            scenario_col=scenario_col,
        )
        for s in child_seeds
    ]
    arr = np.stack([s.values for s in results], axis=0)
    return (
        pd.Series(arr.mean(axis=0), index=years),
        pd.Series(np.percentile(arr, ci_low, axis=0), index=years),
        pd.Series(np.percentile(arr, ci_high, axis=0), index=years),
    )


class AdoptionEngine:
    """Engine for projecting retrofit uptake over time.

    Args:
        propensity_results_path: Path to propensity parquet (output of
            ``PropensityModelEngine``).
        adoption_rates_path: Path to adoption curves JSON.
        propensity_df: Pre-loaded propensity DataFrame (alternative to path).
        params_path: Optional path to params.yaml for loading configuration.
        acceptance_threshold: Default acceptance probability threshold for
            the ``set_threshold`` method.
        floor_threshold: Soft-floor for ``ranked_distribution`` and
            ``dice_roll`` methods.
        adoption_scenario: Name of the adoption curve scenario to use.
        method: Integration method; one of ``"set_threshold"``,
            ``"dice_roll"``, ``"ranked_distribution"``, ``"mc_ensemble"``.
        start_year: First year of the projection.
        end_year: Last year of the projection.
        n_ensemble_runs: Number of dice-roll runs for ``mc_ensemble``.
        random_seed: RNG seed for stochastic methods.
        income_brackets: Income bracket names used in net cost columns.
    """

    def __init__(  # noqa: D107
        self,
        propensity_results_path: str | Path | None = None,
        adoption_rates_path: str | Path | None = None,
        propensity_df: pd.DataFrame | None = None,
        params_path: str | Path | None = None,
        acceptance_threshold: float = 0.50,
        floor_threshold: float = 0.10,
        adoption_scenario: str = "status_quo",
        method: MethodName = "set_threshold",
        start_year: int = 2025,
        end_year: int = 2050,
        n_ensemble_runs: int = 100,
        random_seed: int = 42,
        income_brackets: list[str] | None = None,
    ):
        self.params: dict = {}
        if params_path is not None:
            p = Path(params_path)
            if p.exists():
                with open(p) as f:
                    self.params = yaml.safe_load(f) or {}

        uptake_cfg = self.params.get("uptake", {})
        prop_cfg = self.params.get("propensity", {})

        self.acceptance_threshold = float(
            uptake_cfg.get(
                "acceptance_threshold",
                prop_cfg.get("acceptance_threshold", acceptance_threshold),
            )
        )
        self.floor_threshold = float(uptake_cfg.get("floor_threshold", floor_threshold))
        self.adoption_scenario_name = str(
            uptake_cfg.get("adoption_scenario", adoption_scenario)
        )
        self.method: MethodName = str(uptake_cfg.get("method", method))
        self.start_year = int(uptake_cfg.get("start_year", start_year))
        self.end_year = int(uptake_cfg.get("end_year", end_year))
        self.n_ensemble_runs = int(uptake_cfg.get("n_ensemble_runs", n_ensemble_runs))
        self.random_seed = int(uptake_cfg.get("random_seed", random_seed))
        self.income_brackets = income_brackets or ["IncomeEligible", "AllCustomers"]
        self.threshold_by_scenario: dict[str, float] = uptake_cfg.get(
            "threshold_by_scenario", {}
        )

        # Load propensity data
        if propensity_df is not None:
            self.data = propensity_df.copy()
        elif propensity_results_path is not None:
            self.data = pd.read_parquet(propensity_results_path)
        else:
            msg = "Must provide propensity_results_path or propensity_df"
            raise ValueError(msg)

        # Normalise building_id column
        if (
            "building.id" in self.data.columns
            and "building_id" not in self.data.columns
        ):
            self.data = self.data.rename(columns={"building.id": "building_id"})
        if "building_id" not in self.data.columns:
            self.data["building_id"] = self.data.index.astype(str)

        # Load adoption curves
        rates_path = Path(adoption_rates_path) if adoption_rates_path else None
        if rates_path is None:
            rates_path = Path(
                self.params.get("paths", {}).get(
                    "adoption_rates", "inputs/adoption/adoption_curves.json"
                )
            )
        self.adoption_rates = self._load_adoption_rates(rates_path)
        self.adoption_scenario_obj = self._get_adoption_scenario()
        self.curve: dict[int, float] = {
            int(y): min(float(v), 1.0)
            for y, v in self.adoption_scenario_obj.values.items()
        }
        self.years: list[int] = list(range(self.start_year, self.end_year + 1))

        logger.info(
            f"Loaded {len(self.data)} propensity rows; method={self.method}; scenario={self.adoption_scenario_name}"
        )

    def _load_adoption_rates(self, path: Path) -> dict:
        """Load adoption rates from JSON, or return built-in defaults."""
        if path.exists():
            with open(path) as f:
                return json.load(f)
        logger.warning(f"Adoption rates file not found: {path}, using defaults")
        return {
            "scenarios": {
                "status_quo": {
                    "description": "Business as usual",
                    "curve_type": "linear",
                    "max_adoption": 0.85,
                    "annual_attrition": 0.05,
                    "values": {
                        str(y): round(0.02 + 0.03 * (y - 2024), 4)
                        for y in range(2024, 2051)
                    },
                }
            }
        }

    def _get_adoption_scenario(self) -> AdoptionScenario:
        """Resolve and return the configured adoption scenario object."""
        scenarios = self.adoption_rates.get("scenarios", {})
        name = self.adoption_scenario_name
        if name not in scenarios:
            logger.warning(
                f"Adoption scenario '{name}' not found, using first available"
            )
            name = next(iter(scenarios), "status_quo")
        sd = scenarios.get(name, {})
        return AdoptionScenario(
            name=name,
            description=sd.get("description", ""),
            curve_type=sd.get("curve_type", "linear"),
            max_adoption=float(sd.get("max_adoption", 0.85)),
            annual_attrition=float(sd.get("annual_attrition", 0.05)),
            values=sd.get("values", {}),
        )

    def _get_threshold(self, scenario: str) -> float:
        """Return the acceptance threshold for a given retrofit scenario."""
        return float(
            self.threshold_by_scenario.get(scenario, self.acceptance_threshold)
        )

    def calculate_uptake(self) -> UptakeResult:
        """Calculate retrofit uptake for all buildings and scenarios.

        Dispatches to the configured ``method`` and builds a yearly summary
        and per-scenario statistics.

        Returns:
            ``UptakeResult`` with per-building data, yearly summary DataFrame,
            and scenario statistics dict.
        """
        self.data["passes_threshold"] = (
            self.data.apply(
                lambda r: (
                    r["acceptance_probability"]
                    >= self._get_threshold(r.get("retrofit.scenario", ""))
                ),
                axis=1,
            )
            if "retrofit.scenario" in self.data.columns
            else self.data["acceptance_probability"] >= self.acceptance_threshold
        )

        scenarios = (
            self.data["retrofit.scenario"].unique().tolist()
            if "retrofit.scenario" in self.data.columns
            else ["default"]
        )
        yearly_rows: list[dict] = []

        for scenario in scenarios:
            sub = (
                self.data[self.data["retrofit.scenario"] == scenario]
                if "retrofit.scenario" in self.data.columns
                else self.data
            )
            n = len(sub)
            threshold = self._get_threshold(scenario)

            if self.method == "set_threshold":
                series = method_set_threshold(
                    self.data, scenario, self.curve, self.years, threshold=threshold
                )
                self._append_yearly_rows(
                    yearly_rows, scenario, series, n, method="set_threshold"
                )

            elif self.method == "dice_roll":
                rng = np.random.default_rng(self.random_seed)
                series = method_dice_roll(
                    self.data,
                    scenario,
                    self.curve,
                    self.years,
                    rng=rng,
                    floor_threshold=self.floor_threshold,
                )
                self._append_yearly_rows(
                    yearly_rows, scenario, series, n, method="dice_roll"
                )

            elif self.method == "ranked_distribution":
                series = method_ranked_distribution(
                    self.data,
                    scenario,
                    self.curve,
                    self.years,
                    floor_threshold=self.floor_threshold,
                )
                self._append_yearly_rows(
                    yearly_rows, scenario, series, n, method="ranked_distribution"
                )

            elif self.method == "mc_ensemble":
                mean_s, lo_s, hi_s = method_mc_ensemble(
                    self.data,
                    scenario,
                    self.curve,
                    self.years,
                    n_runs=self.n_ensemble_runs,
                    seed=self.random_seed,
                    floor_threshold=self.floor_threshold,
                )
                self._append_yearly_rows(
                    yearly_rows,
                    scenario,
                    mean_s,
                    n,
                    method="mc_ensemble",
                    p_low=lo_s,
                    p_high=hi_s,
                )

            else:
                logger.warning(
                    f"Unknown method '{self.method}', falling back to set_threshold"
                )
                series = method_set_threshold(
                    self.data, scenario, self.curve, self.years, threshold=threshold
                )
                self._append_yearly_rows(
                    yearly_rows, scenario, series, n, method="set_threshold"
                )

        yearly_summary = pd.DataFrame(yearly_rows)
        scenario_summary = self._build_scenario_summary(scenarios)

        return UptakeResult(
            data=self.data.copy(),
            yearly_summary=yearly_summary,
            scenario_summary=scenario_summary,
            method=self.method,
            adoption_scenario_name=self.adoption_scenario_name,
        )

    def _append_yearly_rows(
        self,
        rows: list[dict],
        scenario: str,
        cumulative_pct: pd.Series,
        n_buildings: int,
        method: str,
        p_low: pd.Series | None = None,
        p_high: pd.Series | None = None,
    ) -> None:
        """Append per-year summary rows for a scenario into ``rows``."""
        prev = 0.0
        for year in self.years:
            cum = float(cumulative_pct.get(year, 0.0))
            yearly_pct = max(0.0, cum - prev)
            row: dict[str, Any] = {
                "year": year,
                "retrofit.scenario": scenario,
                "method": method,
                "adoption_scenario": self.adoption_scenario_name,
                "n_buildings": n_buildings,
                "cumulative_adoption_pct": cum,
                "yearly_adoption_pct": yearly_pct,
                "n_adopting": round(yearly_pct / 100 * n_buildings),
                "curve_target_pct": float(self.curve.get(year, 0.0)) * 100,
            }
            if p_low is not None:
                row["cumulative_adoption_pct_p10"] = float(p_low.get(year, 0.0))
            if p_high is not None:
                row["cumulative_adoption_pct_p90"] = float(p_high.get(year, 0.0))
            rows.append(row)
            prev = cum

    def _build_scenario_summary(self, scenarios: list[str]) -> dict[str, Any]:
        """Build summary statistics dict by scenario."""
        summary: dict[str, Any] = {
            "method": self.method,
            "adoption_scenario": self.adoption_scenario_name,
            "acceptance_threshold": self.acceptance_threshold,
            "floor_threshold": self.floor_threshold,
            "time_horizon": {"start_year": self.start_year, "end_year": self.end_year},
            "by_scenario": {},
        }
        for scenario in scenarios:
            sub = (
                self.data[self.data["retrofit.scenario"] == scenario]
                if "retrofit.scenario" in self.data.columns
                else self.data
            )
            passes = (
                sub["passes_threshold"].sum()
                if "passes_threshold" in sub.columns
                else len(sub)
            )
            summary["by_scenario"][scenario] = {
                "n_total": len(sub),
                "n_passing_threshold": int(passes),
                "pass_rate": float(passes / len(sub)) if len(sub) > 0 else 0.0,
                "mean_acceptance_probability": float(
                    sub["acceptance_probability"].mean()
                ),
            }
        return summary

    # ── Export ────────────────────────────────────────────────────────────────

    def export_results(self, result: UptakeResult, output_path: str | Path) -> None:
        """Export per-building uptake data to parquet."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.data.to_parquet(out, index=False)
        logger.info(f"Exported uptake data to {out}")

    def export_yearly_summary(
        self, result: UptakeResult, output_path: str | Path
    ) -> None:
        """Export yearly adoption summary to parquet."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.yearly_summary.to_parquet(out, index=False)
        logger.info(f"Exported yearly summary to {out}")

    def export_summary(self, result: UptakeResult, output_path: str | Path) -> None:
        """Export scenario summary statistics to JSON."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result.scenario_summary, f, indent=2)
        logger.info(f"Exported summary to {out}")
