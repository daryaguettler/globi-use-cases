"""Look up US census tract demographics for buildings using lat/lon coordinates.

Uses the free Census Geocoder API (no key required) to resolve lat/lon → tract,
then the Census ACS5 API (free key, optional) to fetch income and education
distributions.

For non-US data, ``build_prior_from_sliders`` converts user-specified
distribution parameters into the format expected by PropensityModelEngine.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CensusGeocodeResult(BaseModel):
    """Result from the Census Geocoder API for a single lat/lon point."""

    state: str = Field(description="State FIPS code")
    county: str = Field(description="County FIPS code")
    tract: str = Field(description="Tract FIPS code")
    geoid: str = Field(description="Full 11-digit GEOID (state+county+tract)")

# Census API base URLs
_GEOCODER_URL = (
    "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
)
_ACS_URL = "https://api.census.gov/data/2020/acs/acs5"

# Income bracket midpoints used by the propensity model (k USD)
INCOME_CATEGORIES_K: list[float] = [
    5.0, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5,
    55.0, 67.5, 87.5, 112.5, 137.5, 175.0, 225.0,
]
# Education levels: 1=HS, 2=Associates, 3=Bachelors, 4=Graduate
EDUCATION_CATEGORIES: list[float] = [1.0, 2.0, 3.0, 4.0]


# ── Census Geocoder ────────────────────────────────────────────────────────────

def geocode_point(lat: float, lon: float, retries: int = 3) -> CensusGeocodeResult | None:
    """Return FIPS codes for a lat/lon point, or ``None`` on failure."""
    params = {
        "x": lon,
        "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "layers": "Census Tracts",
        "format": "json",
    }
    for attempt in range(retries):
        try:
            r = requests.get(_GEOCODER_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            geos = (
                data.get("result", {})
                .get("geographies", {})
                .get("Census Tracts", [])
            )
            if geos:
                g = geos[0]
                return CensusGeocodeResult(
                    state=g.get("STATE", ""),
                    county=g.get("COUNTY", ""),
                    tract=g.get("TRACT", ""),
                    geoid=g.get("GEOID", ""),
                )
        except Exception as exc:
            logger.debug(f"Geocoder attempt {attempt + 1} failed: {exc}")
            if attempt < retries - 1:
                time.sleep(1.0)
    return None


def batch_geocode(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    max_buildings: int = 500,
    sleep_between: float = 0.03,
    coord_round_decimals: int = 5,
) -> pd.DataFrame:
    """Geocode a DataFrame of buildings, adding ``state_fips``, ``county_fips``,
    ``tract_fips``, and ``geoid`` columns.

    One Census Geocoder request is made per *distinct* rounded (lat, lon) among
    the first ``max_buildings`` rows — buildings that share the same coordinates
    reuse the same result (typical for portfolios in one tract).

    Args:
        df:             DataFrame with lat/lon columns.
        lat_col:        Name of latitude column.
        lon_col:        Name of longitude column.
        max_buildings:  Cap on number of buildings considered (for performance).
        sleep_between:  Seconds to sleep between distinct geocoder calls.
        coord_round_decimals: Decimal places for deduplicating coordinates (~1.1 m at 5).

    Returns:
        DataFrame with census columns added (NaN where lookup failed).
    """
    out = df.copy()
    for col in ["state_fips", "county_fips", "tract_fips", "geoid"]:
        out[col] = None

    if lat_col not in df.columns or lon_col not in df.columns:
        logger.warning(f"Lat/lon columns '{lat_col}'/'{lon_col}' not found; skipping geocoding.")
        return out

    rows_to_process = min(len(df), max_buildings)
    if rows_to_process < len(df):
        logger.info(f"Geocoding capped at {max_buildings} buildings.")

    # group row indices by rounded coordinate → one API call per distinct location
    buckets: dict[tuple[float, float], list[Any]] = {}
    for i in range(rows_to_process):
        lat = df[lat_col].iloc[i]
        lon = df[lon_col].iloc[i]
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        key = (round(lat_f, coord_round_decimals), round(lon_f, coord_round_decimals))
        buckets.setdefault(key, []).append(df.index[i])

    n_unique = len(buckets)
    logger.info(f"Geocoding {n_unique} distinct coordinate(s) for {rows_to_process} row(s).")

    for (lat_f, lon_f), idx_list in buckets.items():
        result = geocode_point(lat_f, lon_f)
        if result and (result.geoid or result.state):
            for idx in idx_list:
                out.at[idx, "state_fips"] = result.state
                out.at[idx, "county_fips"] = result.county
                out.at[idx, "tract_fips"] = result.tract
                out.at[idx, "geoid"] = result.geoid
        if sleep_between > 0:
            time.sleep(sleep_between)

    geocoded = out["geoid"].notna().sum()
    logger.info(f"Geocoded {geocoded}/{rows_to_process} buildings.")
    return out


# ── ACS demographics ───────────────────────────────────────────────────────────

_ACS_INCOME_VARS = [
    "B19001_002E",  # < $10k
    "B19001_003E",  # $10k–$14,999
    "B19001_004E",  # $15k–$19,999
    "B19001_005E",  # $20k–$24,999
    "B19001_006E",  # $25k–$29,999
    "B19001_007E",  # $30k–$34,999
    "B19001_008E",  # $35k–$39,999
    "B19001_009E",  # $40k–$44,999
    "B19001_010E",  # $45k–$49,999
    "B19001_011E",  # $50k–$59,999
    "B19001_012E",  # $60k–$74,999
    "B19001_013E",  # $75k–$99,999
    "B19001_014E",  # $100k–$124,999
    "B19001_015E",  # $125k–$149,999
    "B19001_016E",  # $150k–$199,999
    "B19001_017E",  # ≥ $200k
]
_ACS_EDU_VARS = [
    "B15003_017E",  # HS diploma
    "B15003_018E",  # GED
    "B15003_019E",  # Some college < 1 yr
    "B15003_020E",  # Some college ≥ 1 yr
    "B15003_021E",  # Associates
    "B15003_022E",  # Bachelors
    "B15003_023E",  # Masters
    "B15003_024E",  # Professional
    "B15003_025E",  # Doctorate
]
# household size (matches propensity engine: 1–7 person categories)
_ACS_HH_VARS = [
    "B11016_002E",
    "B11016_003E",
    "B11016_004E",
    "B11016_005E",
    "B11016_006E",
    "B11016_007E",
    "B11016_008E",
]


def fetch_tract_demographics(
    state_fips: str,
    county_fips: str,
    tract_fips: str,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fetch ACS5 income, education, and household-size distributions for a tract.

    Returns a dict with ``income_probs``, ``education_probs``, ``household_probs``
    (normalised lists), or ``None`` on failure.
    """
    variables = ",".join(_ACS_INCOME_VARS + _ACS_EDU_VARS + _ACS_HH_VARS)
    params: dict[str, str] = {
        "get": variables,
        "for": f"tract:{tract_fips}",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(_ACS_URL, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if len(rows) < 2:
            return None
        headers = rows[0]
        vals = rows[1]
        data = dict(zip(headers, vals))

        income_counts = np.array(
            [max(0, int(data.get(v, 0) or 0)) for v in _ACS_INCOME_VARS],
            dtype=float,
        )
        edu_raw = np.array(
            [max(0, int(data.get(v, 0) or 0)) for v in _ACS_EDU_VARS],
            dtype=float,
        )
        # Collapse education into 4 categories: HS/GED, Associates, Bachelors, Graduate
        edu_counts = np.array([
            edu_raw[0] + edu_raw[1] + edu_raw[2] + edu_raw[3],  # HS / some college
            edu_raw[4],                                           # Associates
            edu_raw[5],                                           # Bachelors
            edu_raw[6] + edu_raw[7] + edu_raw[8],                # Graduate
        ], dtype=float)
        hh_counts = np.array(
            [max(0, int(data.get(v, 0) or 0)) for v in _ACS_HH_VARS],
            dtype=float,
        )

        def _normalise(arr: np.ndarray) -> np.ndarray:
            s = arr.sum()
            return arr / s if s > 0 else np.ones_like(arr) / len(arr)

        return {
            "income_probs": _normalise(income_counts).tolist(),
            "education_probs": _normalise(edu_counts).tolist(),
            "household_probs": _normalise(hh_counts).tolist(),
        }
    except Exception as exc:
        logger.debug(f"ACS fetch failed for {state_fips}/{county_fips}/{tract_fips}: {exc}")
        return None


def enrich_with_census(
    df: pd.DataFrame,
    api_key: str | None = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    max_buildings: int = 500,
) -> pd.DataFrame:
    """Geocode buildings and fetch ACS demographics, returning an enriched DataFrame.

    Adds columns: ``state_fips``, ``county_fips``, ``tract_fips``, ``geoid``,
    ``income_probs``, ``education_probs``, ``household_probs`` (lists per row).
    """
    out = batch_geocode(df, lat_col=lat_col, lon_col=lon_col, max_buildings=max_buildings)

    # fetch demographics per unique tract (not per row)
    tract_cache: dict[tuple, dict | None] = {}
    for col in ["income_probs", "education_probs", "household_probs"]:
        out[col] = None

    for idx in out.index:
        state = out.at[idx, "state_fips"]
        county = out.at[idx, "county_fips"]
        tract = out.at[idx, "tract_fips"]
        if not all([state, county, tract]):
            continue
        key = (state, county, tract)
        if key not in tract_cache:
            tract_cache[key] = fetch_tract_demographics(state, county, tract, api_key)
            time.sleep(0.08)
        demo = tract_cache[key]
        if demo:
            out.at[idx, "income_probs"] = demo["income_probs"]
            out.at[idx, "education_probs"] = demo["education_probs"]
            hp = demo.get("household_probs")
            out.at[idx, "household_probs"] = hp if hp is not None else [1.0 / 7.0] * 7

    return out


def tract_distributions_from_enriched(
    df: pd.DataFrame,
    county_fips_map: dict[int, str] | None = None,
) -> dict[str, dict]:
    """Build ``PropensityModelEngine.tract_distributions`` entries keyed by ``geoid``.

    Uses columns produced by :func:`enrich_with_census`. County names are resolved
    via ``county_fips_map`` when possible (defaults to MA in the app engine).
    """
    county_fips_map = county_fips_map or {}
    required = ("geoid", "income_probs", "education_probs")
    if not all(c in df.columns for c in required):
        return {}
    sub = df[df["geoid"].notna()].copy()
    if sub.empty:
        return {}
    sub = sub.drop_duplicates(subset=["geoid"], keep="first")
    out: dict[str, dict[str, Any]] = {}
    for _, row in sub.iterrows():
        gid = str(row["geoid"]).strip()
        if not gid:
            continue
        inc = row.get("income_probs")
        edu = row.get("education_probs")
        if inc is None or edu is None:
            continue
        cfi = row.get("county_fips")
        county_name = ""
        if cfi is not None and pd.notna(cfi):
            try:
                ck = int(float(str(cfi).split(".")[0]))
                county_name = county_fips_map.get(ck) or ""
            except (TypeError, ValueError):
                county_name = ""
        if not county_name and cfi is not None and pd.notna(cfi):
            county_name = str(cfi).strip()

        hhp = row.get("household_probs")
        if hhp is None or (isinstance(hhp, float) and pd.isna(hhp)):
            hh = np.ones(7, dtype=float) / 7.0
        else:
            hh = np.array(list(hhp), dtype=float)
            hh = hh / hh.sum() if hh.sum() > 0 else np.ones(7, dtype=float) / 7.0

        inc_a = np.array(list(inc), dtype=float)
        edu_a = np.array(list(edu), dtype=float)
        inc_a = inc_a / inc_a.sum() if inc_a.sum() > 0 else np.ones_like(inc_a) / len(inc_a)
        edu_a = edu_a / edu_a.sum() if edu_a.sum() > 0 else np.ones_like(edu_a) / len(edu_a)

        out[gid] = {
            "education": edu_a,
            "household_size": hh,
            "income": inc_a,
            "county": county_name or "Unknown",
        }
    return out


# ── Non-US priors ──────────────────────────────────────────────────────────────

def build_prior_distributions(
    income_mean_k: float,
    income_std_k: float,
    pct_hs: float,
    pct_associates: float,
    pct_bachelors: float,
    pct_graduate: float,
) -> dict[str, list[float]]:
    """Build demographic prior distributions for non-US use cases.

    Generates an income probability vector over ``INCOME_CATEGORIES_K`` bins
    from a lognormal approximation, and normalises the education shares.

    Args:
        income_mean_k:    Mean household income in thousands of USD.
        income_std_k:     Std deviation of household income in thousands of USD.
        pct_hs:           % with high school / some college (0–100).
        pct_associates:   % with associates degree (0–100).
        pct_bachelors:    % with bachelors degree (0–100).
        pct_graduate:     % with graduate degree (0–100).

    Returns:
        Dict with ``income_probs`` and ``education_probs`` (sum to 1).
    """
    # Lognormal income distribution
    bins = np.array(INCOME_CATEGORIES_K, dtype=float)
    sigma = np.sqrt(np.log(1 + (income_std_k / income_mean_k) ** 2))
    mu = np.log(income_mean_k) - 0.5 * sigma ** 2
    # Evaluate lognormal PDF at each bin midpoint
    pdf = (1 / (bins * sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(bins) - mu) ** 2) / (2 * sigma ** 2)
    )
    income_probs = (pdf / pdf.sum()).tolist()

    edu = np.array([pct_hs, pct_associates, pct_bachelors, pct_graduate], dtype=float)
    edu = np.clip(edu, 0, None)
    edu_probs = (edu / edu.sum()).tolist() if edu.sum() > 0 else [0.25, 0.25, 0.25, 0.25]

    return {"income_probs": income_probs, "education_probs": edu_probs}
