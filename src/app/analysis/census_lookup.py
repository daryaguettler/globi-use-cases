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

logger = logging.getLogger(__name__)

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

def geocode_point(lat: float, lon: float, retries: int = 3) -> dict[str, str] | None:
    """Return ``{state, county, tract}`` FIPS codes for a lat/lon point.

    Returns ``None`` on failure.  Rate-limited to avoid hitting API limits.
    """
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
                return {
                    "state": g.get("STATE", ""),
                    "county": g.get("COUNTY", ""),
                    "tract": g.get("TRACT", ""),
                    "geoid": g.get("GEOID", ""),
                }
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
    sleep_between: float = 0.05,
) -> pd.DataFrame:
    """Geocode a DataFrame of buildings, adding ``state``, ``county_fips``,
    ``tract_fips``, and ``geoid`` columns.

    Args:
        df:             DataFrame with lat/lon columns.
        lat_col:        Name of latitude column.
        lon_col:        Name of longitude column.
        max_buildings:  Cap on number of buildings to geocode (for performance).
        sleep_between:  Seconds to sleep between API calls.

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

    for i in range(rows_to_process):
        lat = df[lat_col].iloc[i]
        lon = df[lon_col].iloc[i]
        try:
            lat_f, lon_f = float(lat), float(lon)
        except (TypeError, ValueError):
            continue

        result = geocode_point(lat_f, lon_f)
        if result:
            out.at[df.index[i], "state_fips"] = result["state"]
            out.at[df.index[i], "county_fips"] = result["county"]
            out.at[df.index[i], "tract_fips"] = result["tract"]
            out.at[df.index[i], "geoid"] = result["geoid"]

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


def fetch_tract_demographics(
    state_fips: str,
    county_fips: str,
    tract_fips: str,
    api_key: str | None = None,
) -> dict[str, Any] | None:
    """Fetch ACS5 income and education distributions for a census tract.

    Returns a dict with keys ``income_probs`` and ``education_probs``
    (normalised probability arrays), or ``None`` on failure.
    """
    variables = ",".join(_ACS_INCOME_VARS + _ACS_EDU_VARS)
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

        def _normalise(arr: np.ndarray) -> np.ndarray:
            s = arr.sum()
            return arr / s if s > 0 else np.ones_like(arr) / len(arr)

        return {
            "income_probs": _normalise(income_counts).tolist(),
            "education_probs": _normalise(edu_counts).tolist(),
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
    ``income_probs`` (list), ``education_probs`` (list).
    """
    out = batch_geocode(df, lat_col=lat_col, lon_col=lon_col, max_buildings=max_buildings)

    # Fetch demographics per unique tract
    tract_cache: dict[tuple, dict | None] = {}
    for col in ["income_probs", "education_probs"]:
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
            time.sleep(0.1)
        demo = tract_cache[key]
        if demo:
            out.at[idx, "income_probs"] = demo["income_probs"]
            out.at[idx, "education_probs"] = demo["education_probs"]

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
