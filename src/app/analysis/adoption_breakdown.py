"""Adoption cohort vs demographics and incentive effects (for results UI)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

# aligned with app.analysis.census_lookup.INCOME_CATEGORIES_K (16 midpoints, k$)
INCOME_BIN_LABELS: list[str] = [
    "< $10k", "$10k–$15k", "$15k–$20k", "$20k–$25k", "$25k–$30k", "$30k–$35k", "$35k–$40k", "$40k–$45k", "$45k–$50k", "$50k–$60k", "$60k–$75k", "$75k–$100k", "$100k–$125k", "$125k–$150k", "$150k–$200k", "$200k+",
]
EDU_BIN_LABELS: list[str] = [
    "HS or less (incl. GED)",
    "Associate / some college",
    "Bachelor's",
    "Graduate / prof. / PhD",
]


def _stable_hash_u32(s: str) -> int:
    h = hashlib.md5(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    return int(h, 16)


def _sample_category_index(
    building_id: str, probs: object, n_cats: int, labels: list[str] | None = None
) -> tuple[int, str | None]:
    """Deterministic index into [0, n_cats) from a tract multinomial; falls back to 0 / uniform."""
    if probs is None or (isinstance(probs, float) and pd.isna(probs)):
        p = np.ones(n_cats) / n_cats
    else:
        p = np.asarray(probs, dtype=float).ravel()
        if p.size != n_cats or not np.isfinite(p).all():
            p = np.ones(n_cats) / n_cats
        else:
            s = p.sum()
            p = p / s if s > 0 else np.ones(n_cats) / n_cats
    rng = np.random.default_rng(_stable_hash_u32(str(building_id)))
    k = int(rng.choice(n_cats, p=p))
    lab = labels[k] if labels and k < len(labels) else None
    return k, lab


def propensity_ranked_positions(p_series: pd.Series, n_take: int) -> np.ndarray:
    """I positions of the top n_take rows by propensity (matches emissions: rank first, high first)."""
    n_total = len(p_series)
    if n_total == 0 or n_take <= 0:
        return np.array([], dtype=int)
    n_take = min(n_take, n_total)
    prop_order = p_series.rank(method="first", ascending=False).astype(int).sub(1)
    ranked = prop_order.argsort().values
    return ranked[:n_take]


def n_adopters_from_cumulative(n_rows: int, cum_pct: float) -> int:
    return max(0, min(n_rows, int(round(cum_pct / 100.0 * n_rows))))


def resolve_retrofit_scenario_name(policy: pd.DataFrame, session_name: str) -> str:
    """Session label and policy ``retrofit.scenario`` can drift; prefer policy data."""
    if "retrofit.scenario" not in policy.columns or policy.empty:
        return (session_name or "Retrofit").strip() or "Retrofit"
    u = [str(x) for x in policy["retrofit.scenario"].dropna().unique().tolist()]
    s = (session_name or "").strip()
    if not u:
        return s or "Retrofit"
    if s in u:
        return s
    for x in u:
        if str(x).strip().lower() == s.lower():
            return x
    if len(u) == 1:
        return u[0]
    return s if s else u[0]


def last_year_uptake_row(
    yearly: pd.DataFrame,
    adoption_curve_key: str,
    retrofit_label: str,
) -> pd.Series | None:
    """
    One summary row: final projection year for this adoption-curve and retrofit.
    If ``retrofit_label`` is wrong, use the only unique value in the table when unambiguous.
    """
    if yearly is None or yearly.empty:
        return None
    sub = yearly
    if "adoption_scenario" in sub.columns and adoption_curve_key is not None and str(adoption_curve_key) != "":
        sub = sub[sub["adoption_scenario"].astype(str) == str(adoption_curve_key)]
    if sub.empty:
        return None
    if "retrofit.scenario" in sub.columns and retrofit_label:
        r = str(retrofit_label).strip()
        m = sub["retrofit.scenario"].astype(str) == r
        if m.any():
            sub = sub[m]
        else:
            m2 = sub["retrofit.scenario"].astype(str).str.lower() == r.lower()
            if m2.any():
                sub = sub[m2]
            else:
                uq = sub["retrofit.scenario"].dropna().unique()
                if len(uq) == 1:
                    sub = sub[sub["retrofit.scenario"] == uq[0]]
    if sub.empty:
        return None
    y_max = int(sub["year"].max()) if "year" in sub.columns else None
    if y_max is not None:
        sub = sub[sub["year"] == y_max]
    if sub.empty:
        return None
    return sub.iloc[0]


def n_adopters_from_yearly_row(row: pd.Series) -> int:
    """Adopter count for one yearly_summary row (same n_buildings the engine used)."""
    cum = float(row.get("cumulative_adoption_pct", 0.0) or 0.0)
    nb = int(row.get("n_buildings", 0) or 0)
    if nb <= 0:
        return 0
    return max(0, min(nb, int(round(cum / 100.0 * nb))))


def count_adopters_with_positive_incentive(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    propensity_data: pd.DataFrame,
    id_col: str = "building.id",
    min_usd: float = 0.0,
) -> int:
    """Among top n_adopt by p_col, count with expected_incentive_usd > min_usd (with-incentive run)."""
    if n_adopt <= 0 or p_col not in policy.columns or propensity_data is None or propensity_data.empty:
        return 0
    if "expected_incentive_usd" not in propensity_data.columns or id_col not in propensity_data.columns:
        return 0
    pos = propensity_ranked_positions(policy[p_col], n_adopt)
    ids = policy.iloc[pos][id_col].astype(str)
    s = (
        propensity_data.drop_duplicates(id_col, keep="first")
        .set_index(id_col, drop=False)
        .get("expected_incentive_usd")
    )
    if s is None:
        return 0
    vals = s.reindex(ids)
    return int((vals.fillna(0.0) > min_usd).sum())


def build_adoption_cohort_by_demographics(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    id_col: str = "building.id",
) -> dict[str, Any]:
    """For the top n_adopt rows by p_col, count stable synthetic income / education from tract lists."""
    if p_col not in policy.columns or n_adopt <= 0:
        return {
            "n_adopt": 0,
            "n_rows": len(policy),
            "income_counts": {},
            "education_counts": {},
            "row_positions": np.array([], dtype=int),
        }
    pos = propensity_ranked_positions(policy[p_col], n_adopt)
    sub = policy.iloc[pos]
    inc_counts: dict[str, int] = {}
    edu_counts: dict[str, int] = {}
    for _, row in sub.iterrows():
        bid = str(row.get(id_col, ""))
        ip = row.get("income_probs")
        ep = row.get("education_probs")
        if ip is not None and not (isinstance(ip, float) and pd.isna(ip)):
            _, li = _sample_category_index(bid, ip, len(INCOME_BIN_LABELS), INCOME_BIN_LABELS)
        else:
            li = "Unknown (tract income)"
        if ep is not None and not (isinstance(ep, float) and pd.isna(ep)):
            _, le = _sample_category_index(
                f"{bid}_edu", ep, len(EDU_BIN_LABELS), EDU_BIN_LABELS
            )
        else:
            le = "Unknown (tract education)"
        inc_counts[li] = inc_counts.get(li, 0) + 1
        edu_counts[le] = edu_counts.get(le, 0) + 1
    return {
        "n_adopt": int(len(pos)),
        "n_rows": len(policy),
        "income_counts": inc_counts,
        "education_counts": edu_counts,
        "row_positions": pos,
    }


def expected_incentive_sum_on_adopted_cohort(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    propensity_data: pd.DataFrame,
    id_col: str = "building.id",
) -> float | None:
    """Sum expected_incentive_usd over building ids in the top n_adopt by p_col (merge-safe, not iloc)."""
    if (
        propensity_data.empty
        or "expected_incentive_usd" not in propensity_data.columns
        or id_col not in propensity_data.columns
    ):
        return None
    if p_col not in policy.columns or n_adopt <= 0:
        return 0.0
    pos = propensity_ranked_positions(policy[p_col], n_adopt)
    adopted_ids = policy.iloc[pos][id_col].astype(str)
    s_map = (
        propensity_data.drop_duplicates(id_col, keep="first")
        .set_index(id_col, drop=False)
        .get("expected_incentive_usd")
    )
    if s_map is None:
        return None
    return float(s_map.reindex(adopted_ids).fillna(0.0).sum())


def ranking_displacement_at_equal_n(
    building_ids: pd.Series, p_base: pd.Series, p_inc: pd.Series, n: int
) -> tuple[int, int, int]:
    """
    Building sets: top n by p_inc vs top n by p_base (same n). Returns
    (count only in inc cohort, count only in base cohort, overlap size).
    """
    if (
        p_base is None
        or p_inc is None
        or len(p_base) != len(p_inc)
        or len(building_ids) != len(p_inc)
        or n <= 0
    ):
        return 0, 0, 0
    n = min(n, len(p_inc))
    pos_i = propensity_ranked_positions(p_inc, n)
    pos_b = propensity_ranked_positions(p_base, n)
    set_i = set(building_ids.iloc[pos_i].astype(str))
    set_b = set(building_ids.iloc[pos_b].astype(str))
    only_i = len(set_i - set_b)
    only_b = len(set_b - set_i)
    overlap = len(set_i & set_b)
    return only_i, only_b, overlap