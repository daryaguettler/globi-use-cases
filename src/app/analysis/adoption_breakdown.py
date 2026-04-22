"""Adoption cohort vs demographics and incentive effects (for results UI)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from app.analysis.census_lookup import canonical_census_geoid_str

# aligned with app.analysis.census_lookup.INCOME_CATEGORIES_K (16 midpoints, k$)
INCOME_BIN_LABELS: list[str] = [
    "< $10k", "$10k–$15k", "$15k–$20k", "$20k–$25k", "$25k–$30k", "$30k–$35k", "$35k–$40k", "$40k–$45k", "$45k–$50k", "$50k–$60k", "$60k–$75k", "$75k–$100k", "$100k–$125k", "$125k–$150k", "$150k–$200k", "$200k+",
]
# k$ midpoints — same order as use_cases.apply_propensity.INCOME_CATEGORIES
INCOME_BIN_MIDPOINTS_K: tuple[float, ...] = (
    5.0,
    12.5,
    17.5,
    22.5,
    27.5,
    32.5,
    37.5,
    42.5,
    47.5,
    55.0,
    67.5,
    87.5,
    112.5,
    137.5,
    175.0,
    225.0,
)
EDU_BIN_LABELS: list[str] = [
    "HS or less (incl. GED)",
    "Associate / some college",
    "Bachelor's",
    "Graduate / prof. / PhD",
]


def income_label_to_midpoint_k(label: str) -> float | None:
    try:
        i = INCOME_BIN_LABELS.index(label)
        return INCOME_BIN_MIDPOINTS_K[i]
    except ValueError:
        return None


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


def _lookup_tract_dist_by_geoid(tract_distributions: dict[str, dict], geoid_raw: object) -> dict | None:
    c = canonical_census_geoid_str(geoid_raw)
    if not c:
        return None
    if c in tract_distributions:
        return tract_distributions[c]
    for k, v in tract_distributions.items():
        if canonical_census_geoid_str(k) == c:
            return v
    return None


def _row_geoid_candidates(row: pd.Series) -> list[str]:
    """Ordered unique canonical GEOIDs from common columns (11-digit tract id)."""
    seen: set[str] = set()
    out: list[str] = []
    for c in ("geoid", "GEOID", "GEOID20", "GEOID10"):
        if c not in row.index:
            continue
        g = canonical_census_geoid_str(row.get(c))
        if len(g) == 11 and g not in seen:
            seen.add(g)
            out.append(g)
    for c in ("building.tract_id", "feature.location.tract_id", "tract_id"):
        if c not in row.index:
            continue
        g = canonical_census_geoid_str(row.get(c))
        if len(g) == 11 and g not in seen:
            seen.add(g)
            out.append(g)
    return out


def resolve_row_tract_distribution(
    row: pd.Series,
    tract_distributions: dict[str, dict],
    county_fips_map: dict[int, str] | None = None,
) -> dict | None:
    """Match engine `_sample_demographics` / `_residential_chunk` tract lookup (geoid or county+tract)."""
    county_fips_map = county_fips_map or {}
    if not tract_distributions:
        return None
    for g in _row_geoid_candidates(row):
        hit = _lookup_tract_dist_by_geoid(tract_distributions, g)
        if hit is not None:
            return hit
    county_col = next(
        (c for c in ["building.county", "feature.location.county", "county", "county_fips"] if c in row.index),
        None,
    )
    tract_col = next(
        (c for c in ["building.tract_id", "feature.location.tract_id", "tract_id", "GEOID20", "GEOID"] if c in row.index),
        None,
    )
    county = "Middlesex"
    if county_col is not None:
        cv = row.get(county_col)
        if cv is not None and pd.notna(cv):
            if county_col == "county_fips":
                try:
                    ck = int(float(str(cv).split(".")[0]))
                    county = county_fips_map.get(ck) or str(cv)
                except (TypeError, ValueError):
                    county = str(cv)
            else:
                county = str(cv).strip() or county
    tract_str = ""
    if tract_col is not None:
        tv = row.get(tract_col)
        if tv is not None and pd.notna(tv):
            tid = str(tv).strip()
            tract_str = tid[-6:].zfill(6) if len(tid) >= 6 else tid.zfill(6)
    expected = f"{county}_{tract_str}"
    if expected in tract_distributions:
        return tract_distributions[expected]
    for k, d in tract_distributions.items():
        if isinstance(k, str) and k.endswith(f"_{tract_str}") and str(d.get("county", "")) == county:
            return tract_distributions[k]
    for k, d in tract_distributions.items():
        if str(d.get("county", "")) == county:
            return tract_distributions[k]
    return None


def _pmf_vector_ok(vec: object, n: int) -> bool:
    if vec is None or (isinstance(vec, float) and pd.isna(vec)):
        return False
    try:
        a = np.asarray(vec, dtype=float).ravel()
        return bool(a.size == n and np.isfinite(a).all() and float(a.sum()) > 0)
    except (TypeError, ValueError):
        return False


def _normalize_pmf_to_list(vec: object, n: int) -> list[float] | None:
    try:
        a = np.asarray(vec, dtype=float).ravel()
        if a.size != n or not np.isfinite(a).all():
            return None
        s = float(a.sum())
        if s <= 0:
            return None
        return (a / s).tolist()
    except (TypeError, ValueError):
        return None


def fill_missing_demographic_probs(
    df: pd.DataFrame,
    tract_distributions: dict[str, dict] | None,
    county_fips_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Ensure ``income_probs`` (16) and ``education_probs`` (4) when tract PMFs exist (e.g. census CSV path)."""
    td = tract_distributions or {}
    out = df.copy()
    if "income_probs" not in out.columns:
        out["income_probs"] = None
    if "education_probs" not in out.columns:
        out["education_probs"] = None
    for idx in out.index:
        ip = out.at[idx, "income_probs"]
        ep = out.at[idx, "education_probs"]
        ok_i = _pmf_vector_ok(ip, len(INCOME_BIN_LABELS))
        ok_e = _pmf_vector_ok(ep, len(EDU_BIN_LABELS))
        if ok_i and ok_e:
            continue
        if not td:
            continue
        dist = resolve_row_tract_distribution(out.loc[idx], td, county_fips_map)
        if dist is None:
            continue
        if not ok_i:
            inc_src = dist.get("income")
            if inc_src is None:
                inc_src = dist.get("income_probs")
            inc_list = _normalize_pmf_to_list(inc_src, len(INCOME_BIN_LABELS))
            if inc_list:
                out.at[idx, "income_probs"] = inc_list
        if not ok_e:
            edu_src = dist.get("education")
            if edu_src is None:
                edu_src = dist.get("education_probs")
            edu_list = _normalize_pmf_to_list(edu_src, len(EDU_BIN_LABELS))
            if edu_list:
                out.at[idx, "education_probs"] = edu_list
    out = copy_demographic_probs_from_same_geoid(out)
    return out


def copy_demographic_probs_from_same_geoid(df: pd.DataFrame) -> pd.DataFrame:
    """Copy ``income_probs`` / ``education_probs`` from another row with the same canonical GEOID."""
    out = df
    if "income_probs" not in out.columns:
        return out

    def geoid_key_for_index(i: object) -> str:
        row = out.loc[i]
        cand = _row_geoid_candidates(row)
        return cand[0] if cand else ""

    template: dict[str, tuple[list[float], list[float]]] = {}
    for idx in out.index:
        ip = out.at[idx, "income_probs"]
        ep = out.at[idx, "education_probs"] if "education_probs" in out.columns else None
        if not _pmf_vector_ok(ip, len(INCOME_BIN_LABELS)):
            continue
        if ep is None or not _pmf_vector_ok(ep, len(EDU_BIN_LABELS)):
            continue
        gk = geoid_key_for_index(idx)
        if gk and gk not in template:
            template[gk] = (
                [float(x) for x in np.asarray(ip, dtype=float).ravel()],
                [float(x) for x in np.asarray(ep, dtype=float).ravel()],
            )

    for idx in out.index:
        ip = out.at[idx, "income_probs"]
        ep = out.at[idx, "education_probs"] if "education_probs" in out.columns else None
        ok_i = _pmf_vector_ok(ip, len(INCOME_BIN_LABELS))
        ok_e = _pmf_vector_ok(ep, len(EDU_BIN_LABELS))
        if ok_i and ok_e:
            continue
        gk = geoid_key_for_index(idx)
        if not gk or gk not in template:
            continue
        inc_t, edu_t = template[gk]
        if not ok_i:
            out.at[idx, "income_probs"] = inc_t
        if not ok_e:
            out.at[idx, "education_probs"] = edu_t
    return out


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


def pick_adoption_scenario_for_cohort_breakdown(
    keys_in_both: list[str],
    scenario_results: dict,
    retrofit_label: str,
    *,
    reference_zero_adoption: frozenset[str] = frozenset({"no_adoption"}),
) -> str:
    """Choose which adoption-curve run drives cohort / incentive lift metrics.

    ``adoption_curves.json`` key order usually starts with ``no_adoption`` (0%%
    forever). The first key would make adopter counts and charts all zero even
    when other scenarios have real uptake. Prefer a scenario with the largest
    final-year adopter count, using only non-reference keys when the list is
    non-empty after filtering.
    """

    def _n_final(adoption_key: str) -> int:
        ys = scenario_results[adoption_key]["uptake"].yearly_summary
        row = last_year_uptake_row(ys, adoption_key, retrofit_label)
        return n_adopters_from_yearly_row(row) if row is not None else 0

    if not keys_in_both:
        return ""
    non_ref = [k for k in keys_in_both if k not in reference_zero_adoption]
    pool = non_ref if non_ref else keys_in_both
    return max(pool, key=_n_final)


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


def single_adoption_scenario_finance_row(
    ad_name: str,
    policy_base: pd.DataFrame,
    policy_inc: pd.DataFrame,
    propensity_data: pd.DataFrame,
    scenario_results: dict,
    scenario_results_inc: dict,
    r_label: str,
) -> dict[str, Any]:
    """Per adoption curve: adopters, costs, incentives, ranking (for results table + KPI ranges)."""
    yb = scenario_results[ad_name]["uptake"].yearly_summary
    yi = scenario_results_inc[ad_name]["uptake"].yearly_summary
    row_b = last_year_uptake_row(yb, ad_name, r_label)
    row_i = last_year_uptake_row(yi, ad_name, r_label)
    n_b = n_adopters_from_yearly_row(row_b) if row_b is not None else 0
    n_i = n_adopters_from_yearly_row(row_i) if row_i is not None else 0
    cum_b = float(row_b.get("cumulative_adoption_pct", 0.0) or 0.0) if row_b is not None else 0.0
    cum_i = float(row_i.get("cumulative_adoption_pct", 0.0) or 0.0) if row_i is not None else 0.0
    n_w_incentive = (
        count_adopters_with_positive_incentive(
            policy_inc, "acceptance_probability", n_i, propensity_data, min_usd=0.0
        )
        if row_i is not None
        else 0
    )
    n_eq = min(n_b, n_i)
    only_i: int | None = None
    if n_eq > 0 and n_b and n_i:
        only_i, _, _ = ranking_displacement_at_equal_n(
            policy_inc["building.id"],
            policy_base["acceptance_probability"],
            policy_inc["acceptance_probability"],
            n_eq,
        )
    inc_cohort_usd = (
        expected_incentive_sum_on_adopted_cohort(
            policy_inc, "acceptance_probability", n_i, propensity_data
        )
        if n_i > 0
        else 0.0
    )
    if inc_cohort_usd is None:
        inc_cohort_usd = 0.0
    gross_cohort = (
        gross_upfront_sum_on_adopted_cohort(
            policy_inc, "acceptance_probability", n_i, propensity_data
        )
        if n_i > 0
        else 0.0
    )
    if gross_cohort is None:
        gross_cohort = 0.0
    pos_inc_sum = (
        positive_expected_incentive_sum_on_adopted_cohort(
            policy_inc, "acceptance_probability", n_i, propensity_data
        )
        if n_i > 0
        else 0.0
    )
    if pos_inc_sum is None:
        pos_inc_sum = 0.0
    avg_inc_adopter = float(inc_cohort_usd) / n_i if n_i else 0.0
    avg_inc_recipient = float(pos_inc_sum) / n_w_incentive if n_w_incentive else 0.0
    return {
        "adoption_scenario": ad_name,
        "row_ok": row_b is not None and row_i is not None,
        "n_adopters_with_inc": n_i,
        "n_adopters_no_inc": n_b,
        "n_positive_incentive": n_w_incentive,
        "extra_adopters": n_i - n_b,
        "cum_pct_with_inc": cum_i,
        "cum_pct_no_inc": cum_b,
        "cohort_expected_incentive_usd": float(inc_cohort_usd),
        "cohort_positive_incentive_usd": float(pos_inc_sum),
        "cohort_gross_cost_usd": float(gross_cohort),
        "avg_incentive_per_adopter_usd": avg_inc_adopter,
        "avg_incentive_per_recipient_usd": avg_inc_recipient,
        "n_eq": n_eq,
        "reranked_top_n": only_i,
    }


def build_adoption_finance_breakdown_dataframe(
    policy_base: pd.DataFrame,
    policy_inc: pd.DataFrame,
    propensity_data: pd.DataFrame,
    scenario_results: dict,
    scenario_results_inc: dict,
    session_retrofit_name: str,
) -> tuple[pd.DataFrame | None, str]:
    """All adoption scenarios: finance columns for the results table; (None, '') if not computable."""
    if (
        "acceptance_probability" not in policy_base.columns
        or "acceptance_probability" not in policy_inc.columns
    ):
        return None, ""
    keys_i = sorted(k for k in scenario_results_inc if scenario_results.get(k))
    if not keys_i:
        return None, ""
    r_label = resolve_retrofit_scenario_name(policy_inc, session_retrofit_name)
    rows = [
        single_adoption_scenario_finance_row(
            k,
            policy_base,
            policy_inc,
            propensity_data,
            scenario_results,
            scenario_results_inc,
            r_label,
        )
        for k in keys_i
    ]
    return pd.DataFrame(rows), r_label


def build_adoption_cohort_by_demographics(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    id_col: str = "building.id",
    *,
    expected_incentive_by_id: pd.Series | None = None,
) -> dict[str, Any]:
    """Top n_adopt by p_col: one sampled income/education label per building from tract PMFs; optional incentive split."""
    if p_col not in policy.columns or n_adopt <= 0:
        return {
            "n_adopt": 0,
            "n_rows": len(policy),
            "income_counts": {},
            "education_counts": {},
            "income_by_incentive": None,
            "education_by_incentive": None,
            "row_positions": np.array([], dtype=int),
        }
    pos = propensity_ranked_positions(policy[p_col], n_adopt)
    sub = policy.iloc[pos]
    inc_counts: dict[str, int] = {}
    edu_counts: dict[str, int] = {}
    use_split = expected_incentive_by_id is not None
    inc_pos: dict[str, int] = {}
    inc_zero: dict[str, int] = {}
    edu_pos: dict[str, int] = {}
    edu_zero: dict[str, int] = {}
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
        if use_split:
            amt = 0.0
            if bid and bid in expected_incentive_by_id.index:
                try:
                    amt = float(expected_incentive_by_id.loc[bid])
                except (TypeError, ValueError):
                    amt = 0.0
            if pd.isna(amt):
                amt = 0.0
            subsidized = amt > 0.0
            if subsidized:
                inc_pos[li] = inc_pos.get(li, 0) + 1
                edu_pos[le] = edu_pos.get(le, 0) + 1
            else:
                inc_zero[li] = inc_zero.get(li, 0) + 1
                edu_zero[le] = edu_zero.get(le, 0) + 1
    out: dict[str, Any] = {
        "n_adopt": int(len(pos)),
        "n_rows": len(policy),
        "income_counts": inc_counts,
        "education_counts": edu_counts,
        "row_positions": pos,
    }
    if use_split:
        out["income_by_incentive"] = {"positive": inc_pos, "zero": inc_zero}
        out["education_by_incentive"] = {"positive": edu_pos, "zero": edu_zero}
    else:
        out["income_by_incentive"] = None
        out["education_by_incentive"] = None
    return out


def build_adoption_cohort_by_demographics_with_lift(
    policy_inc: pd.DataFrame,
    policy_base: pd.DataFrame,
    p_col_inc: str,
    p_col_base: str,
    n_adopt_inc: int,
    id_col: str = "building.id",
) -> dict[str, Any]:
    """Top ``n`` by with-incentive propensity vs same ``n`` by no-incentive propensity.

    Let ``n = min(n_adopt_inc, len(policy_inc), len(policy_base))``. **Would adopt** = in the
    with-incentive top-``n`` and also in the **no-incentive** top-``n`` (same cohort size, two
    rankings). **Added** = in the with-incentive top-``n`` but not in the no-incentive top-``n``
    (lift from cost reduction / reordering at fixed adoption count).

    Using the baseline run's *total* adopters ``n_b`` for the comparison set is wrong when
    ``n_b`` is large: top-``n_b`` by base propensity then includes almost the whole portfolio, so
    **Added** collapses to zero.
    """
    if (
        p_col_inc not in policy_inc.columns
        or p_col_base not in policy_base.columns
        or n_adopt_inc <= 0
    ):
        return {
            "n_adopt_inc": 0,
            "n_rank_compare": 0,
            "income_would_adopt": {},
            "income_added": {},
            "education_would_adopt": {},
            "education_added": {},
            "income_counts": {},
            "education_counts": {},
            "row_positions": np.array([], dtype=int),
        }
    n_take = min(int(n_adopt_inc), len(policy_inc), len(policy_base))
    base_pos = propensity_ranked_positions(policy_base[p_col_base], n_take)
    base_ids = set(policy_base.iloc[base_pos][id_col].astype(str))

    pos = propensity_ranked_positions(policy_inc[p_col_inc], n_take)
    sub = policy_inc.iloc[pos]
    inc_would: dict[str, int] = {}
    inc_added: dict[str, int] = {}
    edu_would: dict[str, int] = {}
    edu_added: dict[str, int] = {}
    inc_total: dict[str, int] = {}
    edu_total: dict[str, int] = {}

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
        inc_total[li] = inc_total.get(li, 0) + 1
        edu_total[le] = edu_total.get(le, 0) + 1
        in_base = bool(bid) and bid in base_ids
        if in_base:
            inc_would[li] = inc_would.get(li, 0) + 1
            edu_would[le] = edu_would.get(le, 0) + 1
        else:
            inc_added[li] = inc_added.get(li, 0) + 1
            edu_added[le] = edu_added.get(le, 0) + 1

    return {
        "n_adopt_inc": int(len(pos)),
        "n_rank_compare": int(n_take),
        "income_would_adopt": inc_would,
        "income_added": inc_added,
        "education_would_adopt": edu_would,
        "education_added": edu_added,
        "income_counts": inc_total,
        "education_counts": edu_total,
        "row_positions": pos,
    }


def build_income_portfolio_adoption_profile(
    policy_inc: pd.DataFrame,
    policy_base: pd.DataFrame,
    p_col: str,
    n_rank_compare: int,
    incentive_usd_by_income_k: dict[float, float] | None,
    id_col: str = "building.id",
) -> dict[str, Any]:
    """Per income bin (one sampled label per building): portfolio share and adoption rates.

    - **portfolio_pct**: % of all buildings in each bin (sums to 100).
    - **pct_adopt_base**: % of buildings in the bin that are in the top ``n_rank_compare`` by
      no-incentive ``p_col``.
    - **pct_adopt_lift**: % of buildings in the bin in the with-incentive top ``n_rank_compare``
      but not the no-incentive top ``n_rank_compare``, **only counted** when the bin’s income
      midpoint has strictly positive incentive in ``incentive_usd_by_income_k`` (Configure tab).
    - **pct_stack_remainder**: ``100 - pct_adopt_base - pct_adopt_lift`` (did not adopt at this
      cutoff / not attributed lift), for 100% stacked bars per bin.

    **Order of operations:** rank all buildings once by **no-incentive** propensity and take
    top ``n`` → **base adopters** per sampled income bin. Then rank by **with-incentive**
    propensity and take top ``n``; buildings that enter this set but not the base set **and**
    whose bin has Configure incentive ``> 0`` → **lift** segment. Remainder did not rank in
    the base top ``n`` and are not counted as policy lift (including ineligible tiers).
    """
    inc_map = incentive_usd_by_income_k or {}
    out_empty: dict[str, Any] = {
        "labels": [],
        "n_portfolio": 0,
        "n_rank_compare": 0,
        "count_in_bin": [],
        "portfolio_pct": [],
        "pct_adopt_base": [],
        "pct_adopt_lift": [],
        "pct_stack_remainder": [],
        "tier_has_incentive": [],
    }
    if (
        p_col not in policy_inc.columns
        or p_col not in policy_base.columns
        or n_rank_compare <= 0
        or policy_inc.empty
    ):
        return out_empty
    n_take = min(int(n_rank_compare), len(policy_inc), len(policy_base))
    if n_take <= 0:
        return out_empty

    base_pos = propensity_ranked_positions(policy_base[p_col], n_take)
    inc_pos = propensity_ranked_positions(policy_inc[p_col], n_take)
    base_ids = set(policy_base.iloc[base_pos][id_col].astype(str))
    inc_ids = set(policy_inc.iloc[inc_pos][id_col].astype(str))

    n_bin: dict[str, int] = {}
    adopt_b: dict[str, int] = {}
    lift_elig: dict[str, int] = {}

    for _, row in policy_inc.iterrows():
        bid = str(row.get(id_col, ""))
        ip = row.get("income_probs")
        if ip is not None and not (isinstance(ip, float) and pd.isna(ip)):
            _, li = _sample_category_index(bid, ip, len(INCOME_BIN_LABELS), INCOME_BIN_LABELS)
        else:
            li = "Unknown (tract income)"
        n_bin[li] = n_bin.get(li, 0) + 1
        if bid and bid in base_ids:
            adopt_b[li] = adopt_b.get(li, 0) + 1
        if bid and bid in inc_ids and bid not in base_ids:
            mid = income_label_to_midpoint_k(li)
            if mid is not None and float(inc_map.get(mid, 0.0)) > 0.0:
                lift_elig[li] = lift_elig.get(li, 0) + 1

    labels = [c for c in INCOME_BIN_LABELS if n_bin.get(c, 0) > 0] + sorted(
        k for k in n_bin if k not in INCOME_BIN_LABELS
    )
    n_tot = int(sum(n_bin.values()))
    count_in_bin = [n_bin[l] for l in labels]
    port_pct = [100.0 * c / n_tot for c in count_in_bin] if n_tot else []
    pct_b = [100.0 * adopt_b.get(l, 0) / n_bin[l] if n_bin[l] else 0.0 for l in labels]
    pct_lift = [100.0 * lift_elig.get(l, 0) / n_bin[l] if n_bin[l] else 0.0 for l in labels]
    pct_rem = [
        max(0.0, 100.0 - pct_b[i] - pct_lift[i]) for i in range(len(labels))
    ]
    tier_inc = []
    for l in labels:
        mid = income_label_to_midpoint_k(l)
        tier_inc.append(
            bool(mid is not None and float(inc_map.get(mid, 0.0)) > 0.0)
        )

    return {
        "labels": labels,
        "n_portfolio": n_tot,
        "n_rank_compare": int(n_take),
        "count_in_bin": count_in_bin,
        "portfolio_pct": port_pct,
        "pct_adopt_base": pct_b,
        "pct_adopt_lift": pct_lift,
        "pct_stack_remainder": pct_rem,
        "tier_has_incentive": tier_inc,
    }


def retrofit_cost_column_for_policy(policy: pd.DataFrame) -> tuple[pd.Series, str]:
    """Per-row total retrofit deal cost (USD) and which column was used."""
    if "gross_upfront_usd" in policy.columns:
        s = pd.to_numeric(policy["gross_upfront_usd"], errors="coerce").fillna(0.0)
        if float(s.sum()) > 0 or float(s.max()) > 0:
            return s, "gross_upfront_usd"
    for k in ("adjusted_net_cost.AllCustomers", "net_cost.AllCustomers", "cost.Total"):
        if k in policy.columns:
            s = pd.to_numeric(policy[k], errors="coerce").fillna(0.0)
            return s, k
    return pd.Series(0.0, index=policy.index), ""


def build_retrofit_cost_bucket_incentive_table(
    policy_inc: pd.DataFrame,
    *,
    n_adopters: int,
    p_col: str = "acceptance_probability",
    incentive_col: str = "expected_incentive_usd",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Quantile buckets of retrofit cost: building counts, % adopted (top n by propensity), % with adopt + incentive > 0.

    Adopters match the finance / energy model: top ``n_adopters`` rows by ``p_col``.
    """
    empty = pd.DataFrame(
        columns=[
            "cost_bucket",
            "n_buildings",
            "n_adopted",
            "pct_adopted",
            "n_incentivized_adopters",
            "pct_incentivized_adopters_in_bucket",
        ]
    )
    if policy_inc.empty or n_adopters <= 0 or p_col not in policy_inc.columns:
        return empty
    cost, _src = retrofit_cost_column_for_policy(policy_inc)
    if float(cost.max()) <= 0:
        return empty

    pos = propensity_ranked_positions(policy_inc[p_col], n_adopters)
    adopted = np.zeros(len(policy_inc), dtype=bool)
    adopted[pos] = True

    inc = (
        pd.to_numeric(policy_inc[incentive_col], errors="coerce").fillna(0.0)
        if incentive_col in policy_inc.columns
        else pd.Series(0.0, index=policy_inc.index)
    )
    incent_adopt = adopted & (inc > 0)

    nq = max(1, min(int(n_bins), len(policy_inc)))
    try:
        bucket = pd.qcut(cost, q=nq, duplicates="drop")
    except (ValueError, TypeError):
        bucket = pd.Series(["all"] * len(policy_inc), index=policy_inc.index, dtype=object)

    df = pd.DataFrame(
        {
            "cost_bucket": bucket.astype(str),
            "adopted": adopted,
            "incentivized_adopter": incent_adopt,
        }
    )
    g = df.groupby("cost_bucket", sort=True, observed=False)
    out = g.agg(
        n_buildings=("adopted", "count"),
        n_adopted=("adopted", "sum"),
        n_incentivized_adopters=("incentivized_adopter", "sum"),
    ).reset_index()
    out["pct_adopted"] = np.where(
        out["n_buildings"] > 0,
        100.0 * out["n_adopted"].astype(float) / out["n_buildings"].astype(float),
        0.0,
    )
    out["pct_incentivized_adopters_in_bucket"] = np.where(
        out["n_buildings"] > 0,
        100.0 * out["n_incentivized_adopters"].astype(float) / out["n_buildings"].astype(float),
        0.0,
    )
    return out


def _adopted_building_ids_by_propensity(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    id_col: str = "building.id",
) -> pd.Series:
    """building.id for the top n_adopt rows by *p_col* (same order as uptake ranking)."""
    if p_col not in policy.columns or n_adopt <= 0:
        return pd.Series([], dtype=str)
    pos = propensity_ranked_positions(policy[p_col], n_adopt)
    return policy.iloc[pos][id_col].astype(str)


def gross_upfront_sum_on_adopted_cohort(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    propensity_data: pd.DataFrame,
    id_col: str = "building.id",
) -> float | None:
    """Sum gross_upfront_usd for the top n_adopt buildings by *p_col*."""
    if (
        propensity_data.empty
        or "gross_upfront_usd" not in propensity_data.columns
        or id_col not in propensity_data.columns
    ):
        return None
    if p_col not in policy.columns or n_adopt <= 0:
        return 0.0
    adopted_ids = _adopted_building_ids_by_propensity(policy, p_col, n_adopt, id_col)
    s_map = (
        propensity_data.drop_duplicates(id_col, keep="first")
        .set_index(id_col, drop=False)
        .get("gross_upfront_usd")
    )
    if s_map is None:
        return None
    return float(s_map.reindex(adopted_ids).fillna(0.0).sum())


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
    adopted_ids = _adopted_building_ids_by_propensity(policy, p_col, n_adopt, id_col)
    s_map = (
        propensity_data.drop_duplicates(id_col, keep="first")
        .set_index(id_col, drop=False)
        .get("expected_incentive_usd")
    )
    if s_map is None:
        return None
    return float(s_map.reindex(adopted_ids).fillna(0.0).sum())


def positive_expected_incentive_sum_on_adopted_cohort(
    policy: pd.DataFrame,
    p_col: str,
    n_adopt: int,
    propensity_data: pd.DataFrame,
    id_col: str = "building.id",
) -> float | None:
    """Sum expected_incentive_usd only where > 0 among the adopted cohort."""
    if (
        propensity_data.empty
        or "expected_incentive_usd" not in propensity_data.columns
        or id_col not in propensity_data.columns
    ):
        return None
    if p_col not in policy.columns or n_adopt <= 0:
        return 0.0
    adopted_ids = _adopted_building_ids_by_propensity(policy, p_col, n_adopt, id_col)
    s_map = (
        propensity_data.drop_duplicates(id_col, keep="first")
        .set_index(id_col, drop=False)
        .get("expected_incentive_usd")
    )
    if s_map is None:
        return None
    vals = s_map.reindex(adopted_ids).fillna(0.0).astype(float)
    return float(vals[vals > 0].sum())


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