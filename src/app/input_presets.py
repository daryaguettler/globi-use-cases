"""Serializable energy input layouts (browse paths / batch filenames per year)."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PRESET_VERSION = 1

MODE_BROWSE = "Browse data inputs"
MODE_BATCH = "Upload many, pick per year"
MODE_PER_YEAR = "Upload files (per year)"


@dataclass
class ApplyResult:
    ok: bool
    error: str | None = None
    year_files: dict[int, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    n_years: int = 1
    selected_years: list[int] = field(default_factory=list)
    scenario_name: str = "Retrofit"
    upload_mode: str = MODE_BROWSE
    building_area_unit: str = "sqm"


def _norm_rel(s: str) -> str:
    return str(s).replace("\\", "/")


def build_input_preset_dict(
    upload_mode: str,
    n_years: int,
    selected_years: list[int],
    scenario_name: str,
    year_files: dict[int, Any],
    building_area_unit: str = "sqm",
) -> dict[str, Any]:
    assignments: dict[str, dict[str, str]] = {}
    for yr in selected_years:
        e = year_files.get(yr) or {}
        if upload_mode == MODE_BROWSE:
            br, sr = e.get("baseline_rel"), e.get("scenario_rel")
            if isinstance(br, str) and isinstance(sr, str) and br and sr:
                ys = str(int(yr))
                assignments[ys] = {
                    "kind": "browse",
                    "baseline": _norm_rel(br),
                    "scenario": _norm_rel(sr),
                }
        elif upload_mode == MODE_BATCH:
            bl = e.get("baseline_label")
            sl = e.get("scenario_label")
            if isinstance(bl, str) and isinstance(sl, str) and bl and sl:
                ys = str(int(yr))
                assignments[ys] = {
                    "kind": "batch",
                    "baseline": bl,
                    "scenario": sl,
                }
    return {
        "version": PRESET_VERSION,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "upload_mode": upload_mode,
        "n_years": int(n_years),
        "selected_years": [int(y) for y in selected_years],
        "scenario_name": (scenario_name or "Retrofit").strip() or "Retrofit",
        "building_area_unit": str(building_area_unit or "sqm").lower()
        if str(building_area_unit or "sqm").lower() in ("sqm", "sqft")
        else "sqm",
        "assignments": assignments,
    }


def apply_input_preset(
    preset: dict[str, Any],
    data_inputs_dir: Path,
    uploaded_library: dict[str, bytes] | None,
) -> ApplyResult:
    lib = uploaded_library or {}
    try:
        ver = int(preset.get("version", 1))
        if ver != PRESET_VERSION:
            return ApplyResult(
                ok=False,
                error=f"unsupported preset version {ver} (expected {PRESET_VERSION})",
            )
        upload_mode = str(preset.get("upload_mode") or MODE_BROWSE)
        if upload_mode not in (MODE_BROWSE, MODE_BATCH, MODE_PER_YEAR):
            return ApplyResult(ok=False, error=f"unknown upload_mode {upload_mode!r}")
        n_years = int(preset.get("n_years", 1))
        raw_years = preset.get("selected_years") or [2025]
        selected_years = [int(y) for y in raw_years]
        scenario_name = str(preset.get("scenario_name") or "Retrofit").strip() or "Retrofit"
        raw_bau = str(preset.get("building_area_unit") or "sqm").lower()
        building_area_unit = raw_bau if raw_bau in ("sqm", "sqft") else "sqm"
        assignments: dict[str, Any] = preset.get("assignments") or {}
    except (TypeError, ValueError) as exc:
        return ApplyResult(ok=False, error=f"invalid preset: {exc}")

    warnings: list[str] = []
    year_files: dict[int, dict[str, Any]] = {}
    data_inputs_dir = data_inputs_dir.resolve()

    if upload_mode == MODE_BROWSE:
        for yr in selected_years:
            ys = str(int(yr))
            spec = assignments.get(ys)
            if not spec or spec.get("kind") != "browse":
                continue
            br = spec.get("baseline")
            sr = spec.get("scenario")
            if not isinstance(br, str) or not isinstance(sr, str):
                warnings.append(f"year {yr}: invalid browse paths in preset")
                continue
            br_n, sr_n = _norm_rel(br), _norm_rel(sr)
            pb = (data_inputs_dir / br_n).resolve()
            ps = (data_inputs_dir / sr_n).resolve()
            try:
                pb.relative_to(data_inputs_dir)
                ps.relative_to(data_inputs_dir)
            except ValueError:
                warnings.append(f"year {yr}: path escapes data/inputs")
                continue
            if not pb.is_file():
                warnings.append(f"year {yr}: missing baseline {br_n}")
                continue
            if not ps.is_file():
                warnings.append(f"year {yr}: missing scenario {sr_n}")
                continue
            year_files[int(yr)] = {
                "baseline": pb.read_bytes(),
                "scenario": ps.read_bytes(),
                "baseline_rel": br_n,
                "scenario_rel": sr_n,
            }

    elif upload_mode == MODE_BATCH:
        for yr in selected_years:
            ys = str(int(yr))
            spec = assignments.get(ys)
            if not spec or spec.get("kind") != "batch":
                continue
            bl = spec.get("baseline")
            sl = spec.get("scenario")
            if not isinstance(bl, str) or not isinstance(sl, str):
                warnings.append(f"year {yr}: invalid batch labels in preset")
                continue
            entry: dict[str, Any] = {
                "baseline_label": bl,
                "scenario_label": sl,
            }
            if bl in lib:
                entry["baseline"] = lib[bl]
            else:
                warnings.append(
                    f"year {yr}: batch pool has no file {bl!r} — upload it, then load preset again"
                )
            if sl in lib:
                entry["scenario"] = lib[sl]
            else:
                warnings.append(
                    f"year {yr}: batch pool has no file {sl!r} — upload it, then load preset again"
                )
            year_files[int(yr)] = entry

    else:
        # per-year upload: only restore layout; user re-uploads files
        pass

    return ApplyResult(
        ok=True,
        year_files=year_files,
        warnings=warnings,
        n_years=n_years,
        selected_years=selected_years,
        scenario_name=scenario_name,
        upload_mode=upload_mode,
        building_area_unit=building_area_unit,
    )


def preset_json_dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2)


def preset_json_loads(raw: str) -> dict[str, Any]:
    return json.loads(raw)
