"""Split a multi-scenario globi parquet into per-scenario EnergyAndPeak files.

A file like 342501.pq contains all scenarios stacked in a MultiIndex that
includes a ``retrofit.scenario`` level.  This script filters each requested
scenario, samples a single montecarlo ID, and writes one parquet per scenario.

Usage
-----
    python src/tools/split_scenarios.py \\
        data/inputs/globi_outputs/342501.pq \\
        --scenarios Baseline ASHP \\
        --out-dir data/inputs/globi_outputs/split

    # Split every scenario found in the file
    python src/tools/split_scenarios.py data/inputs/globi_outputs/342501.pq

    # Use a fixed random seed for reproducibility
    python src/tools/split_scenarios.py data/inputs/globi_outputs/342501.pq --seed 42

Output filenames: <scenario>_EnergyAndPeak.pq
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


SCENARIO_LEVEL = "retrofit.scenario"
MONTECARLO_LEVEL = "montecarlo.id"


def split_scenarios(
    src: Path,
    scenarios: list[str] | None,
    out_dir: Path,
    seed: int | None = None,
) -> list[Path]:
    """Read *src*, split by scenario, sample one montecarlo ID, write parquets.

    Parameters
    ----------
    src:
        Path to the multi-scenario parquet (e.g. ``342501.pq``).
    scenarios:
        Scenario names to extract.  ``None`` means extract all.
    out_dir:
        Directory to write output files into (created if absent).
    seed:
        Random seed for montecarlo ID sampling (``None`` = non-deterministic).

    Returns
    -------
    List of paths to the written files.
    """
    df = pd.read_parquet(src)

    if SCENARIO_LEVEL not in df.index.names:
        raise ValueError(
            f"Expected index level '{SCENARIO_LEVEL}' not found in {src}.\n"
            f"Available levels: {df.index.names}"
        )

    available = df.index.get_level_values(SCENARIO_LEVEL).unique().tolist()
    targets = scenarios if scenarios is not None else available

    missing = [s for s in targets if s not in available]
    if missing:
        raise ValueError(
            f"Scenarios not found in {src}: {missing}\n"
            f"Available: {available}"
        )

    # Pick one montecarlo ID to use across all scenarios
    if MONTECARLO_LEVEL in df.index.names:
        all_mc_ids = df.index.get_level_values(MONTECARLO_LEVEL).unique().tolist()
        rng = random.Random(seed)
        chosen_mc_id = rng.choice(all_mc_ids)
        print(f"Sampled montecarlo.id = {chosen_mc_id}  (seed={seed}, pool size={len(all_mc_ids)})")
    else:
        chosen_mc_id = None
        print(f"Warning: '{MONTECARLO_LEVEL}' not found in index; skipping montecarlo sampling.")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for scenario in targets:
        subset = df.xs(scenario, level=SCENARIO_LEVEL)
        if chosen_mc_id is not None:
            subset = subset.xs(chosen_mc_id, level=MONTECARLO_LEVEL)
        out_path = out_dir / f"{scenario}_EnergyAndPeak.pq"
        subset.to_parquet(out_path)
        print(f"Wrote {out_path}  ({len(subset):,} rows × {subset.shape[1]} cols)")
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path, help="Multi-scenario parquet file")
    parser.add_argument(
        "--scenarios", "-s",
        nargs="+",
        default=None,
        help="Scenario names to extract (default: all)",
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: same directory as src)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for montecarlo ID sampling (default: non-deterministic)",
    )
    args = parser.parse_args()

    src: Path = args.src.resolve()
    out_dir: Path = args.out_dir.resolve() if args.out_dir else src.parent

    split_scenarios(src=src, scenarios=args.scenarios, out_dir=out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
