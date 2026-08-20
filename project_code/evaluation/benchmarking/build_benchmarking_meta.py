#!/usr/bin/env python3
"""Build a benchmark test set with no overlap to ABB4, ABB3, or Boltz training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_TEST_META = Path(
    "/opig-shared/users/lina4783/structures_final/test_meta.csv"
)
DEFAULT_OUTPUT_META = Path(
    "/opig-shared/users/lina4783/structures_final/benchmarking_meta.csv"
)
DEFAULT_ABB4_TRAIN = Path(
    "/opig-shared/users/lina4783/ABB4/data/Exp_strucs_stage1.csv"
)
DEFAULT_ABB3_SPLIT = Path(
    "/opig-shared/users/lina4783/abodybuilder3/data/split.csv"
)
DEFAULT_DEPO_META = Path(
    "/opig-shared/users/lina4783/data/split_v01/df_filtered_ab_split_v4.csv"
)
DEFAULT_BENCHMARK_DIR = Path(
    "/opig-shared/users/lina4783/abb4_experiments/evaluation/benchmarking"
)

BOLTZ_CUTOFF = pd.Timestamp("2023-06-01")

ABB3_SPOT_CHECKS = {
    "6i8s_G0-K0": "pdb_00006i8s:G::K",
    "6hxw_H0-L0": "pdb_00006hxw:H::L",
    "1b2w_H0-L0": "pdb_00001b2w:H::L",
}


def normalize_abb3_chain(chain: str) -> str:
    if chain.endswith("0") and len(chain) > 1:
        return chain[:-1]
    return chain


def abb3_structure_to_pdb_name(structure: str) -> str | None:
    if "_" not in structure or "-" not in structure:
        return None
    pdb_id, chain_part = structure.split("_", 1)
    heavy_raw, light_raw = chain_part.split("-", 1)
    heavy = normalize_abb3_chain(heavy_raw)
    light = normalize_abb3_chain(light_raw)
    return f"pdb_{pdb_id.lower().zfill(8)}:{heavy}::{light}"


def load_abb4_train_cdrs(path: Path) -> set[str]:
    abb4 = pd.read_csv(path, usecols=["concat_CDR", "split"])
    train = abb4.loc[abb4["split"] == "train", "concat_CDR"].dropna()
    return set(train)


def load_abb3_train_pdb_names(path: Path) -> tuple[set[str], pd.DataFrame]:
    split = pd.read_csv(path, usecols=["structure", "split"])
    train = split.loc[split["split"] == "train", "structure"].dropna().astype(str)
    converted = train.map(abb3_structure_to_pdb_name)
    mapping = pd.DataFrame(
        {
            "structure": train.values,
            "pdb_name": converted.values,
        }
    )
    failed = mapping["pdb_name"].isna()
    if failed.any():
        print(
            "Warning: failed to convert "
            f"{failed.sum()} ABB3 train structure IDs; examples: "
            f"{mapping.loc[failed, 'structure'].head(5).tolist()}"
        )
    return set(mapping["pdb_name"].dropna()), mapping


def load_pdb_deposition_dates(path: Path) -> pd.Series:
    depo = pd.read_csv(path, usecols=["INSTANCE", "PDBdepo"])
    depo["PDBdepo"] = pd.to_datetime(depo["PDBdepo"], format="%Y-%m-%d", errors="coerce")
    depo = depo.drop_duplicates(subset="INSTANCE", keep="first")
    return depo.set_index("INSTANCE")["PDBdepo"]


def apply_filters(
    test_df: pd.DataFrame,
    abb4_train_cdrs: set[str],
    abb3_train_pdb_names: set[str],
    pdb_depo: pd.Series,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    work = test_df.copy()
    work["PDBdepo"] = work["pdb_name"].map(pdb_depo)

    work["exclude_abb4_cdr"] = work["concat_CDR"].isin(abb4_train_cdrs)
    work["exclude_abb3_structure"] = work["pdb_name"].isin(abb3_train_pdb_names)
    work["exclude_boltz_date"] = work["PDBdepo"].isna() | (work["PDBdepo"] <= cutoff)
    work["exclude_any"] = (
        work["exclude_abb4_cdr"]
        | work["exclude_abb3_structure"]
        | work["exclude_boltz_date"]
    )

    excluded = work.loc[work["exclude_any"]].copy()
    filtered = work.loc[~work["exclude_any"], test_df.columns].copy()

    summary = {
        "input_rows": int(len(test_df)),
        "removed_abb4_cdr_overlap": int(work["exclude_abb4_cdr"].sum()),
        "removed_abb3_structure_overlap": int(work["exclude_abb3_structure"].sum()),
        "removed_boltz_date": int(work["exclude_boltz_date"].sum()),
        "removed_any": int(work["exclude_any"].sum()),
        "output_rows": int(len(filtered)),
        "removed_only_abb4": int(
            (
                work["exclude_abb4_cdr"]
                & ~work["exclude_abb3_structure"]
                & ~work["exclude_boltz_date"]
            ).sum()
        ),
        "removed_only_abb3": int(
            (
                ~work["exclude_abb4_cdr"]
                & work["exclude_abb3_structure"]
                & ~work["exclude_boltz_date"]
            ).sum()
        ),
        "removed_only_boltz": int(
            (
                ~work["exclude_abb4_cdr"]
                & ~work["exclude_abb3_structure"]
                & work["exclude_boltz_date"]
            ).sum()
        ),
        "missing_pdbdepo": int(work["PDBdepo"].isna().sum()),
        "boltz_cutoff": cutoff.strftime("%Y-%m-%d"),
    }
    return filtered, excluded, summary


def validate_abb3_conversion() -> None:
    mismatches = []
    for abb3_id, expected in ABB3_SPOT_CHECKS.items():
        actual = abb3_structure_to_pdb_name(abb3_id)
        if actual != expected:
            mismatches.append((abb3_id, expected, actual))
    if mismatches:
        raise ValueError(f"ABB3 conversion spot-checks failed: {mismatches}")
    print("ABB3 conversion spot-checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-meta", type=Path, default=DEFAULT_TEST_META)
    parser.add_argument("--output-meta", type=Path, default=DEFAULT_OUTPUT_META)
    parser.add_argument("--abb4-train", type=Path, default=DEFAULT_ABB4_TRAIN)
    parser.add_argument("--abb3-split", type=Path, default=DEFAULT_ABB3_SPLIT)
    parser.add_argument("--depo-meta", type=Path, default=DEFAULT_DEPO_META)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    parser.add_argument(
        "--boltz-cutoff",
        type=str,
        default=BOLTZ_CUTOFF.strftime("%Y-%m-%d"),
        help="Exclude rows with PDBdepo on or before this date (YYYY-MM-DD).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.boltz_cutoff)

    args.benchmark_dir.mkdir(parents=True, exist_ok=True)
    (args.benchmark_dir / "logs").mkdir(parents=True, exist_ok=True)

    validate_abb3_conversion()

    test_df = pd.read_csv(args.test_meta)
    abb4_train_cdrs = load_abb4_train_cdrs(args.abb4_train)
    abb3_train_pdb_names, abb3_mapping = load_abb3_train_pdb_names(args.abb3_split)
    pdb_depo = load_pdb_deposition_dates(args.depo_meta)

    filtered, excluded, summary = apply_filters(
        test_df,
        abb4_train_cdrs,
        abb3_train_pdb_names,
        pdb_depo,
        cutoff,
    )

    summary["abb4_train_cdr_count"] = len(abb4_train_cdrs)
    summary["abb3_train_structure_count"] = int(len(abb3_mapping))
    summary["abb3_train_pdb_name_count"] = len(abb3_train_pdb_names)

    filtered.to_csv(args.output_meta, index=False)

    excluded_out = excluded[
        [
            "pdb_name",
            "concat_CDR",
            "PDBdepo",
            "exclude_abb4_cdr",
            "exclude_abb3_structure",
            "exclude_boltz_date",
        ]
    ].sort_values("pdb_name")
    excluded_out.to_csv(args.benchmark_dir / "excluded_rows.csv", index=False)

    summary_path = args.benchmark_dir / "filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Benchmarking test-set filter summary")
    print("-" * 40)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("-" * 40)
    print(f"Wrote filtered metadata to {args.output_meta}")
    print(f"Wrote exclusion audit to {args.benchmark_dir / 'excluded_rows.csv'}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
