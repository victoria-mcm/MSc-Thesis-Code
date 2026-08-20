#!/usr/bin/env python3
"""Subsampled global pairwise H_cdr3 RMSDs (upper-bound reference for cluster diversity plots)."""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_cluster_pairwise_rmsd import compute_pairwise_rmsds_from_pkl  # noqa: E402


def load_members(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, index_col=0)
    rows = []
    for record in meta.itertuples(index=False):
        path = record.processed_path
        if not isinstance(path, str) or not os.path.isfile(path):
            continue
        rows.append({"pdb_name": record.pdb_name, "structure_path": path})
    members = pd.DataFrame(rows)
    if members.empty:
        raise RuntimeError("No structures with valid processed_path found")
    return members.drop_duplicates(subset=["pdb_name"]).reset_index(drop=True)


def compute_pair_row(path_a: str, path_b: str, name_a: str, name_b: str, replicate: int) -> dict:
    row = {
        "replicate": replicate,
        "pdb_name_a": name_a,
        "pdb_name_b": name_b,
        "H_cdr3": np.nan,
        "status": "error",
        "error": "",
    }
    try:
        metrics = compute_pairwise_rmsds_from_pkl(path_a, path_b)
        row["H_cdr3"] = metrics.get("H_cdr3", np.nan)
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = str(exc)
    return row


def run_replicate(
    replicate: int,
    members: pd.DataFrame,
    n_samples: int,
    base_seed: int,
) -> list[dict]:
    rng = np.random.default_rng(base_seed + replicate)
    if n_samples > len(members):
        raise ValueError(f"n_samples={n_samples} exceeds available structures ({len(members)})")
    sample_idx = rng.choice(len(members), size=n_samples, replace=False)
    sample = members.iloc[sample_idx].reset_index(drop=True)

    pair_rows = []
    for i, j in itertools.combinations(range(len(sample)), 2):
        a = sample.iloc[i]
        b = sample.iloc[j]
        pair_rows.append(
            compute_pair_row(
                a.structure_path,
                b.structure_path,
                a.pdb_name,
                b.pdb_name,
                replicate,
            )
        )
    return pair_rows


def run(
    meta_path: str,
    output_dir: str,
    n_samples: int,
    n_replicates: int,
    base_seed: int,
    n_jobs: int,
) -> None:
    members = load_members(meta_path)
    print(f"Universe: {len(members)} structures with valid pkls")
    print(f"Plan: {n_replicates} replicates × {n_samples} samples × {n_samples * (n_samples - 1) // 2} pairs")

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_replicate)(rep, members, n_samples, base_seed)
        for rep in tqdm(range(n_replicates), desc="Replicates")
    )

    rows = [row for chunk in results for row in chunk]
    out_df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "global_subsample_H_cdr3_pairs.csv")
    out_df.to_csv(out_path, index=False)
    ok = out_df[out_df["status"] == "ok"]
    print(f"Saved {out_path} ({len(out_df)} rows, {len(ok)} ok, median H_cdr3={ok['H_cdr3'].median():.3f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Global subsampled pairwise H_cdr3 for upper-bound reference.")
    parser.add_argument(
        "--meta_path",
        default="/opig-shared/users/lina4783/structures_final/metadata_no_nano.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="/opig-shared/users/lina4783/abb4_experiments/evaluation/cluster_intra_rmsd/outputs/bounds",
    )
    parser.add_argument("--n_samples", type=int, default=80)
    parser.add_argument("--n_replicates", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        n_replicates=args.n_replicates,
        base_seed=args.base_seed,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
