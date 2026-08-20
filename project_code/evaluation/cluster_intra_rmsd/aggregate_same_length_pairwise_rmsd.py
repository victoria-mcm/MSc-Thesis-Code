#!/usr/bin/env python3
"""Get pairwise RMSDs for same-length CDR pairs in clusters"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ABB4_ROOT = SCRIPT_DIR.parents[2] / "ABB4"
if str(ABB4_ROOT) not in sys.path:
    sys.path.insert(0, str(ABB4_ROOT))

from abb4.analysis.postproc.calculate_struc_pred_metrics import METRIC_KEYS  # noqa: E402

from pair_length_filters import (  # noqa: E402
    SAME_LENGTH_CDR_METRICS,
    build_length_lookup,
    filter_ok_pairs_for_metric,
    load_meta_for_lengths,
)


def aggregate(
    pairs_path: str,
    meta_path: str,
    output_path: str,
    cluster_rmsd_path: str | None,
) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_path, low_memory=False)
    meta = load_meta_for_lengths(meta_path)
    length_lookup = build_length_lookup(meta)

    if cluster_rmsd_path and os.path.isfile(cluster_rmsd_path):
        n_members = pd.read_csv(cluster_rmsd_path)[["cluster_ids", "n_members"]]
    else:
        n_members = pairs.groupby("cluster_ids")["pdb_name_a"].nunique().reset_index()
        n_members = n_members.rename(columns={"pdb_name_a": "n_members"})

    ok_all = pairs[pairs["status"] == "ok"]
    print(f"Loaded {len(pairs):,} pair rows ({len(ok_all):,} ok) from {pairs_path}")

    rows = []
    for cluster_id, cluster_pairs in pairs.groupby("cluster_ids"):
        nm = n_members.loc[n_members["cluster_ids"] == cluster_id, "n_members"]
        if len(nm):
            n_m = int(nm.iloc[0])
        else:
            n_m = int(cluster_pairs["pdb_name_a"].nunique())
        row = {
            "cluster_ids": cluster_id,
            "n_members": n_m,
            "n_pairs_expected": n_m * (n_m - 1) // 2,
        }
        ok_cluster = cluster_pairs[cluster_pairs["status"] == "ok"]
        row["n_pairs_computed"] = len(ok_cluster)

        for key in METRIC_KEYS:
            if key in SAME_LENGTH_CDR_METRICS:
                subset = filter_ok_pairs_for_metric(cluster_pairs, key, length_lookup)
            else:
                subset = ok_cluster
            row[f"n_pairs_computed_{key}"] = len(subset)
            if key not in subset.columns:
                vals = pd.Series(dtype=float)
            else:
                vals = subset[key].dropna()
            row[f"mean_pairwise_{key}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"std_pairwise_{key}"] = float(vals.std(ddof=0)) if len(vals) else np.nan
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cluster_ids")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} clusters)")

    for metric in SAME_LENGTH_CDR_METRICS:
        n_ok = len(ok_all)
        kept = sum(
            len(filter_ok_pairs_for_metric(g, metric, length_lookup))
            for _, g in pairs.groupby("cluster_ids")
        )
        pct = 100.0 * kept / n_ok if n_ok else 0.0
        print(f"  {metric}: kept {kept:,}/{n_ok:,} ok pairs ({pct:.1f}%)")
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate cluster RMSDs from pair CSV with same-length CDR filters."
    )
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument(
        "--meta_path",
        default="/opig-shared/users/lina4783/structures_final/test_meta.csv",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--cluster_rmsd_path",
        default=None,
        help="Optional existing cluster summary for n_members (else inferred from pairs).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    aggregate(
        pairs_path=args.pairs_path,
        meta_path=args.meta_path,
        output_path=args.output_path,
        cluster_rmsd_path=args.cluster_rmsd_path,
    )


if __name__ == "__main__":
    main()
