#!/usr/bin/env python3
"""Merge SPACE2 cluster assignments back into metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pdb_filename_to_pdb_name(filename: str) -> str:
    """Convert sanitized PDB filename back to metadata pdb_name."""
    stem = Path(filename).stem
    if "__" in stem:
        parts = stem.split("__")
        if len(parts) == 3:
            return f"{parts[0]}:{parts[1]}::{parts[2]}"
    return stem


def cluster_id_to_pdb_name(cluster_id: str, manifest: pd.DataFrame | None) -> str | None:
    """Resolve SPACE2 ID (path or filename) to metadata pdb_name."""
    cluster_id = str(cluster_id)
    if manifest is not None:
        path_lookup = dict(
            zip(manifest["pdb_path"].astype(str), manifest["pdb_name"].astype(str))
        )
        if cluster_id in path_lookup:
            return path_lookup[cluster_id]

        filename_lookup = dict(
            zip(manifest["pdb_filename"].astype(str), manifest["pdb_name"].astype(str))
        )
        basename = Path(cluster_id).name
        if basename in filename_lookup:
            return filename_lookup[basename]

    return pdb_filename_to_pdb_name(cluster_id)


def load_clusters(clusters_path: Path, manifest_path: Path | None) -> pd.DataFrame:
    clusters = pd.read_csv(clusters_path)
    required = {"ID", "cluster_by_length", "cluster_by_rmsd"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Clusters CSV missing columns: {sorted(missing)}")

    manifest = pd.read_csv(manifest_path) if manifest_path is not None else None

    clusters = clusters.copy()
    clusters["pdb_name"] = clusters["ID"].astype(str).map(
        lambda cid: cluster_id_to_pdb_name(cid, manifest)
    )
    missing_names = clusters["pdb_name"].isna().sum()
    if missing_names:
        raise ValueError(
            f"Failed to map {missing_names} SPACE2 IDs to pdb_name; "
            "provide --manifest from pkl_to_space2_pdb.py"
        )

    clusters = clusters.rename(
        columns={
            "cluster_by_length": "space2_cluster_by_length",
            "cluster_by_rmsd": "space2_cluster_by_rmsd",
        }
    )
    clusters["space2_representative"] = clusters["space2_cluster_by_rmsd"].astype(str).map(
        lambda cid: cluster_id_to_pdb_name(cid, manifest)
    )
    return clusters[
        [
            "pdb_name",
            "space2_cluster_by_length",
            "space2_cluster_by_rmsd",
            "space2_representative",
        ]
    ]


def merge_metadata(
    metadata_path: Path,
    clusters_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    clusters = load_clusters(clusters_path, manifest_path)

    merged = metadata.merge(clusters, on="pdb_name", how="left")
    n_assigned = int(merged["space2_cluster_by_rmsd"].notna().sum())
    print(
        f"Merged {n_assigned}/{len(merged)} rows with SPACE2 clusters "
        f"-> {output_path}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="export_manifest.csv from pkl_to_space2_pdb.py (recommended)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_metadata(args.metadata, args.clusters, args.output, args.manifest)


if __name__ == "__main__":
    main()
