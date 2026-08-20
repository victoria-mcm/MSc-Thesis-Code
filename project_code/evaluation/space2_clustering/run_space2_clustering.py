#!/usr/bin/env python3
"""Run SPACE2 agglomerative clustering on exported PDB files."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import SPACE2


def load_pdb_paths(manifest_path: Path | None, pdb_dir: Path | None) -> list[str]:
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path)
        ok = manifest.loc[manifest["status"] == "ok", "pdb_path"].dropna().astype(str)
        paths = [str(Path(p)) for p in ok if Path(p).is_file()]
        if paths:
            return paths
        raise ValueError(f"No successful PDB exports found in manifest: {manifest_path}")

    if pdb_dir is None:
        raise ValueError("Provide --manifest or --pdb-dir")

    paths = sorted(str(p) for p in pdb_dir.glob("*.pdb"))
    if not paths:
        raise ValueError(f"No PDB files found in {pdb_dir}")
    return paths


def run_clustering(
    pdb_paths: list[str],
    cutoff: float,
    n_jobs: int,
) -> pd.DataFrame:
    print(f"Clustering {len(pdb_paths)} structures (cutoff={cutoff} A, n_jobs={n_jobs})")
    t0 = time.time()
    clustered = SPACE2.agglomerative_clustering(
        pdb_paths,
        cutoff=cutoff,
        d_metric="rmsd",
        n_jobs=n_jobs,
    )
    elapsed = time.time() - t0
    print(f"Clustering finished in {elapsed:.1f}s")
    return clustered


def write_summary(
    clustered: pd.DataFrame,
    summary_path: Path,
    pdb_paths: list[str],
    cutoff: float,
    n_jobs: int,
    elapsed_sec: float,
) -> None:
    summary = {
        "n_structures": len(pdb_paths),
        "n_length_groups": clustered["cluster_by_length"].nunique(),
        "n_rmsd_clusters": clustered["cluster_by_rmsd"].nunique(),
        "cutoff_angstrom": cutoff,
        "d_metric": "rmsd",
        "n_jobs": n_jobs,
        "elapsed_sec": elapsed_sec,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="export_manifest.csv from pkl_to_space2_pdb.py",
    )
    source.add_argument(
        "--pdb-dir",
        type=Path,
        help="Directory containing exported PDB files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV for SPACE2 cluster assignments",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path (default: output dir / space2_cluster_summary.json)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.25,
        help="Agglomerative clustering distance threshold in Angstroms (default: 1.25)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for SPACE2 (-1 = all CPUs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdb_paths = load_pdb_paths(args.manifest, args.pdb_dir)

    t0 = time.time()
    clustered = run_clustering(pdb_paths, cutoff=args.cutoff, n_jobs=args.n_jobs)
    elapsed = time.time() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clustered.to_csv(args.output, index=False)
    print(f"Clusters written to {args.output}")

    summary_path = args.summary
    if summary_path is None:
        summary_path = args.output.parent / "space2_cluster_summary.json"
    write_summary(
        clustered,
        summary_path,
        pdb_paths,
        args.cutoff,
        args.n_jobs,
        elapsed,
    )


if __name__ == "__main__":
    main()
