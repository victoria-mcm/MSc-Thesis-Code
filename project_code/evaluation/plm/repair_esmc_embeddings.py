#!/usr/bin/env python3
"""Compact bloated .pt embedding files and report corrupt/missing entries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_MANIFEST = "/opig-shared/users/lina4783/abb4_experiments/plm/esmc_300m_concat_cdr.csv"
DEFAULT_MIN_BYTES = 300_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair ESM-C embedding .pt files on disk.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=DEFAULT_MIN_BYTES,
        help="Re-save .pt files at or above this size after cloning tensor storage.",
    )
    parser.add_argument(
        "--delete-corrupt",
        action="store_true",
        help="Delete unreadable .pt files so extraction can regenerate them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)

    corrupt: list[str] = []
    compacted = 0
    skipped = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Checking embeddings"):
        pdb_name = str(row["pdb_name"])
        path = Path(row["embedding_path"])

        if not path.exists():
            corrupt.append(pdb_name)
            continue

        try:
            embedding = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            corrupt.append(pdb_name)
            if args.delete_corrupt:
                path.unlink(missing_ok=True)
            continue

        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
            corrupt.append(pdb_name)
            if args.delete_corrupt:
                path.unlink(missing_ok=True)
            continue

        if path.stat().st_size >= args.min_bytes:
            compact = embedding.detach().cpu().float().clone()
            torch.save(compact, path)
            compacted += 1
        else:
            skipped += 1

    print(f"Checked {len(manifest)} embeddings", flush=True)
    print(f"Compacted {compacted} bloated files", flush=True)
    print(f"Skipped {skipped} already-compact files", flush=True)
    print(f"Corrupt/missing: {len(corrupt)}", flush=True)
    if corrupt:
        print("Examples:", ", ".join(corrupt[:10]), flush=True)
        if args.delete_corrupt:
            print("Deleted corrupt files; re-run extract_esmc_concat_cdr_embeddings.py to regenerate.", flush=True)


if __name__ == "__main__":
    main()
