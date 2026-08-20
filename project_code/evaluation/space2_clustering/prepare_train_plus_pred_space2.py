#!/usr/bin/env python3
"""Assemble SPACE2 inputs: train ground-truth PDBs + representative predicted test PDBs.

Train PDBs are reused from the existing full PKL export (already IMGT H/L).
Test PDBs are the representative sample from a prediction directory (already IMGT H/L).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

BASE = Path("/opig-shared/users/lina4783")
DEFAULT_PRED_PATH = (
    BASE / "abb4_experiments/evaluation/predictions_ckpt_5139_imgt"
)
DEFAULT_TRAIN_MANIFEST = (
    BASE
    / "abb4_experiments/evaluation/space2_clustering/outputs/full/export_manifest.csv"
)
DEFAULT_METADATA = BASE / "structures_final/metadata.csv"
DEFAULT_TEST_META = BASE / "structures_final/test_meta.csv"
DEFAULT_OUT_DIR = (
    BASE
    / "abb4_experiments/evaluation/space2_clustering/outputs/train_plus_pred_test"
)


def sanitize_pdb_filename(pdb_name: str) -> str:
    """Convert metadata pdb_name to a filesystem-safe PDB filename."""
    parts = pdb_name.split(":")
    if len(parts) >= 4 and parts[2] == "":
        prefix, heavy, light = parts[0], parts[1], parts[3]
        return f"{prefix}__{heavy}__{light}.pdb"
    return pdb_name.replace(":", "_") + ".pdb"


def _link_or_copy(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dest)
        return "copy"


def assemble(
    pred_path: Path,
    metrics_summary: Path,
    test_meta_path: Path,
    train_manifest_path: Path,
    metadata_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    pdb_dir = out_dir / "pdbs"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path, usecols=["pdb_name", "split"])
    train_names = set(metadata.loc[metadata["split"].eq("train"), "pdb_name"].astype(str))

    train_manifest = pd.read_csv(train_manifest_path)
    train_ok = train_manifest.loc[
        train_manifest["status"].eq("ok")
        & train_manifest["pdb_name"].astype(str).isin(train_names)
    ].copy()

    metrics = pd.read_csv(
        metrics_summary,
        usecols=["pdb_name", "representative_sample", "status"],
    )
    metrics = metrics.loc[metrics["status"].eq("ok")].copy()
    test_meta = pd.read_csv(test_meta_path, usecols=["pdb_name"])
    test_names = set(test_meta["pdb_name"].astype(str))
    metrics = metrics.loc[metrics["pdb_name"].astype(str).isin(test_names)].copy()

    records: list[dict] = []

    for _, row in train_ok.iterrows():
        src = Path(str(row["pdb_path"]))
        pdb_filename = str(row["pdb_filename"])
        dest = pdb_dir / pdb_filename
        record = {
            "pdb_name": row["pdb_name"],
            "pdb_filename": pdb_filename,
            "pdb_path": str(dest),
            "source": "train_gt",
            "representative_sample": "",
            "src_path": str(src),
            "status": "pending",
            "error": "",
        }
        if not src.is_file():
            record["status"] = "skipped"
            record["error"] = "train PDB missing"
            records.append(record)
            continue
        try:
            _link_or_copy(src, dest)
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
        records.append(record)

    for _, row in metrics.iterrows():
        pdb_name = str(row["pdb_name"])
        sample = str(row["representative_sample"])
        src = pred_path / pdb_name / sample
        pdb_filename = sanitize_pdb_filename(pdb_name)
        dest = pdb_dir / pdb_filename
        record = {
            "pdb_name": pdb_name,
            "pdb_filename": pdb_filename,
            "pdb_path": str(dest),
            "source": "test_pred",
            "representative_sample": sample,
            "src_path": str(src),
            "status": "pending",
            "error": "",
        }
        if not src.is_file():
            record["status"] = "skipped"
            record["error"] = f"predicted PDB missing: {src}"
            records.append(record)
            continue
        try:
            _link_or_copy(src, dest)
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
        records.append(record)

    manifest = pd.DataFrame(records)
    manifest_path = out_dir / "export_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    n_ok = int(manifest["status"].eq("ok").sum())
    n_train = int(((manifest["source"] == "train_gt") & (manifest["status"] == "ok")).sum())
    n_pred = int(((manifest["source"] == "test_pred") & (manifest["status"] == "ok")).sum())
    n_fail = int(manifest["status"].eq("failed").sum())
    n_skip = int(manifest["status"].eq("skipped").sum())
    print(
        f"Assembled {n_ok} PDBs ({n_train} train GT, {n_pred} predicted test); "
        f"{n_fail} failed, {n_skip} skipped -> {manifest_path}"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pred-path",
        type=Path,
        default=DEFAULT_PRED_PATH,
        help="Root directory of IMGT-numbered predictions (one subdir per pdb_name)",
    )
    parser.add_argument(
        "--metrics-summary",
        type=Path,
        default=None,
        help="struc_pred_metrics_test_summary.csv (default: <pred-path>/struc_pred_metrics_test_summary.csv)",
    )
    parser.add_argument("--test-meta", type=Path, default=DEFAULT_TEST_META)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_summary = args.metrics_summary
    if metrics_summary is None:
        metrics_summary = args.pred_path / "struc_pred_metrics_test_summary.csv"
    assemble(
        pred_path=args.pred_path,
        metrics_summary=metrics_summary,
        test_meta_path=args.test_meta,
        train_manifest_path=args.train_manifest,
        metadata_path=args.metadata,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
