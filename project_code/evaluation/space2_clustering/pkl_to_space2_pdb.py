#!/usr/bin/env python3
"""Export ABB4 processed PKL files to IMGT-numbered H/L PDBs for SPACE2."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from abb4.data import protein, residue_constants

CA_IDX = residue_constants.atom_order["CA"]
LIGHT_CHAIN_OFFSET = 1000
IMGT_MIN = 1
IMGT_MAX = 128


def sanitize_pdb_filename(pdb_name: str) -> str:
    """Convert metadata pdb_name to a filesystem-safe PDB filename."""
    # e.g. pdb_00001kc5:H::L -> pdb_00001kc5__H__L.pdb
    parts = pdb_name.split(":")
    if len(parts) >= 4 and parts[2] == "":
        prefix, heavy, light = parts[0], parts[1], parts[3]
        return f"{prefix}__{heavy}__{light}.pdb"
    return pdb_name.replace(":", "_") + ".pdb"


def pdb_filename_to_pdb_name(filename: str) -> str:
    """Convert sanitized PDB filename back to metadata pdb_name."""
    stem = Path(filename).stem
    if "__" in stem:
        parts = stem.split("__")
        if len(parts) == 3:
            return f"{parts[0]}:{parts[1]}::{parts[2]}"
    return stem


def load_pkl(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def export_pkl_to_space2_pdb(feats: dict, out_path: Path) -> tuple[int, int]:
    """Write one PKL to a SPACE2-compatible PDB. Returns (n_heavy, n_light) residue counts."""
    imgt = np.asarray(feats["residue_index"], dtype=int)
    atom_pos = np.asarray(feats["atom_positions"], dtype=float)
    atom_mask = np.asarray(feats["atom_mask"], dtype=float)
    aatype = np.asarray(feats["aatype"], dtype=int)

    if "bb_mask" in feats:
        modeled = np.asarray(feats["bb_mask"], dtype=bool)
    else:
        modeled = atom_mask[:, CA_IDX] > 0.5

    ca_present = atom_mask[:, CA_IDX] > 0.5
    keep = np.where(modeled & ca_present)[0]
    if keep.size == 0:
        raise ValueError("No modeled CA atoms found")

    sel_imgt = imgt[keep]
    chain_index = np.where(sel_imgt < LIGHT_CHAIN_OFFSET, 0, 1)
    pdb_residue_index = np.where(
        sel_imgt < LIGHT_CHAIN_OFFSET, sel_imgt, sel_imgt - LIGHT_CHAIN_OFFSET
    )

    valid = (pdb_residue_index >= IMGT_MIN) & (pdb_residue_index <= IMGT_MAX)
    if not np.any(valid):
        raise ValueError("No residues in IMGT range 1-128")

    chain_index = chain_index[valid]
    if not (0 in chain_index and 1 in chain_index):
        raise ValueError("Structure must contain both heavy and light chain CA atoms")

    prot = protein.Protein(
        atom_positions=atom_pos[keep][valid],
        atom_mask=atom_mask[keep][valid],
        aatype=aatype[keep][valid],
        residue_index=pdb_residue_index[valid].astype(int),
        chain_index=chain_index.astype(int),
        b_factors=np.zeros((int(valid.sum()), atom_mask.shape[1]), dtype=float),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        handle.write(protein.to_pdb(prot))

    n_heavy = int(np.sum(chain_index == 0))
    n_light = int(np.sum(chain_index == 1))
    return n_heavy, n_light


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Keep two-chain Fv entries suitable for SPACE2."""
    out = df.copy()
    out = out[out["num_chains"] == 2]
    out = out[out["Lchain"].notna() & (out["Lchain"].astype(str).str.strip() != "")]
    if "VL_seq" in out.columns:
        out = out[out["VL_seq"].notna() & (out["VL_seq"].astype(str).str.strip() != "")]
    return out.reset_index(drop=True)


def process_metadata(
    metadata_path: Path,
    pdb_dir: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    df = filter_metadata(df)

    records: list[dict] = []
    for _, row in df.iterrows():
        pdb_name = row["pdb_name"]
        processed_path = Path(row["processed_path"])
        pdb_filename = sanitize_pdb_filename(pdb_name)
        pdb_path = pdb_dir / pdb_filename

        record = {
            "pdb_name": pdb_name,
            "pdb_filename": pdb_filename,
            "pdb_path": str(pdb_path.resolve()),
            "processed_path": str(processed_path),
            "status": "pending",
            "n_res_H": np.nan,
            "n_res_L": np.nan,
            "error": "",
        }

        if not processed_path.is_file():
            record["status"] = "skipped"
            record["error"] = "processed_path missing"
            records.append(record)
            continue

        try:
            feats = load_pkl(processed_path)
            n_h, n_l = export_pkl_to_space2_pdb(feats, pdb_path)
            record["status"] = "ok"
            record["n_res_H"] = n_h
            record["n_res_L"] = n_l
        except Exception as exc:  # noqa: BLE001 - collect per-row failures
            record["status"] = "failed"
            record["error"] = str(exc)

        records.append(record)

    manifest = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    n_ok = int((manifest["status"] == "ok").sum())
    n_fail = int((manifest["status"] == "failed").sum())
    n_skip = int((manifest["status"] == "skipped").sum())
    print(
        f"Export complete: {n_ok} ok, {n_fail} failed, {n_skip} skipped "
        f"(manifest: {manifest_path})"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Input metadata CSV (e.g. baby_test_meta.csv or metadata.csv)",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        required=True,
        help="Output directory for SPACE2-ready PDB files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest CSV (default: <pdb-dir>/../export_manifest.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.pdb_dir.parent / "export_manifest.csv"
    process_metadata(args.metadata, args.pdb_dir, manifest_path)


if __name__ == "__main__":
    main()
