#!/usr/bin/env python3
"""Compute mean pairwise intra-cluster RMSDs for ground truth (pkl) and predictions (PDB)."""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import Superimposer
from Bio.PDB.Atom import Atom
from joblib import Parallel, delayed
from tqdm import tqdm

# Reuse metric definitions from the existing evaluation script.
ABB4_ROOT = Path(__file__).resolve().parents[3] / "ABB4"
if str(ABB4_ROOT) not in sys.path:
    sys.path.insert(0, str(ABB4_ROOT))

from abb4.analysis.postproc.calculate_struc_pred_metrics import (  # noqa: E402
    METRIC_KEYS,
    PDB_PARSER,
    chain_rmsds,
)

from pair_length_filters import (  # noqa: E402
    SAME_LENGTH_CDR_METRICS,
    build_length_lookup,
    filter_ok_pairs_for_metric,
    load_meta_for_lengths,
)

N_IDX, CA_IDX, C_IDX = 0, 1, 2
BACKBONE_ATOM_IDXS = (N_IDX, CA_IDX, C_IDX)
CHAIN_TERM = 1000

IMGT_CDR_RANGES = {
    "cdr1": (27, 38),
    "cdr2": (56, 65),
    "cdr3": (105, 117),
}


def imgt_region_from_number(imgt_num: int) -> str:
    for region, (start, end) in IMGT_CDR_RANGES.items():
        if start <= imgt_num <= end:
            return region
    return "framework"


def imgt_number(residue_index: int) -> int:
    return int(residue_index) if residue_index < CHAIN_TERM else int(residue_index) - CHAIN_TERM


def apply_rotran(coords: np.ndarray, rotran) -> np.ndarray:
    rot, tran = rotran
    return np.dot(coords, rot) + tran


def load_pkl_chains(pkl_path: str) -> dict[str, dict]:
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle)

    modeled = data["modeled_idx"]
    residue_index = data["residue_index"][modeled].astype(int)
    positions = np.asarray(data["atom_positions"][modeled], dtype=float)
    atom_mask = np.asarray(data["atom_mask"][modeled], dtype=float) > 0

    chains = {}
    for label, chain_sel in (("H", residue_index < CHAIN_TERM), ("L", residue_index >= CHAIN_TERM)):
        if not np.any(chain_sel):
            continue
        chains[label] = {
            "residue_index": residue_index[chain_sel],
            "positions": positions[chain_sel],
            "mask": atom_mask[chain_sel],
        }
    return chains


def _backbone_points(positions: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if not all(mask[idx] for idx in BACKBONE_ATOM_IDXS):
        return None
    return positions[list(BACKBONE_ATOM_IDXS), :]


def _region_rmsd(ref_points: list[np.ndarray], mob_points: list[np.ndarray]) -> float:
    if not ref_points:
        return np.nan
    ref_xyz = np.concatenate(ref_points, axis=0)
    mob_xyz = np.concatenate(mob_points, axis=0)
    return float(np.sqrt(np.mean(np.sum((ref_xyz - mob_xyz) ** 2, axis=1))))


def chain_rmsds_from_pkl(ref_chain: dict, mob_chain: dict, chain_label: str) -> dict[str, float]:
    ref_map = {int(r): i for i, r in enumerate(ref_chain["residue_index"])}
    mob_map = {int(r): i for i, r in enumerate(mob_chain["residue_index"])}
    common_ids = sorted(set(ref_map) & set(mob_map))

    region_ids = {region: [] for region in ["framework", "cdr1", "cdr2", "cdr3"]}
    for residue_id in common_ids:
        region_ids[imgt_region_from_number(imgt_number(residue_id))].append(residue_id)

    fit_ref = []
    fit_mob = []
    for residue_id in region_ids["framework"]:
        ref_bb = _backbone_points(
            ref_chain["positions"][ref_map[residue_id]],
            ref_chain["mask"][ref_map[residue_id]],
        )
        mob_bb = _backbone_points(
            mob_chain["positions"][mob_map[residue_id]],
            mob_chain["mask"][mob_map[residue_id]],
        )
        if ref_bb is None or mob_bb is None:
            continue
        fit_ref.append(ref_bb)
        fit_mob.append(mob_bb)

    if len(fit_ref) < 1:
        raise ValueError(f"Not enough framework backbone atoms for chain {chain_label}")

    fit_ref_xyz = np.concatenate(fit_ref, axis=0)
    fit_mob_xyz = np.concatenate(fit_mob, axis=0)
    if fit_ref_xyz.shape[0] < 3:
        raise ValueError(f"Not enough framework backbone atoms for chain {chain_label}")

    fixed_atoms = [
        Atom("X", coord, 0, 0, " ", " X ", i, "C") for i, coord in enumerate(fit_ref_xyz)
    ]
    mobile_atoms = [
        Atom("X", coord, 0, 0, " ", " X ", i, "C") for i, coord in enumerate(fit_mob_xyz)
    ]
    imposer = Superimposer()
    imposer.set_atoms(fixed_atoms, mobile_atoms)
    imposer.apply(mobile_atoms)
    rotran = imposer.rotran

    mob_positions_aligned = mob_chain["positions"].copy()
    for residue_id in common_ids:
        mob_positions_aligned[mob_map[residue_id]] = apply_rotran(
            mob_chain["positions"][mob_map[residue_id]], rotran
        )

    def collect_region_points(region_key: str):
        ref_points = []
        mob_points = []
        for residue_id in region_ids[region_key]:
            ref_bb = _backbone_points(
                ref_chain["positions"][ref_map[residue_id]],
                ref_chain["mask"][ref_map[residue_id]],
            )
            mob_bb = _backbone_points(
                mob_positions_aligned[mob_map[residue_id]],
                mob_chain["mask"][mob_map[residue_id]],
            )
            if ref_bb is None or mob_bb is None:
                continue
            ref_points.append(ref_bb)
            mob_points.append(mob_bb)
        return ref_points, mob_points

    cdr_all_ids = region_ids["cdr1"] + region_ids["cdr2"] + region_ids["cdr3"]
    metrics = {}
    for region_key in ["framework", "cdr1", "cdr2", "cdr3"]:
        ref_points, mob_points = collect_region_points(region_key)
        metrics[f"{chain_label}_{region_key}"] = _region_rmsd(ref_points, mob_points)

    ref_all = []
    mob_all = []
    for residue_id in cdr_all_ids:
        ref_bb = _backbone_points(
            ref_chain["positions"][ref_map[residue_id]],
            ref_chain["mask"][ref_map[residue_id]],
        )
        mob_bb = _backbone_points(
            mob_positions_aligned[mob_map[residue_id]],
            mob_chain["mask"][mob_map[residue_id]],
        )
        if ref_bb is None or mob_bb is None:
            continue
        ref_all.append(ref_bb)
        mob_all.append(mob_bb)
    metrics[f"{chain_label}_cdr_all"] = _region_rmsd(ref_all, mob_all)
    return metrics


def compute_pairwise_rmsds_from_pkl(ref_pkl: str, mob_pkl: str) -> dict[str, float]:
    ref_chains = load_pkl_chains(ref_pkl)
    mob_chains = load_pkl_chains(mob_pkl)
    metrics: dict[str, float] = {}

    if "H" in ref_chains and "H" in mob_chains:
        metrics.update(chain_rmsds_from_pkl(ref_chains["H"], mob_chains["H"], "H"))
    if "L" in ref_chains and "L" in mob_chains:
        metrics.update(chain_rmsds_from_pkl(ref_chains["L"], mob_chains["L"], "L"))
    if not metrics:
        raise ValueError("No common chains found between pkl structures")
    return metrics


def compute_pairwise_rmsds_from_pdb(ref_pdb: str, mob_pdb: str) -> dict[str, float]:
    ref_model = next(PDB_PARSER.get_structure("ref", ref_pdb).get_models())
    mob_model = next(PDB_PARSER.get_structure("mob", mob_pdb).get_models())
    metrics: dict[str, float] = {}
    metrics.update(chain_rmsds(ref_model["H"], mob_model["H"], "H"))
    if "L" in ref_model and "L" in mob_model:
        metrics.update(chain_rmsds(ref_model["L"], mob_model["L"], "L"))
    return metrics


def load_metadata(
    meta_path: str, summary_path: str | None, cluster_col: str, mode: str
) -> pd.DataFrame:
    meta = pd.read_csv(meta_path, index_col=0)
    if summary_path is not None and mode == "prediction":
        summary = pd.read_csv(summary_path)
        ok_names = summary.loc[summary["status"] == "ok", "pdb_name"].astype(str)
        meta = meta[meta["pdb_name"].isin(ok_names)].copy()
    if cluster_col not in meta.columns:
        raise ValueError(f"Column {cluster_col} not found in metadata")
    meta = meta.dropna(subset=[cluster_col, "pdb_name"]).copy()
    return meta


def compute_pair_metrics(mode: str, ref_row: pd.Series, mob_row: pd.Series) -> dict:
    try:
        if mode == "ground_truth":
            metrics = compute_pairwise_rmsds_from_pkl(ref_row.structure_path, mob_row.structure_path)
        else:
            metrics = compute_pairwise_rmsds_from_pdb(ref_row.structure_path, mob_row.structure_path)
        for key in METRIC_KEYS:
            metrics.setdefault(key, np.nan)
        metrics["status"] = "ok"
        metrics["error"] = ""
    except Exception as exc:
        metrics = {key: np.nan for key in METRIC_KEYS}
        metrics["status"] = "error"
        metrics["error"] = str(exc)
    metrics.update(
        {
            "cluster_ids": ref_row.cluster_ids,
            "pdb_name_a": ref_row.pdb_name,
            "pdb_name_b": mob_row.pdb_name,
        }
    )
    return metrics


def summarize_cluster(
    pair_df: pd.DataFrame,
    cluster_id,
    n_members: int,
    length_lookup: dict | None = None,
) -> dict:
    ok_pairs = pair_df[pair_df["status"] == "ok"]
    summary = {
        "cluster_ids": cluster_id,
        "n_members": n_members,
        "n_pairs_expected": n_members * (n_members - 1) // 2,
        "n_pairs_computed": len(ok_pairs),
    }
    for key in METRIC_KEYS:
        if length_lookup is not None and key in SAME_LENGTH_CDR_METRICS:
            subset = filter_ok_pairs_for_metric(pair_df, key, length_lookup)
            summary[f"n_pairs_computed_{key}"] = len(subset)
        else:
            subset = ok_pairs
        if key not in subset.columns:
            values = pd.Series(dtype=float)
        else:
            values = subset[key].dropna()
        summary[f"mean_pairwise_{key}"] = float(values.mean()) if len(values) else np.nan
        summary[f"std_pairwise_{key}"] = float(values.std(ddof=0)) if len(values) else np.nan
    return summary


def process_cluster(cluster_id, cluster_members: pd.DataFrame, mode: str, length_lookup: dict | None = None):
    if len(cluster_members) < 2:
        return [], None

    pair_rows = []
    indexed = cluster_members.reset_index(drop=True)
    for i, j in itertools.combinations(range(len(indexed)), 2):
        pair_rows.append(compute_pair_metrics(mode, indexed.iloc[i], indexed.iloc[j]))

    pair_df = pd.DataFrame(pair_rows)
    summary = summarize_cluster(pair_df, cluster_id, len(cluster_members), length_lookup)
    return pair_rows, summary


def prepare_members(meta: pd.DataFrame, mode: str, pred_path: str | None, summary_path: str | None, cluster_col: str):
    rows = []
    summary = None
    skip_not_in_summary = 0
    skip_missing_file = 0
    if mode == "prediction":
        summary = pd.read_csv(summary_path).set_index("pdb_name")

    for record in meta.itertuples(index=False):
        cluster_id = getattr(record, cluster_col)
        if mode == "ground_truth":
            structure_path = record.processed_path
        else:
            if summary is None or record.pdb_name not in summary.index:
                skip_not_in_summary += 1
                continue
            rep = summary.loc[record.pdb_name, "representative_sample"]
            structure_path = os.path.join(pred_path, record.pdb_name, rep)

        if not os.path.isfile(structure_path):
            skip_missing_file += 1
            continue
        rows.append(
            {
                "pdb_name": record.pdb_name,
                "cluster_ids": cluster_id,
                "structure_path": structure_path,
            }
        )
    return pd.DataFrame(rows)


def run(
    mode: str,
    meta_path: str,
    output_dir: str,
    cluster_col: str,
    pred_path: str | None,
    summary_path: str | None,
    save_pairs: bool,
    n_jobs: int,
    same_length_only: bool,
    length_meta_path: str | None,
):
    meta = load_metadata(meta_path, summary_path, cluster_col, mode)
    members = prepare_members(meta, mode, pred_path, summary_path, cluster_col)
    if members.empty:
        raise RuntimeError(
            "No valid structures found for analysis. "
            f"After metadata filters: {len(meta)} rows; prepared members: 0. "
            "For prediction mode, --meta_path must list the same pdb_name set as "
            "--summary_path (e.g. test_meta.csv with struc_pred_metrics_test_summary.csv)."
        )

    length_lookup = None
    if same_length_only:
        lmeta_path = length_meta_path or meta_path
        length_lookup = build_length_lookup(load_meta_for_lengths(lmeta_path))
        print(f"Same-length CDR filtering enabled (lengths from {lmeta_path})")

    cluster_groups = {
        cluster_id: group for cluster_id, group in members.groupby("cluster_ids") if len(group) >= 2
    }
    print(
        f"Mode={mode}: {len(members)} structures across {members['cluster_ids'].nunique()} clusters; "
        f"{len(cluster_groups)} clusters with >=2 members"
    )

    cluster_ids = list(cluster_groups.keys())
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_cluster)(cluster_id, cluster_groups[cluster_id], mode, length_lookup)
        for cluster_id in tqdm(cluster_ids, desc=f"Clusters ({mode})")
    )

    pair_rows = []
    summaries = []
    for pair_chunk, summary in results:
        pair_rows.extend(pair_chunk)
        if summary is not None:
            summaries.append(summary)

    os.makedirs(output_dir, exist_ok=True)
    prefix = "cluster_gt" if mode == "ground_truth" else "cluster_pred"
    suffix = "_same_len" if same_length_only else ""
    summary_df = pd.DataFrame(summaries).sort_values("cluster_ids")
    summary_path_out = os.path.join(output_dir, f"{prefix}_pairwise_rmsd{suffix}.csv")
    summary_df.to_csv(summary_path_out, index=False)
    print(f"Saved cluster summaries to {summary_path_out} ({len(summary_df)} clusters)")

    if save_pairs:
        pairs_df = pd.DataFrame(pair_rows)
        pairs_path_out = os.path.join(output_dir, f"{prefix}_pairwise_pairs.csv")
        pairs_df.to_csv(pairs_path_out, index=False)
        print(f"Saved pair details to {pairs_path_out} ({len(pairs_df)} pairs)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute mean pairwise intra-cluster RMSDs for GT pkls or predicted PDBs."
    )
    parser.add_argument("--mode", choices=["ground_truth", "prediction"], required=True)
    parser.add_argument(
        "--meta_path",
        default="/opig-shared/users/lina4783/structures_final/test_meta.csv",
    )
    parser.add_argument("--pred_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--cluster_col", default="cluster_ids")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_pairs", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=10)
    parser.add_argument(
        "--same_length_only",
        action="store_true",
        help="Mean/std per CDR metric uses only pairs with equal CDR sequence length for that metric.",
    )
    parser.add_argument(
        "--length_meta_path",
        default=None,
        help="CSV with CDRH/CDRL columns for length lookup (default: --meta_path).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "prediction":
        if not args.pred_path or not args.summary_path:
            raise ValueError("prediction mode requires --pred_path and --summary_path")

    run(
        mode=args.mode,
        meta_path=args.meta_path,
        output_dir=args.output_dir,
        cluster_col=args.cluster_col,
        pred_path=args.pred_path,
        summary_path=args.summary_path,
        save_pairs=args.save_pairs,
        n_jobs=args.n_jobs,
        same_length_only=args.same_length_only,
        length_meta_path=args.length_meta_path,
    )


if __name__ == "__main__":
    main()
