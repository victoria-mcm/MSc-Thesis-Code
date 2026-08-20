#!/usr/bin/env python3
"""Calculate framework-aligned CDR RMSDs for antibody structure predictions."""

import argparse
import glob
import os
import re
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import Superimposer
from joblib import Parallel, delayed
from tqdm import tqdm

# Backbone atoms used for superposition and RMSD (N-CA-C only).
BACKBONE_ATOMS = ("N", "CA", "C")

# IMGT CDR boundaries (same for heavy and light chains).
IMGT_CDR_RANGES = {
    "cdr1": (27, 38),
    "cdr2": (56, 65),
    "cdr3": (105, 117),
}

PDB_PARSER = PDB.PDBParser(QUIET=True)
CIF_PARSER = PDB.MMCIFParser(QUIET=True)


def imgt_region(residue):
    """Map a Bio.PDB residue to IMGT region label (cdr1/cdr2/cdr3/framework)."""
    residue_number = residue.id[1]
    for region, (start, end) in IMGT_CDR_RANGES.items():
        if start <= residue_number <= end:
            return region
    return "framework"


def residue_dict(chain):
    """Build residue-id -> Residue lookup, excluding hetero/water entries."""
    return {res.id: res for res in chain.get_residues() if res.id[0] == " "}


def atom_pairs(truth_res, pred_res, residue_ids):
    """Collect paired backbone atoms for truth (fixed) and prediction (mobile)."""
    fixed_atoms = []
    moved_atoms = []

    for residue_id in residue_ids:
        true_residue = truth_res[residue_id]
        pred_residue = pred_res[residue_id]
        for atom_name in BACKBONE_ATOMS:
            if atom_name in true_residue and atom_name in pred_residue:
                fixed_atoms.append(true_residue[atom_name])
                moved_atoms.append(pred_residue[atom_name])

    return fixed_atoms, moved_atoms


def rmsd_from_pairs(truth_res, pred_res, residue_ids):
    """Compute backbone RMSD over the given residue IDs (pred must already be aligned)."""
    fixed_atoms, moved_atoms = atom_pairs(truth_res, pred_res, residue_ids)
    if not fixed_atoms:
        return np.nan

    fixed_xyz = np.array([atom.get_coord() for atom in fixed_atoms], dtype=float)
    moved_xyz = np.array([atom.get_coord() for atom in moved_atoms], dtype=float)
    return float(np.sqrt(np.mean(np.sum((fixed_xyz - moved_xyz) ** 2, axis=1))))


def chain_rmsds(true_chain, pred_chain, chain_label):
    """Superimpose on framework, then report per-region RMSDs for one chain (H or L)."""
    truth_res = residue_dict(true_chain)
    pred_res = residue_dict(pred_chain)
    common_ids = [res_id for res_id in truth_res if res_id in pred_res]

    region_ids = {
        "framework": [],
        "cdr1": [],
        "cdr2": [],
        "cdr3": [],
    }
    for res_id in common_ids:
        region_ids[imgt_region(truth_res[res_id])].append(res_id)

    cdr_all_ids = region_ids["cdr1"] + region_ids["cdr2"] + region_ids["cdr3"]

    fit_fixed, fit_moved = atom_pairs(truth_res, pred_res, region_ids["framework"])
    if len(fit_fixed) < 3:
        raise ValueError(
            f"Not enough framework backbone atoms to superimpose chain {chain_label}"
        )

    pred_chain_copy = deepcopy(pred_chain)
    pred_res_aligned = residue_dict(pred_chain_copy)

    imposer = Superimposer()
    imposer.set_atoms(fit_fixed, fit_moved)
    imposer.apply(pred_chain_copy.get_atoms())

    return {
        f"{chain_label}_framework": rmsd_from_pairs(
            truth_res, pred_res_aligned, region_ids["framework"]
        ),
        f"{chain_label}_cdr1": rmsd_from_pairs(truth_res, pred_res_aligned, region_ids["cdr1"]),
        f"{chain_label}_cdr2": rmsd_from_pairs(truth_res, pred_res_aligned, region_ids["cdr2"]),
        f"{chain_label}_cdr3": rmsd_from_pairs(truth_res, pred_res_aligned, region_ids["cdr3"]),
        f"{chain_label}_cdr_all": rmsd_from_pairs(truth_res, pred_res_aligned, cdr_all_ids),
    }


def compute_rmsds(pred_file, true_file, h_chain_id, l_chain_id):
    """Score one prediction PDB against one ground-truth mmCIF (H always; L if present)."""
    true_structure = CIF_PARSER.get_structure("true", true_file)
    pred_structure = PDB_PARSER.get_structure("pred", pred_file)

    true_model = next(true_structure.get_models())
    pred_model = next(pred_structure.get_models())

    metrics = {}
    metrics.update(
        chain_rmsds(true_model[h_chain_id], pred_model["H"], "H")
    )
    if not pd.isna(l_chain_id):
        metrics.update(
            chain_rmsds(true_model[l_chain_id], pred_model["L"], "L")
        )
    return metrics


# Column order for per-chain RMSD metrics in output CSVs.
METRIC_KEYS = [
    "H_framework", "H_cdr1", "H_cdr2", "H_cdr3", "H_cdr_all",
    "L_framework", "L_cdr1", "L_cdr2", "L_cdr3", "L_cdr_all",
]

METRIC_ORDER = [
    "H_framework", "H_cdr1", "H_cdr2", "H_cdr3", "H_cdr_all",
    "L_framework", "L_cdr1", "L_cdr2", "L_cdr3", "L_cdr_all",
]

def metric_keys_present(candidate_metrics):
    """Return ordered metric keys present in the first valid scored conformation."""
    valid = [m for m in candidate_metrics if m is not None]
    if not valid:
        raise ValueError("No valid ground-truth conformations could be scored")

    # Remove non-RMSD bookkeeping fields and preserve a stable order.
    return [k for k in METRIC_ORDER if k in valid[0]]


def build_sequence_candidates(dataset):
    """Map full_seq to all ground-truth structures with that sequence."""
    candidates = {}
    for row in dataset.itertuples(index=False):
        entry = {
            "pdb_name": row.pdb_name,
            "raw_path": row.raw_path,
            "Hchain": row.Hchain,
            "Lchain": row.Lchain,
        }
        candidates.setdefault(row.full_seq, []).append(entry)
    return candidates


def sample_index(sample_name):
    """Parse sample index from filenames like sample_3.pdb."""
    match = re.search(r"sample_(\d+)\.pdb$", sample_name)
    if match is None:
        raise ValueError(f"Cannot parse sample index from {sample_name}")
    return int(match.group(1))


def list_prediction_samples(pred_dir, pdb_name, all_samples=False, sample_name="sample_0.pdb"):
    """Return one or all sample_*.pdb filenames for a structure subdirectory."""
    if not all_samples:
        return [sample_name]

    pred_subdir = os.path.join(pred_dir, pdb_name)
    sample_files = glob.glob(os.path.join(pred_subdir, "sample_*.pdb"))
    if not sample_files:
        raise FileNotFoundError(f"No sample_*.pdb files found in {pred_subdir}")

    return sorted(
        (os.path.basename(path) for path in sample_files),
        key=sample_index,
    )


def load_pred_chains(pred_file):
    """Load H (and L if present) chains from a prediction PDB."""
    structure = PDB_PARSER.get_structure("pred", pred_file)
    model = next(structure.get_models())
    chains = {"H": model["H"]}
    if "L" in model:
        chains["L"] = model["L"]
    return chains


def framework_residue_ids(chain):
    return [
        res_id
        for res_id, residue in residue_dict(chain).items()
        if imgt_region(residue) == "framework"
    ]


def align_pred_chain_to_reference(ref_chain, mobile_chain):
    ref_res = residue_dict(ref_chain)
    mobile_res = residue_dict(mobile_chain)
    framework_ids = [
        res_id for res_id in framework_residue_ids(ref_chain) if res_id in mobile_res
    ]

    fit_fixed, fit_moved = atom_pairs(ref_res, mobile_res, framework_ids)
    if len(fit_fixed) < 3:
        raise ValueError("Not enough framework backbone atoms to superimpose predictions")

    mobile_copy = deepcopy(mobile_chain)
    imposer = Superimposer()
    imposer.set_atoms(fit_fixed, fit_moved)
    imposer.apply(mobile_copy.get_atoms())
    return mobile_copy


def align_pred_chains_to_reference(ref_chains, mobile_chains):
    aligned = {}
    for chain_id, ref_chain in ref_chains.items():
        if chain_id not in mobile_chains:
            continue
        aligned[chain_id] = align_pred_chain_to_reference(ref_chain, mobile_chains[chain_id])
    return aligned


def backbone_coords(chain):
    return {
        res.id: res["CA"].get_coord()
        for res in chain.get_residues()
        if res.id[0] == " " and "CA" in res
    }


def select_representative_sample(pred_dir, pdb_name, sample_names):
    if len(sample_names) == 1:
        return sample_names[0], 0.0

    ref_file = os.path.join(pred_dir, pdb_name, sample_names[0])
    ref_chains = load_pred_chains(ref_file)
    aligned_coords = []

    for sample_name in sample_names:
        pred_file = os.path.join(pred_dir, pdb_name, sample_name)
        mobile_chains = load_pred_chains(pred_file)
        if sample_name == sample_names[0]:
            aligned_chains = {chain_id: deepcopy(chain) for chain_id, chain in ref_chains.items()}
        else:
            aligned_chains = align_pred_chains_to_reference(ref_chains, mobile_chains)

        sample_coords = {
            chain_id: backbone_coords(chain) for chain_id, chain in aligned_chains.items()
        }
        aligned_coords.append(sample_coords)

    common_residue_ids = {}
    for chain_id in ref_chains:
        residue_ids = set(aligned_coords[0][chain_id])
        for sample_coords in aligned_coords[1:]:
            residue_ids &= set(sample_coords.get(chain_id, {}))
        common_residue_ids[chain_id] = sorted(residue_ids)

    centroid_coords = {}
    for chain_id, residue_ids in common_residue_ids.items():
        centroid_coords[chain_id] = {}
        for residue_id in residue_ids:
            points = np.array(
                [sample_coords[chain_id][residue_id] for sample_coords in aligned_coords],
                dtype=float,
            )
            centroid_coords[chain_id][residue_id] = points.mean(axis=0)

    sample_rmsds = []
    for sample_coords in aligned_coords:
        diffs = []
        for chain_id, residue_ids in common_residue_ids.items():
            for residue_id in residue_ids:
                diffs.append(sample_coords[chain_id][residue_id] - centroid_coords[chain_id][residue_id])
        if not diffs:
            sample_rmsds.append(np.nan)
        else:
            diff_array = np.array(diffs, dtype=float)
            sample_rmsds.append(
                float(np.sqrt(np.mean(np.sum(diff_array ** 2, axis=1))))
            )

    best_idx = int(np.nanargmin(sample_rmsds))
    return sample_names[best_idx], sample_rmsds[best_idx]


def summarize_multi_sample_results(sample_results, representative_sample):
    rep_row = next(row for row in sample_results if row["sample"] == representative_sample)
    summary = {
        "pdb_name": rep_row["pdb_name"],
        "representative_sample": representative_sample,
        "full_seq": rep_row.get("full_seq"),
        "best_match_pdb_name": rep_row.get("best_match_pdb_name"),
        "n_conformations_scored": rep_row.get("n_conformations_scored"),
        "n_conformations_total": rep_row.get("n_conformations_total"),
        "error": rep_row.get("error", ""),
    }

    for key in METRIC_KEYS:
        if key in rep_row:
            summary[key] = rep_row[key]

    for key in METRIC_KEYS:
        values = [
            row[key]
            for row in sample_results
            if key in row and pd.notna(row[key])
        ]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_var"] = float(np.var(values))
        else:
            summary[f"{key}_mean"] = np.nan
            summary[f"{key}_var"] = np.nan

    return summary


def select_best_metrics(candidate_metrics):
    """selecs closest matching ground truth conformation based on mean RMSD across all regions."""
    valid = [metrics for metrics in candidate_metrics if metrics is not None]
    if not valid:
        raise ValueError("No valid ground-truth conformations could be scored")

    metric_keys = metric_keys_present(valid)

    #best = {key: min(metrics[key] for metrics in valid) for key in metric_keys} #this picks min for each region independently

    mean_rmsds = []
    for metrics in valid:
        mean_rmsds.append((metrics, np.mean([metrics[key] for key in metric_keys])))
    best_match_metrics, _ = min(mean_rmsds, key=lambda item: item[1])
    #best_match_metrics = min(valid,key=lambda m: m["H_cdr3"])
    best = {key: best_match_metrics[key] for key in metric_keys}
    best["best_match_pdb_name"] = best_match_metrics["matched_pdb_name"]
    best["n_conformations_scored"] = len(valid)
    return best


def calculate_prediction_metrics(
    pdb_name,
    pred_dir,
    gt_candidates,
    sample_name="sample_0.pdb",
):
    pred_file = os.path.join(pred_dir, pdb_name, sample_name)
    if not os.path.isfile(pred_file):
        raise FileNotFoundError(f"Missing prediction file: {pred_file}")

    candidate_metrics = []
    candidate_errors = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for candidate in gt_candidates:
            if not os.path.isfile(candidate["raw_path"]):
                candidate_errors.append(
                    f"{candidate['pdb_name']}: missing file {candidate['raw_path']}"
                )
                candidate_metrics.append(None)
                continue
            try:
                metrics = compute_rmsds(
                    pred_file,
                    candidate["raw_path"],
                    candidate["Hchain"],
                    candidate["Lchain"],
                )
                metrics["matched_pdb_name"] = candidate["pdb_name"]
                candidate_metrics.append(metrics)
            except Exception as exc:
                candidate_errors.append(f"{candidate['pdb_name']}: {exc}")
                candidate_metrics.append(None)

    best = select_best_metrics(candidate_metrics)
    if candidate_errors:
        best["conformation_errors"] = "; ".join(candidate_errors)
    return best


def score_sample(row, pred_dir, sample_name, gt_candidates):
    metrics = calculate_prediction_metrics(
        pdb_name=row.pdb_name,
        pred_dir=pred_dir,
        gt_candidates=gt_candidates,
        sample_name=sample_name,
    )
    metrics["status"] = "ok"
    metrics["error"] = metrics.pop("conformation_errors", "")
    metrics["full_seq"] = row.full_seq
    metrics["n_conformations_total"] = len(gt_candidates)
    metrics["pdb_name"] = row.pdb_name
    metrics["sample"] = sample_name
    metrics["sample_idx"] = sample_index(sample_name)
    return metrics


def process_row(row, pred_dir, all_samples, sample_name, seq_candidates):
    long_rows = []
    summary_row = None

    try:
        gt_candidates = seq_candidates[row.full_seq]
        sample_names = list_prediction_samples(
            pred_dir, row.pdb_name, all_samples=all_samples, sample_name=sample_name
        )
    except Exception as exc:
        error_row = {
            "status": "error",
            "error": str(exc),
            "full_seq": row.full_seq,
            "pdb_name": row.pdb_name,
        }
        if all_samples:
            return [error_row], {
                **error_row,
                "n_samples_scored": 0,
            }
        error_row["sample"] = sample_name
        error_row["sample_idx"] = sample_index(sample_name)
        return [error_row], None

    for current_sample in sample_names:
        try:
            long_rows.append(score_sample(row, pred_dir, current_sample, gt_candidates))
        except Exception as exc:
            error_row = {
                "status": "error",
                "error": str(exc),
                "full_seq": row.full_seq,
                "pdb_name": row.pdb_name,
                "sample": current_sample,
                "sample_idx": sample_index(current_sample),
            }
            long_rows.append(error_row)

    if all_samples:
        ok_rows = [row_metrics for row_metrics in long_rows if row_metrics.get("status") == "ok"]
        if ok_rows:
            try:
                representative_sample, centroid_rmsd = select_representative_sample(
                    pred_dir,
                    row.pdb_name,
                    [row_metrics["sample"] for row_metrics in ok_rows],
                )
                summary_row = summarize_multi_sample_results(ok_rows, representative_sample)
                summary_row["centroid_rmsd"] = centroid_rmsd
                summary_row["n_samples_scored"] = len(ok_rows)
                summary_row["status"] = "ok"
            except Exception as exc:
                summary_row = {
                    "pdb_name": row.pdb_name,
                    "full_seq": row.full_seq,
                    "status": "error",
                    "error": str(exc),
                    "n_samples_scored": len(ok_rows),
                }
        else:
            summary_row = {
                "pdb_name": row.pdb_name,
                "full_seq": row.full_seq,
                "status": "error",
                "error": long_rows[0].get("error", "No samples could be scored"),
                "n_samples_scored": 0,
            }

    return long_rows, summary_row


def main(
    pred_path,
    meta_path,
    dset,
    sample_name="sample_0.pdb",
    all_samples=False,
    output_path=None,
    summary_output_path=None,
    n_jobs=1,
):
    dataset = pd.read_csv(meta_path, index_col=0)
    dataset = dataset[dataset.split == dset].copy()
    seq_candidates = build_sequence_candidates(dataset)
    n_multi = sum(1 for entries in seq_candidates.values() if len(entries) > 1)
    print(
        f"Evaluating {len(dataset)} structures across "
        f"{len(seq_candidates)} unique sequences "
        f"({n_multi} sequences with multiple conformations)"
    )

    if output_path is None:
        suffix = "all_samples" if all_samples else ""
        output_path = os.path.join(pred_path, f"struc_pred_metrics_{dset}_{suffix}.csv")
    if summary_output_path is None and all_samples:
        summary_output_path = os.path.join(pred_path, f"struc_pred_metrics_{dset}_summary.csv")

    rows = list(dataset.itertuples(index=False))
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row, pred_path, all_samples, sample_name, seq_candidates)
        for row in tqdm(rows, total=len(rows), desc="Calculating RMSDs")
    )

    long_results = []
    summary_results = []
    for long_rows, summary_row in results:
        long_results.extend(long_rows)
        if summary_row is not None:
            summary_results.append(summary_row)

    results_df = pd.DataFrame(long_results).set_index("pdb_name")
    results_df.to_csv(output_path)
    print(f"Saved metrics for {len(results_df)} rows to {output_path}")
    print(
        f"Successful: {(results_df['status'] == 'ok').sum()} | "
        f"Failed: {(results_df['status'] == 'error').sum()}"
    )

    if all_samples and summary_results:
        summary_df = pd.DataFrame(summary_results).set_index("pdb_name")
        summary_df.to_csv(summary_output_path)
        print(f"Saved summary metrics for {len(summary_df)} structures to {summary_output_path}")
        print(
            f"Summary successful: {(summary_df['status'] == 'ok').sum()} | "
            f"Summary failed: {(summary_df['status'] == 'error').sum()}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate framework-aligned CDR RMSDs for structure predictions."
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Directory containing per-structure prediction subdirectories.",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        default="/opig-shared/users/lina4783/structures_final/test_meta.csv",
        help="CSV with pdb_name, raw_path, Hchain, Lchain, full_seq, and split columns.",
    )
    parser.add_argument(
        "--dset",
        type=str,
        required=True,
        help="Dataset split to evaluate (e.g. test).",
    )
    parser.add_argument(
        "--sample_name",
        type=str,
        default="sample_0.pdb",
        help="Prediction filename inside each pdb_name subdirectory.",
    )
    parser.add_argument(
        "--all_samples",
        action="store_true",
        help="Score all sample_*.pdb files in each pdb_name subdirectory.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output CSV path (defaults to pred_path/struc_pred_metrics_<dset>_<suffix>.csv).",
    )
    parser.add_argument(
        "--summary_output_path",
        type=str,
        default=None,
        help="Summary CSV path when --all_samples is set.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=10,
        help="Number of parallel jobs.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Deprecated; kept for compatibility with existing sbatch scripts.",
    )
    args = parser.parse_args()
    main(
        pred_path=args.pred_path,
        meta_path=args.meta_path,
        dset=args.dset,
        sample_name=args.sample_name,
        all_samples=args.all_samples,
        output_path=args.output_path,
        summary_output_path=args.summary_output_path,
        n_jobs=args.n_jobs,
    )
