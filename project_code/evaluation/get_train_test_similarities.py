#!/usr/bin/env python3
"""
cdrh3_train_test_alignment_identity.py

Compute train-vs-test CDRH3 similarity using global pairwise alignment.

Outputs:
  - similarity_matrix.csv
  - nearest_train_per_test.csv
  - similarity_summary.json

Similarity types:
  identity
    identity = matches / alignment_columns
    where alignment_columns includes matches, mismatches, and gaps in the
    final global alignment core region.

  blosum
    mean BLOSUM substitution-matrix score over aligned residue pairs in the
    core region (gaps excluded from the mean).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Literal, Tuple, Optional

import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Align import PairwiseAligner, substitution_matrices

SIMILARITY_TYPES = ("identity", "blosum")
SimilarityType = Literal["identity", "blosum"]


def normalize_seq(seq: object) -> str:
    """Uppercase and keep only letters."""
    if pd.isna(seq):
        return ""
    s = str(seq).upper().strip()
    return re.sub(r"[^A-Z]", "", s)


def read_table(path: Path, id_col: Optional[str], seq_col: str) -> List[Tuple[str, str]]:
    """Read CSV/TSV file and return (id, seq) pairs."""
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    if seq_col not in df.columns:
        raise ValueError(f"Sequence column '{seq_col}' not found in {path}. Columns: {list(df.columns)}")

    if id_col is not None and id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in {path}. Columns: {list(df.columns)}")

    items: List[Tuple[str, str]] = []
    for i, row in df.iterrows():
        seq_id = str(row[id_col]) if id_col is not None else f"row_{i}"
        seq = normalize_seq(row[seq_col])
        if seq:
            items.append((seq_id, seq))
    return items


def _core_alignment_columns(aln1: str, aln2: str) -> Tuple[str, str]:
    """Return the aligned core span shared by both sequences."""
    shared_cols = [
        i for i, (a, b) in enumerate(zip(aln1, aln2))
        if a != "-" and b != "-"
    ]

    if shared_cols:
        start = shared_cols[0]
        stop = shared_cols[-1] + 1
        return aln1[start:stop], aln2[start:stop]
    return aln1, aln2


def align_and_identity(
    seq1: str,
    seq2: str,
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
) -> Tuple[float, int, str, str]:
    """
    Global alignment + percent identity.

    Denominator rule:
      - do NOT count leading/trailing overhangs
      - do count internal gaps/mismatches inside the aligned core region

    Returns:
      identity (0-1),
      effective_alignment_length,
      aligned_seq1,
      aligned_seq2
    """
    alignments = pairwise2.align.globalms(
        seq1,
        seq2,
        match_score,
        mismatch_score,
        gap_open,
        gap_extend,
        one_alignment_only=True,
    )

    if not alignments:
        return 0.0, 0, "", ""

    aln1, aln2, _, _, _ = alignments[0]
    core1, core2 = _core_alignment_columns(aln1, aln2)

    matches = 0
    alignment_cols = 0

    for a, b in zip(core1, core2):
        if a == "-" and b == "-":
            continue
        alignment_cols += 1
        if a == b and a != "-":
            matches += 1

    identity = matches / alignment_cols if alignment_cols else 0.0
    return identity, alignment_cols, aln1, aln2


def _blosum_pair_score(matrix, residue_a: str, residue_b: str) -> float:
    try:
        return float(matrix[(residue_a, residue_b)])
    except KeyError:
        return float(matrix[(residue_b, residue_a)])


def align_and_blosum(
    seq1: str,
    seq2: str,
    blosum_matrix: str = "BLOSUM62",
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
) -> Tuple[float, int, str, str]:
    """
    Global alignment + mean BLOSUM score over aligned residue pairs.

    Returns:
      mean_blosum_score,
      number_of_scored_residue_pairs,
      aligned_seq1,
      aligned_seq2
    """
    matrix = substitution_matrices.load(blosum_matrix)
    aligner = PairwiseAligner()
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    aligner.mode = "global"

    alignments = aligner.align(seq1, seq2)
    if not alignments:
        return 0.0, 0, "", ""

    aln1 = str(alignments[0][0])
    aln2 = str(alignments[0][1])
    core1, core2 = _core_alignment_columns(aln1, aln2)

    scores: List[float] = []
    for a, b in zip(core1, core2):
        if a == "-" or b == "-":
            continue
        scores.append(_blosum_pair_score(matrix, a, b))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return mean_score, len(scores), aln1, aln2


def align_pair(
    similarity_type: SimilarityType,
    seq1: str,
    seq2: str,
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    blosum_matrix: str = "BLOSUM62",
    blosum_gap_open: float = -10.0,
    blosum_gap_extend: float = -0.5,
) -> Tuple[float, int, str, str]:
    if similarity_type == "identity":
        return align_and_identity(
            seq1,
            seq2,
            match_score=match_score,
            mismatch_score=mismatch_score,
            gap_open=gap_open,
            gap_extend=gap_extend,
        )
    if similarity_type == "blosum":
        return align_and_blosum(
            seq1,
            seq2,
            blosum_matrix=blosum_matrix,
            gap_open=blosum_gap_open,
            gap_extend=blosum_gap_extend,
        )
    raise ValueError(f"Unsupported similarity_type: {similarity_type}")


def _summary_prefix(similarity_type: SimilarityType) -> str:
    return "identity" if similarity_type == "identity" else "blosum_similarity"


def compute_cross_alignment(
    train: List[Tuple[str, str]],
    test: List[Tuple[str, str]],
    similarity_type: SimilarityType = "identity",
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    blosum_matrix: str = "BLOSUM62",
    blosum_gap_open: float = -10.0,
    blosum_gap_extend: float = -0.5,
) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Compute test x train similarity matrix and nearest training example per test.
    """
    n_test = len(test)
    n_train = len(train)

    sim_mat = np.zeros((n_test, n_train), dtype=float)
    nearest_rows = []
    all_sims = []

    best_overall = {
        "test_id": None,
        "train_id": None,
        "test_seq": None,
        "train_seq": None,
        "similarity": -math.inf,
        "alignment_length": None,
    }

    for i, (test_id, test_seq) in enumerate(test):
        best_j = -1
        best_sim = -math.inf
        best_aln_len = None

        for j, (train_id, train_seq) in enumerate(train):
            similarity, aln_len, _, _ = align_pair(
                similarity_type,
                test_seq,
                train_seq,
                match_score=match_score,
                mismatch_score=mismatch_score,
                gap_open=gap_open,
                gap_extend=gap_extend,
                blosum_matrix=blosum_matrix,
                blosum_gap_open=blosum_gap_open,
                blosum_gap_extend=blosum_gap_extend,
            )

            sim_mat[i, j] = similarity
            all_sims.append(similarity)

            if similarity > best_sim:
                best_sim = similarity
                best_aln_len = aln_len
                best_j = j

            if similarity > best_overall["similarity"]:
                best_overall = {
                    "test_id": test_id,
                    "train_id": train_id,
                    "test_seq": test_seq,
                    "train_seq": train_seq,
                    "similarity": similarity,
                    "alignment_length": aln_len,
                }

        row = {
            "test_id": test_id,
            "test_seq": test_seq,
            "best_train_id": train[best_j][0],
            "best_train_seq": train[best_j][1],
            "best_alignment_length": best_aln_len,
        }
        if similarity_type == "identity":
            row["best_identity"] = best_sim
            row["best_identity_percent"] = best_sim * 100.0
        else:
            row["best_blosum_similarity"] = best_sim

        nearest_rows.append(row)

    metric = _summary_prefix(similarity_type)
    summary = {
        "similarity_type": similarity_type,
        "n_train": n_train,
        "n_test": n_test,
        "n_pairs": int(n_train * n_test),
        f"max_{metric}": float(best_overall["similarity"]),
        f"max_{metric}_test_id": best_overall["test_id"],
        f"max_{metric}_train_id": best_overall["train_id"],
        f"max_{metric}_alignment_length": best_overall["alignment_length"],
        f"mean_{metric}": float(np.mean(all_sims)) if all_sims else None,
        f"std_{metric}": float(np.std(all_sims)) if all_sims else None,
        f"min_{metric}": float(np.min(all_sims)) if all_sims else None,
        f"median_{metric}": float(np.median(all_sims)) if all_sims else None,
        f"p05_{metric}": float(np.quantile(all_sims, 0.05)) if all_sims else None,
        f"p25_{metric}": float(np.quantile(all_sims, 0.25)) if all_sims else None,
        f"p75_{metric}": float(np.quantile(all_sims, 0.75)) if all_sims else None,
        f"p95_{metric}": float(np.quantile(all_sims, 0.95)) if all_sims else None,
    }

    if similarity_type == "identity":
        summary["max_identity_percent"] = float(best_overall["similarity"] * 100.0)

    if similarity_type == "blosum":
        summary["blosum_matrix"] = blosum_matrix

    if all_sims:
        if similarity_type == "identity":
            hist_range = (0.0, 1.0)
        else:
            hist_range = (float(np.min(all_sims)), float(np.max(all_sims)))
            if hist_range[0] == hist_range[1]:
                hist_range = (hist_range[0] - 1.0, hist_range[1] + 1.0)

        hist_counts, hist_edges = np.histogram(all_sims, bins=20, range=hist_range)
        summary["histogram"] = [
            {
                "bin_left": float(hist_edges[k]),
                "bin_right": float(hist_edges[k + 1]),
                "count": int(hist_counts[k]),
            }
            for k in range(len(hist_counts))
        ]
    else:
        summary["histogram"] = []

    nearest_df = pd.DataFrame(nearest_rows)
    return sim_mat, nearest_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute train-test CDRH3 similarity with global pairwise alignment."
    )
    parser.add_argument("--train", required=True, type=Path, help="Train CSV/TSV file")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV/TSV file")
    parser.add_argument("--seq-col", default="cdrh3", help="Sequence column name in both files")
    parser.add_argument("--id-col", default=None, help="ID column name in both files (optional)")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--similarity-type",
        choices=SIMILARITY_TYPES,
        default="identity",
        help="Similarity metric: percent identity or mean BLOSUM score",
    )
    parser.add_argument("--match-score", type=float, default=1.0, help="Match score for identity alignment")
    parser.add_argument("--mismatch-score", type=float, default=0.0, help="Mismatch score for identity alignment")
    parser.add_argument("--gap-open", type=float, default=-2.0, help="Gap open penalty for identity alignment")
    parser.add_argument("--gap-extend", type=float, default=-0.5, help="Gap extend penalty for identity alignment")
    parser.add_argument("--blosum-matrix", default="BLOSUM62", help="Substitution matrix for BLOSUM similarity")
    parser.add_argument("--blosum-gap-open", type=float, default=-10.0, help="Gap open penalty for BLOSUM alignment")
    parser.add_argument("--blosum-gap-extend", type=float, default=-0.5, help="Gap extend penalty for BLOSUM alignment")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    train = read_table(args.train, args.id_col, args.seq_col)
    test = read_table(args.test, args.id_col, args.seq_col)

    if not train:
        raise ValueError("No valid train sequences found after cleaning.")
    if not test:
        raise ValueError("No valid test sequences found after cleaning.")

    sim_mat, nearest_df, summary = compute_cross_alignment(
        train=train,
        test=test,
        similarity_type=args.similarity_type,
        match_score=args.match_score,
        mismatch_score=args.mismatch_score,
        gap_open=args.gap_open,
        gap_extend=args.gap_extend,
        blosum_matrix=args.blosum_matrix,
        blosum_gap_open=args.blosum_gap_open,
        blosum_gap_extend=args.blosum_gap_extend,
    )

    train_ids = [x[0] for x in train]
    test_ids = [x[0] for x in test]

    sim_df = pd.DataFrame(sim_mat, index=test_ids, columns=train_ids)
    sim_df.to_csv(args.outdir / "similarity_matrix.csv")

    nearest_df.to_csv(args.outdir / "nearest_train_per_test.csv", index=False)

    with open(args.outdir / "similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    metric = _summary_prefix(args.similarity_type)
    print("Done.")
    print(f"Similarity type: {args.similarity_type}")
    print(f"Train sequences: {summary['n_train']}")
    print(f"Test sequences:  {summary['n_test']}")
    print(f"Pairwise pairs:   {summary['n_pairs']}")
    print(f"Max {metric}:     {summary[f'max_{metric}']:.4f}")
    print(f"Saved to:         {args.outdir}")


if __name__ == "__main__":
    main()
