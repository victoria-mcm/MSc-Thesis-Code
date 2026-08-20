#!/usr/bin/env python3
"""
Compute train-vs-test similarities between ESM-C residue embeddings.

Embedding modes (--embedding-mode):
  - cdr: embeddings are already concat_CDR residues (default)
  - vhvl_cdr: load full VH+VL chain embeddings and extract CDR slices from metadata

Supported metrics (--similarity-type):
  - wasserstein: uniform OT over per-residue embeddings (lower = more similar)
  - mean_cosine: mean-pool residues then cosine similarity (higher = more similar)
  - cdr_mean_cosine: per-CDR mean-pool, concat, then cosine (higher = more similar)

Outputs:
  - similarity_matrix.csv
  - nearest_train_per_test.csv
  - similarity_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import ot
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from tqdm import tqdm

SIMILARITY_TYPES = ("wasserstein", "mean_cosine", "cdr_mean_cosine")
SimilarityType = Literal["wasserstein", "mean_cosine", "cdr_mean_cosine"]
EMBEDDING_MODES = ("cdr", "vhvl_cdr")
EmbeddingMode = Literal["cdr", "vhvl_cdr"]
CDR_COLS = ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"]
REGION_COLS = [
    "fwr1_h", "CDRH1", "fwr2_h", "CDRH2", "fwr3_h", "CDRH3", "fwr4_h",
    "fwr1_l", "CDRL1", "fwr2_l", "CDRL2", "fwr3_l", "CDRL3", "fwr4_l",
]

_TRAIN_IDS: List[str] = []
_TRAIN_REPS: List[np.ndarray] = []
_SIMILARITY_TYPE: SimilarityType = "wasserstein"
_LOWER_IS_MORE_SIMILAR: bool = True


@dataclass(frozen=True)
class SplitRecord:
    id: str
    seq: str
    cdr_lengths: Tuple[int, ...]
    cdr_slices: Tuple[Tuple[int, int], ...] = ()


def _init_worker(
    train_ids: List[str],
    train_reps: List[np.ndarray],
    similarity_type: SimilarityType,
    lower_is_more_similar: bool,
) -> None:
    global _TRAIN_IDS, _TRAIN_REPS, _SIMILARITY_TYPE, _LOWER_IS_MORE_SIMILAR
    _TRAIN_IDS = train_ids
    _TRAIN_REPS = train_reps
    _SIMILARITY_TYPE = similarity_type
    _LOWER_IS_MORE_SIMILAR = lower_is_more_similar


def normalize_seq(seq: object) -> str:
    """Uppercase and keep only letters."""
    if pd.isna(seq):
        return ""
    s = str(seq).upper().strip()
    return re.sub(r"[^A-Z]", "", s)


def sanitize_checkpoint_id(pdb_name: str) -> str:
    return pdb_name.replace(":", "_")


def _summary_metric_name(similarity_type: SimilarityType) -> str:
    return {
        "wasserstein": "wasserstein_distance",
        "mean_cosine": "mean_cosine_similarity",
        "cdr_mean_cosine": "cdr_mean_cosine_similarity",
    }[similarity_type]


def _best_metric_field(similarity_type: SimilarityType) -> str:
    return f"best_{_summary_metric_name(similarity_type)}"


def _lower_is_more_similar(similarity_type: SimilarityType) -> bool:
    return similarity_type == "wasserstein"


def _extreme_prefix(lower_is_more_similar: bool) -> str:
    return "min" if lower_is_more_similar else "max"


def _embedding_source(embedding_mode: EmbeddingMode) -> str:
    return {
        "cdr": "esmc_300m_concat_cdr",
        "vhvl_cdr": "esmc_300m_vhvl_cdr_extracted",
    }[embedding_mode]


def compute_cdr_slices(row: pd.Series) -> Tuple[Tuple[int, int], ...]:
    """Return (start, end) slices into a full VH+VL chain for each CDR region."""
    slices: List[Tuple[int, int]] = []
    offset = 0
    for col in REGION_COLS:
        region_seq = normalize_seq(row[col])
        if col in CDR_COLS:
            slices.append((offset, offset + len(region_seq)))
        offset += len(region_seq)
    return tuple(slices)


def validate_region_layout(row: pd.Series, seq_id: str, concat_cdr: str) -> int:
    """Validate FW/CDR columns and return full-chain length."""
    missing = [col for col in REGION_COLS if col not in row.index]
    if missing:
        raise ValueError(f"{seq_id}: missing region columns: {missing}")

    region_seqs = [normalize_seq(row[col]) for col in REGION_COLS]
    full_from_regions = "".join(region_seqs)
    full_len = len(full_from_regions)

    if "full_seq_concat" in row.index:
        full_seq_concat = normalize_seq(row["full_seq_concat"])
        if full_seq_concat and full_seq_concat != full_from_regions:
            raise ValueError(
                f"{seq_id}: full_seq_concat does not match concatenated region columns "
                f"(len {len(full_seq_concat)} vs {full_len})"
            )
    elif {"VH_seq", "VL_seq"}.issubset(row.index):
        vh_vl = normalize_seq(row["VH_seq"]) + normalize_seq(row["VL_seq"])
        if vh_vl != full_from_regions:
            raise ValueError(
                f"{seq_id}: VH_seq+VL_seq does not match concatenated region columns "
                f"(len {len(vh_vl)} vs {full_len})"
            )

    cdr_from_regions = "".join(
        normalize_seq(row[col]) for col in CDR_COLS if col in row.index
    )
    if cdr_from_regions != concat_cdr:
        raise ValueError(
            f"{seq_id}: concat_CDR does not match CDR columns from region layout "
            f"(len {len(concat_cdr)} vs {len(cdr_from_regions)})"
        )

    return full_len


def extract_cdr_embedding(
    full_embed: np.ndarray,
    cdr_slices: Tuple[Tuple[int, int], ...],
) -> np.ndarray:
    if not cdr_slices:
        raise ValueError("cdr_slices must be non-empty for vhvl_cdr extraction")
    parts = [full_embed[start:end] for start, end in cdr_slices]
    return np.concatenate(parts, axis=0)


def read_split_table(
    path: Path,
    id_col: Optional[str],
    seq_col: str,
    require_cdr_lengths: bool = False,
    embedding_mode: EmbeddingMode = "cdr",
) -> List[SplitRecord]:
    """Read CSV/TSV file and return split records with optional CDR lengths."""
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    if seq_col not in df.columns:
        raise ValueError(
            f"Sequence column '{seq_col}' not found in {path}. Columns: {list(df.columns)}"
        )

    if id_col is not None and id_col not in df.columns:
        raise ValueError(
            f"ID column '{id_col}' not found in {path}. Columns: {list(df.columns)}"
        )

    if require_cdr_lengths or embedding_mode == "vhvl_cdr":
        missing_cdr = [col for col in CDR_COLS if col not in df.columns]
        if missing_cdr:
            raise ValueError(
                f"CDR columns required for {embedding_mode} missing in {path}: {missing_cdr}"
            )

    if embedding_mode == "vhvl_cdr":
        missing_regions = [col for col in REGION_COLS if col not in df.columns]
        if missing_regions:
            raise ValueError(
                f"Region columns required for vhvl_cdr missing in {path}: {missing_regions}"
            )

    records: List[SplitRecord] = []
    for i, row in df.iterrows():
        seq_id = str(row[id_col]) if id_col is not None else f"row_{i}"
        seq = normalize_seq(row[seq_col])
        if not seq:
            continue

        cdr_lengths: Tuple[int, ...] = ()
        cdr_slices: Tuple[Tuple[int, int], ...] = ()
        if require_cdr_lengths or embedding_mode == "vhvl_cdr" or all(
            col in df.columns for col in CDR_COLS
        ):
            cdr_seqs = [normalize_seq(row[col]) for col in CDR_COLS]
            cdr_lengths = tuple(len(cdr) for cdr in cdr_seqs)
            expected = "".join(cdr_seqs)
            if expected != seq:
                raise ValueError(
                    f"{seq_id}: {seq_col} does not match concatenated CDR columns "
                    f"(expected len {len(expected)}, got {len(seq)})"
                )
            if sum(cdr_lengths) != len(seq):
                raise ValueError(
                    f"{seq_id}: sum of CDR lengths ({sum(cdr_lengths)}) "
                    f"!= {seq_col} length ({len(seq)})"
                )

        if embedding_mode == "vhvl_cdr":
            validate_region_layout(row, seq_id, seq)
            cdr_slices = compute_cdr_slices(row)

        records.append(
            SplitRecord(id=seq_id, seq=seq, cdr_lengths=cdr_lengths, cdr_slices=cdr_slices)
        )
    return records


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"pdb_name", "embedding_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Manifest {manifest_path} missing columns: {sorted(missing)}"
        )
    if manifest["pdb_name"].duplicated().any():
        raise ValueError("Manifest contains duplicate pdb_name values.")
    return manifest.set_index("pdb_name", drop=False)


def load_embedding(manifest: pd.DataFrame, pdb_name: str) -> np.ndarray:
    if pdb_name not in manifest.index:
        raise KeyError(f"{pdb_name!r} not found in embedding manifest.")

    path = Path(manifest.loc[pdb_name, "embedding_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file for {pdb_name}: {path}")

    embedding = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(embedding, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(embedding)}")
    if embedding.ndim != 2:
        raise ValueError(f"Expected 2D tensor in {path}, got shape {tuple(embedding.shape)}")
    return embedding.detach().cpu().float().clone().numpy()


def validate_embeddings(manifest: pd.DataFrame, pdb_names: set[str]) -> None:
    bad: List[str] = []

    def check(pdb_name: str) -> Tuple[str, Optional[str]]:
        path = Path(manifest.loc[pdb_name, "embedding_path"])
        if not path.exists():
            return pdb_name, "missing file"
        try:
            embedding = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2:
                return pdb_name, f"invalid tensor shape {getattr(embedding, 'shape', type(embedding))}"
        except Exception as exc:
            return pdb_name, str(exc)
        return pdb_name, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(check, pdb_name) for pdb_name in sorted(pdb_names)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Validating embeddings"):
            pdb_name, error = future.result()
            if error is not None:
                bad.append(f"{pdb_name}: {error}")

    if bad:
        preview = "\n  ".join(bad[:10])
        raise RuntimeError(
            f"{len(bad)} embedding files are missing or corrupt. Examples:\n  {preview}\n"
            "Run repair_esmc_embeddings.py --delete-corrupt, then re-run "
            "extract_esmc_concat_cdr_embeddings.py for missing entries."
        )


def mean_pool(embed: np.ndarray) -> np.ndarray:
    if embed.shape[0] == 0:
        return np.zeros(embed.shape[1], dtype=np.float64)
    return embed.mean(axis=0)


def cdr_mean_concat(embed: np.ndarray, cdr_lengths: Tuple[int, ...]) -> np.ndarray:
    if embed.shape[0] == 0:
        return np.zeros(len(cdr_lengths) * embed.shape[1], dtype=np.float64)

    parts: List[np.ndarray] = []
    start = 0
    for length in cdr_lengths:
        end = start + length
        block = embed[start:end]
        parts.append(mean_pool(block))
        start = end

    if start != embed.shape[0]:
        raise ValueError(
            f"CDR lengths sum to {start} but embedding has {embed.shape[0]} residues"
        )
    return np.concatenate(parts, axis=0)


def build_representation(
    embed: np.ndarray,
    similarity_type: SimilarityType,
    cdr_lengths: Tuple[int, ...] = (),
) -> np.ndarray:
    if similarity_type == "wasserstein":
        return embed
    if similarity_type == "mean_cosine":
        return mean_pool(embed)
    if similarity_type == "cdr_mean_cosine":
        if not cdr_lengths:
            raise ValueError("cdr_mean_cosine requires CDR lengths")
        return cdr_mean_concat(embed, cdr_lengths)
    raise ValueError(f"Unsupported similarity_type: {similarity_type}")


def build_all_representations(
    embeddings: Dict[str, np.ndarray],
    records: List[SplitRecord],
    similarity_type: SimilarityType,
) -> Dict[str, np.ndarray]:
    representations: Dict[str, np.ndarray] = {}
    for record in records:
        representations[record.id] = build_representation(
            embeddings[record.id],
            similarity_type,
            record.cdr_lengths,
        )
    return representations


def extract_cdr_embeddings_from_full_chain(
    embeddings: Dict[str, np.ndarray],
    records: List[SplitRecord],
    manifest: pd.DataFrame,
) -> None:
    """In-place: replace full VH+VL embeddings with extracted CDR residue embeddings."""
    for record in records:
        if not record.cdr_slices:
            raise ValueError(f"{record.id}: missing cdr_slices for vhvl_cdr extraction")

        full_embed = embeddings[record.id]
        if record.id in manifest.index and "seq_len" in manifest.columns:
            manifest_len = int(manifest.loc[record.id, "seq_len"])
            if full_embed.shape[0] != manifest_len:
                raise ValueError(
                    f"{record.id}: embedding length {full_embed.shape[0]} != "
                    f"manifest seq_len {manifest_len}"
                )

        cdr_len = sum(end - start for start, end in record.cdr_slices)
        if cdr_len != len(record.seq):
            raise ValueError(
                f"{record.id}: extracted CDR length {cdr_len} != "
                f"concat_CDR length {len(record.seq)}"
            )

        embeddings[record.id] = extract_cdr_embedding(full_embed, record.cdr_slices)


def wasserstein2_uniform(embed_a: np.ndarray, embed_b: np.ndarray) -> float:
    """Wasserstein-2 distance between uniform residue distributions."""
    n_a = embed_a.shape[0]
    n_b = embed_b.shape[0]
    if n_a == 0 or n_b == 0:
        return float("nan")

    weights_a = np.full(n_a, 1.0 / n_a, dtype=np.float64)
    weights_b = np.full(n_b, 1.0 / n_b, dtype=np.float64)
    cost = cdist(embed_a, embed_b, metric="sqeuclidean")
    w2_squared = ot.emd2(weights_a, weights_b, cost)
    return float(math.sqrt(max(w2_squared, 0.0)))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return float("nan")
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def pairwise_score(
    rep_a: np.ndarray,
    rep_b: np.ndarray,
    similarity_type: SimilarityType,
) -> float:
    if similarity_type == "wasserstein":
        return wasserstein2_uniform(rep_a, rep_b)
    return cosine_similarity(rep_a, rep_b)


def _compute_test_row_worker(
    test_id: str,
    test_seq: str,
    test_rep: np.ndarray,
) -> Tuple[str, str, np.ndarray, int, float, Optional[int], Optional[int]]:
    return _compute_test_row(
        test_id,
        test_seq,
        test_rep,
        _TRAIN_IDS,
        _TRAIN_REPS,
        _SIMILARITY_TYPE,
        _LOWER_IS_MORE_SIMILAR,
    )


def _compute_test_row(
    test_id: str,
    test_seq: str,
    test_rep: np.ndarray,
    train_ids: List[str],
    train_reps: List[np.ndarray],
    similarity_type: SimilarityType,
    lower_is_more_similar: bool,
) -> Tuple[str, str, np.ndarray, int, float, Optional[int], Optional[int]]:
    row = np.empty(len(train_ids), dtype=np.float64)
    best_j = -1
    if lower_is_more_similar:
        best_score = math.inf
    else:
        best_score = -math.inf

    for j, train_rep in enumerate(train_reps):
        score = pairwise_score(test_rep, train_rep, similarity_type)
        row[j] = score
        if lower_is_more_similar:
            if score < best_score:
                best_score = score
                best_j = j
        elif score > best_score:
            best_score = score
            best_j = j

    n_test_residues: Optional[int] = None
    n_best_train_residues: Optional[int] = None
    if similarity_type == "wasserstein":
        n_test_residues = int(test_rep.shape[0])
        n_best_train_residues = int(train_reps[best_j].shape[0])

    return (
        test_id,
        test_seq,
        row,
        best_j,
        best_score,
        n_test_residues,
        n_best_train_residues,
    )


def _checkpoint_row_path(checkpoint_dir: Path, test_id: str) -> Path:
    return checkpoint_dir / "rows" / f"{sanitize_checkpoint_id(test_id)}.npy"


def _load_checkpoint_rows(
    checkpoint_dir: Path,
    test_ids: List[str],
) -> Dict[str, np.ndarray]:
    rows: Dict[str, np.ndarray] = {}
    for test_id in test_ids:
        path = _checkpoint_row_path(checkpoint_dir, test_id)
        if path.exists():
            rows[test_id] = np.load(path)
    return rows


def _save_checkpoint_row(
    checkpoint_dir: Path,
    test_id: str,
    row: np.ndarray,
    nearest_record: dict,
) -> None:
    rows_dir = checkpoint_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    np.save(_checkpoint_row_path(checkpoint_dir, test_id), row)
    with open(checkpoint_dir / "nearest.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(nearest_record) + "\n")


def _load_checkpoint_nearest(checkpoint_dir: Path) -> Dict[str, dict]:
    nearest: Dict[str, dict] = {}
    nearest_path = checkpoint_dir / "nearest.jsonl"
    if not nearest_path.exists():
        return nearest
    with open(nearest_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            nearest[str(record["test_id"])] = record
    return nearest


def _build_nearest_record(
    test_id: str,
    test_seq: str,
    train: List[SplitRecord],
    train_ids: List[str],
    best_j: int,
    best_score: float,
    similarity_type: SimilarityType,
    n_test_residues: Optional[int],
    n_best_train_residues: Optional[int],
) -> dict:
    record = {
        "test_id": test_id,
        "test_seq": test_seq,
        "best_train_id": train_ids[best_j],
        "best_train_seq": train[best_j].seq,
        _best_metric_field(similarity_type): best_score,
    }
    if similarity_type == "wasserstein":
        record["n_test_residues"] = n_test_residues
        record["n_train_residues"] = n_best_train_residues
    return record


def _is_better(score: float, current_best: float, lower_is_more_similar: bool) -> bool:
    if lower_is_more_similar:
        return score < current_best
    return score > current_best


def compute_cross_similarity(
    train: List[SplitRecord],
    test: List[SplitRecord],
    representations: Dict[str, np.ndarray],
    similarity_type: SimilarityType,
    embedding_mode: EmbeddingMode = "cdr",
    n_jobs: int = 1,
    checkpoint_dir: Optional[Path] = None,
    resume: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame, dict]:
    n_test = len(test)
    n_train = len(train)
    train_ids = [item.id for item in train]
    train_reps = [representations[train_id] for train_id in train_ids]
    test_ids = [item.id for item in test]
    lower_is_more_similar = _lower_is_more_similar(similarity_type)
    metric = _summary_metric_name(similarity_type)
    best_field = _best_metric_field(similarity_type)
    extreme = _extreme_prefix(lower_is_more_similar)

    score_mat = np.zeros((n_test, n_train), dtype=np.float64)
    nearest_by_test: Dict[str, dict] = {}
    all_scores: List[float] = []

    if checkpoint_dir is not None and resume:
        completed_rows = _load_checkpoint_rows(checkpoint_dir, test_ids)
        nearest_by_test = _load_checkpoint_nearest(checkpoint_dir)
        for i, test_id in enumerate(test_ids):
            if test_id in completed_rows:
                score_mat[i, :] = completed_rows[test_id]
                all_scores.extend(completed_rows[test_id].tolist())
        print(
            f"Resumed from checkpoint with {len(completed_rows)} completed test rows",
            flush=True,
        )
    else:
        completed_rows = {}

    if lower_is_more_similar:
        best_overall_score = math.inf
    else:
        best_overall_score = -math.inf

    best_overall = {
        "test_id": None,
        "train_id": None,
        "test_seq": None,
        "train_seq": None,
        "score": best_overall_score,
        "n_test_residues": None,
        "n_train_residues": None,
    }

    for record in nearest_by_test.values():
        score = float(record[best_field])
        if _is_better(score, best_overall["score"], lower_is_more_similar):
            best_overall = {
                "test_id": record["test_id"],
                "train_id": record["best_train_id"],
                "test_seq": record["test_seq"],
                "train_seq": record["best_train_seq"],
                "score": score,
                "n_test_residues": record.get("n_test_residues"),
                "n_train_residues": record.get("n_train_residues"),
            }

    test_items = [
        (record.id, record.seq, representations[record.id])
        for record in test
        if record.id not in completed_rows
    ]

    if test_items:
        if checkpoint_dir is not None and not resume:
            nearest_path = checkpoint_dir / "nearest.jsonl"
            if nearest_path.exists():
                nearest_path.unlink()
            rows_dir = checkpoint_dir / "rows"
            if rows_dir.exists():
                for path in rows_dir.glob("*.npy"):
                    path.unlink()

        desc = f"Computing {similarity_type} similarities"
        if n_jobs <= 1:
            pending_results = [
                _compute_test_row(
                    *item,
                    train_ids,
                    train_reps,
                    similarity_type,
                    lower_is_more_similar,
                )
                for item in tqdm(test_items, desc=desc)
            ]
        else:
            pending_results = []
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_init_worker,
                initargs=(train_ids, train_reps, similarity_type, lower_is_more_similar),
            ) as executor:
                futures = {
                    executor.submit(_compute_test_row_worker, *item): item[0]
                    for item in test_items
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    pending_results.append(future.result())

        for (
            test_id,
            test_seq,
            row,
            best_j,
            best_score,
            n_test_residues,
            n_best_train_residues,
        ) in pending_results:
            i = test_ids.index(test_id)
            score_mat[i, :] = row
            all_scores.extend(row.tolist())
            nearest_record = _build_nearest_record(
                test_id,
                test_seq,
                train,
                train_ids,
                best_j,
                best_score,
                similarity_type,
                n_test_residues,
                n_best_train_residues,
            )
            nearest_by_test[test_id] = nearest_record

            if checkpoint_dir is not None:
                _save_checkpoint_row(checkpoint_dir, test_id, row, nearest_record)

            if _is_better(best_score, best_overall["score"], lower_is_more_similar):
                best_overall = {
                    "test_id": test_id,
                    "train_id": train_ids[best_j],
                    "test_seq": test_seq,
                    "train_seq": train[best_j].seq,
                    "score": best_score,
                    "n_test_residues": n_test_residues,
                    "n_train_residues": n_best_train_residues,
                }

    nearest_rows = [nearest_by_test[test_id] for test_id in test_ids]

    summary: dict = {
        "similarity_type": similarity_type,
        "embedding_mode": embedding_mode,
        "lower_is_more_similar": lower_is_more_similar,
        "embedding_source": _embedding_source(embedding_mode),
        "n_train": n_train,
        "n_test": n_test,
        "n_pairs": int(n_train * n_test),
        f"{extreme}_{metric}": float(best_overall["score"]),
        f"{extreme}_{metric}_test_id": best_overall["test_id"],
        f"{extreme}_{metric}_train_id": best_overall["train_id"],
        f"mean_{metric}": float(np.mean(all_scores)) if all_scores else None,
        f"std_{metric}": float(np.std(all_scores)) if all_scores else None,
        f"median_{metric}": float(np.median(all_scores)) if all_scores else None,
        f"p05_{metric}": float(np.quantile(all_scores, 0.05)) if all_scores else None,
        f"p25_{metric}": float(np.quantile(all_scores, 0.25)) if all_scores else None,
        f"p75_{metric}": float(np.quantile(all_scores, 0.75)) if all_scores else None,
        f"p95_{metric}": float(np.quantile(all_scores, 0.95)) if all_scores else None,
    }

    if similarity_type == "wasserstein":
        summary["cost_metric"] = "sqeuclidean"
        summary[f"{extreme}_{metric}_n_test_residues"] = best_overall["n_test_residues"]
        summary[f"{extreme}_{metric}_n_train_residues"] = best_overall["n_train_residues"]
        opposite = "max"
    else:
        opposite = "min"

    if all_scores:
        summary[f"{opposite}_{metric}"] = float(
            np.max(all_scores) if opposite == "max" else np.min(all_scores)
        )

    if representations:
        first = next(iter(representations.values()))
        if similarity_type == "wasserstein":
            summary["embed_dim"] = int(first.shape[1])
        else:
            summary["representation_dim"] = int(first.shape[0])

    if all_scores:
        if similarity_type in {"mean_cosine", "cdr_mean_cosine"}:
            hist_range = (-1.0, 1.0)
        else:
            hist_range = (float(np.min(all_scores)), float(np.max(all_scores)))
            if hist_range[0] == hist_range[1]:
                hist_range = (hist_range[0] - 1.0, hist_range[1] + 1.0)

        hist_counts, hist_edges = np.histogram(all_scores, bins=20, range=hist_range)
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
    return score_mat, nearest_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute train-test embedding similarities on ESM-C antibody embeddings."
    )
    parser.add_argument("--train", required=True, type=Path, help="Train CSV/TSV file")
    parser.add_argument("--test", required=True, type=Path, help="Test CSV/TSV file")
    parser.add_argument("--manifest", required=True, type=Path, help="Embedding manifest CSV")
    parser.add_argument(
        "--seq-col",
        default="concat_CDR",
        help="Sequence column for labels/validation (concat_CDR recommended)",
    )
    parser.add_argument("--id-col", default="pdb_name", help="ID column name")
    parser.add_argument("--outdir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--embedding-mode",
        choices=EMBEDDING_MODES,
        default="cdr",
        help="cdr: embeddings are concat_CDR residues; "
        "vhvl_cdr: load full VH+VL chain and extract CDR slices",
    )
    parser.add_argument(
        "--similarity-type",
        choices=SIMILARITY_TYPES,
        default="wasserstein",
        help="Similarity metric: wasserstein, mean_cosine, or cdr_mean_cosine",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers over test sequences",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint rows in outdir/.checkpoint/<mode>/<type> if present.",
    )
    args = parser.parse_args()

    similarity_type: SimilarityType = args.similarity_type
    embedding_mode: EmbeddingMode = args.embedding_mode
    require_cdr = similarity_type == "cdr_mean_cosine" or embedding_mode == "vhvl_cdr"

    args.outdir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.outdir / ".checkpoint" / embedding_mode / similarity_type

    train = read_split_table(
        args.train, args.id_col, args.seq_col, require_cdr, embedding_mode
    )
    test = read_split_table(
        args.test, args.id_col, args.seq_col, require_cdr, embedding_mode
    )
    if not train:
        raise ValueError("No valid train sequences found after cleaning.")
    if not test:
        raise ValueError("No valid test sequences found after cleaning.")

    manifest = load_manifest(args.manifest)
    needed_ids = {item.id for item in train} | {item.id for item in test}
    missing_ids = sorted(needed_ids - set(manifest.index))
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise FileNotFoundError(
            f"{len(missing_ids)} IDs missing from manifest. Examples: {preview}"
        )

    print(f"Validating {len(needed_ids)} required embeddings...", flush=True)
    validate_embeddings(manifest, needed_ids)

    print(f"Loading {len(needed_ids)} embeddings from {args.manifest}", flush=True)
    with ThreadPoolExecutor(max_workers=min(8, len(needed_ids))) as executor:
        futures = {
            executor.submit(load_embedding, manifest, pdb_name): pdb_name
            for pdb_name in needed_ids
        }
        embeddings: Dict[str, np.ndarray] = {}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading embeddings"):
            pdb_name = futures[future]
            embeddings[pdb_name] = future.result()

    all_records = train + test
    if embedding_mode == "vhvl_cdr":
        print("Extracting CDR embeddings from full VH+VL chains...", flush=True)
        extract_cdr_embeddings_from_full_chain(embeddings, all_records, manifest)

    print(f"Building {similarity_type} representations...", flush=True)
    representations = build_all_representations(embeddings, all_records, similarity_type)

    score_mat, nearest_df, summary = compute_cross_similarity(
        train=train,
        test=test,
        representations=representations,
        similarity_type=similarity_type,
        embedding_mode=embedding_mode,
        n_jobs=args.n_jobs,
        checkpoint_dir=checkpoint_dir,
        resume=args.resume,
    )

    train_ids = [item.id for item in train]
    test_ids = [item.id for item in test]

    score_df = pd.DataFrame(score_mat, index=test_ids, columns=train_ids)
    score_df.to_csv(args.outdir / "similarity_matrix.csv")

    nearest_df.to_csv(args.outdir / "nearest_train_per_test.csv", index=False)

    with open(args.outdir / "similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    metric = _summary_metric_name(similarity_type)
    extreme = _extreme_prefix(summary["lower_is_more_similar"])
    print("Done.", flush=True)
    print(f"Embedding mode:  {summary['embedding_mode']}", flush=True)
    print(f"Similarity type: {summary['similarity_type']}", flush=True)
    print(f"Train sequences: {summary['n_train']}", flush=True)
    print(f"Test sequences:  {summary['n_test']}", flush=True)
    print(f"Pairwise pairs:  {summary['n_pairs']}", flush=True)
    print(f"{extreme.title()} {metric}: {summary[f'{extreme}_{metric}']:.4f}", flush=True)
    print(f"Saved to:        {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
