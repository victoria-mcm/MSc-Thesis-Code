"""Per-metric CDR sequence length rules for filtering intra-cluster pairwise RMSD pairs."""

from __future__ import annotations

import pandas as pd

# CDR metrics that use same-length pair filtering (framework excluded).
SAME_LENGTH_CDR_METRICS: tuple[str, ...] = (
    "H_cdr1",
    "H_cdr2",
    "H_cdr3",
    "H_cdr_all",
    "L_cdr1",
    "L_cdr2",
    "L_cdr3",
    "L_cdr_all",
)

_META_COL = {
    "H_cdr1": "CDRH1",
    "H_cdr2": "CDRH2",
    "H_cdr3": "CDRH3",
    "L_cdr1": "CDRL1",
    "L_cdr2": "CDRL2",
    "L_cdr3": "CDRL3",
}


def _as_seq(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def load_meta_for_lengths(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    if "pdb_name" not in meta.columns:
        raise ValueError(f"pdb_name column required in {meta_path}")
    return meta.drop_duplicates(subset=["pdb_name"], keep="first")


def build_length_lookup(meta: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Map pdb_name -> metric_key -> sequence length used for pair filtering."""
    lookup: dict[str, dict[str, int]] = {}
    for row in meta.itertuples(index=False):
        h_concat = _as_seq(row.CDRH1) + _as_seq(row.CDRH2) + _as_seq(row.CDRH3)
        l_concat = _as_seq(row.CDRL1) + _as_seq(row.CDRL2) + _as_seq(row.CDRL3)
        lengths = {"H_cdr_all": len(h_concat), "L_cdr_all": len(l_concat)}
        for metric, col in _META_COL.items():
            lengths[metric] = len(_as_seq(getattr(row, col)))
        lookup[row.pdb_name] = lengths
    return lookup


def pair_ok_for_metric(
    pdb_name_a: str,
    pdb_name_b: str,
    metric: str,
    length_lookup: dict[str, dict[str, int]],
) -> bool:
    if metric not in SAME_LENGTH_CDR_METRICS:
        return True
    la = length_lookup.get(pdb_name_a)
    lb = length_lookup.get(pdb_name_b)
    if la is None or lb is None:
        return False
    if metric not in la or metric not in lb:
        return False
    return la[metric] == lb[metric]


def filter_ok_pairs_for_metric(
    pair_df: pd.DataFrame,
    metric: str,
    length_lookup: dict[str, dict[str, int]],
) -> pd.DataFrame:
    ok = pair_df[pair_df["status"] == "ok"].copy()
    if metric not in SAME_LENGTH_CDR_METRICS:
        return ok
    mask = [
        pair_ok_for_metric(a, b, metric, length_lookup)
        for a, b in zip(ok["pdb_name_a"], ok["pdb_name_b"])
    ]
    return ok.loc[mask]
