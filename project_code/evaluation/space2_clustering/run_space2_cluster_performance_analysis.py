"""SPACE2 cluster vs test H_cdr3 performance — batch runner (notebook companion)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import binomtest, chi2, kruskal, mannwhitneyu, pearsonr, spearmanr, t as t_dist

COLOR_TEST_ONLY_CLUSTER = "tomato"
COLOR_TRAIN_OVERLAP_CLUSTER = "lightskyblue"

BASE = Path("/opig-shared/users/lina4783")
EVAL = BASE / "abb4_experiments/evaluation/space2_clustering"
OUT_DIR = EVAL / "outputs"
FIG_DIR = EVAL / "figures"

METRICS_PATH = (
    BASE
    / "abb4_experiments/evaluation/predictions_ckpt_5139_imgt/struc_pred_metrics_test_summary.csv"
)
ALL_SAMPLES_PATH = (
    BASE
    / "abb4_experiments/evaluation/predictions_ckpt_5139_imgt/struc_pred_metrics_test_all_samples.csv"
)
# Sequence baseline boxplot port: evaluation_checks.ipynb (all-samples → pdb_summary → sns.boxplot).
EVAL_CHECKS_NOTEBOOK = BASE / "abb4_experiments/evaluation/evaluation_checks.ipynb"
SPACE2_META_PATH = EVAL / "outputs/full/metadata_with_space2.csv"
SPACE2_PRED_META_PATH = EVAL / "outputs/train_plus_pred_test/metadata_with_space2.csv"
TEST_META_PATH = BASE / "structures_final/test_meta.csv"

TARGET = "H_cdr3"
MERGED_OUT = OUT_DIR / "space2_test_performance_merged.csv"
CLUSTER_STATS_OUT = OUT_DIR / "space2_cluster_summary_stats.csv"
CORR_SUMMARY_OUT = OUT_DIR / "space2_performance_correlation_summary.csv"
WITHIN_SPACE2_OUT = OUT_DIR / "space2_within_cluster_stats.csv"
WITHIN_SEQ_OUT = OUT_DIR / "seq_within_cluster_stats.csv"
SPACE2_PLOT_CLUSTER_INDEX_OUT = OUT_DIR / "space2_analysis_a_cluster_plot_index.csv"
PRED_VS_GT_OVERLAP_OUT = OUT_DIR / "space2_pred_vs_gt_train_overlap.csv"
LENGTH_GROUP_OVERLAP_OUT = OUT_DIR / "space2_length_group_train_overlap.csv"
CLUSTER_SIZE_FIG = FIG_DIR / "space2_cluster_size_distribution_train_test.png"
LENGTH_GROUP_OVERLAP_FIG = FIG_DIR / "analysis_e_length_group_train_overlap.png"


def _emit_figure(fig, path: Path | None, show: bool) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def is_space2_assigned(df: pd.DataFrame) -> pd.Series:
    return df["space2_cluster_by_length"].notna() & df["space2_representative"].notna()


def filter_space2_assigned(merged: pd.DataFrame) -> pd.DataFrame:
    """Test rows with a real SPACE2 assignment (excludes synthetic ``nan|nan`` bucket)."""
    return merged.loc[is_space2_assigned(merged)].copy()


def univariate_linear_regression(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(y)
    if n < 3:
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r2": np.nan, "adj_r2": np.nan, "p_value": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - 2)
    df = n - 2
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    se_slope = np.sqrt(ss_res / df / denom) if denom else np.nan
    t_stat = slope / se_slope if se_slope else np.nan
    p_value = 2.0 * t_dist.sf(abs(t_stat), df) if np.isfinite(t_stat) else np.nan
    return {
        "n": n,
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "adj_r2": adj_r2,
        "r2_percent": 100.0 * r2,
        "p_value": p_value,
    }


def correlation_summary(x, y, label: str, cluster_type: str = "space2") -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ols = univariate_linear_regression(x, y)
    pr, pp = pearsonr(x, y) if len(x) > 1 else (np.nan, np.nan)
    sr, sp = spearmanr(x, y) if len(x) > 1 else (np.nan, np.nan)
    return {
        "cluster_type": cluster_type,
        "level": label,
        "pearson_r": pr,
        "pearson_p": pp,
        "spearman_rho": sr,
        "spearman_p": sp,
        **ols,
    }


def make_space2_cluster_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["space2_cluster_by_length"].astype(str)
        + "|"
        + df["space2_representative"].astype(str)
    )


def within_cluster_stats(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    rows = []
    for gid, g in df.groupby(group_col, sort=False):
        y = g[value_col].astype(float)
        y = y[np.isfinite(y)]
        n = len(y)
        if n == 0:
            continue
        q75, q25 = np.percentile(y, [75, 25])
        rows.append(
            {
                group_col: gid,
                "n_test": n,
                "mean": float(y.mean()),
                "median": float(np.median(y)),
                "std": float(y.std(ddof=1)) if n > 1 else 0.0,
                "iqr": float(q75 - q25),
                "min": float(y.min()),
                "max": float(y.max()),
                "range": float(y.max() - y.min()),
            }
        )
    return pd.DataFrame(rows)


def _cluster_x_label(group_col: str, gid) -> str:
    if group_col == "space2_cluster_id":
        s = str(gid)
        return s.split("|", 1)[1] if "|" in s else s
    if isinstance(gid, (int, np.integer)):
        return str(int(gid))
    if isinstance(gid, float) and np.isfinite(gid) and gid == int(gid):
        return str(int(gid))
    return str(gid)


def build_seq_cluster_pdb_summary_evaluation_checks() -> pd.DataFrame:
    """Per-PDB mean H_cdr3 for sequence clusters (clusters with >2 PDB IDs).

    Mirrors ``evaluation_checks.ipynb``: map ``cluster_ids`` from ``test_meta``,
    filter ``cluster_size > 2`` (via all-sample row counts / 10), then
    ``groupby(['cluster_id', 'pdb_name']).mean()``.
    """
    test_meta = pd.read_csv(TEST_META_PATH, usecols=["pdb_name", "cluster_ids"])
    cluster_map = test_meta.set_index("pdb_name")["cluster_ids"]
    rmsds_df = pd.read_csv(ALL_SAMPLES_PATH, usecols=["pdb_name", TARGET, "status"])
    rmsds_df = rmsds_df.loc[rmsds_df["status"].eq("ok")].copy()
    rmsds_df["cluster_id"] = rmsds_df["pdb_name"].map(cluster_map)
    rmsds_df = rmsds_df.dropna(subset=["cluster_id"])

    cluster_sizes = rmsds_df.groupby("cluster_id").size()
    cluster_sizes = cluster_sizes[cluster_sizes > 2]
    rmsds_df["cluster_size"] = np.int64(rmsds_df["cluster_id"].map(cluster_sizes) / 10)
    usable = rmsds_df.loc[rmsds_df["cluster_size"] > 2]

    return (
        usable.groupby(["cluster_id", "pdb_name"], as_index=False)
        .agg(mean_rmsd=(TARGET, "mean"))
    )


def build_space2_pdb_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per (SPACE2 cluster, PDB) with mean_rmsd (= summary H_cdr3)."""
    n_pdbs = merged.groupby("space2_cluster_id")["pdb_name"].transform("nunique")
    usable = merged.loc[n_pdbs > 2].copy()
    summary = (
        usable.groupby(["space2_cluster_id", "pdb_name"], as_index=False)
        .agg(mean_rmsd=(TARGET, "mean"))
    )
    summary = summary.rename(columns={"space2_cluster_id": "cluster_id"})
    return summary


def plot_mean_rmsds_by_cluster(
    pdb_summary: pd.DataFrame,
    title: str,
    out_path: Path | None,
    *,
    use_rep_labels: bool = False,
    numeric_x_labels: bool = False,
    cluster_has_train: pd.Series | None = None,
    plot_index_path: Path | None = None,
    show_figures: bool = False,
) -> int:
    """``mean_rmsds_clusters.png`` style plot from ``evaluation_checks.ipynb``."""
    cluster_means = pdb_summary.groupby("cluster_id")["mean_rmsd"].mean()
    cluster_order = cluster_means.sort_values().index
    n_clusters = len(cluster_order)
    if n_clusters == 0:
        return 0

    plot_df = pdb_summary.copy()
    if numeric_x_labels:
        number_map = {gid: i + 1 for i, gid in enumerate(cluster_order)}
        plot_df["cluster_plot"] = plot_df["cluster_id"].map(number_map)
        x_col = "cluster_plot"
        order = list(range(1, n_clusters + 1))
        index_rows = []
        for plot_num, gid in enumerate(cluster_order, start=1):
            row = {
                "plot_cluster_number": plot_num,
                "space2_cluster_id": gid,
                "representative_pdb": _cluster_x_label("space2_cluster_id", gid),
                "cluster_mean_H_cdr3": float(cluster_means[gid]),
            }
            if cluster_has_train is not None:
                row["has_train_member"] = bool(cluster_has_train.get(gid, False))
            index_rows.append(row)
        index_path = plot_index_path or SPACE2_PLOT_CLUSTER_INDEX_OUT
        pd.DataFrame(index_rows).to_csv(index_path, index=False)
    elif use_rep_labels:
        label_map = {gid: _cluster_x_label("space2_cluster_id", gid) for gid in cluster_order}
        plot_df["cluster_plot"] = plot_df["cluster_id"].map(label_map)
        x_col = "cluster_plot"
        order = [label_map[gid] for gid in cluster_order]
    else:
        x_col = "cluster_id"
        order = list(cluster_order)

    per_cluster_w = 0.045 if numeric_x_labels else 0.09
    fig_w = max(18.0, per_cluster_w * n_clusters)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    default_color = COLOR_TEST_ONLY_CLUSTER if cluster_has_train is not None else "tomato"
    sns.boxplot(
        data=plot_df,
        x=x_col,
        y="mean_rmsd",
        order=order,
        color=default_color,
        fliersize=2,
        ax=ax,
    )
    if cluster_has_train is not None:
        box_colors = [
            COLOR_TRAIN_OVERLAP_CLUSTER if bool(cluster_has_train.get(cid, False)) else COLOR_TEST_ONLY_CLUSTER
            for cid in cluster_order
        ]
        for patch, color in zip(ax.patches[:n_clusters], box_colors):
            patch.set_facecolor(color)
        ax.legend(
            handles=[
                Patch(
                    facecolor=COLOR_TRAIN_OVERLAP_CLUSTER,
                    label="Train structure(s) in SPACE2 cluster",
                ),
                Patch(
                    facecolor=COLOR_TEST_ONLY_CLUSTER,
                    label="No train structures (test-only cluster)",
                ),
            ],
            loc="upper left",
        )
    ax.axhline(y=2.5, color="red", linestyle="--")
    tick_rotation = 90
    plt.setp(ax.get_xticklabels(), rotation=tick_rotation)
    ax.tick_params(axis='x', which='major', labelsize=7)
    if numeric_x_labels:
        ax.set_xlabel("SPACE2 cluster")
    else:
        ax.set_xlabel("SPACE2 cluster")
    ax.set_ylabel("Mean CDRH3 RMSD (n=10 per PDB ID) (Angstroms)")
    ax.set_title(title)
    fig.tight_layout()
    _emit_figure(fig, out_path, show_figures)
    return n_clusters


def plot_sorted_cluster_boxplot(
    df: pd.DataFrame,
    group_col: str,
    title: str,
    out_path: Path,
    min_n: int = 3,
) -> int:
    """Deprecated wrapper — prefer ``plot_mean_rmsds_by_cluster`` + pdb_summary builders."""
    n_pdbs = df.groupby(group_col)["pdb_name"].transform("nunique")
    usable = df.loc[n_pdbs > min_n - 1]
    pdb_summary = (
        usable.groupby([group_col, "pdb_name"], as_index=False)
        .agg(mean_rmsd=(TARGET, "mean"))
        .rename(columns={group_col: "cluster_id"})
    )
    return plot_mean_rmsds_by_cluster(
        pdb_summary,
        title,
        out_path,
        numeric_x_labels=(group_col == "space2_cluster_id"),
    )


def plot_space2_cluster_size_distribution(
    space2_meta: pd.DataFrame,
    fig_path: Path | None = CLUSTER_SIZE_FIG,
    *,
    show_figures: bool = False,
) -> dict:
    """Histogram of SPACE2 cluster sizes (train + test), excluding unassigned structures."""
    cohort = space2_meta.loc[space2_meta["split"].isin(["train", "test"])].copy()
    n_cohort = len(cohort)
    assigned = cohort.loc[is_space2_assigned(cohort)]
    n_unassigned = n_cohort - len(assigned)
    sizes = assigned.groupby("space2_cluster_id")["pdb_name"].nunique()
    sizes.name = "n_members"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(sizes.values, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Structures per SPACE2 cluster")
    axes[0].set_ylabel("Number of clusters")
    axes[0].set_title(f"Cluster size distribution (n={len(sizes)} clusters)")

    positive = sizes.loc[sizes > 0]
    axes[1].hist(positive.values, bins=50, color="steelblue", edgecolor="white")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Structures per cluster (log x)")
    axes[1].set_ylabel("Number of clusters")
    axes[1].set_title("Same distribution (log-x scale)")
    fig.suptitle("SPACE2 cluster sizes — train + test, assigned structures only", y=1.02)
    fig.tight_layout()
    _emit_figure(fig, fig_path, show_figures)

    meta = {
        "n_train_test_structures": n_cohort,
        "n_unassigned_excluded": int(n_unassigned),
        "pct_unassigned": 100.0 * n_unassigned / n_cohort if n_cohort else 0.0,
        "n_space2_clusters": len(sizes),
    }
    print(
        f"SPACE2 cluster sizes: excluded {meta['n_unassigned_excluded']} unassigned "
        f"({meta['pct_unassigned']:.1f}% of train+test metadata)"
    )
    return meta


def load_and_merge() -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "pdb_name",
        "split",
        "cluster_ids",
        "space2_cluster_by_length",
        "space2_representative",
    ]
    space2_meta = pd.read_csv(SPACE2_META_PATH, usecols=usecols)
    space2_meta["space2_cluster_id"] = make_space2_cluster_id(space2_meta)

    metrics = pd.read_csv(METRICS_PATH, usecols=["pdb_name", TARGET, "status"])
    metrics = metrics.loc[metrics["status"].eq("ok")].copy()
    metrics[TARGET] = pd.to_numeric(metrics[TARGET], errors="coerce")
    metrics = metrics.loc[np.isfinite(metrics[TARGET])]

    test_meta = space2_meta.loc[space2_meta["split"].eq("test")].copy()
    merged = metrics.merge(
        test_meta.drop_duplicates("pdb_name"),
        on="pdb_name",
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == len(metrics), "unexpected row loss on merge"

    assigned_meta = space2_meta.loc[is_space2_assigned(space2_meta)]
    train_counts = (
        assigned_meta.loc[assigned_meta["split"].eq("train")]
        .groupby("space2_cluster_id")["pdb_name"]
        .nunique()
        .rename("n_train_space2_cluster")
    )
    test_assigned = test_meta.loc[is_space2_assigned(test_meta)]
    test_counts = (
        test_assigned.groupby("space2_cluster_id")["pdb_name"]
        .nunique()
        .rename("n_test_space2_cluster")
    )
    total_counts = (
        assigned_meta.groupby("space2_cluster_id")["pdb_name"]
        .nunique()
        .rename("n_total_space2_cluster")
    )
    cluster_stats = (
        pd.concat([train_counts, test_counts, total_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    merged = merged.merge(cluster_stats, on="space2_cluster_id", how="left")
    for col in ("n_train_space2_cluster", "n_test_space2_cluster", "n_total_space2_cluster"):
        merged[col] = merged[col].fillna(0).astype(int)
    merged["train_absent_space2"] = merged["n_train_space2_cluster"].eq(0)

    seq_test_size = (
        test_meta.groupby("cluster_ids")["pdb_name"].nunique().rename("n_test_seq_cluster")
    )
    merged = merged.merge(
        seq_test_size.reset_index(),
        on="cluster_ids",
        how="left",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_OUT, index=False)
    cluster_stats.to_csv(CLUSTER_STATS_OUT, index=False)
    return merged, space2_meta


def analysis_a(merged: pd.DataFrame, show_figures: bool = False) -> dict:
    os.makedirs(FIG_DIR, exist_ok=True)
    merged_s2 = filter_space2_assigned(merged)
    n_excluded = len(merged) - len(merged_s2)

    space2_within = within_cluster_stats(merged_s2, "space2_cluster_id", TARGET)
    space2_within = space2_within.merge(
        merged_s2.groupby("space2_cluster_id")["n_train_space2_cluster"].first().reset_index(),
        on="space2_cluster_id",
        how="left",
    )
    space2_within.to_csv(WITHIN_SPACE2_OUT, index=False)

    seq_within = within_cluster_stats(merged, "cluster_ids", TARGET)
    seq_within.to_csv(WITHIN_SEQ_OUT, index=False)

    s2_summary = build_space2_pdb_summary(merged_s2)
    train_flag = (
        merged_s2.groupby("space2_cluster_id")["n_train_space2_cluster"]
        .first()
        .gt(0)
    )
    train_flag.index = train_flag.index.rename("cluster_id")
    n_space2_plotted = plot_mean_rmsds_by_cluster(
        s2_summary,
        "Performance (CDRH3 RMSD) by SPACE2 cluster",
        FIG_DIR / "analysis_a_space2_within_cluster_boxplots.png",
        numeric_x_labels=True,
        cluster_has_train=train_flag,
        show_figures=show_figures,
    )
    seq_pdb_summary = build_seq_cluster_pdb_summary_evaluation_checks()
    n_seq_plotted = plot_mean_rmsds_by_cluster(
        seq_pdb_summary,
        "Performance (CDRH3 RMSD) by Cluster",
        FIG_DIR / "analysis_a_seq_within_cluster_boxplots.png",
        show_figures=show_figures,
    )

    kw_space2 = kw_seq = np.nan
    kw_space2_p = kw_seq_p = np.nan
    groups_s = [
        g[TARGET].values
        for _, g in merged_s2.groupby("space2_cluster_id")
        if len(g) > 2
    ]
    if len(groups_s) >= 2:
        kw_space2, kw_space2_p = kruskal(*groups_s)

    groups_c = [
        g[TARGET].values
        for _, g in merged.groupby("cluster_ids")
        if len(g) > 2
    ]
    if len(groups_c) >= 2:
        kw_seq, kw_seq_p = kruskal(*groups_c)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, within, col in [
        ("SPACE2", space2_within, "std"),
        ("sequence cluster_ids", seq_within, "std"),
    ]:
        sub = within.loc[within["n_test"] > 2]
        ax.hist(
            sub[col],
            bins=30,
            alpha=0.5,
            label=f"{label} (n={len(sub)} clusters)",
            density=True,
        )
    ax.set_xlabel(f"within-cluster std({TARGET})")
    ax.set_ylabel("density")
    ax.set_title("Distribution of within-cluster spread (n_test>2)")
    ax.legend()
    fig.tight_layout()
    _emit_figure(fig, FIG_DIR / "analysis_a_within_std_comparison.png", show_figures)

    return {
        "n_test_excluded_unassigned_space2": n_excluded,
        "n_space2_clusters_plotted": n_space2_plotted,
        "n_seq_clusters_plotted": n_seq_plotted,
        "kruskal_space2_H": kw_space2,
        "kruskal_space2_p": kw_space2_p,
        "kruskal_seq_H": kw_seq,
        "kruskal_seq_p": kw_seq_p,
    }


def analysis_b(merged: pd.DataFrame, show_figures: bool = False) -> list[dict]:
    merged = filter_space2_assigned(merged)
    x_col = "n_train_space2_cluster"
    y = merged[TARGET].values
    x = merged[x_col].values
    stats_rows = [
        correlation_summary(x, y, "per_sample", "space2"),
    ]

    cluster_df = (
        merged.groupby("space2_cluster_id", as_index=False)
        .agg(
            n_samples=("pdb_name", "count"),
            H_cdr3_median=(TARGET, "median"),
            n_train_space2_cluster=(x_col, "first"),
            n_test_space2_cluster=("n_test_space2_cluster", "first"),
        )
    )
    stats_rows.append(
        correlation_summary(
            cluster_df["n_train_space2_cluster"],
            cluster_df["H_cdr3_median"],
            "per_cluster_median",
            "space2",
        )
    )

    zero = merged.loc[merged[x_col].eq(0), TARGET]
    pos = merged.loc[merged[x_col].gt(0), TARGET]
    if len(zero) and len(pos):
        u, p = mannwhitneyu(zero, pos, alternative="two-sided")
        stats_rows.append(
            {
                "cluster_type": "space2",
                "level": "mannwhitney_n_train_0_vs_gt0",
                "n": len(zero) + len(pos),
                "mannwhitney_u": u,
                "mannwhitney_p": p,
                "median_n0": float(zero.median()),
                "median_n_gt0": float(pos.median()),
            }
        )

    os.makedirs(FIG_DIR, exist_ok=True)
    global_median = float(np.median(y))
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = merged.loc[merged[x_col].notna()].copy()
    order = sorted(v for v in plot_df[x_col].unique() if np.isfinite(v))
    if len(order) > 15:
        order = order[:15]
        plot_df = plot_df.loc[plot_df[x_col].isin(order)]
    data = [plot_df.loc[plot_df[x_col] == v, TARGET].values for v in order]
    ax.boxplot(data, positions=range(len(order)), widths=0.6, showfliers=False)
    rng = np.random.default_rng(0)
    for i, v in enumerate(order):
        ys = plot_df.loc[plot_df[x_col] == v, TARGET].values
        jitter = rng.uniform(-0.15, 0.15, size=len(ys))
        ax.scatter(np.full(len(ys), i) + jitter, ys, s=10, alpha=0.4, edgecolors="none")
    ax.axhline(global_median, color="gray", ls="--", lw=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(int(v)) for v in order])
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{TARGET} (Å)")
    ax.set_title("Test H_cdr3 vs SPACE2 train-set cluster size (first 15 counts if many)")
    fig.tight_layout()
    _emit_figure(fig, FIG_DIR / "analysis_b_train_coverage_boxplot.png", show_figures)

    fig, ax = plt.subplots(figsize=(8, 5))
    jitter = rng.uniform(-0.15, 0.15, size=len(x))
    ax.scatter(x + jitter, y, s=12, alpha=0.35, edgecolors="none")
    ols = univariate_linear_regression(x, y)
    xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    ax.plot(xline, ols["slope"] * xline + ols["intercept"], color="C3", lw=2)
    sr, sp = spearmanr(x, y)
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{TARGET} (Å)")
    ax.set_title(f"SPACE2 train coverage vs {TARGET} (Spearman rho={sr:.3f}, p={sp:.2g})")
    fig.tight_layout()
    _emit_figure(fig, FIG_DIR / "analysis_b_train_coverage_scatter.png", show_figures)

    return stats_rows


def analysis_c(merged: pd.DataFrame, show_figures: bool = False) -> list[dict]:
    merged = filter_space2_assigned(merged)
    rows = []
    absent = merged.loc[merged["train_absent_space2"]]
    present = merged.loc[~merged["train_absent_space2"]]
    if len(absent) and len(present):
        u, p = mannwhitneyu(absent[TARGET], present[TARGET], alternative="two-sided")
        rows.append(
            {
                "cluster_type": "space2",
                "level": "train_absent_vs_present_structures",
                "n_absent": len(absent),
                "n_present": len(present),
                "median_absent": float(absent[TARGET].median()),
                "median_present": float(present[TARGET].median()),
                "mannwhitney_u": u,
                "mannwhitney_p": p,
            }
        )

    test_only_clusters = merged.loc[merged["n_train_space2_cluster"].eq(0), "space2_cluster_id"].nunique()
    rows.append(
        {
            "cluster_type": "space2",
            "level": "test_only_space2_cluster_count",
            "n_clusters": int(test_only_clusters),
            "n_test_structures": int(len(absent)),
        }
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        [present[TARGET].values, absent[TARGET].values],
        tick_labels=["train present in SPACE2 cluster", "train absent (test-only cluster)"],
    )
    ax.set_ylabel(f"CDRH3 RMSD (Å)")
    ax.set_title("CDRH3 RMSD by SPACE2 train presence")
    fig.tight_layout()
    _emit_figure(fig, FIG_DIR / "analysis_c_train_absent_boxplot.png", show_figures)

    return rows


def _test_train_share_table(space2_meta: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Per test structure: SPACE2 cluster id and train-member count for one clustering."""
    meta = space2_meta.copy()
    meta["space2_cluster_id"] = make_space2_cluster_id(meta)
    assigned = meta.loc[is_space2_assigned(meta)]
    train_counts = (
        assigned.loc[assigned["split"].eq("train")]
        .groupby("space2_cluster_id")["pdb_name"]
        .nunique()
        .rename(f"n_train_{prefix}")
    )
    test = assigned.loc[assigned["split"].eq("test"), ["pdb_name", "space2_cluster_id"]].drop_duplicates(
        "pdb_name"
    )
    test = test.merge(train_counts.reset_index(), on="space2_cluster_id", how="left")
    test[f"n_train_{prefix}"] = test[f"n_train_{prefix}"].fillna(0).astype(int)
    test[f"shares_train_{prefix}"] = test[f"n_train_{prefix}"].gt(0)
    return test.rename(columns={"space2_cluster_id": f"space2_cluster_id_{prefix}"})


def _mcnemar_two_sided(n01: int, n10: int) -> dict:
    """McNemar test on discordant pairs (n01 = GT yes/pred no, n10 = GT no/pred yes)."""
    n_disc = n01 + n10
    if n_disc == 0:
        return {"mcnemar_n_discordant": 0, "mcnemar_p": 1.0, "mcnemar_stat": 0.0}
    # Exact binomial test of P(pred yes | discordant) = 0.5
    result = binomtest(n10, n=n_disc, p=0.5, alternative="two-sided")
    stat = ((abs(n01 - n10) - 1) ** 2) / n_disc if n_disc else np.nan
    return {
        "mcnemar_n_discordant": n_disc,
        "mcnemar_stat": float(stat),
        "mcnemar_p": float(result.pvalue),
        "chi2_p_continuity": float(chi2.sf(stat, 1)) if np.isfinite(stat) else np.nan,
    }


def analysis_length_group_train_overlap(
    gt_meta: pd.DataFrame | None = None,
    pred_meta: pd.DataFrame | None = None,
    show_figures: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pre-clustering ceiling: test structures in train-seen SPACE2 length groups."""
    usecols = ["pdb_name", "split", "space2_cluster_by_length", "space2_representative"]
    if gt_meta is None:
        gt_meta = pd.read_csv(SPACE2_META_PATH, usecols=usecols)
    else:
        gt_meta = gt_meta[usecols].copy()

    datasets: list[tuple[str, pd.DataFrame]] = [("gt", gt_meta)]
    if pred_meta is not None:
        datasets.append(("pred", pred_meta[usecols].copy()))
    elif SPACE2_PRED_META_PATH.is_file():
        datasets.append(("pred", pd.read_csv(SPACE2_PRED_META_PATH, usecols=usecols)))

    test_tables: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for source, meta in datasets:
        assigned = meta.loc[is_space2_assigned(meta)].copy()
        train_groups = set(
            assigned.loc[assigned["split"].eq("train"), "space2_cluster_by_length"].astype(str)
        )
        test = assigned.loc[assigned["split"].eq("test"), ["pdb_name", "space2_cluster_by_length"]].copy()
        test["space2_cluster_by_length"] = test["space2_cluster_by_length"].astype(str)
        test = test.drop_duplicates("pdb_name")
        test["in_train_length_group"] = test["space2_cluster_by_length"].isin(train_groups)
        test["source"] = source
        test_tables.append(test)

        test_groups = set(test["space2_cluster_by_length"])
        n_test = len(test)
        n_in_train = int(test["in_train_length_group"].sum())
        n_not_in_train = n_test - n_in_train
        summary_rows.append(
            {
                "source": source,
                "n_assigned_test_structures": n_test,
                "n_test_in_train_length_group": n_in_train,
                "frac_test_in_train_length_group": (n_in_train / n_test) if n_test else np.nan,
                "n_test_not_in_train_length_group": n_not_in_train,
                "n_train_length_groups": len(train_groups),
                "n_test_length_groups": len(test_groups),
                "n_shared_length_groups": len(train_groups & test_groups),
                "n_test_only_length_groups": len(test_groups - train_groups),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    test_df = pd.concat(test_tables, ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(LENGTH_GROUP_OVERLAP_OUT, index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Ground-truth test" if s == "gt" else "Predicted test" for s in summary_df["source"]]
    fracs = summary_df["frac_test_in_train_length_group"].astype(float).values
    counts = summary_df["n_test_in_train_length_group"].astype(int).values
    totals = summary_df["n_assigned_test_structures"].astype(int).values
    colors = ["tomato" if s == "gt" else "lightskyblue" for s in summary_df["source"]]
    bars = ax.bar(labels, fracs, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fraction of test in train-seen CDR-length group")
    ax.set_ylim(0, min(1.0, max(fracs) * 1.2 + 0.05))
    ax.set_title("SPACE2 pre-clustering CDR-length group overlap")
    for bar, count, total, frac in zip(bars, counts, totals, fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}/{total}\n({100*frac:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    _emit_figure(fig, LENGTH_GROUP_OVERLAP_FIG, show_figures)

    print("Analysis E (length-group overlap):", json.dumps(summary_rows, indent=2, default=str))
    return test_df, summary_df


def analysis_d(gt_meta: pd.DataFrame | None = None, show_figures: bool = False) -> tuple[pd.DataFrame, dict]:
    """Compare GT vs predicted test structures sharing a SPACE2 cluster with train."""
    if not SPACE2_PRED_META_PATH.is_file():
        raise FileNotFoundError(
            f"Predicted-test SPACE2 merge not found: {SPACE2_PRED_META_PATH}\n"
            "Run run_space2_train_plus_pred.sbatch first."
        )
    usecols = ["pdb_name", "split", "space2_cluster_by_length", "space2_representative"]
    if gt_meta is None:
        gt_meta = pd.read_csv(SPACE2_META_PATH, usecols=usecols)
    else:
        gt_meta = gt_meta[usecols].copy()
    pred_meta = pd.read_csv(SPACE2_PRED_META_PATH, usecols=usecols)

    gt_test = _test_train_share_table(gt_meta, "gt")
    pred_test = _test_train_share_table(pred_meta, "pred")
    paired = gt_test.merge(pred_test, on="pdb_name", how="inner")
    if paired.empty:
        raise ValueError("No overlapping assigned test pdb_names between GT and pred SPACE2 runs")

    gt_share = paired["shares_train_gt"]
    pred_share = paired["shares_train_pred"]
    n = len(paired)
    n_gt = int(gt_share.sum())
    n_pred = int(pred_share.sum())
    n_both = int((gt_share & pred_share).sum())
    n_neither = int((~gt_share & ~pred_share).sum())
    n_gt_only = int((gt_share & ~pred_share).sum())
    n_pred_only = int((~gt_share & pred_share).sum())
    novel_gt = paired.loc[~gt_share]
    n_novel_gt = len(novel_gt)
    n_novel_pulled = int(novel_gt["shares_train_pred"].sum())

    mcnemar = _mcnemar_two_sided(n_gt_only, n_pred_only)
    summary = {
        "n_paired_test": n,
        "n_gt_share_train": n_gt,
        "n_pred_share_train": n_pred,
        "frac_gt_share_train": n_gt / n,
        "frac_pred_share_train": n_pred / n,
        "n_both_share": n_both,
        "n_neither_share": n_neither,
        "n_gt_only": n_gt_only,
        "n_pred_only": n_pred_only,
        "n_gt_novel": n_novel_gt,
        "n_gt_novel_pred_shares_train": n_novel_pulled,
        "frac_gt_novel_pred_shares_train": (n_novel_pulled / n_novel_gt) if n_novel_gt else np.nan,
        **mcnemar,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired.to_csv(PRED_VS_GT_OVERLAP_OUT, index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Ground-truth test", "Predicted test"]
    counts = [n_gt, n_pred]
    fracs = [n_gt / n, n_pred / n]
    bars = ax.bar(labels, fracs, color=["tomato", "lightskyblue"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Fraction sharing a SPACE2 cluster with train")
    ax.set_ylim(0, min(1.0, max(fracs) * 1.25 + 0.05))
    ax.set_title(f"Train-cluster overlap (n={n} paired test structures)")
    for bar, count, frac in zip(bars, counts, fracs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}/{n}\n({100 * frac:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    _emit_figure(fig, FIG_DIR / "analysis_d_train_overlap_bar.png", show_figures)

    table = np.array([[n_both, n_gt_only], [n_pred_only, n_neither]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        table,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=["Pred shares train", "Pred test-only"],
        yticklabels=["GT shares train", "GT test-only"],
        ax=ax,
    )
    ax.set_title(f"Paired train-overlap (McNemar p={summary['mcnemar_p']:.2g})")
    fig.tight_layout()
    _emit_figure(fig, FIG_DIR / "analysis_d_train_overlap_2x2.png", show_figures)

    print("Analysis D:", json.dumps(summary, indent=2, default=str))
    return paired, summary


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    merged, space2_meta = load_and_merge()
    print(f"Merged test rows: {len(merged)} -> {MERGED_OUT}")
    plot_space2_cluster_size_distribution(space2_meta)

    a_meta = analysis_a(merged)
    print("Analysis A:", json.dumps(a_meta, indent=2, default=str))

    stats = analysis_b(merged)
    stats.extend(analysis_c(merged))

    corr_df = pd.DataFrame(stats)
    corr_df.to_csv(CORR_SUMMARY_OUT, index=False)
    print(f"Saved summary: {CORR_SUMMARY_OUT}")
    print(corr_df.to_string(index=False))

    d_summary = None
    if SPACE2_PRED_META_PATH.is_file():
        _, d_summary = analysis_d(gt_meta=space2_meta, show_figures=False)
    else:
        print(
            f"Skipping Analysis D (predicted SPACE2 merge missing): {SPACE2_PRED_META_PATH}"
        )
    _, e_summary = analysis_length_group_train_overlap(gt_meta=space2_meta, show_figures=False)
    return merged, corr_df, a_meta, d_summary, e_summary


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
