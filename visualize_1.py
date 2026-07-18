import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR     = Path("general_result")
CLUSTERING_DIR  = Path("cluster_res")
ENRICHMENT_DIR  = Path("ppi_res")
PERMUTATION_DIR = Path("results/sc2gc_stats")
MUTATION_DIR    = Path("F:/TruongHongKiet/chạy_code/data/TCGA Cancer")
OUTPUT_DIR      = Path("figures_p1_final_new")

GENE_RESULT_FILENAME = "sc2gc_results_1.csv"
GENE_COLUMN_NAME     = "Genes"

CLUSTERING_KNOWN_SUFFIX = "known"
CLUSTERING_OTHER_SUFFIX = "other"

HIGHLIGHT_CANCERS = [
    "Breast Invasive Carcinoma (BRCA)",
    "Colorectal Adenocarcinoma (COADREAD)",
    "Glioblastoma Multiforme (GBM)",
    "Brain Lower Grade Glioma (LGG)",
    "Liver Hepatocellular Carcinoma (LIHC)",
    "Acute Myeloid Leukemia (LAML)",
]

PALETTE = {
    "primary":   "#2D6A4F",
    "secondary": "#52B788",
    "accent":    "#D4A017",
    "danger":    "#C1121F",
    "light":     "#F0F4F8",
    "dark":      "#1A1A2E",
    "lloyd":     "#4361EE",
    "gmm":       "#F72585",
    "fcm":       "#7209B7",
    "known":     "#06D6A0",
    "other":     "#FFB703",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cancer_list(base_dir: Path) -> list[str]:
    if not base_dir.exists():
        print(f"[WARN] Directory not found: {base_dir}")
        return []
    return sorted([d.name for d in base_dir.iterdir() if d.is_dir()])


def load_gene_list(cancer: str) -> list[str]:
    path = RESULTS_DIR / cancer / GENE_RESULT_FILENAME
    if not path.exists():
        files = list((RESULTS_DIR / cancer).glob("*.csv"))
        if not files:
            return []
        path = files[0]
    try:
        df = pd.read_csv(path)
        col = GENE_COLUMN_NAME if GENE_COLUMN_NAME in df.columns else df.columns[0]
        return df[col].dropna().str.strip().tolist()
    except Exception as e:
        print(f"[WARN] load_gene_list({cancer}): {e}")
        return []


def load_clustering(cancer: str, matrix_type: str = "known") -> pd.DataFrame | None:
    folder = CLUSTERING_DIR / cancer
    if not folder.exists():
        return None

    filename = f"{matrix_type}_interactions_all_labels.csv"
    file_path = folder / filename

    if not file_path.exists():
        files = list(folder.glob(f"*{matrix_type}*interactions*all_labels*.csv"))
        if not files:
            return None
        file_path = files[0]

    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        if df.columns[0] != "Gene" and "Sample_Name" in df.columns[0] or df.columns[0] == "Sample_Name":
            df = df.rename(columns={df.columns[0]: "Gene"})
        return df
    except Exception as e:
        print(f"[WARN] load_clustering({cancer}, {matrix_type}): {e}")
        return None


def parse_permutation_report(cancer: str) -> dict:
    result = {
        "cancer":               cancer,
        "n_genes":              None,
        "original_pvalue":      None,
        "n_overlap_groundtruth": None,
        "avg_pvalue_shuffle":   None,
        "better_shuffles":      None,
        "empirical_pvalue":     None,
        "total_permutations":   None,
    }

    txt_files = []
    folder = PERMUTATION_DIR / cancer
    if folder.exists():
        txt_files = list(folder.glob("*.txt")) + list(folder.glob("*.log"))

    if not txt_files and PERMUTATION_DIR.exists():
        for f in PERMUTATION_DIR.glob("*.txt"):
            if f"({cancer})" in f.name or f"{cancer}" in f.name:
                txt_files.append(f)

    if not txt_files:
        return result

    content = txt_files[0].read_text(encoding="utf-8", errors="ignore")

    patterns = {
        "n_genes":               r"Original results:\s*(\d+)\s*genes",
        "original_pvalue":       r"P-value of original matrix:\s*([\deE+\-\.]+)",
        "n_overlap_groundtruth": r"Number of genes overlap with ground truth.*?:\s*(\d+)",
        "avg_pvalue_shuffle":    r"Average p-value:\s*([\deE+\-\.]+)",
        "better_shuffles":       r"Better Shuffles Found:\s*(\d+)",
        "empirical_pvalue":      r"Final Empirical P-value:\s*([\deE+\-\.]+)",
        "total_permutations":    r"Total Completed Permutations:\s*(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            try:
                result[key] = float(m.group(1))
            except Exception:
                result[key] = m.group(1)

    return result


def load_enrichment(cancer: str) -> pd.DataFrame | None:
    folder = ENRICHMENT_DIR / cancer
    if not folder.exists():
        return None

    dfs = []
    for matrix_folder in folder.iterdir():
        if not matrix_folder.is_dir():
            continue
        matrix_type = matrix_folder.name

        for algo_folder in matrix_folder.iterdir():
            if not algo_folder.is_dir():
                continue
            algo_name = algo_folder.name.replace("_label", "")
            file_path = algo_folder / "p-value_sum.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    df["matrix_type"] = matrix_type
                    df["algorithm"]   = algo_name
                    df["source_file"] = "p-value_sum.csv"
                    dfs.append(df)
                except Exception as e:
                    print(f"[WARN] Error reading {file_path}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_mutation_frequency(cancer: str) -> pd.Series | None:
    if not MUTATION_DIR or not MUTATION_DIR.exists():
        return None

    cancer_dir = MUTATION_DIR / cancer
    match = re.search(r'\((.*?)\)', cancer)
    prefix = match.group(1) if match else cancer

    expected_file = cancer_dir / f"{prefix}_mutation_matched_expression_zscore_all.csv"
    candidates = []
    if expected_file.exists():
        candidates.append(expected_file)
    elif cancer_dir.exists():
        candidates = list(cancer_dir.glob("*_mutation_matched_expression_zscore_all.csv"))

    if not candidates:
        return None

    try:
        freq = None
        n_chunks = 0
        for chunk in pd.read_csv(candidates[0], index_col=0, chunksize=500):
            chunk_mean = chunk.mean(numeric_only=True)
            freq = chunk_mean if freq is None else freq.add(chunk_mean, fill_value=0)
            n_chunks += 1
        return freq / n_chunks
    except Exception as e:
        print(f"[WARN] load_mutation_frequency({cancer}): {e}")
        return None


def save_fig(fig: plt.Figure, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(path).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Permutation test empirical p-value (all cancers)
# ---------------------------------------------------------------------------

def fig1_permutation_pvalue(all_cancers: list[str]):
    print("\n[Figure 1] Permutation test empirical p-values...")

    records = [parse_permutation_report(c) for c in all_cancers]
    df = pd.DataFrame(records).dropna(subset=["empirical_pvalue"])

    if df.empty:
        print("  [SKIP] No permutation data available.")
        return

    df["label"] = df["cancer"].apply(
        lambda c: re.search(r'\((.*?)\)', c).group(1) if re.search(r'\((.*?)\)', c) else c
    )
    df = df.sort_values("empirical_pvalue", ascending=True)
    df["highlight"] = df["label"].isin(HIGHLIGHT_CANCERS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    colors = [PALETTE["primary"] if h else PALETTE["secondary"] for h in df["highlight"]]
    ax.bar(range(len(df)), df["empirical_pvalue"], color=colors,
           edgecolor="white", linewidth=0.5, width=0.75)
    ax.axhline(0.05, color=PALETTE["danger"], linestyle="--", linewidth=1.5,
               label="p = 0.05 threshold", zorder=5)

    for i, (_, row) in enumerate(df.iterrows()):
        if row["empirical_pvalue"] == 0.0:
            ax.text(i, 0.002, "p=0", ha="center", va="bottom",
                    fontsize=6.5, color="white", fontweight="bold", rotation=90)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Empirical P-value", fontsize=12, labelpad=8)
    ax.set_xlabel("Cancer Type", fontsize=12, labelpad=8)
    ax.set_title("Empirical P-values from 1,000 Random Permutation Tests",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(-0.005, max(0.08, df["empirical_pvalue"].max() * 1.15))
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(color=PALETTE["primary"],   label="Highlighted cancers (n=5)"),
        mpatches.Patch(color=PALETTE["secondary"], label="Other cancers"),
        plt.Line2D([0], [0], color=PALETTE["danger"], linestyle="--", label="p = 0.05"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    ax2 = axes[1]
    ax2.axis("off")

    n_zero  = (df["empirical_pvalue"] == 0).sum()
    n_sig   = (df["empirical_pvalue"] < 0.05).sum()
    n_total = len(df)

    summary_data = [
        ["Total cancer types analyzed",  f"{n_total}"],
        ["Empirical p-value = 0.000",    f"{n_zero} ({n_zero / n_total * 100:.1f}%)"],
        ["Empirical p-value < 0.05",     f"{n_sig} ({n_sig / n_total * 100:.1f}%)"],
        ["Mean empirical p-value",       f"{df['empirical_pvalue'].mean():.4f}"],
        ["Median empirical p-value",     f"{df['empirical_pvalue'].median():.4f}"],
    ]
    if "original_pvalue" in df.columns and df["original_pvalue"].notna().any():
        mean_orig = df["original_pvalue"].dropna().mean()
        summary_data.append(["Mean original p-value", f"{mean_orig:.2e}"])

    tbl = ax2.table(cellText=summary_data, colLabels=["Metric", "Value"],
                    cellLoc="left", loc="center", colWidths=[0.65, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(PALETTE["dark"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4F8")
        cell.set_edgecolor("#CCCCCC")
    ax2.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    save_fig(fig, "figure1_permutation_pvalue.png")


# ---------------------------------------------------------------------------
# Figure 2 — PPI enrichment q-value (highlighted cancers)
# ---------------------------------------------------------------------------

def fig2_ppi_enrichment(cancers: list[str]):
    print("\n[Figure 2] PPI enrichment q-values...")

    all_rows = []
    for cancer in cancers:
        df = load_enrichment(cancer)
        if df is None:
            continue
        df["cancer"] = cancer
        all_rows.append(df)

    if not all_rows:
        print("  [SKIP] No enrichment data available.")
        return

    data = pd.concat(all_rows, ignore_index=True)

    col_map = {}
    for c in data.columns:
        cl = c.lower().strip()
        if "q_value" in cl or "q-value" in cl or cl == "qvalue":
            col_map[c] = "q_value"
        elif "p_value" in cl or "p-value" in cl or cl == "pvalue":
            col_map[c] = "p_value"
        elif "algorithm" in cl or "method" in cl or "cluster_algo" in cl:
            col_map[c] = "algorithm"
        elif "matrix" in cl or "interaction" in cl:
            col_map[c] = "matrix_type"
        elif "cluster" in cl and "id" in cl:
            col_map[c] = "cluster_id"
    data = data.rename(columns=col_map)

    val_col = "q_value" if "q_value" in data.columns else "p_value"
    if val_col not in data.columns:
        print("  [SKIP] Column q_value/p_value not found.")
        return

    data["-log10_q"] = -np.log10(data[val_col].clip(lower=1e-300))

    algo_col   = "algorithm"   if "algorithm"   in data.columns else None
    matrix_col = "matrix_type" if "matrix_type" in data.columns else "source_file"

    n_cols = 4
    n_rows = (len(cancers) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows), sharey=True)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    threshold_line = -np.log10(0.05)

    for i, ax in enumerate(axes):
        if i >= len(cancers):
            ax.axis("off")
            continue

        cancer = cancers[i]
        sub = data[data["cancer"] == cancer]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cancer)
            continue

        if algo_col and matrix_col in data.columns:
            grouped = sub.groupby([algo_col, matrix_col])["-log10_q"].mean().reset_index()
            algos    = grouped[algo_col].unique()
            matrices = [CLUSTERING_KNOWN_SUFFIX, CLUSTERING_OTHER_SUFFIX]
            x = np.arange(len(algos))
            width = 0.35
            for j, mat in enumerate(matrices):
                vals = [
                    grouped[
                        (grouped[algo_col] == a)
                        & (grouped[matrix_col].str.contains(mat, case=False))
                    ]["-log10_q"].mean()
                    if not grouped[
                        (grouped[algo_col] == a)
                        & (grouped[matrix_col].str.contains(mat, case=False))
                    ].empty
                    else 0
                    for a in algos
                ]
                color = PALETTE["known"] if mat == CLUSTERING_KNOWN_SUFFIX else PALETTE["other"]
                ax.bar(x + j * width, vals, width, label=f"{mat.capitalize()} interactions",
                       color=color, edgecolor="white", alpha=0.9)
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels([a[:5] for a in algos], fontsize=9)
        else:
            vals = sub["-log10_q"].values
            ax.bar(range(len(vals)), vals, color=PALETTE["primary"], edgecolor="white", alpha=0.9)

        ax.axhline(threshold_line, color=PALETTE["danger"], linestyle="--",
                   linewidth=1.2, label="q = 0.05")
        ax.set_title(cancer, fontsize=12, fontweight="bold")
        ax.set_xlabel("Clustering Algorithm", fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    for i in range(0, len(cancers), n_cols):
        if i < len(axes):
            axes[i].set_ylabel("-log₁₀(q-value)", fontsize=12)

    handles = [
        mpatches.Patch(color=PALETTE["known"],  label="Known interactions"),
        mpatches.Patch(color=PALETTE["other"],  label="Other interactions"),
        plt.Line2D([0], [0], color=PALETTE["danger"], linestyle="--", label="q = 0.05"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    save_fig(fig, "figure2_ppi_enrichment.png")


def fig2_ppi_enrichment_new(
    cancers: list[str],
    i: int,
    load_enrichment_fn=None,
    CLUSTERING_KNOWN_SUFFIX: str = "known",
    CLUSTERING_OTHER_SUFFIX: str = "other",
    PALETTE: dict = None,
    OUTPUT_DIR: Path = Path("figures"),
    save_fig_fn=None,
):
    """PPI enrichment plot with broken y-axis and value annotations."""
    if PALETTE is None:
        PALETTE = {
            "known":   "#06D6A0",
            "other":   "#FFB703",
            "danger":  "#C1121F",
            "primary": "#2D6A4F",
        }

    print("\n[Figure 2] PPI enrichment q-values (broken axis)...")

    all_rows = []
    for cancer in cancers:
        df = load_enrichment_fn(cancer)
        if df is None:
            continue
        df["cancer"] = cancer
        all_rows.append(df)

    if not all_rows:
        print("  [SKIP] No enrichment data available.")
        return

    data = pd.concat(all_rows, ignore_index=True)

    col_map = {}
    for c in data.columns:
        cl = c.lower().strip()
        if any(x in cl for x in ("q_value", "q-value", "qvalue")):
            col_map[c] = "q_value"
        elif any(x in cl for x in ("p_value", "p-value", "pvalue")):
            col_map[c] = "p_value"
        elif any(x in cl for x in ("algorithm", "method", "algo")):
            col_map[c] = "algorithm"
        elif any(x in cl for x in ("matrix", "interaction")):
            col_map[c] = "matrix_type"
    data = data.rename(columns=col_map)

    val_col = (
        "q_value" if "q_value" in data.columns
        else "p_value" if "p_value" in data.columns
        else None
    )
    if val_col is None:
        print("  [SKIP] Column q_value/p_value not found.")
        return

    data["-log10_q"] = -np.log10(data[val_col].clip(lower=1e-300))

    algo_col   = "algorithm"   if "algorithm"   in data.columns else None
    matrix_col = "matrix_type" if "matrix_type" in data.columns else "source_file"

    all_vals   = data["-log10_q"].replace([np.inf, -np.inf], np.nan).dropna()
    global_max = all_vals.max()
    pct85      = float(np.percentile(all_vals, 85))
    break_lower  = max(10.0, np.ceil(pct85 / 5) * 5)
    y_bottom_max = break_lower * 1.05
    y_top_min    = break_lower
    y_top_max    = global_max * 1.12
    threshold_line = -np.log10(0.05)

    n_cols   = 4
    n_rows   = (len(cancers) + n_cols - 1) // n_cols
    fig      = plt.figure(figsize=(5.5 * n_cols, 6.5 * n_rows))
    outer_gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    algos_nice = {"lloyd": "Lloyd", "gmm": "GMM", "fcm": "FCM"}
    bar_width  = 0.32

    for idx, cancer in enumerate(cancers):
        row_i = idx // n_cols
        col_i = idx % n_cols

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer_gs[row_i, col_i],
            height_ratios=[1, 2.5],
            hspace=0.06,
        )
        ax_top = fig.add_subplot(inner_gs[0])
        ax_bot = fig.add_subplot(inner_gs[1])

        sub   = data[data["cancer"] == cancer]
        m     = re.search(r'\(([^)]+)\)', cancer)
        short = m.group(1) if m else cancer

        if sub.empty:
            for ax in (ax_top, ax_bot):
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.axis("off")
            ax_top.set_title(short, fontsize=11, fontweight="bold")
            continue

        if algo_col and matrix_col in data.columns:
            raw_algos = sub[algo_col].str.lower().unique()
            algos = [a for a in ("lloyd", "gmm", "fcm") if a in raw_algos] or list(raw_algos)
            mats  = [CLUSTERING_KNOWN_SUFFIX, CLUSTERING_OTHER_SUFFIX]
            x     = np.arange(len(algos))

            bar_data = {}
            for a in algos:
                for mat in mats:
                    mask = (
                        sub[algo_col].str.lower().str.contains(a)
                        & sub[matrix_col].str.lower().str.contains(mat)
                    )
                    v = sub.loc[mask, "-log10_q"].mean()
                    bar_data[(a, mat)] = v if not np.isnan(v) else 0.0

            def draw_bars(ax):
                for j, mat in enumerate(mats):
                    color   = PALETTE["known"] if mat == CLUSTERING_KNOWN_SUFFIX else PALETTE["other"]
                    offsets = x + (j - 0.5) * bar_width
                    vals    = [bar_data[(a, mat)] for a in algos]
                    ax.bar(offsets, vals, bar_width, color=color,
                           edgecolor="white", alpha=0.92,
                           label=f"{mat.capitalize()} interactions")

            draw_bars(ax_top)
            draw_bars(ax_bot)

            for j, mat in enumerate(mats):
                for ai, a in enumerate(algos):
                    v      = bar_data[(a, mat)]
                    offset = x[ai] + (j - 0.5) * bar_width
                    if v > break_lower:
                        ax_top.text(offset, v + y_top_max * 0.01,
                                    f"{v:.1f}", ha="center", va="bottom",
                                    fontsize=6.5, fontweight="bold", color="#1A1A2E")
                    else:
                        ax_bot.text(offset, v + y_bottom_max * 0.015,
                                    f"{v:.1f}", ha="center", va="bottom",
                                    fontsize=6.5, fontweight="bold", color="#1A1A2E")

            ax_bot.set_xticks(x)
            ax_bot.set_xticklabels([algos_nice.get(a, a.upper()) for a in algos], fontsize=8.5)
            ax_top.set_xticks(x)
            ax_top.set_xticklabels([])
        else:
            vals = sub["-log10_q"].values
            for _ax in (ax_top, ax_bot):
                _ax.bar(range(len(vals)), vals, color=PALETTE["primary"],
                        edgecolor="white", alpha=0.9)
            for k, v in enumerate(vals):
                target = ax_top if v > break_lower else ax_bot
                target.text(k, v + 0.5, f"{v:.1f}", ha="center",
                            va="bottom", fontsize=6.5)

        ax_bot.set_ylim(0, y_bottom_max)
        ax_top.set_ylim(y_top_min, y_top_max)

        ax_bot.axhline(threshold_line, color=PALETTE["danger"],
                       linestyle="--", linewidth=1.3, zorder=5)
        ax_bot.text(
            1.01, threshold_line,
            f"q=0.05\n({threshold_line:.2f})",
            transform=blended_transform_factory(ax_bot.transAxes, ax_bot.transData),
            va="center", ha="left", fontsize=6.5,
            color=PALETTE["danger"], fontweight="bold", linespacing=1.3,
        )

        ax_bot.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax_top.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))

        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False)

        d = 0.012
        kwargs_break = dict(transform=ax_top.transAxes, color="#555555",
                            clip_on=False, linewidth=1.2)
        ax_top.plot((-d, +d), (-d, +d), **kwargs_break)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs_break)
        kwargs_break.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs_break)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_break)

        for ax in (ax_top, ax_bot):
            ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        ax_top.set_title(short, fontsize=11, fontweight="bold", pad=6)

        if col_i == 0:
            ax_bot.set_ylabel(r"$-\log_{10}(q\text{-value})$", fontsize=10)

        ax_bot.set_xlabel("Clustering Algorithm", fontsize=8.5, labelpad=4)

    total_cells = n_rows * n_cols
    for idx in range(len(cancers), total_cells):
        row_i = idx // n_cols
        col_i = idx % n_cols
        dummy_inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row_i, col_i]
        )
        for k in range(2):
            fig.add_subplot(dummy_inner[k]).axis("off")

    handles = [
        mpatches.Patch(color=PALETTE["known"],  label="Known interactions"),
        mpatches.Patch(color=PALETTE["other"],  label="Other interactions"),
        plt.Line2D([0], [0], color=PALETTE["danger"],
                   linestyle="--", linewidth=1.3, label="q = 0.05"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=10, bbox_to_anchor=(0.5, -0.015),
               frameon=True, edgecolor="#CCCCCC")

    if save_fig_fn:
        save_fig_fn(fig, f"figure2_ppi_enrichment_{i}.png")
    else:
        plt.tight_layout()
        out = OUTPUT_DIR / "figure2_ppi_enrichment.png"
        plt.savefig(str(out), dpi=300, bbox_inches="tight", facecolor="white")
        plt.savefig(str(out).replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
        print(f"  Saved: {out}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Clustering heatmap (selected cancers, known + other)
# ---------------------------------------------------------------------------

def fig3_clustering_heatmap(i: int, cancers: list[str]):
    print("\n[Figure 3] Clustering assignment heatmaps...")

    fig, axes = plt.subplots(
        len(cancers), 2,
        figsize=(12, 3.2 * len(cancers)),
        squeeze=False,
    )

    for row_i, cancer in enumerate(cancers):
        for col_i, mat_type in enumerate([CLUSTERING_KNOWN_SUFFIX, CLUSTERING_OTHER_SUFFIX]):
            ax = axes[row_i][col_i]
            df = load_clustering(cancer, mat_type)

            if df is None or df.empty:
                ax.text(0.5, 0.5, f"No data\n({mat_type})",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, color="gray")
                ax.set_title(f"{cancer} – {mat_type.capitalize()} Interactions", fontsize=10)
                ax.axis("off")
                continue

            cluster_cols = [
                c for c in df.columns
                if any(a in c.lower() for a in ["lloyd", "gmm", "fcm"])
            ]
            if not cluster_cols:
                cluster_cols = [c for c in df.columns if c != "Gene" and "label" in c.lower()]
            if not cluster_cols:
                cluster_cols = [c for c in df.columns if c not in ["Gene", "Sample_Name"]][:3]

            gene_col = "Gene" if "Gene" in df.columns else df.columns[0]
            plot_df  = df.set_index(gene_col)[cluster_cols].copy()
            plot_df.columns = [
                c.replace("_label", "").replace("_cluster", "").upper()
                for c in plot_df.columns
            ]

            n_genes = len(plot_df)
            im = ax.imshow(plot_df.values.T, aspect="auto",
                           cmap="tab10", interpolation="nearest", vmin=0, vmax=9)

            ax.set_yticks(range(len(plot_df.columns)))
            ax.set_yticklabels(plot_df.columns, fontsize=9, fontweight="bold")
            ax.set_xticks(range(n_genes))
            ax.set_xticklabels(plot_df.index.tolist(), rotation=60, ha="right", fontsize=6.5)

            mat_label = (
                "Known Interactions" if mat_type == CLUSTERING_KNOWN_SUFFIX
                else "Other Interactions"
            )
            ax.set_title(f"{cancer} — {mat_label}", fontsize=10, fontweight="bold", pad=6)

            cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.8)
            cbar.set_label("Cluster ID", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    save_fig(fig, f"figure3_clustering_heatmap_{i}.png")


# ---------------------------------------------------------------------------
# Figure 4 — Mutation frequency heatmap (all cancers × top genes)
# ---------------------------------------------------------------------------

def fig4_mutation_frequency_heatmap(all_cancers: list[str], top_n: int = 10):
    print(f"\n[Figure 4] Mutation frequency heatmap (top {top_n} genes)...")

    cancer_genes     = {}
    gene_cancer_count = Counter()

    for cancer in all_cancers:
        genes = load_gene_list(cancer)
        cancer_genes[cancer] = genes
        for g in genes:
            gene_cancer_count[g] += 1

    if not any(cancer_genes.values()):
        print("  [SKIP] No gene data available.")
        return

    top_genes   = [g for g, _ in gene_cancer_count.most_common(top_n)]
    other_label = f"Other Genes\n(n={len(gene_cancer_count) - top_n})"

    freq_matrix    = {}
    has_real_freq  = False

    for cancer in all_cancers:
        freq = load_mutation_frequency(cancer)
        if freq is not None:
            has_real_freq = True
            freq_matrix[cancer] = freq
        else:
            genes = cancer_genes.get(cancer, [])
            freq_matrix[cancer] = pd.Series({g: 1.0 for g in genes})

    rows = []
    for cancer in all_cancers:
        freq = freq_matrix.get(cancer, pd.Series(dtype=float))
        row  = {}
        for g in top_genes:
            if has_real_freq and g in freq.index:
                row[g] = float(freq[g])
            elif g in cancer_genes.get(cancer, []):
                row[g] = 1.0
            else:
                row[g] = 0.0

        other_genes = [g for g in cancer_genes.get(cancer, []) if g not in top_genes]
        if has_real_freq and other_genes:
            other_vals = [float(freq[g]) for g in other_genes if g in freq.index]
            row[other_label] = np.mean(other_vals) if other_vals else 0.0
        else:
            row[other_label] = len(other_genes) / max(1, len(cancer_genes.get(cancer, [])))
        rows.append(row)

    heatmap_df = pd.DataFrame(rows, index=all_cancers, columns=top_genes + [other_label])

    fig = plt.figure(figsize=(max(14, top_n * 1.1 + 4), max(10, len(all_cancers) * 0.45)))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[top_n, 1.8], wspace=0.02)

    ax_main  = fig.add_subplot(gs[0])
    ax_other = fig.add_subplot(gs[1])

    top_data   = heatmap_df[top_genes]
    other_data = heatmap_df[[other_label]]

    cmap_main = LinearSegmentedColormap.from_list(
        "mutation_freq", ["#FFFFFF", "#A8D8B9", "#2D6A4F"], N=256
    )
    im_main = ax_main.imshow(
        top_data.values, aspect="auto", cmap=cmap_main,
        vmin=0, vmax=max(top_data.values.max(), 0.01),
        interpolation="nearest", rasterized=True,
    )
    ax_main.set_xticks(range(len(top_genes)))
    ax_main.set_xticklabels(top_genes, rotation=45, ha="right", fontsize=9, fontweight="bold")
    ax_main.set_yticks(range(len(all_cancers)))
    ax_main.set_yticklabels(all_cancers, fontsize=9)
    ax_main.set_title(f"Top {top_n} Most Frequent Driver Genes", fontsize=11, fontweight="bold")
    ax_main.set_xlabel("Gene", fontsize=11, labelpad=8)
    ax_main.set_ylabel("Cancer Type", fontsize=11, labelpad=8)

    for k, cancer in enumerate(all_cancers):
        if cancer in HIGHLIGHT_CANCERS:
            ax_main.get_yticklabels()[k].set_fontweight("bold")
            ax_main.get_yticklabels()[k].set_color(PALETTE["primary"])

    if has_real_freq:
        for k in range(len(all_cancers)):
            for j in range(len(top_genes)):
                val = top_data.values[k, j]
                if val > 0:
                    text_color = "white" if val > top_data.values.max() * 0.6 else "#333333"
                    ax_main.text(j, k, f"{val:.2f}", ha="center", va="center",
                                 fontsize=6, color=text_color)

    cbar1 = plt.colorbar(im_main, ax=ax_main, orientation="vertical",
                         pad=0.01, shrink=0.8, fraction=0.03)
    cbar1_label = "Mutation\nFrequency" if has_real_freq else "Presence\nin Output"
    cbar1.ax.set_title(cbar1_label, fontsize=8, pad=10)
    cbar1.ax.tick_params(labelsize=7)

    cmap_other = LinearSegmentedColormap.from_list(
        "other", ["#FFFFFF", "#FFB703", "#E85D04"], N=256
    )
    im_other = ax_other.imshow(
        other_data.values, aspect="auto", cmap=cmap_other,
        vmin=0, vmax=max(other_data.values.max(), 0.01),
        interpolation="nearest", rasterized=True,
    )
    ax_other.set_xticks([0])
    ax_other.set_xticklabels([other_label], fontsize=9, rotation=30, ha="right")
    ax_other.set_yticks(range(len(all_cancers)))
    ax_other.set_yticklabels([], fontsize=9)
    ax_other.set_title("Remaining\nGenes", fontsize=11, fontweight="bold")

    n_other_per_cancer = {
        c: len([g for g in cancer_genes.get(c, []) if g not in top_genes])
        for c in all_cancers
    }
    for k, cancer in enumerate(all_cancers):
        ax_other.text(0, k, f"n={n_other_per_cancer[cancer]}", ha="center",
                      va="center", fontsize=7.5, color="#333333")

    cbar2 = plt.colorbar(im_other, ax=ax_other, orientation="vertical",
                         pad=0.02, shrink=0.8, fraction=0.12)
    cbar2.ax.set_title("Avg. Freq.\n/ Presence", fontsize=7, pad=10)
    cbar2.ax.tick_params(labelsize=7)

    for ax_sep in [ax_main, ax_other]:
        for spine in ax_sep.spines.values():
            spine.set_edgecolor("#CCCCCC")

    save_fig(fig, "figure4_mutation_frequency_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 5 — Summary table (highlighted cancers)
# ---------------------------------------------------------------------------

def fig5_summary_table(cancers: list[str]):
    print("\n[Figure 5] Summary statistics table for highlighted cancers...")

    rows = []
    for cancer in cancers:
        genes     = load_gene_list(cancer)
        clust     = load_clustering(cancer, CLUSTERING_KNOWN_SUFFIX)
        n_clusters = (
            clust[["lloyd_label", "gmm_label", "fcm_label"]].nunique().to_dict()
            if clust is not None
            and all(c in clust.columns for c in ["lloyd_label", "gmm_label", "fcm_label"])
            else {}
        )
        match      = re.search(r'\((.*?)\)', cancer)
        short_name = match.group(1) if match else cancer

        rows.append({
            "Cancer":         short_name,
            "Num Genes":      len(genes) if genes else "—",
            "#Clusters\n(Lloyd)": n_clusters.get("lloyd_label", "—"),
            "#Clusters\n(GMM)":   n_clusters.get("gmm_label",  "—"),
            "#Clusters\n(FCM)":   n_clusters.get("fcm_label",  "—"),
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(18, 3.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.07, 0.06, 0.28, 0.09, 0.09, 0.09, 0.08, 0.07, 0.07],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.8)

    row_colors = ["#FFFFFF", "#F0F4F8"]
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor(PALETTE["dark"])
            cell.set_text_props(color="white", fontweight="bold", fontsize=8.5)
        else:
            cancer_name = df.iloc[r - 1]["Cancer"] if r - 1 < len(df) else ""
            cell.set_facecolor(
                "#E8F5E9" if cancer_name in HIGHLIGHT_CANCERS
                else row_colors[(r - 1) % 2]
            )

    save_fig(fig, "figure5_summary_table.png")


# ---------------------------------------------------------------------------
# Figure 6 — Gene occurrence across cancers
# ---------------------------------------------------------------------------

def fig6_gene_occurrence(all_cancers: list[str]):
    print("\n[Figure 6] Gene occurrence across cancers...")

    gene_cancer_map = defaultdict(set)
    for cancer in all_cancers:
        for gene in load_gene_list(cancer):
            gene_cancer_map[gene].add(cancer)

    if not gene_cancer_map:
        print("  [SKIP] No data available.")
        return

    freq       = {g: len(cancers) for g, cancers in gene_cancer_map.items()}
    freq_series = pd.Series(freq).sort_values(ascending=False)
    top30      = freq_series.head(30)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    colors = [
        PALETTE["primary"]   if freq_series[g] >= 5 else
        PALETTE["secondary"] if freq_series[g] >= 2 else
        "#CCCCCC"
        for g in top30.index
    ]
    ax.barh(range(len(top30)), top30.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Cancer Types", fontsize=11)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    for k, (gene, val) in enumerate(zip(top30.index, top30.values)):
        ax.text(val + 0.05, k, str(val), va="center", fontsize=8, color=PALETTE["dark"])

    legend_handles = [
        mpatches.Patch(color=PALETTE["primary"],   label="≥5 cancer types"),
        mpatches.Patch(color=PALETTE["secondary"], label="2–4 cancer types"),
        mpatches.Patch(color="#CCCCCC",            label="1 cancer type"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    ax2     = axes[1]
    counts  = list(freq.values())
    n_single   = sum(1 for v in counts if v == 1)
    n_moderate = sum(1 for v in counts if 2 <= v <= 4)
    n_high     = sum(1 for v in counts if v >= 5)
    total_genes = len(counts)

    labels  = ["Cancer-specific\n(1 Type)", "Shared\n(2-4 Types)", "Highly Recurrent\n(≥5 Types)"]
    sizes   = [n_single, n_moderate, n_high]
    colors  = ["#CCCCCC", PALETTE["secondary"], PALETTE["primary"]]
    explode = (0, 0.05, 0.15)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=labels, autopct="%1.1f%%",
        shadow=False, startangle=140, colors=colors,
        textprops={"fontsize": 10, "fontweight": "bold"},
        pctdistance=0.75,
    )
    for autotext in autotexts:
        autotext.set_color("white")

    ax2.add_artist(plt.Circle((0, 0), 0.60, fc="white"))
    ax2.text(0, 0, f"Total Unique\nGenes\n{total_genes}",
             ha="center", va="center", fontsize=11,
             fontweight="bold", color=PALETTE["dark"])

    plt.tight_layout()
    save_fig(fig, "figure6_gene_occurrence.png")


def fig6_gene_occurrence_pie(all_cancers: list[str]):
    print("\n[Figure 6] Gene occurrence across cancers (Pie Chart)...")

    gene_cancer_map = defaultdict(set)
    for cancer in all_cancers:
        for gene in load_gene_list(cancer):
            gene_cancer_map[gene].add(cancer)

    if not gene_cancer_map:
        print("  [SKIP] No data available.")
        return

    counts     = [len(cancers) for cancers in gene_cancer_map.values()]
    n_single   = sum(1 for v in counts if v == 1)
    n_moderate = sum(1 for v in counts if 2 <= v <= 4)
    n_high     = sum(1 for v in counts if v >= 5)
    total_genes = len(counts)

    labels  = ["Cancer-specific\n(1 Type)", "Shared\n(2-4 Types)", "Highly Recurrent\n(≥5 Types)"]
    sizes   = [n_single, n_moderate, n_high]
    colors  = ["#CCCCCC", PALETTE["secondary"], PALETTE["primary"]]
    explode = (0, 0.05, 0.15)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, autopct="%1.1f%%",
        shadow=False, startangle=140, colors=colors,
        textprops={"fontsize": 11, "fontweight": "bold"},
        pctdistance=0.82,
    )
    for autotext in autotexts:
        autotext.set_color("white")

    fig.gca().add_artist(plt.Circle((0, 0), 0.65, fc="white"))
    ax.text(0, 0, f"Total Unique\nGenes\n{total_genes}",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=PALETTE["dark"])

    plt.tight_layout()
    save_fig(fig, "figure6_gene_occurrence_pie.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  CANCER DRIVER GENE — RESULTS VISUALIZATION")
    print("=" * 60)

    all_cancers = [
        "Acute Myeloid Leukemia (LAML)",
        "Bladder Urothelial Carcinoma (BLCA)",
        "Brain Lower Grade Glioma (LGG)",
        "Breast Invasive Carcinoma (BRCA)",
        "Cervical Squamous Cell Carcinoma (CESC)",
        "Colorectal Adenocarcinoma (COADREAD)",
        "Esophageal Adenocarcinoma (ESCA)",
        "Glioblastoma Multiforme (GBM)",
        "Head and Neck Squamous Cell Carcinoma (HNSC)",
        "Kidney Renal Clear Cell Carcinoma (KIRC)",
        "Kidney Renal Papillary Cell Carcinoma (KIRP)",
        "Liver Hepatocellular Carcinoma (LIHC)",
        "Lung Adenocarcinoma (LUAD)",
        "Lung Squamous Cell Carcinoma (LUSC)",
        "Ovarian Serous Cystadenocarcinoma (OV)",
        "Pancreatic Adenocarcinoma (PAAD)",
        "Pheochromocytoma and Paraganglioma (PCPG)",
        "Prostate Adenocarcinoma (PRAD)",
        "Sarcoma (SARC)",
        "Skin Cutaneous Melanoma (SKCM)",
        "Stomach Adenocarcinoma (STAD)",
        "Testicular Germ Cell Tumors (TGCT)",
        "Thymoma (THYM)",
        "Thyroid Carcinoma (THCA)",
        "Uterine Corpus Endometrial Carcinoma (UCEC)",
    ]

    if not all_cancers:
        print(f"\n[ERROR] No cancer types found in: {RESULTS_DIR}")
        return

    print(f"\nDetected {len(all_cancers)} cancer types.")

    highlight = [c for c in HIGHLIGHT_CANCERS if c in all_cancers]
    if not highlight:
        highlight = all_cancers[:6]
        print(f"  [INFO] Using first 6 cancers as highlight: {highlight}")
    else:
        print(f"  [INFO] Highlighted cancers: {highlight}")

    print(f"\nOutput directory: {OUTPUT_DIR}/\n")

    fig4_mutation_frequency_heatmap(all_cancers, top_n=20)
    fig5_summary_table(all_cancers)
    fig6_gene_occurrence(all_cancers)

    print("\n" + "=" * 60)
    print("  Done! All figures saved to: figures/")
    print("  Each figure is saved as .png (screen) and .pdf (paper).")
    print("=" * 60)


if __name__ == "__main__":
    main()
