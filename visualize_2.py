import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import hypergeom

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 150
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["font.family"] = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GENERAL_RESULTS_DIR    = "general_result"
CLUSTERING_RESULTS_DIR = "cluster_res"
PPI_ENRICHMENT_DIR     = "ppi_res"
PERMUTATION_DIR        = "results/sc2gc_stats"
DATA_ROOT              = "F:/TruongHongKiet/chạy_code/data/TCGA Cancer"
OUTPUT_DIR             = "figures_p2_final_new"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CANCER_COLORS = {
    "Bladder Urothelial Carcinoma (BLCA)":                 "#3cb44b",
    "Breast Invasive Carcinoma (BRCA)":                    "#ffe119",
    "Cervical Squamous Cell Carcinoma (CESC)":             "#4363d8",
    "Colorectal Adenocarcinoma (COADREAD)":                "#911eb4",
    "Esophageal Adenocarcinoma (ESCA)":                    "#f032e6",
    "Glioblastoma Multiforme (GBM)":                       "#bfef45",
    "Head and Neck Squamous Cell Carcinoma (HNSC)":        "#fabed4",
    "Kidney Renal Clear Cell Carcinoma (KIRC)":            "#635cb9",
    "Kidney Renal Papillary Cell Carcinoma (KIRP)":        "#9A6324",
    "Acute Myeloid Leukemia (LAML)":                       "#1f1e33",
    "Brain Lower Grade Glioma (LGG)":                      "#800000",
    "Liver Hepatocellular Carcinoma (LIHC)":               "#aaffc3",
    "Lung Adenocarcinoma (LUAD)":                          "#808000",
    "Lung Squamous Cell Carcinoma (LUSC)":                 "#ffd8b1",
    "Ovarian Serous Cystadenocarcinoma (OV)":              "#a9a9a9",
    "Pancreatic Adenocarcinoma (PAAD)":                    "#0ad195",
    "Pheochromocytoma and Paraganglioma (PCPG)":           "#bc5af9",
    "Prostate Adenocarcinoma (PRAD)":                      "#aa6e28",
    "Sarcoma (SARC)":                                      "#008080",
    "Skin Cutaneous Melanoma (SKCM)":                      "#0075dc",
    "Stomach Adenocarcinoma (STAD)":                       "#993F00",
    "Testicular Germ Cell Tumors (TGCT)":                  "#4C005C",
    "Thymoma (THYM)":                                      "#191919",
    "Thyroid Carcinoma (THCA)":                            "#005C31",
    "Uterine Corpus Endometrial Carcinoma (UCEC)":         "#FFCC99",
}

CLUSTERING_ALGORITHMS = ["Lloyd", "GMM", "FCM"]
INTERACTION_TYPES     = ["known_interactions", "other_interactions"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_cancers(base_dir: str) -> list[str]:
    if not os.path.isdir(base_dir):
        print(f"[WARN] Directory not found: {base_dir}")
        return []
    return sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])


def save_fig(fig: plt.Figure, name: str, subdir: str = "") -> str:
    folder = os.path.join(OUTPUT_DIR, subdir) if subdir else OUTPUT_DIR
    os.makedirs(folder, exist_ok=True)

    path_png = os.path.join(folder, f"{name}.png")
    path_pdf = os.path.join(folder, f"{name}.pdf")

    fig.savefig(path_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(path_pdf, bbox_inches="tight", facecolor=fig.get_facecolor(), format="pdf")

    print(f"Saved: {name}.png and {name}.pdf")
    plt.close(fig)
    return path_png


def load_general_results(cancer: str) -> pd.DataFrame | None:
    folder = os.path.join(GENERAL_RESULTS_DIR, cancer)
    csvs   = glob.glob(os.path.join(folder, "*.csv"))
    return pd.read_csv(csvs[0]) if csvs else None


def load_clustering_results(cancer: str, algo: str, interaction: str) -> pd.DataFrame | None:
    folder    = os.path.join(CLUSTERING_RESULTS_DIR, cancer)
    file_path = os.path.join(folder, f"{interaction}_all_labels.csv")

    if not os.path.exists(file_path):
        return None

    df       = pd.read_csv(file_path)
    algo_key = algo.lower()
    gene_col = next(
        (c for c in df.columns if "Sample_Name" in c.lower()), df.columns[0]
    )
    label_col = next(
        (c for c in df.columns if algo_key in c.lower() and "label" in c.lower()), None
    )

    if label_col:
        res_df = df[[gene_col, label_col]].copy()
        return res_df.rename(columns={label_col: f"{algo}_cluster"})

    return df


def load_ppi_enrichment(cancer: str) -> pd.DataFrame | None:
    folder  = os.path.join(PPI_ENRICHMENT_DIR, cancer)
    pattern = os.path.join(folder, "*", "*", "p-value_sum.csv")
    csvs    = glob.glob(pattern)

    if not csvs:
        return None

    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def load_permutation(cancer: str) -> pd.DataFrame | None:
    folder = os.path.join(PERMUTATION_DIR, cancer)
    csvs   = glob.glob(os.path.join(folder, "*.csv"))
    return pd.read_csv(csvs[0]) if csvs else None


# ---------------------------------------------------------------------------
# Section 2 — Data Overview
# ---------------------------------------------------------------------------

def plot_mutation_frequency_scatter(mutation_csv: str, top_n: int = 20, cancer_name: str = ""):
    print(f"[2.1] Mutation Frequency Scatter - {cancer_name}")
    df          = pd.read_csv(mutation_csv, index_col=0)
    n_patients  = df.shape[0]
    mut_counts  = df.sum(axis=0).sort_values(ascending=False)
    mut_freq    = mut_counts / n_patients * 100

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    sc = ax.scatter(
        np.arange(1, len(mut_counts) + 1), mut_freq.values,
        c=mut_freq.values, cmap="plasma",
        s=18, alpha=0.75, linewidths=0, zorder=3, rasterized=True,
    )

    try:
        from adjustText import adjust_text
        texts = [
            ax.text(
                mut_freq.index.get_loc(gene) + 1, freq, gene,
                color="black", fontsize=5, fontweight="bold",
            )
            for gene, freq in mut_freq.head(top_n).items()
        ]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    except ImportError:
        for gene, freq in mut_freq.head(top_n).items():
            r = mut_freq.index.get_loc(gene) + 1
            ax.annotate(gene, (r, freq), color="black", fontsize=5, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")

    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Mutation Frequency (%)", color="black", fontsize=10)
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

    ax.set_xlabel("Gene Rank (by mutation count)", color="black", fontsize=11)
    ax.set_ylabel("Mutation Frequency (%)", color="black", fontsize=11)
    ax.tick_params(colors="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.grid(True, color="#e0e0e0", linewidth=0.5, alpha=0.7)

    return save_fig(fig, f"mutation_freq_scatter_{cancer_name}", subdir="data_overview")


def plot_mutation_burden_barplot(cancer_list: list[str], count_col: str = "n_mutations"):
    print("[2.2] Mutation Burden Barplot")
    records = []
    for cancer in cancer_list:
        df = load_general_results(cancer)
        if df is None:
            continue
        val = df[count_col].sum() if count_col in df.columns else len(df)
        records.append({"cancer": cancer, "value": val})

    if not records:
        return

    data   = pd.DataFrame(records).sort_values("value", ascending=False)
    colors = [CANCER_COLORS.get(c, "#888") for c in data["cancer"]]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    ax.set_facecolor("white")
    bars = ax.bar(data["cancer"], data["value"], color=colors, edgecolor="none", width=0.7)

    for bar, val in zip(bars, data["value"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(int(val)), ha="center", va="bottom", color="black", fontsize=7)

    ax.set_xlabel("Cancer Type", color="black", fontsize=11)
    ax.set_ylabel("Number of Selected Driver Genes", color="black", fontsize=11)
    ax.tick_params(colors="black", axis="both")
    ax.set_xticklabels(data["cancer"], rotation=45, ha="right", fontsize=8, color="black")
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
    ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.5, alpha=0.7)

    return save_fig(fig, "mutation_burden_barplot", subdir="data_overview")


def plot_patient_gene_heatmap(
    mutation_csv: str,
    cancer_name: str = "",
    max_genes: int = 80,
    max_patients: int = 100,
):
    print(f"[2.3] Mutation Heatmap — {cancer_name}")
    df        = pd.read_csv(mutation_csv, index_col=0)
    top_genes = df.sum(axis=0).nlargest(max_genes).index
    sub       = df[top_genes].iloc[:max_patients].T

    fig, ax = plt.subplots(
        figsize=(min(sub.shape[1] * 0.12 + 2, 20), min(sub.shape[0] * 0.25 + 2, 14)),
        facecolor="white",
    )
    ax.set_facecolor("white")

    cmap_bin = LinearSegmentedColormap.from_list("bin", ["#f4f6f8", "#e94560"])
    ax.imshow(sub.values, aspect="auto", cmap=cmap_bin, interpolation="none", rasterized=True)

    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index, fontsize=6, color="black")
    ax.set_xticks([])
    ax.set_xlabel(f"Patients (n={sub.shape[1]})", color="black", fontsize=10)

    patch_mut = mpatches.Patch(color="#e94560", label="Mutated")
    patch_wt  = mpatches.Patch(color="#f4f6f8", label="Wild-type")
    ax.legend(handles=[patch_mut, patch_wt], loc="upper right",
              facecolor="white", edgecolor="#cccccc", labelcolor="black", fontsize=8)

    return save_fig(fig, f"mutation_heatmap_{cancer_name}", subdir="data_overview")


# ---------------------------------------------------------------------------
# Section 3 — Algorithm results
# ---------------------------------------------------------------------------

def plot_oncoprint(
    cancer: str,
    driver_genes: list[str] | None = None,
    mutation_csv: str | None = None,
):
    print(f"[3.1] OncoPrint — {cancer}")
    if driver_genes is None:
        df_res = load_general_results(cancer)
        if df_res is None:
            return
        gene_col     = next((c for c in df_res.columns if "gene" in c.lower()), df_res.columns[0])
        driver_genes = df_res[gene_col].dropna().tolist()

    if mutation_csv and os.path.isfile(mutation_csv):
        mut      = pd.read_csv(mutation_csv, index_col=0)
        common   = [g for g in driver_genes if g in mut.columns]
        sub      = mut[common].T
        has_data = True
    else:
        sub = pd.DataFrame(
            np.random.choice([0, 1], size=(len(driver_genes), 30), p=[0.7, 0.3]),
            index=driver_genes,
            columns=[f"P{i + 1}" for i in range(30)],
        )
        has_data = False

    sub = sub.loc[sub.sum(axis=1).sort_values(ascending=False).index]
    sub = sub[sub.sum(axis=0).sort_values(ascending=False).index]

    n_genes, n_patients = sub.shape
    fig_h = max(4, n_genes * 0.35 + 2)
    fig_w = max(8, n_patients * 0.15 + 3)

    fig = plt.figure(figsize=(min(fig_w, 22), min(fig_h, 16)), facecolor="white")
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1, n_genes], hspace=0.05)

    ax_top = fig.add_subplot(gs[0])
    ax_top.set_facecolor("white")
    burden = sub.sum(axis=0)
    ax_top.bar(range(n_patients), burden.values, color="#e94560", width=0.8, rasterized=True)
    ax_top.set_xlim(-0.5, n_patients - 0.5)
    ax_top.set_xticks([])
    ax_top.set_ylabel("# Mut", color="black", fontsize=8)
    ax_top.tick_params(colors="black", labelsize=7)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    for sp in ["bottom", "left"]:
        ax_top.spines[sp].set_edgecolor("#cccccc")
    title_suffix = "" if has_data else " [Simulated — mutation_csv required]"
    ax_top.set_title(f"OncoPrint — {cancer}{title_suffix}", color="black", fontsize=12, pad=8)

    ax_main = fig.add_subplot(gs[1])
    ax_main.set_facecolor("#f4f6f8")
    cell_h  = 0.8

    for i, gene in enumerate(sub.index):
        for j in range(len(sub.columns)):
            if sub.iloc[i, j] == 1:
                ax_main.add_patch(mpatches.FancyBboxPatch(
                    (j - 0.4, i - cell_h / 2), 0.8, cell_h,
                    boxstyle="square,pad=0.02",
                    facecolor="#e94560", edgecolor="none", rasterized=True,
                ))

    ax_main.set_xlim(-0.5, n_patients - 0.5)
    ax_main.set_ylim(-0.5, n_genes - 0.5)
    ax_main.set_yticks(range(n_genes))
    ax_main.set_yticklabels(sub.index, fontsize=min(8, 100 // n_genes + 4), color="black")
    ax_main.set_xticks([])
    ax_main.set_xlabel(f"Patients (n={n_patients})", color="black", fontsize=10)
    ax_main.invert_yaxis()

    for i in range(n_genes + 1):
        ax_main.axhline(i - 0.5, color="white", linewidth=0.7)

    ax_freq = ax_main.twinx()
    freq = sub.sum(axis=1) / n_patients * 100
    ax_freq.barh(range(n_genes), freq.values, color="#e1b12c",
                 alpha=0.5, height=0.5, rasterized=True)
    ax_freq.set_ylim(-0.5, n_genes - 0.5)
    ax_freq.invert_yaxis()
    ax_freq.set_yticks([])
    ax_freq.set_xlabel("Freq %", color="#d35400", fontsize=8)
    ax_freq.tick_params(colors="#d35400", labelsize=7)
    for sp in ax_freq.spines.values():
        sp.set_edgecolor("#cccccc")

    return save_fig(fig, f"oncoprint_{cancer}", subdir="general_results")


def plot_driver_gene_summary_all_cancers(cancer_list: list[str]):
    print("[3.2] Driver Gene Summary — all cancers")
    n, ncols = len(cancer_list), 4
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5), facecolor="white")
    axes_flat = axes.flatten()

    for idx, cancer in enumerate(cancer_list):
        ax = axes_flat[idx]
        ax.set_facecolor("#f8f9fa")
        df = load_general_results(cancer)
        if df is None:
            ax.text(0.5, 0.5, f"{cancer}\n(no data)",
                    ha="center", va="center", color="#666", transform=ax.transAxes)
            ax.axis("off")
            continue

        gene_col  = next((c for c in df.columns if "gene" in c.lower()), df.columns[0])
        score_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["score", "freq", "count", "pval"])),
            None,
        )
        genes  = df[gene_col].dropna().head(20).tolist()
        vals   = df[score_col].dropna().head(20).values if score_col else np.ones(len(genes))
        color  = CANCER_COLORS.get(cancer, "#888")

        ax.barh(range(len(genes)), vals[::-1], color=color, alpha=0.85, edgecolor="none")
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes[::-1], fontsize=6, color="black")
        ax.tick_params(colors="black", labelsize=6)
        ax.set_xlabel(score_col if score_col else "count", color="#555", fontsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
        ax.grid(True, axis="x", color="#e0e0e0", linewidth=0.4)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Driver Genes Selected per Cancer Type",
                 color="black", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return save_fig(fig, "driver_gene_summary_all", subdir="general_results")


def plot_gene_frequency_across_cancers(cancer_list: list[str], top_n_genes: int = 30):
    print("[3.3] Gene Frequency Across Cancers (Dot Plot)")
    gene_cancer_map = {}
    for cancer in cancer_list:
        df = load_general_results(cancer)
        if df is None:
            continue
        gene_col = next((c for c in df.columns if "gene" in c.lower()), df.columns[0])
        for g in df[gene_col].dropna():
            gene_cancer_map.setdefault(g, set()).add(cancer)

    top_genes = sorted(
        gene_cancer_map, key=lambda g: len(gene_cancer_map[g]), reverse=True
    )[:top_n_genes]
    matrix = pd.DataFrame(0, index=top_genes, columns=cancer_list)
    for gene in top_genes:
        for cancer in gene_cancer_map.get(gene, []):
            if cancer in matrix.columns:
                matrix.loc[gene, cancer] = 1

    fig, ax = plt.subplots(
        figsize=(max(14, len(cancer_list) * 0.5), max(8, top_n_genes * 0.35)),
        facecolor="white",
    )
    ax.set_facecolor("white")

    for i, gene in enumerate(matrix.index):
        for j, cancer in enumerate(matrix.columns):
            if matrix.loc[gene, cancer]:
                ax.scatter(j, i, s=120, color=CANCER_COLORS.get(cancer, "#888"),
                           zorder=3, edgecolors="white", linewidths=0.3, rasterized=True)
            else:
                ax.scatter(j, i, s=20, color="#e0e0e0", zorder=2, rasterized=True)

    ax.set_xticks(range(len(cancer_list)))
    ax.set_xticklabels(cancer_list, rotation=45, ha="right", fontsize=8, color="black")
    ax.set_yticks(range(len(top_genes)))
    ax.set_yticklabels(top_genes, fontsize=8, color="black")
    ax.set_xlim(-0.5, len(cancer_list) - 0.5)
    ax.set_ylim(-0.5, top_n_genes - 0.5)
    ax.grid(True, color="#f0f0f0", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(top_genes)))
    ax2.set_yticklabels(
        [f"n={len(gene_cancer_map.get(g, []))}" for g in top_genes],
        fontsize=7, color="#d35400",
    )
    ax2.tick_params(colors="#d35400")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#cccccc")

    return save_fig(fig, "gene_frequency_across_cancers", subdir="general_results")


def plot_advanced_multicancer_oncoprint(
    gene_list: list[str],
    cancer_list: list[str],
    title: str = "SWI / SNF Complex",
):
    """Multi-cancer OncoPrint from real mutation and CNA data."""
    print(f"[3.4] Advanced Multi-Cancer OncoPrint — {title}")
    import re

    AMP_THRESH = 1.0
    DEL_THRESH = -1.0

    patient_cancers = {}
    combined_data   = {}

    for cancer_full_name in cancer_list:
        match   = re.search(r'\((.*?)\)', cancer_full_name)
        acronym = match.group(1) if match else cancer_full_name
        folder  = os.path.join(DATA_ROOT, cancer_full_name)

        if not os.path.isdir(folder):
            matched = glob.glob(os.path.join(DATA_ROOT, f"*{acronym}*"))
            if not matched:
                continue
            folder = matched[0]

        mut_files = (
            glob.glob(os.path.join(folder, f"*{acronym}_mutation*zscore_all.csv"))
            or glob.glob(os.path.join(folder, f"*{acronym}_mutation*.csv"))
        )
        cna_files = (
            glob.glob(os.path.join(folder, f"*{acronym}_expression_log2_cna.csv"))
            or glob.glob(os.path.join(folder, f"*{acronym}*cna*.csv"))
        )

        if not mut_files or not cna_files:
            continue

        df_mut = pd.read_csv(mut_files[0], index_col=0)
        df_cna = pd.read_csv(cna_files[0], index_col=0)

        for p in set(df_mut.index) & set(df_cna.index):
            pid = f"{acronym}_{p}"
            patient_cancers[pid] = cancer_full_name
            combined_data[pid]   = {}

            for gene in gene_list:
                mut_val = df_mut.loc[p, gene] if gene in df_mut.columns else 0
                cna_val = df_cna.loc[p, gene] if gene in df_cna.columns else 0.0

                state = ""
                if mut_val > 0:
                    state += "SNV"
                if cna_val >= AMP_THRESH:
                    state += "_AMP" if state else "AMP"
                elif cna_val <= DEL_THRESH:
                    state += "_DEL" if state else "DEL"

                combined_data[pid][gene] = state

    if not combined_data:
        print("  [SKIP] Insufficient real data.")
        return

    matrix = pd.DataFrame(combined_data).fillna("")
    has_mut = matrix.apply(lambda col: col.str.contains("SNV|AMP|DEL").any(), axis=0)
    matrix  = matrix.loc[:, has_mut]

    if matrix.empty:
        print(f"  [SKIP] No patients with mutations in cluster {title}.")
        return

    sorted_patients = sorted(
        matrix.columns,
        key=lambda x: (patient_cancers[x], matrix[x].value_counts().get("", 100)),
    )
    matrix = matrix[sorted_patients]

    n_genes, n_pats = matrix.shape
    fig_w = min(14, max(8, n_pats * 0.03))
    fig_h = max(3, n_genes * 0.25)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax  = fig.add_subplot(gridspec.GridSpec(1, 1)[0])
    ax.set_facecolor("#f9f9f9")

    cell_w, cell_h = 1.0, 0.8
    for i, gene in enumerate(matrix.index):
        for j, patient in enumerate(matrix.columns):
            mut   = str(matrix.loc[gene, patient])
            if not mut:
                continue
            color = CANCER_COLORS.get(patient_cancers[patient], "#333333")

            if "SNV" in mut:
                ax.add_patch(mpatches.Rectangle(
                    (j - cell_w / 2, i - cell_h / 2), cell_w, cell_h,
                    facecolor=color, edgecolor="none", rasterized=True,
                ))
            if "AMP" in mut:
                ax.add_patch(mpatches.Rectangle(
                    (j - cell_w / 2, i), cell_w, cell_h / 2,
                    facecolor=color, edgecolor="none", zorder=2, rasterized=True,
                ))
            elif "DEL" in mut:
                ax.add_patch(mpatches.Rectangle(
                    (j - cell_w / 2, i - cell_h / 2), cell_w, cell_h / 2,
                    facecolor=color, edgecolor="none", zorder=2, rasterized=True,
                ))

    ax.set_xlim(-0.5, n_pats - 0.5)
    ax.set_ylim(-0.5, n_genes - 0.5)
    ax.set_yticks(range(n_genes))
    mut_counts = (matrix != "").sum(axis=1)
    ax.set_yticklabels([f"{g} ({mut_counts[g]})" for g in matrix.index], fontsize=8, color="black")
    ax.set_xticks([])
    ax.invert_yaxis()

    for i in range(n_genes + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(" ", pad=70)

    mut_elements = [
        mpatches.Patch(facecolor="#888888", label="SNV (Full box)"),
        mpatches.Patch(facecolor="#888888", label="Amplification (Top half)"),
        mpatches.Patch(facecolor="#888888", label="Deletion (Bottom half)"),
    ]
    leg_mut = ax.legend(
        handles=mut_elements, loc="lower right", bbox_to_anchor=(1.0, 1.15),
        ncol=3, frameon=False, fontsize=7,
        title="Mutation Types", title_fontproperties={"weight": "bold", "size": 8},
    )
    ax.add_artist(leg_mut)

    plotted_cancers = sorted({patient_cancers[p] for p in matrix.columns})
    cancer_elements = []
    for c in plotted_cancers:
        m_label = re.search(r'\((.*?)\)', c)
        label   = m_label.group(1) if m_label else c
        cancer_elements.append(mpatches.Patch(facecolor=CANCER_COLORS.get(c, "#333333"), label=label))

    n_cols = min(10, max(4, len(cancer_elements) // 2))
    ax.legend(
        handles=cancer_elements, loc="lower right", bbox_to_anchor=(1.0, 1.02),
        ncol=n_cols, frameon=False, fontsize=7,
        title="Cancer Types", title_fontproperties={"weight": "bold", "size": 8},
    )

    return save_fig(
        fig,
        f"advanced_oncoprint_{title.replace(' ', '_').replace('/', '')}",
        subdir="general_results",
    )


# ---------------------------------------------------------------------------
# Section 4 — Clustering results
# ---------------------------------------------------------------------------

def plot_cluster_heatmap(
    cancer: str,
    algo: str = "Lloyd",
    interaction: str = "known_interactions",
):
    print(f"[4.1] Cluster Heatmap — {cancer} / {algo} / {interaction}")
    df = load_clustering_results(cancer, algo, interaction)
    if df is None:
        return

    cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
    gene_col    = next((c for c in df.columns if "gene"    in c.lower()), None)
    if not cluster_col or not gene_col:
        return

    clusters  = df[cluster_col].unique()
    all_genes = df[gene_col].unique()
    matrix    = pd.DataFrame(0, index=clusters, columns=all_genes)
    for _, row in df.iterrows():
        matrix.loc[row[cluster_col], row[gene_col]] = 1

    fig, ax = plt.subplots(
        figsize=(max(10, len(all_genes) * 0.25), max(4, len(clusters) * 0.6)),
        facecolor="white",
    )
    ax.set_facecolor("white")
    cmap_cl = LinearSegmentedColormap.from_list("cl", ["#f4f6f8", "#00b4d8"])
    ax.imshow(matrix.values, aspect="auto", cmap=cmap_cl, interpolation="none")

    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([f"Cluster {c}" for c in clusters], color="black", fontsize=9)

    if len(all_genes) <= 50:
        ax.set_xticks(range(len(all_genes)))
        ax.set_xticklabels(all_genes, rotation=90, fontsize=7, color="black")
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Genes (n={len(all_genes)})", color="black", fontsize=10)

    return save_fig(fig, f"cluster_heatmap_{cancer}_{algo}_{interaction}", subdir="clustering")


def plot_cluster_size_comparison(cancer_list: list[str]):
    print("[4.2] Cluster Size Comparison")
    records = []
    for cancer in cancer_list:
        for algo in CLUSTERING_ALGORITHMS:
            for itype in INTERACTION_TYPES:
                df = load_clustering_results(cancer, algo, itype)
                if df is None:
                    continue
                cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
                n_clusters  = df[cluster_col].nunique() if cluster_col else 0
                avg_size    = len(df) / max(n_clusters, 1) if cluster_col else 0
                records.append({
                    "cancer":     cancer,
                    "algo":       algo,
                    "interaction": itype,
                    "n_clusters": n_clusters,
                    "avg_size":   avg_size,
                })

    if not records:
        return

    data        = pd.DataFrame(records)
    algo_colors = {"Lloyd": "#e94560", "GMM": "#00b4d8", "FCM": "#f7b731"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")
    for ax in axes:
        ax.set_facecolor("#f8f9fa")

    for ax, metric, label in zip(
        axes,
        ["n_clusters", "avg_size"],
        ["Number of Clusters", "Average Cluster Size (genes)"],
    ):
        x, width = np.arange(len(cancer_list)), 0.12
        for i, algo in enumerate(CLUSTERING_ALGORITHMS):
            for j, itype in enumerate(INTERACTION_TYPES):
                subset = data[(data["algo"] == algo) & (data["interaction"] == itype)]
                vals   = [
                    subset[subset["cancer"] == c][metric].values[0]
                    if len(subset[subset["cancer"] == c]) > 0
                    else 0
                    for c in cancer_list
                ]
                offset = (i * 2 + j - 2.5) * width
                alpha  = 0.9 if itype == "known_interactions" else 0.4
                ax.bar(x + offset, vals, width, color=algo_colors[algo],
                       alpha=alpha, edgecolor="none")

        ax.set_xticks(x)
        ax.set_xticklabels(cancer_list, rotation=45, ha="right", fontsize=7, color="black")
        ax.set_ylabel(label, color="black", fontsize=10)
        ax.tick_params(colors="black")
        for sp in ax.spines.values():
            sp.set_edgecolor("#cccccc")
        ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.5)

    axes[0].set_title("Number of Clusters per Cancer", color="black", fontsize=12)
    axes[1].set_title("Average Cluster Size per Cancer", color="black", fontsize=12)

    handles = [mpatches.Patch(color=c, label=a) for a, c in algo_colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               facecolor="white", edgecolor="#cccccc", labelcolor="black",
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    return save_fig(fig, "cluster_size_comparison", subdir="clustering")


# ---------------------------------------------------------------------------
# Section 5 — PPI enrichment
# ---------------------------------------------------------------------------

def plot_ppi_pvalue_heatmap(cancer_list: list[str]):
    print("[5.1] PPI Enrichment Q-value Heatmap")
    records = []
    for cancer in cancer_list:
        df = load_ppi_enrichment(cancer)
        if df is None:
            continue
        use_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["q_val", "qval", "p_val", "pval"])),
            None,
        )
        if use_col:
            records.append({"cancer": cancer, "median_q": df[use_col].dropna().median()})

    if not records:
        return

    data     = pd.DataFrame(records).set_index("cancer")
    log_vals = -np.log10(data["median_q"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(3, max(6, len(cancer_list) * 0.35)), facecolor="white")
    ax.set_facecolor("white")
    im = ax.imshow(
        log_vals.values.reshape(-1, 1), aspect="auto",
        cmap="YlOrRd", interpolation="none",
        vmin=0, vmax=max(log_vals.max(), 5),
    )
    ax.set_xticks([0])
    ax.set_xticklabels(["Median Q"], color="black", fontsize=9)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data.index, fontsize=8, color="black")

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("-log₁₀(q-value)", color="black", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")

    return save_fig(fig, "ppi_enrichment_qvalue_heatmap", subdir="ppi_enrichment")


def plot_ppi_cluster_significance(cancer: str):
    print(f"[5.2] PPI Cluster Significance — {cancer}")
    df = load_ppi_enrichment(cancer)
    if df is None:
        return

    pval_col    = next(
        (c for c in df.columns if any(k in c.lower() for k in ["q_val", "qval", "p_val", "pval"])),
        df.columns[-1],
    )
    cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
    labels      = df[cluster_col].astype(str).tolist() if cluster_col else [f"Row {i + 1}" for i in range(len(df))]
    vals        = -np.log10(df[pval_col].clip(lower=1e-300).values)

    fig, ax = plt.subplots(figsize=(max(8, len(vals) * 0.5), 5), facecolor="white")
    ax.set_facecolor("#f8f9fa")

    colors_bar = ["#e94560" if v >= -np.log10(0.05) else "#00b4d8" for v in vals]
    ax.bar(range(len(vals)), vals, color=colors_bar, edgecolor="none")
    ax.axhline(-np.log10(0.05), color="#e84118", linewidth=1.5,
               linestyle="--", label="q = 0.05 threshold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="black")
    ax.set_ylabel("-log₁₀(q-value)", color="black", fontsize=10)
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.5)
    ax.legend(facecolor="white", edgecolor="#cccccc", labelcolor="black", fontsize=9)

    return save_fig(fig, f"ppi_cluster_significance_{cancer}", subdir="ppi_enrichment")


def plot_ppi_summary_volcano(cancer_list: list[str]):
    print("[5.3] PPI Summary Volcano Plot")
    records = []
    for cancer in cancer_list:
        df = load_ppi_enrichment(cancer)
        if df is None:
            continue
        pval_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["q_val", "qval", "p_val", "pval"])),
            df.columns[-1],
        )
        size_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["size", "count", "n_gene", "nodes"])),
            None,
        )
        for _, row in df.iterrows():
            pv   = float(row[pval_col]) if not pd.isna(row[pval_col]) else 1.0
            size = float(row[size_col]) if size_col and not pd.isna(row[size_col]) else 10
            records.append({"cancer": cancer, "pvalue": pv, "size": size})

    if not records:
        return

    data           = pd.DataFrame(records)
    data["log_q"]  = -np.log10(data["pvalue"].clip(lower=1e-300))
    data["log_sz"] = np.log2(data["size"].clip(lower=1))
    data["sig"]    = data["pvalue"] < 0.05

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("white")

    for sig, group in data.groupby("sig"):
        color = "#e94560" if sig else "#b0b8c1"
        ax.scatter(group["log_sz"], group["log_q"], c=color, s=40, alpha=0.75,
                   edgecolors="white", linewidths=0.5,
                   label="Significant (q<0.05)" if sig else "Not significant")

    ax.axhline(-np.log10(0.05), color="#e84118", linestyle="--", linewidth=1.2, label="q=0.05")
    ax.set_xlabel("log₂(Cluster Size / Nodes)", color="black", fontsize=11)
    ax.set_ylabel("-log₁₀(FDR q-value)", color="black", fontsize=11)
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.grid(True, color="#e0e0e0", linewidth=0.5, alpha=0.6)
    ax.legend(facecolor="white", edgecolor="#cccccc", labelcolor="black", fontsize=9)

    return save_fig(fig, "ppi_summary_volcano", subdir="ppi_enrichment")


# ---------------------------------------------------------------------------
# Section 6 — Permutation test
# ---------------------------------------------------------------------------

def plot_permutation_pvalue_summary(cancer_list: list[str]):
    print("[6.1] Permutation P-value Summary")
    records = []
    for cancer in cancer_list:
        df = load_permutation(cancer)
        if df is None:
            continue
        pv = float(df["empirical_pvalue"].iloc[0])
        if pv == 0.0:
            pv = 0.001
        records.append({"cancer": cancer, "pvalue": pv})

    data = pd.DataFrame(records).dropna(subset=["pvalue"])
    if data.empty:
        return

    data     = data.sort_values("pvalue")
    data["log_p"]   = -np.log10(data["pvalue"].clip(lower=1e-10))
    colors_bar = [CANCER_COLORS.get(c, "#888") for c in data["cancer"]]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    ax.set_facecolor("#f8f9fa")
    bars = ax.bar(data["cancer"], data["log_p"], color=colors_bar, edgecolor="none", width=0.7)
    ax.axhline(-np.log10(0.05), color="#e84118", linestyle="--", linewidth=1.5, label="p = 0.05")

    for bar, pv in zip(bars, data["pvalue"]):
        label = f"{pv:.3f}" if pv >= 0.001 else "<0.001"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                label, ha="center", va="bottom", color="black", fontsize=6, rotation=90)

    ax.set_xlabel("Cancer Type", color="black", fontsize=11)
    ax.set_ylabel("-log₁₀(Empirical p-value)", color="black", fontsize=11)
    ax.set_xticklabels(data["cancer"], rotation=45, ha="right", fontsize=8, color="black")
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.grid(True, axis="y", color="#e0e0e0", linewidth=0.5)
    ax.legend(facecolor="white", edgecolor="#cccccc", labelcolor="black", fontsize=9)

    return save_fig(fig, "permutation_pvalue_summary", subdir="permutation")


def plot_permutation_distribution(cancer: str, original_pvalue: float | None = None):
    print(f"  [SKIP] Permutation Distribution — {cancer}: distribution data not available in .txt format.")
    return None


def plot_permutation_pvalue_ranking(cancer_list: list[str]):
    print("[6.3] Permutation P-value Lollipop Ranking")
    records = []
    for cancer in cancer_list:
        df = load_permutation(cancer)
        pv = float(df["empirical_pvalue"].iloc[0]) if df is not None else np.nan
        if pv == 0.0:
            pv = 0.001
        records.append({"cancer": cancer, "pvalue": pv})

    data = pd.DataFrame(records).sort_values("pvalue", na_position="last")

    fig, ax = plt.subplots(figsize=(6, max(8, len(data) * 0.4)), facecolor="white")
    ax.set_facecolor("white")

    for i, row in data.iterrows():
        pv    = row["pvalue"] if not np.isnan(row["pvalue"]) else 1.0
        color = CANCER_COLORS.get(row["cancer"], "#888")
        rank  = list(data.index).index(i)
        ax.plot([0, pv], [rank, rank], color="#cccccc", linewidth=1)
        ax.scatter(pv, rank, s=80, color=color, edgecolors="white", linewidths=0.5, zorder=3)

    ax.axvline(0.05, color="#e84118", linewidth=1.2, linestyle="--", label="p = 0.05")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["cancer"].tolist(), fontsize=8, color="black")
    ax.set_xlabel("Empirical P-value (permutation)", color="black", fontsize=10)
    ax.tick_params(colors="black")
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.grid(True, axis="x", color="#e0e0e0", linewidth=0.5)
    ax.legend(facecolor="white", edgecolor="#cccccc", labelcolor="black", fontsize=9)

    return save_fig(fig, "permutation_pvalue_ranking", subdir="permutation")


# ---------------------------------------------------------------------------
# Section 7 — Summary dashboard
# ---------------------------------------------------------------------------

def plot_summary_dashboard(cancer_list: list[str]):
    print("[7.1] Summary Dashboard")
    fig = plt.figure(figsize=(22, 16), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32,
                            left=0.07, right=0.95, top=0.93, bottom=0.08)

    # Panel A — driver gene count
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor("#f8f9fa")
    gene_counts = [
        len(load_general_results(c)) if load_general_results(c) is not None else 0
        for c in cancer_list
    ]
    ax_a.bar(cancer_list, gene_counts,
             color=[CANCER_COLORS.get(c, "#888") for c in cancer_list], edgecolor="none")
    ax_a.set_xticklabels(cancer_list, rotation=45, ha="right", fontsize=7, color="black")
    ax_a.set_ylabel("# Driver Genes Selected", color="black", fontsize=9)
    ax_a.set_title("A   Driver Gene Count per Cancer", color="black",
                   fontsize=11, fontweight="bold", loc="left", pad=8)
    ax_a.tick_params(colors="black", labelsize=7)
    for sp in ax_a.spines.values():
        sp.set_edgecolor("#cccccc")
    ax_a.grid(True, axis="y", color="#e0e0e0", linewidth=0.5)

    # Panel B — PPI enrichment significance
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor("#f8f9fa")
    sig_rates = []
    for cancer in cancer_list:
        df = load_ppi_enrichment(cancer)
        if df is None:
            sig_rates.append(0)
            continue
        pval_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["q_val", "qval", "p_val", "pval"])),
            df.columns[-1],
        )
        sig_rates.append((df[pval_col] < 0.05).mean() * 100)
    ax_b.bar(cancer_list, sig_rates,
             color=[CANCER_COLORS.get(c, "#888") for c in cancer_list], edgecolor="none")
    ax_b.axhline(50, color="#e84118", linewidth=1, linestyle="--", alpha=0.7)
    ax_b.set_xticklabels(cancer_list, rotation=45, ha="right", fontsize=7, color="black")
    ax_b.set_ylabel("% Clusters with q < 0.05", color="black", fontsize=9)
    ax_b.set_title("B   PPI Enrichment Significance", color="black",
                   fontsize=11, fontweight="bold", loc="left", pad=8)
    ax_b.tick_params(colors="black", labelsize=7)
    for sp in ax_b.spines.values():
        sp.set_edgecolor("#cccccc")
    ax_b.grid(True, axis="y", color="#e0e0e0", linewidth=0.5)

    # Panel C — permutation test lollipop
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor("white")
    perm_pvals = []
    for cancer in cancer_list:
        df = load_permutation(cancer)
        pv = float(df["empirical_pvalue"].iloc[0]) if df is not None else np.nan
        perm_pvals.append(pv if not np.isnan(pv) else 1.0)

    sort_idx       = np.argsort(perm_pvals)
    sorted_cancers = [cancer_list[i] for i in sort_idx]
    sorted_pvals   = [perm_pvals[i]  for i in sort_idx]

    for rank, (cancer, pv) in enumerate(zip(sorted_cancers, sorted_pvals)):
        ax_c.plot([0, pv], [rank, rank], color="#cccccc", linewidth=1)
        ax_c.scatter(pv, rank, s=60, color=CANCER_COLORS.get(cancer, "#888"),
                     edgecolors="white", linewidths=0.4, zorder=3)

    ax_c.axvline(0.05, color="#e84118", linewidth=1.2, linestyle="--")
    ax_c.set_yticks(range(len(sorted_cancers)))
    ax_c.set_yticklabels(sorted_cancers, fontsize=7, color="black")
    ax_c.set_xlabel("Empirical P-value", color="black", fontsize=9)
    ax_c.set_title("C   Permutation Test Robustness", color="black",
                   fontsize=11, fontweight="bold", loc="left", pad=8)
    ax_c.tick_params(colors="black", labelsize=7)
    for sp in ax_c.spines.values():
        sp.set_edgecolor("#cccccc")
    ax_c.grid(True, axis="x", color="#e0e0e0", linewidth=0.5)

    # Panel D — Jaccard similarity heatmap
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor("#f8f9fa")

    gene_sets = {
        cancer: (
            set(
                load_general_results(cancer)[
                    next(
                        (c for c in load_general_results(cancer).columns if "gene" in c.lower()),
                        load_general_results(cancer).columns[0],
                    )
                ].dropna().tolist()
            )
            if load_general_results(cancer) is not None
            else set()
        )
        for cancer in cancer_list
    }

    n       = len(cancer_list)
    jaccard = np.zeros((n, n))
    for i, c1 in enumerate(cancer_list):
        for j, c2 in enumerate(cancer_list):
            s1, s2 = gene_sets[c1], gene_sets[c2]
            union  = len(s1 | s2)
            jaccard[i, j] = len(s1 & s2) / union if union > 0 else 0

    im = ax_d.imshow(jaccard, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")
    ax_d.set_xticks(range(n))
    ax_d.set_yticks(range(n))
    ax_d.set_xticklabels(cancer_list, rotation=90, fontsize=6, color="black")
    ax_d.set_yticklabels(cancer_list, fontsize=6, color="black")

    cb = fig.colorbar(im, ax=ax_d, pad=0.01, fraction=0.046)
    cb.set_label("Jaccard Similarity", color="black", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="black")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="black")
    ax_d.set_title("D   Driver Gene Overlap (Jaccard)", color="black",
                   fontsize=11, fontweight="bold", loc="left", pad=8)

    fig.suptitle("Comprehensive Results Dashboard — Cancer Driver Gene Detection",
                 color="black", fontsize=16, fontweight="bold", y=0.975)
    return save_fig(fig, "summary_dashboard", subdir="")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(
    general_results_dir: str | None = None,
    clustering_dir:      str | None = None,
    ppi_dir:             str | None = None,
    permutation_dir:     str | None = None,
    draw_data_overview: bool = True,
    draw_general:       bool = True,
    draw_clustering:    bool = True,
    draw_ppi:           bool = True,
    draw_permutation:   bool = True,
    draw_dashboard:     bool = True,
):
    global GENERAL_RESULTS_DIR, CLUSTERING_RESULTS_DIR, PPI_ENRICHMENT_DIR, PERMUTATION_DIR
    if general_results_dir:  GENERAL_RESULTS_DIR    = general_results_dir
    if clustering_dir:       CLUSTERING_RESULTS_DIR = clustering_dir
    if ppi_dir:              PPI_ENRICHMENT_DIR      = ppi_dir
    if permutation_dir:      PERMUTATION_DIR         = permutation_dir

    cancer_list = list(CANCER_COLORS.keys())
    if not cancer_list:
        print("[ERROR] CANCER_COLORS is empty.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Found {len(cancer_list)} cancer types.")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 60}\n")

    def get_mutation_file(cancer_full_name: str) -> str | None:
        import re
        match   = re.search(r'\((.*?)\)', cancer_full_name)
        acronym = match.group(1) if match else cancer_full_name
        folder  = os.path.join(DATA_ROOT, cancer_full_name)

        if not os.path.isdir(folder):
            matched = glob.glob(os.path.join(DATA_ROOT, f"*{acronym}*"))
            if not matched:
                return None
            folder = matched[0]

        zscore_files = glob.glob(os.path.join(folder, f"{acronym}_mutation*zscore_all.csv"))
        if zscore_files:
            return zscore_files[0]

        fallback = glob.glob(os.path.join(folder, f"{acronym}_mutation*.csv"))
        return fallback[0] if fallback else None

    if draw_data_overview:
        print("\n[== SECTION 2: DATA OVERVIEW ==]")
        for cancer in cancer_list:
            mut_csv = get_mutation_file(cancer)
            if mut_csv:
                plot_mutation_frequency_scatter(mut_csv, top_n=25, cancer_name=cancer)
                plot_patient_gene_heatmap(mut_csv, cancer_name=cancer)
            else:
                print(f"  [SKIP] Data overview: mutation file not found for {cancer}")

    if draw_general:
        print("\n[== SECTION 3: GENERAL RESULTS ==]")
        plot_mutation_burden_barplot(cancer_list)
        plot_driver_gene_summary_all_cancers(cancer_list)
        plot_gene_frequency_across_cancers(cancer_list, top_n_genes=30)

        for cancer in cancer_list:
            mut_csv = get_mutation_file(cancer)
            plot_oncoprint(cancer, mutation_csv=mut_csv if mut_csv else None)

        from collections import Counter
        all_driver_genes = []
        for cancer in cancer_list:
            df_res = load_general_results(cancer)
            if df_res is not None:
                gene_col = next(
                    (c for c in df_res.columns if "gene" in c.lower()), df_res.columns[0]
                )
                all_driver_genes.extend(df_res[gene_col].dropna().tolist())

        top_genes = [gene for gene, _ in Counter(all_driver_genes).most_common(50)]
        if top_genes:
            plot_advanced_multicancer_oncoprint(
                gene_list=top_genes,
                cancer_list=cancer_list,
                title=f"Top {len(top_genes)} Pan-Cancer Driver Genes",
            )

    if draw_clustering:
        print("\n[== SECTION 4: CLUSTERING ==]")
        plot_cluster_size_comparison(cancer_list)
        for cancer in cancer_list:
            for algo in CLUSTERING_ALGORITHMS:
                for itype in INTERACTION_TYPES:
                    plot_cluster_heatmap(cancer, algo=algo, interaction=itype)

    if draw_ppi:
        print("\n[== SECTION 5: PPI ENRICHMENT ==]")
        plot_ppi_pvalue_heatmap(cancer_list)
        plot_ppi_summary_volcano(cancer_list)
        for cancer in cancer_list:
            plot_ppi_cluster_significance(cancer)

    if draw_permutation:
        print("\n[== SECTION 6: PERMUTATION TEST ==]")
        plot_permutation_pvalue_summary(cancer_list)
        plot_permutation_pvalue_ranking(cancer_list)
        for cancer in cancer_list:
            plot_permutation_distribution(cancer)

    if draw_dashboard:
        print("\n[== SECTION 7: SUMMARY DASHBOARD ==]")
        plot_summary_dashboard(cancer_list)

    print(f"\n{'=' * 60}")
    print(f"  Done! All figures saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    run_all(
        draw_data_overview=True,
        draw_general=True,
        draw_clustering=True,
        draw_ppi=True,
        draw_permutation=False,
        draw_dashboard=True,
    )
