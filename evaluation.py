import os
import re

import pandas as pd
from scipy.stats import hypergeom

# Configuration
CLUSTER_BASE_FOLDER = "cluster_res"
PPI_BASE_FOLDER = "ppi_res"
GROUND_TRUTH_FOLDER = "F:/TruongHongKiet/chạy_code/data/Ground Truth"
GT_COLUMN_NAME = "SYMBOL"

MATRIX_TYPES = ["known_interactions", "other_interactions"]
CLUSTER_METHODS = ["lloyd_label", "gmm_label", "fcm_label"]
LABEL_FILE_NAMES = {
    "known_interactions": "known_interactions_all_labels.csv",
    "other_interactions": "other_interactions_all_labels.csv",
}

TOTAL_GENES_COUNT = 20511

OUTPUT_DIR = "eval_dir_new_final_test"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = f"{OUTPUT_DIR}/clustering_evaluation_results.csv"
OUTPUT_TXT = f"{OUTPUT_DIR}/clustering_evaluation_averages.txt"

DISEASES = [
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


def extract_abbreviation(disease_full_name: str) -> str:
    match = re.search(r"\((.*?)\)", disease_full_name)
    return match.group(1).strip() if match else disease_full_name


def load_ground_truth(disease_full_name: str) -> set:
    abbr = extract_abbreviation(disease_full_name)
    file_path = os.path.join(GROUND_TRUTH_FOLDER, f"{abbr}_driver_genes.csv")
    try:
        df = pd.read_csv(file_path)
        return set(df[GT_COLUMN_NAME].dropna().astype(str).str.upper().unique())
    except FileNotFoundError:
        print(f"  [!] Ground truth file not found: {file_path}")
        return set()
    except KeyError:
        print(f"  [!] Column '{GT_COLUMN_NAME}' not found in {file_path}")
        return set()


def load_label_file(disease_full_name: str, matrix_type: str) -> pd.DataFrame | None:
    path = os.path.join(CLUSTER_BASE_FOLDER, disease_full_name, LABEL_FILE_NAMES[matrix_type])
    if not os.path.exists(path):
        print(f"    [!] Missing label file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"    [!] Failed to read {path}: {exc}")
        return None


def load_pvalue_file(disease_full_name: str, matrix_type: str, method_col: str) -> pd.DataFrame | None:
    path = os.path.join(PPI_BASE_FOLDER, disease_full_name, matrix_type, method_col, "p-value_sum.csv")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"  [!] P-value file not found: {path}")
        return None
    except Exception as exc:
        print(f"  [!] Failed to read p-value file {path}: {exc}")
        return None


def average_pvalue(pvalue_df: pd.DataFrame | None) -> float:
    if pvalue_df is None or pvalue_df.empty:
        return float("nan")
    return float(pvalue_df["p_value"].mean())


def calculate_classification_metrics(label_df: pd.DataFrame, method_col: str, gt_genes: set):
    """Pool genes across all clusters and compute precision, recall, F1, MCC."""
    clustered_genes = set(
        label_df.loc[label_df[method_col].notna(), "Sample_Name"].astype(str).str.upper()
    )

    tp = len(clustered_genes & gt_genes)
    fp = len(clustered_genes) - tp
    fn = len(gt_genes) - tp
    tn = max(TOTAL_GENES_COUNT - tp - fp - fn, 0)

    precision = tp / len(clustered_genes) if clustered_genes else 0.0
    recall = tp / len(gt_genes) if gt_genes else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denominator if mcc_denominator > 0 else 0.0

    return precision, recall, f1_score, mcc, clustered_genes


def calculate_hypergeom_score(selected_genes: set, gt_genes: set) -> float:
    n = len(selected_genes)
    if n == 0:
        return 1.0
    k = len(selected_genes & gt_genes)
    K = len(gt_genes)
    return hypergeom.sf(k - 1, TOTAL_GENES_COUNT, K, n)


def calculate_per_cluster_enrichment(label_df: pd.DataFrame, method_col: str, gt_genes: set, clustered_genes: set):
    cluster_percentages = []
    for cluster_id in label_df[method_col].dropna().unique():
        cluster_genes = set(
            label_df.loc[label_df[method_col] == cluster_id, "Sample_Name"].astype(str).str.upper()
        )
        if cluster_genes:
            cluster_percentages.append(len(cluster_genes & gt_genes) / len(cluster_genes) * 100)

    avg_gt_percentage = sum(cluster_percentages) / len(cluster_percentages) if cluster_percentages else 0.0
    f_test_score = calculate_hypergeom_score(clustered_genes, gt_genes)
    return avg_gt_percentage, f_test_score


def run_pipeline() -> pd.DataFrame:
    records = []
    print("Computing clustering evaluation metrics...\n")

    for disease in DISEASES:
        abbr = extract_abbreviation(disease)
        gt_genes = load_ground_truth(disease)
        if not gt_genes:
            print(f"  [Skip] {abbr}: no ground truth available.\n")
            continue

        print(f"  Processing {abbr} ({len(gt_genes)} ground-truth genes)")

        for matrix_type in MATRIX_TYPES:
            label_df = load_label_file(disease, matrix_type)
            if label_df is None:
                continue

            for method_col in CLUSTER_METHODS:
                if method_col not in label_df.columns:
                    print(f"    [!] Column '{method_col}' missing in label file for {abbr}/{matrix_type}")
                    continue

                precision, recall, f1_score, mcc, clustered_genes = calculate_classification_metrics(
                    label_df, method_col, gt_genes
                )
                avg_ppi = average_pvalue(load_pvalue_file(disease, matrix_type, method_col))
                avg_gt_percentage, f_test_score = calculate_per_cluster_enrichment(
                    label_df, method_col, gt_genes, clustered_genes
                )

                records.append({
                    "Cancer": abbr,
                    "matrix_type": matrix_type,
                    "cluster_method": method_col.replace("_label", ""),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1_score, 4),
                    "mcc": round(mcc, 4),
                    "avg_ppi": round(avg_ppi, 6) if not pd.isna(avg_ppi) else float("nan"),
                    "avg_gt_percentage": round(avg_gt_percentage, 4),
                    "f_test_score": f_test_score,
                })

    print("\nAll diseases processed.\n")
    return pd.DataFrame(records)


def write_results(results_df: pd.DataFrame) -> None:
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Results saved to: {OUTPUT_CSV}")

    metric_columns = ["precision", "recall", "f1", "mcc", "avg_ppi", "avg_gt_percentage", "f_test_score"]
    avg_df = results_df.groupby(["matrix_type", "cluster_method"])[metric_columns].mean().reset_index()

    lines = ["=" * 65, f"AVERAGE METRICS ACROSS {results_df['Cancer'].nunique()} DISEASES", "=" * 65]
    for _, row in avg_df.iterrows():
        lines.append(f"\n[Matrix: {row['matrix_type']}]  [Method: {row['cluster_method']}]")
        lines.append(f"  avg precision       : {row['precision']:.4f}")
        lines.append(f"  avg recall          : {row['recall']:.4f}")
        lines.append(f"  avg f1              : {row['f1']:.4f}")
        lines.append(f"  avg mcc             : {row['mcc']:.4f}")
        lines.append(f"  avg ppi             : {row['avg_ppi']:.6f}")
        lines.append(f"  avg % GT in cluster : {row['avg_gt_percentage']:.4f}%")
        lines.append(f"  avg F-test score    : {row['f_test_score']:.2e}")
    lines.append("\n" + "=" * 65)

    summary_text = "\n".join(lines)
    print(summary_text)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n[OK] Averages saved to: {OUTPUT_TXT}")


def main() -> None:
    results_df = run_pipeline()
    if results_df.empty:
        print("[!] No valid results were produced.")
        return
    write_results(results_df)


if __name__ == "__main__":
    main()
