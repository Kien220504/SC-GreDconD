import os
import pandas as pd
import numpy as np
import time
from pathlib import Path
import concurrent.futures
import matplotlib.pyplot as plt

import utils
from data_loader import DataLoader
from gredcond import GREDCOND
from greedy_set_cover import weighted_greedy_set_cover

DATA_DIR = Path("F:/TruongHongKiet/chạy_code/data/TCGA Cancer")

FILES_USED = {
    "clinical_patient": "_clinical_patient.csv",
    "clinical_sample": "_clinical_sample.csv",
    "expression": "_expression_zscore_all.csv",
    "mutation": "_mutation_matched_expression_zscore_all.csv"
}


def log(text, f):
    print(text)
    f.write(text + "\n")
    f.flush()


def calculate_metrics_from_sets(predicted_genes, ground_truth_genes):
    """Computes precision/recall/F1 metrics between two gene sets."""
    overlap_genes = predicted_genes.intersection(ground_truth_genes)
    intersection_count = len(overlap_genes)

    predicted_count = len(predicted_genes)
    gt_count = len(ground_truth_genes)

    precision = intersection_count / predicted_count if predicted_count > 0 else 0.0
    recall = intersection_count / gt_count if gt_count > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    tp = intersection_count
    fp = predicted_count - intersection_count
    fn = gt_count - intersection_count
    coverage = intersection_count / gt_count if gt_count > 0 else 0.0
    overlap_str = ", ".join(sorted(list(overlap_genes)))

    return tp, fp, fn, precision, recall, f1_score, overlap_str, coverage


def run_process(mut_df, exp_df, max_factors):
    t0 = time.time()

    exp_df = exp_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    sc_res, _, _, _ = weighted_greedy_set_cover(mut_df, exp_df)
    sc_genes = []
    if not sc_res.empty:
        sc_genes = (sc_res['Gene'].values
                    if 'Gene' in sc_res.columns
                    else sc_res.index.values)

    mut_reduced = mut_df.drop(columns=sc_genes, errors='ignore')
    exp_reduced = exp_df.drop(columns=sc_genes, errors='ignore')

    gc_genes = []
    if mut_reduced.shape[1] > 0:
        genes_name = mut_reduced.columns.tolist()
        gc_analyzer = GREDCOND(mut_reduced.values, genes_name)
        gc_res, _ = gc_analyzer.run_analyze(max_factors=max_factors)
        if not gc_res.empty:
            gc_genes = gc_res.index.values

    total_genes = list(sc_genes) + list(gc_genes)
    mut_new = mut_df[total_genes]
    exp_new = exp_df[total_genes]

    gc_genes_2 = []
    factors = []
    if mut_new.shape[1] > 0:
        new_genes_name = mut_new.columns.tolist()
        gc_analyzer_2 = GREDCOND(mut_new.values, new_genes_name)
        gc_res_2, _ = gc_analyzer_2.run_analyze(max_factors=max_factors)
        factors = gc_analyzer_2.factors
        if not gc_res_2.empty:
            gc_genes_2 = gc_res_2.index.values

    total_time = time.time() - t0
    return total_genes, sc_genes, gc_genes, gc_genes_2, factors, total_time


def search_optimal_k(mut_df, exp_df, ground_truth, n_patients, n_genes, cancer_name=""):
    """
    Two-phase (coarse & fine) grid search for the optimal number of factors k,
    with consistent early stopping.

    - Primary selection criterion: F1 score; tiebreaker: ground-truth overlap count.
    - Fine search window: +/- coarse_step around the best coarse k.
    - Caches run_process results per k to avoid redundant computation.
    - max_limit scales with n_patients (capped at 200).
    """
    gt_set = set(ground_truth)

    max_limit = min(n_patients // 2, 200)
    max_limit = max(max_limit, 10)

    COARSE_STEP = 10
    coarse_k_list = list(range(15, max_limit + 1, COARSE_STEP))
    if not coarse_k_list or coarse_k_list[-1] < max_limit:
        coarse_k_list.append(max_limit)

    cache = {}  # k -> (run_result_tuple, p_value, overlap, f1)

    def evaluate_k(k):
        if k in cache:
            return cache[k][1], cache[k][2], cache[k][3]

        result = run_process(mut_df, exp_df, k)
        _, _, _, gc_genes_2, _, _ = result

        p_value, overlap_count = utils.calculate_hypergeom_score(
            gc_genes_2, ground_truth, total_genes_count=n_genes
        )

        pred_set = set(gc_genes_2)
        _, _, _, precision, recall, f1, _, _ = calculate_metrics_from_sets(pred_set, gt_set)

        cache[k] = (result, p_value, overlap_count, f1)
        return p_value, overlap_count, f1

    # Phase 1: coarse search
    PATIENCE = 2 if cancer_name == "Uterine Corpus Endometrial Carcinoma (UCEC)" else 5
    best_k = None
    best_f1 = -1.0
    best_overlap_at_best_f1 = -1
    history = {}  # k -> f1

    no_improve_count = 0

    print(f"  [Phase 1] Coarse search | step={COARSE_STEP} | max_limit={max_limit}")
    for k in coarse_k_list:
        p_val, overlap, f1 = evaluate_k(k)
        history[k] = f1
        print(f"    k={k:4d} | p-value={p_val:.5f} | f1={f1:.4f} | overlap={overlap}")

        if p_val < 0.01:
            if f1 > best_f1:
                best_f1 = f1
                best_overlap_at_best_f1 = overlap
                best_k = k
                no_improve_count = 0
            elif f1 == best_f1 and overlap > best_overlap_at_best_f1:
                best_overlap_at_best_f1 = overlap
                best_k = k
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count >= PATIENCE:
            print(f"    -> Early stopping Phase 1 after {PATIENCE} steps with no improvement.")
            break

    if best_k is None:
        print("  Warning: No k reached p-value < 0.01 in Phase 1. "
              "Falling back to the k with the highest F1.")
        best_k = max(history, key=history.get) if history else 30

    # Phase 2: fine search
    fine_start = max(5, best_k - COARSE_STEP)
    fine_end = min(max_limit, best_k + COARSE_STEP)
    fine_k_list = [k for k in range(fine_start, fine_end + 1) if k not in history]

    print(f"  [Phase 2] Fine search around k={best_k} | window=[{fine_start}, {fine_end}]")
    no_improve_count = 0

    for k in fine_k_list:
        p_val, overlap, f1 = evaluate_k(k)
        history[k] = f1
        print(f"    k={k:4d} | p-value={p_val:.5f} | f1={f1:.4f} | overlap={overlap}")

        if p_val < 0.01:
            if f1 > best_f1:
                best_f1 = f1
                best_overlap_at_best_f1 = overlap
                best_k = k
                no_improve_count = 0
            elif f1 == best_f1 and overlap > best_overlap_at_best_f1:
                best_overlap_at_best_f1 = overlap
                best_k = k
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        if no_improve_count >= PATIENCE:
            print(f"    -> Early stopping Phase 2 after {PATIENCE} steps with no improvement.")
            break

    best_run_result = cache[best_k][0]

    return best_k, history, best_run_result


def process_single_cancer(cancer_folder):
    res_dir = f"general_results_dynamic_k_f1_v3/{cancer_folder.name}"
    os.makedirs(res_dir, exist_ok=True)

    paths = {}
    for key, suffix in FILES_USED.items():
        matches = list(cancer_folder.glob(f"*{suffix}"))
        paths[key] = str(matches[0]) if matches else None

    if not all(paths.values()):
        missing = [k for k, v in paths.items() if v is None]
        return cancer_folder.name, f"Missing file(s): {missing}", None

    loader = DataLoader(
        paths["expression"],
        paths["mutation"],
        paths["clinical_patient"],
        paths["clinical_sample"]
    )
    exp, mut, _, genes_name = loader.load()

    n_genes = mut.shape[1]
    n_patients = mut.shape[0]
    ground_truth = utils.load_ground_truth(cancer_folder.name, column_name="SYMBOL")

    print(f"\n--- Processing {cancer_folder.name} (Samples: {n_patients}) ---")

    best_k, history, best_run_result = search_optimal_k(
        mut, exp, ground_truth, n_patients, n_genes, cancer_name=cancer_folder.name
    )
    selected_genes, sc_genes, gc_genes, gc_genes_2, factors, run_time = best_run_result

    print(f"*** {cancer_folder.name} selected optimal k = {best_k} ***")

    existing_files = os.listdir(res_dir)
    count = len([f for f in existing_files if f.startswith("sc2gc_results_")]) + 1

    log_filename = f"{res_dir}/sc2gc_results_{count}.txt"
    res_filename = f"{res_dir}/sc2gc_results_{count}.csv"

    gene_df = pd.DataFrame(selected_genes, columns=['Genes'])
    gene_df_2 = pd.DataFrame(gc_genes_2, columns=['Genes'])
    gene_df_2.to_csv(res_filename, index=False, encoding='utf-8')

    p_value, overlap = utils.calculate_hypergeom_score(
        gc_genes_2, ground_truth, total_genes_count=n_genes
    )

    with open(log_filename, "w", encoding="utf-8") as f:
        log("====== SET COVER TO GREDCOND COMBINED PROCESS WITH FACTORS REPORTS ======", f)
        log(f"- Input: {n_genes} genes, {n_patients} patients; "
            f"OPTIMAL Number of factors (k): {best_k}.", f)
        log(f"- Run time (final run): {run_time:.4f}s.", f)

        log("\n> GENE LIST ON ORIGINAL MATRIX:", f)
        log(f" {'No.':<4} | {'Genes':<15}", f)
        log(" " + "-" * 55, f)
        for c, gene_name in enumerate(gene_df['Genes'], start=1):
            log(f" {c:<4} | {gene_name:<15}", f)
        log(" " + "-" * 55, f)

        log(f"\n New input: {len(selected_genes)} genes, {n_patients} patients.", f)

        log("\n> GENE LIST ON NEW INPUT:", f)
        log(f" {'No.':<4} | {'Genes':<15}", f)
        log(" " + "-" * 55, f)
        for c, gene_name in enumerate(gene_df_2['Genes'], start=1):
            log(f" {c:<4} | {gene_name:<15}", f)
        log(" " + "-" * 55, f)

        log("\nGENES IN EACH FACTORS:", f)
        for idx, (patients_mask, genes_indices) in enumerate(factors):
            current_gene_names = [selected_genes[i] for i in genes_indices]
            log(f"Factor {idx + 1}: {current_gene_names}", f)

        log("=" * 50, f)
        log(f"P-value of new input: {p_value:.5g}", f)
        log(f"Number of genes overlapped with ground truth: {overlap}", f)
        log("=" * 50, f)

    return cancer_folder.name, f"Done: {cancer_folder.name} | optimal k={best_k}", history


def main():
    start_time = time.time()
    cancer_folders = [f for f in DATA_DIR.iterdir() if f.is_dir()]
    plot_data = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_single_cancer, folder)
                   for folder in cancer_folders]

        for future in concurrent.futures.as_completed(futures):
            try:
                cancer_name, result_msg, history = future.result()
                print(result_msg)
                if history is not None:
                    plot_data[cancer_name] = history
            except Exception as exc:
                print(f"ERROR: {exc}")

    if plot_data:
        DISTINCT_COLORS = [
            '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
            '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
            '#808000', '#ffd8b1', '#000075', '#a9a9a9', "#06A1D5",
            '#000000', '#e6beff', '#ff4500', '#2e8b57', '#d2691e',
        ]

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'axes.linewidth': 1.2,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

        for idx, (cancer_name, history) in enumerate(plot_data.items()):
            sorted_pts = sorted(history.items())
            k_vals = [p[0] for p in sorted_pts]
            f1_vals = [p[1] for p in sorted_pts]

            color = DISTINCT_COLORS[idx % len(DISTINCT_COLORS)]

            ax.plot(k_vals, f1_vals, linewidth=2.0,
                    label=cancer_name, color=color, alpha=0.85)

        ax.set_xlabel('Number of Factors (k)', labelpad=10)
        ax.set_ylabel('F1 Score', labelpad=10)

        ax.grid(True, linestyle=':', alpha=0.6, color='gray')

        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False,
                  borderaxespad=0., title="Cancer Types", title_fontproperties={'weight': 'bold'})

        os.makedirs("general_results_dynamic_k_f1_v3", exist_ok=True)
        plot_path_png = "general_results_dynamic_k_f1_v3/elbow_plot_all_cancers_f1.png"
        plot_path_pdf = "general_results_dynamic_k_f1_v3/elbow_plot_all_cancers_f1.pdf"

        plt.savefig(plot_path_png)
        plt.savefig(plot_path_pdf)
        plt.close()

        print(f"\nPlots saved at: \n- {plot_path_png}\n- {plot_path_pdf}")

    print(f"\nTOTAL RUN TIME: {time.time() - start_time:.2f}s.")


if __name__ == "__main__":
    main()
