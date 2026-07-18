import sys
import os
import numpy as np
import pandas as pd
import time
from pathlib import Path
from data_loader import DataLoader
from gredcond import GREDCOND
from greedy_set_cover import weighted_greedy_set_cover
import utils

if len(sys.argv) < 3:
    print("Error: Missing matrix ID argument")
    sys.exit(1)

MATRIX_ID = int(sys.argv[1])
CANCER_TYPE = sys.argv[2]
DATA_DIR = Path("data/TCGA Cancer")
SHUFFLED_DIR = Path("data/shuffled_matrices")
RESULT_DIR = Path("data/shuffle_results")

FILES_USED = {
    "clinical_patient": "_clinical_patient.csv",
    "clinical_sample": "_clinical_sample.csv",
    "expression": "_expression_zscore_all.csv",
    "mutation": "_mutation_matched_expression_zscore_all.csv"
}

def run_process(mut_df, exp_df, max_factors=30):
    t0 = time.time()

    exp_df = exp_df.apply(pd.to_numeric, errors='coerce')
    exp_df = exp_df.fillna(0)

    # Set Cover
    sc_res, _, _, _ = weighted_greedy_set_cover(mut_df, exp_df)
    sc_genes = []
    if not sc_res.empty:
        if 'Gene' in sc_res.columns: sc_genes = sc_res['Gene'].values
        else: sc_genes = sc_res.index.values

    # Remove genes
    mut_reduced = mut_df.drop(columns=sc_genes, errors='ignore')
    exp_reduced = exp_df.drop(columns=sc_genes, errors='ignore')

    # GREDCOND
    gc_genes = []
    if mut_reduced.shape[1] > 0:
        I = mut_reduced.values
        exp_mat = exp_reduced.values
        genes_name = mut_reduced.columns.tolist()

        gc_analyzer = GREDCOND(I, genes_name)
        gc_res, _ = gc_analyzer.run_analyze(max_factors=max_factors)

        if not gc_res.empty:
            gc_genes = gc_res.index.values

    # New data
    total_genes = list(sc_genes) + list(gc_genes)
    mut_new = mut_df[total_genes]
    exp_new = exp_df[total_genes]

    # GREDCOND
    gc_genes_2 = []
    if mut_new.shape[1] >0:
        I_new = mut_new.values
        exp_mat = exp_new.values
        new_genes_name = mut_new.columns.tolist()

        gc_analyzer_2 = GREDCOND(I_new, new_genes_name)
        gc_res_2, _ = gc_analyzer_2.run_analyze(max_factors=max_factors)
        factors = gc_analyzer_2.factors

        if not gc_res_2.empty:
            gc_genes_2 = gc_res_2.index.values

    total_time = time.time() - t0

    return total_genes, sc_genes, gc_genes, gc_genes_2, factors, total_time

def main():
    
    shuffled_file = SHUFFLED_DIR / CANCER_TYPE / f"matrix_{MATRIX_ID}.npz"
    if not shuffled_file.exists():
        print(f"File not found: {shuffled_file}")
        return
    
    cancer_raw_dir = DATA_DIR / CANCER_TYPE
    if not cancer_raw_dir.exists():
        print(f"File not found: {cancer_raw_dir}")
        return

    paths = {}
    for key, suffix in FILES_USED.items():
        matches = list(cancer_raw_dir.glob(f"*{suffix}"))
        paths[key] = str(matches[0]) if matches else None

    if not all(paths.values()):
        return

    loader = DataLoader(
        paths["expression"],
        paths["mutation"],
        paths["clinical_patient"],
        paths["clinical_sample"]
    )
    exp, _, _, _ = loader.load()

    ground_truth = utils.load_ground_truth(CANCER_TYPE)

    data_pack = np.load(shuffled_file, allow_pickle=True)
    mut_shuffled_df = pd.DataFrame(
        data=data_pack['data'],
        index=data_pack['index'],
        columns=data_pack['columns']
    )
    
    _, _, _, res_genes, _, run_time = run_process(mut_shuffled_df, exp)
    
    total_genes = mut_shuffled_df.shape[1]
    p_value = utils.calculate_hypergeom_score(res_genes, ground_truth, total_genes_count=total_genes)
        
    cancer_res_dir = RESULT_DIR / CANCER_TYPE
    cancer_res_dir.mkdir(parents=True, exist_ok=True)
    res_file = cancer_res_dir / f"result_{MATRIX_ID}.txt"

    with open(res_file, "w") as f:
        f.write(f"{MATRIX_ID},{p_value},{run_time}\n")
    
    print(f"Finished Matrix {MATRIX_ID}: P-value={p_value:.5g}, Time={run_time:.2f}s")

if __name__ == "__main__":
    main()
