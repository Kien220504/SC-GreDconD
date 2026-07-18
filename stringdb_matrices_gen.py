import requests
import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
import concurrent.futures


def get_string_matrices(gene_list, species=9606):
    base_url = "https://string-db.org/api/json/network"
    params = {
        "identifiers": "%0d".join(gene_list),
        "species": species,
        "caller_identity": "user_matrix_project"
    }

    try:
        response = requests.post(base_url, data=params, timeout=30)
    except Exception as e:
        print("Request error:", e)
        return None, None

    if response.status_code != 200:
        print("ERROR:", response.status_code)
        return None, None

    data = response.json()

    if len(data) == 0:
        return None, None

    original_genes = set(gene_list)
    found_genes = set()

    for interaction in data:
        found_genes.add(interaction['preferredName_A'])
        found_genes.add(interaction['preferredName_B'])

    all_genes = sorted(list(original_genes.union(found_genes)))
    n = len(all_genes)
    gene_index = {gene: i for i, gene in enumerate(all_genes)}

    matrix_1 = np.zeros((n, n))  # Known interactions weight (combined score)
    matrix_2 = np.zeros((n, n))  # Count of other evidence-type links

    print(f"Processing {len(data)} interactions...")

    for item in data:
        idx_a = gene_index[item['preferredName_A']]
        idx_b = gene_index[item['preferredName_B']]

        escore = item.get('escore', 0)  # Experimental
        dscore = item.get('dscore', 0)  # Database
        combined_score = item.get('score', 0)  # Combined PPI score

        if escore > 0 or dscore > 0:
            matrix_1[idx_a][idx_b] = combined_score
            matrix_1[idx_b][idx_a] = combined_score

        other_keys = [
            'nscore',  # Neighborhood
            'fscore',  # Fusions
            'pscore',  # Co-occurrence
            'tscore',  # Textmining
            'ascore',  # Co-expression
            'escore',
            'dscore'
        ]

        count_weight = 0
        for key in other_keys:
            if item.get(key, 0) > 0:
                count_weight += 1

        matrix_2[idx_a][idx_b] = count_weight
        matrix_2[idx_b][idx_a] = count_weight

    df_matrix_1 = pd.DataFrame(matrix_1, index=all_genes, columns=all_genes)
    df_matrix_2 = pd.DataFrame(matrix_2, index=all_genes, columns=all_genes)

    return df_matrix_1, df_matrix_2


def process_single_folder(cancer_folder):
    path = cancer_folder / "sc2gc_results_1.csv"
    try:
        df = pd.read_csv(path)
        gene_list = df['Genes'].tolist()
    except Exception as e:
        return f"Error {cancer_folder.name}: {e}"

    result_dir = Path(f'data/stringdb_matrices_mcc_v2/Cancer_Type/{cancer_folder.name}')
    result_dir.mkdir(parents=True, exist_ok=True)

    m1, m2 = get_string_matrices(gene_list)

    time.sleep(1.0)

    if m1 is not None:
        m1.to_csv(result_dir / "matrix_known_interactions.csv")
        m2.to_csv(result_dir / "matrix_other_counts.csv")
        return f"Done! Matrices saved: {cancer_folder.name} ({len(gene_list)} genes)"
    else:
        return f"Cannot create matrices: {cancer_folder.name}"


if __name__ == "__main__":
    DATA_DIR = Path("general_results_dynamic_k_mcc_v2")

    start_time = time.time()
    folders_to_process = [f for f in DATA_DIR.iterdir() if f.is_dir()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_folder, folder): folder for folder in folders_to_process}

        for future in concurrent.futures.as_completed(futures):
            folder = futures[future]
            try:
                result_message = future.result()
                print(result_message)
            except Exception as exc:
                print(f"Error at {folder.name}: {exc}")

    print(f"\nCompleted in {time.time() - start_time:.2f} seconds.")
