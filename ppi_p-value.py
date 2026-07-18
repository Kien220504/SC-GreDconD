import requests
import json
import pandas as pd
import os
import time
import warnings
from pathlib import Path
import concurrent.futures
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path("mcc_v2_result_clustering")


def get_ppi_enrichment(gene_list, species_id=9606, caller_id="my_cluster_test"):
    url = "https://string-db.org/api/json/ppi_enrichment"

    params = {
        "identifiers": "\r".join(gene_list),
        "species": species_id,
        "caller_identity": caller_id
    }

    try:
        response = requests.post(url, data=params)
        response.raise_for_status()

        result = response.json()

        if result and isinstance(result, list):
            p_value = result[0].get("p_value")
            number_of_nodes = result[0].get("number_of_nodes")
            number_of_edges = result[0].get("number_of_edges")
            expected_edges = result[0].get("expected_number_of_edges")

            return {
                "p_value": p_value,
                "nodes": number_of_nodes,
                "edges": number_of_edges,
                "expected_edges": expected_edges
            }
        else:
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None


def analyze_one_cluster(cluster, log_dir, algo_name, cluster_id):
    enrichment_result = get_ppi_enrichment(cluster)
    time.sleep(1.0)
    log_file_path = f"{log_dir}/cluster_{cluster_id}.txt"
    algo_dict = {
        "lloyd_label": "Lloyd",
        "gmm_label": "Gaussian Mixture",
        "fcm_label": "Fuzzy clustering"
    }

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Algorithm: {algo_dict.get(algo_name)}\n")
        f.write(f"Cluster ID: {cluster_id}\n")
        gene_str = ", ".join(str(g) for g in cluster)
        f.write(f"Genes: {gene_str}\n")
        f.write("-" * 40 + "\n")

        if enrichment_result:
            f.write(f"Number of genes: {enrichment_result['nodes']}\n")
            f.write(f"Real number of interactions: {enrichment_result['edges']}\n")
            f.write(f"Expected number of interactions: {enrichment_result['expected_edges']}\n")
            f.write(f"--> PPI Enrichment p-value: {enrichment_result['p_value']}\n")
            return enrichment_result
        else:
            f.write("Cannot retrieve data.\n")
            return None


def process_single_folder(cancer_folder):
    res_dir = f"mcc_v2_result_ppi_enrichment_p-value/{cancer_folder.name}"
    os.makedirs(res_dir, exist_ok=True)

    matrix_types = ['known_interactions', 'other_interactions']

    for mat_type in matrix_types:
        csv_path = cancer_folder / f"{mat_type}_all_labels.csv"
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        algo_columns = ['lloyd_label', 'gmm_label', 'fcm_label']

        for algo in algo_columns:
            log_dir = f"{res_dir}/{mat_type}/{algo}"
            os.makedirs(log_dir, exist_ok=True)
            sum_file = f"{log_dir}/p-value_sum.csv"

            unique_cluster_ids = df[algo].unique()
            enrichment_list = []
            for c_id in unique_cluster_ids:
                if pd.isna(c_id):
                    continue

                genes_in_cluster = df[df[algo] == c_id]['Sample_Name'].tolist()

                if len(genes_in_cluster) < 2:
                    continue

                enrichment_result = analyze_one_cluster(
                    cluster=genes_in_cluster,
                    log_dir=log_dir,
                    algo_name=algo,
                    cluster_id=int(c_id)
                )

                if enrichment_result is not None:
                    enrichment_result["cluster_id"] = c_id
                    enrichment_list.append(enrichment_result)
                else:
                    enrichment_list.append({
                        "p_value": pd.NA,
                        "nodes": pd.NA,
                        "edges": pd.NA,
                        "expected_edges": pd.NA,
                        "cluster_id": c_id
                    })

            if enrichment_list:
                df_res = pd.DataFrame(enrichment_list)
                cols = ["cluster_id"] + [c for c in df_res.columns if c != "cluster_id"]
                df_res = df_res[cols]

                valid_mask = df_res["p_value"].notna()

                if valid_mask.any():
                    alpha = 0.05
                    reject, q_values, _, _ = multipletests(
                        df_res.loc[valid_mask, "p_value"],
                        alpha=alpha,
                        method='fdr_bh'
                    )
                    df_res.loc[valid_mask, "FDR_q_value"] = q_values
                    df_res.loc[valid_mask, "is_significant"] = df_res.loc[valid_mask, "FDR_q_value"] < alpha
                else:
                    df_res["FDR_q_value"] = pd.NA
                    df_res["is_significant"] = pd.NA

                df_save = df_res.sort_values(by='p_value', na_position='last').reset_index(drop=True)
                df_save.to_csv(sum_file, index=False)

    return f"Done processed {cancer_folder.name}"


def main():
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


if __name__ == "__main__":
    main()
