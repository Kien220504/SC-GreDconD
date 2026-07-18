import numpy as np
import skfuzzy as fuzz
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, ClusterWarning
import os
import time
import warnings
from pathlib import Path
import concurrent.futures

warnings.filterwarnings("ignore", category=ClusterWarning)

DATA_DIR = Path("data/stringdb_matrices_mcc_v2/Cancer_Type")


def print_cluster_info(method_name, k, labels, centers, feature_names, save_file, sample_names):
    def log(text):
        print(text)
        save_file.write(text + "\n")

    log(f"\n{'=' * 30}")
    log(f"METHOD: {method_name} | K = {k}")
    log(f"{'=' * 30}")

    unique_labels, counts = np.unique(labels, return_counts=True)

    for i, label in enumerate(unique_labels):
        log(f"\n--- Cluster {label} (Number of Genes: {counts[i]}) ---")
        indices = np.where(labels == label)[0]
        genes_in_cluster = sample_names[indices]
        gene_str = ", ".join(str(x) for x in genes_in_cluster)
        log(f"Genes: [{gene_str}]")

        log("Centroid:")
        current_center = centers[label]
        for name, val in zip(feature_names, current_center):
            log(f"  > {name}: {val:.4f}")

    save_file.write("\n" + "-" * 50 + "\n")


def find_best_k(algo_name, X):
    """Finds the largest k (up to min(50, len(X)//2)) for which every cluster has at least 2 members."""
    best_k = 2
    if len(X) < 4:
        return 2

    k = 2
    max_k = min(50, len(X) // 2)

    while k <= max_k:
        try:
            if algo_name == 'kmeans':
                model = KMeans(n_clusters=k, init='random', n_init=1, algorithm='lloyd', random_state=42)
                labels = model.fit_predict(X)
            elif algo_name == 'gmm':
                model = GaussianMixture(n_components=k, random_state=42)
                labels = model.fit_predict(X)
            elif algo_name == 'fcm':
                np.random.seed(42)
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data=X.T, c=k, m=2, error=0.005, maxiter=1000, init=None
                )
                labels = np.argmax(u, axis=0)

            unique, counts = np.unique(labels, return_counts=True)

            if np.min(counts) < 2:
                best_k = k - 1 if k > 2 else 2
                break

            best_k = k
            k += 1

        except Exception:
            best_k = k - 1 if k > 2 else 2
            break

    return best_k


def run_algorithms(df, type_of_matrice, RESULT_DIR):
    df_numeric = df.select_dtypes(include=[np.number])
    X = df_numeric.to_numpy()
    feature_names = df_numeric.columns

    df_obj = df.select_dtypes(exclude=[np.number])
    if not df_obj.empty:
        sample_names = df_obj.iloc[:, 0].astype(str).values
    else:
        sample_names = df.index.astype(str).values

    k_lloyd = find_best_k('kmeans', X)
    print(f" -> Best K for Lloyd: {k_lloyd}")
    k_gmm = find_best_k('gmm', X)
    print(f" -> Best K for GMM: {k_gmm}")
    k_fcm = find_best_k('fcm', X)
    print(f" -> Best K for FCM: {k_fcm}")

    txt_file_path = f"{RESULT_DIR}/{type_of_matrice}_cluster_info.txt"

    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(f"CLUSTERING RESULT: {type_of_matrice}\n")
        f.write("=" * 50 + "\n")

        start_time = time.time()

        # Lloyd's algorithm (K-Means)
        lloyd = KMeans(n_clusters=k_lloyd, init='random', n_init=1, algorithm='lloyd', random_state=42)
        lloyd_labels = lloyd.fit_predict(X)
        lloyd_centers = lloyd.cluster_centers_
        print_cluster_info("Lloyd (K-Means)", k_lloyd, lloyd_labels, lloyd_centers, feature_names, f, sample_names)

        # Soft clustering: Gaussian Mixture Model
        gmm = GaussianMixture(n_components=k_gmm, random_state=42)
        gmm.fit(X)
        gmm_labels = gmm.predict(X)
        print_cluster_info("Gaussian Mixture Model", k_gmm, gmm_labels, gmm.means_, feature_names, f, sample_names)

        # Soft clustering: Fuzzy C-Means
        np.random.seed(42)
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            data=X.T, c=k_fcm, m=2, error=0.005, maxiter=100000, init=None
        )
        fcm_labels = np.argmax(u, axis=0)
        print_cluster_info("Fuzzy C-Means", k_fcm, fcm_labels, cntr, feature_names, f, sample_names)

        elapsed_time = time.time() - start_time
        f.write(f"\nTOTAL RUNTIME: {elapsed_time:.2f} seconds.")

    df_master_labels = pd.DataFrame({
        'Sample_Name': sample_names,
        'lloyd_label': lloyd_labels,
        'gmm_label': gmm_labels,
        'fcm_label': fcm_labels
    })
    df_master_labels.to_csv(f"{RESULT_DIR}/{type_of_matrice}_all_labels.csv", index=False)


def process_single_folder(cancer_folder):
    res_dir = f"mcc_v2_result_clustering/{cancer_folder.name}"
    os.makedirs(res_dir, exist_ok=True)

    try:
        df_1 = pd.read_csv(cancer_folder / "matrix_known_interactions.csv")
        df_2 = pd.read_csv(cancer_folder / "matrix_other_counts.csv")
    except FileNotFoundError:
        print(f"Cannot find files in {cancer_folder.name}.")
        return

    data_dict = {
        'known_interactions': df_1,
        'other_interactions': df_2
    }

    for type_of_matrice, df_data in data_dict.items():
        run_algorithms(df_data, type_of_matrice, res_dir)

    return f"Done: {cancer_folder.name}"


def main():
    start_time = time.time()
    folders_to_process = [f for f in DATA_DIR.iterdir() if f.is_dir()]

    if not folders_to_process:
        print("Cannot find any folders!")
        return

    print(f"Found {len(folders_to_process)} cancer folders. Processing...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(process_single_folder, folder): folder
            for folder in folders_to_process
        }

        for future in concurrent.futures.as_completed(futures):
            folder = futures[future]
            try:
                result_message = future.result()
                print(result_message)
            except Exception as exc:
                print(f"Error at {folder.name}: {exc}")

    elapsed_time = time.time() - start_time

    print(f"DONE! TOTAL ELAPSED TIME: {elapsed_time} s.")


if __name__ == "__main__":
    main()
