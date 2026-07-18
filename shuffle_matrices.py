import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data_loader import DataLoader
import utils
import time
from pathlib import Path

DATA_DIR = Path("data/TCGA Cancer")
RES_DIR = Path("data/shuffled_matrices")

FILES_USED = {
    "clinical_patient": "_clinical_patient.csv",
    "clinical_sample": "_clinical_sample.csv",
    "expression": "_expression_zscore_all.csv",
    "mutation": "_mutation_matched_expression_zscore_all.csv"
}

def generate_one_matrix(idx, mut_df, res_dir):
    mut_shuffled = utils.shuffle_matrix(mut_df)
    save_path = res_dir / f"matrix_{idx}.npz"
    
    np.savez_compressed(
        save_path, 
        data=mut_shuffled.values, 
        index=mut_shuffled.index.values, 
        columns=mut_shuffled.columns.values
    )
    return save_path

def main():
    total_matrices = 1000
    n_jobs = -1
    for cancer_folder in DATA_DIR.iterdir():
        if cancer_folder.is_dir():
            paths = {}

            for key, suffix in FILES_USED.items():
                matches = list(cancer_folder.glob(f"*{suffix}"))
                if matches:
                    paths[key] = str(matches[0])
                else:
                    paths[key] = None

            if all(paths.values()):
                loader = DataLoader(
                    paths["expression"],
                    paths["mutation"],
                    paths["clinical_patient"],
                    paths["clinical_sample"]
                )

                print(f"> Processing cancer: {cancer_folder.name}...")

                _, mut, _, _ = loader.load()

                res_dir = RES_DIR / cancer_folder.name
                res_dir.mkdir(parents=True, exist_ok=True)
                
                print(f">>> Generating {total_matrices} shuffled matrices using {n_jobs} cores...")
                start = time.time()
                
                Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(generate_one_matrix)(i, mut, res_dir) for i in range(total_matrices)
                )
                
                print(f">>> Done. Time: {time.time() - start:.2f}s.\n")

            else:
                print(f">>> Cannot found folder :{cancer_folder.name}.\n")

if __name__ == "__main__":
    main()