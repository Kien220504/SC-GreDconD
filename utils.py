import pandas as pd
import numpy as np
import os
import re
from scipy.stats import hypergeom

def load_ground_truth(folder_name, column_name="SYMBOL"):
    match = re.search(r'\((.*?)\)', folder_name)

    if match:
        disease_name = match.group(1).strip()
    else:
        return None
    
    filename = f"{disease_name}_driver_genes.csv" 
    file_path = os.path.join("data/Ground Truth", filename)
    
    try:
        df = pd.read_csv(file_path)
        gt_genes = df[column_name].dropna().unique().tolist()
        return set(gt_genes)
    except FileNotFoundError:
        print(f"Cannot find file {file_path}")
        return set()
    except KeyError:
        print(f"Cannot find column '{column_name}' in {filename}")
        return set() 

def calculate_hypergeom_score(selected_genes, ground_truth_genes, total_genes_count=25000):
    """
    Calculate with hypergeom distribution
    N: total_genes_count
    K: len(ground_truth_genes)
    n: len(selected_genes)
    k: overlap genes
    """
    selected_set = set(selected_genes)
    overlap = len(selected_set.intersection(ground_truth_genes))
    
    K = len(ground_truth_genes)
    n = len(selected_set)
    k = overlap
    
    # P-value: P(X >= k) = sf(k-1)
    if n == 0: return 1.0
    
    p_value = hypergeom.sf(k - 1, total_genes_count, K, n)
    return p_value, k

def shuffle_matrix(df_mutation):
    arr = df_mutation.values.copy()
    rng = np.random.default_rng()
    
    for i in range(arr.shape[0]):
        rng.shuffle(arr[i])
        
    return pd.DataFrame(arr, index=df_mutation.index, columns=df_mutation.columns)