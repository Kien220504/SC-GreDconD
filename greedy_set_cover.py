import pandas as pd
import numpy as np
import time
import os
from data_loader import DataLoader 

def weighted_greedy_set_cover(mut_df, exp_df=None):
    patients_with_mutations = mut_df.index[mut_df.sum(axis=1) > 0].tolist()
    universe = set(patients_with_mutations)
    
    covered_patients = set()
    selected_genes = []
    
    #abs_exp_df = exp_df.abs()
    
    start_time = time.time()
    
    while len(covered_patients) < len(universe):
        best_gene = None
        best_score = -1.0
        best_new_cover = set()
        
        current_uncovered = universe - covered_patients
        
        if not current_uncovered:
            break
            
        for gene in mut_df.columns:
            if gene in selected_genes:
                continue
                
            mutated_indices = mut_df.index[mut_df[gene] == 1]
            
            new_cover = set(mutated_indices).intersection(current_uncovered)
            
            n_new = len(new_cover)
            
            if n_new == 0:
                continue
            
            # Signal
            #avg_signal = abs_exp_df.loc[list(new_cover), gene].mean()
            
            # Cover * (1 + Signal)
            #current_score = n_new * (1 + avg_signal)
            current_score = n_new
            
            if current_score > best_score:
                best_score = current_score
                best_gene = gene
                best_new_cover = new_cover
        
        if best_gene is None:
            break
            
        selected_genes.append({
            'Gene': best_gene,
            'New_Patients_Covered': len(best_new_cover),
            'Avg_Expression_Signal': best_score / len(best_new_cover) - 1,
            'Composite_Score': best_score
        })
        covered_patients.update(best_new_cover)

    elapsed_time = time.time() - start_time
    
    return pd.DataFrame(selected_genes), len(universe), len(covered_patients), elapsed_time


def main():
    EXP_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_expression_zscore_all.csv' 
    MUT_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_mutation_matched_expression_zscore_all.csv'
    CLIN_P_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_clinical_patient.csv'
    CLIN_S_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_clinical_sample.csv'
    
    log_dir = 'results/set_cover'

    if not os.path.exists(log_dir): os.makedirs(log_dir)
    count = len([f for f in os.listdir(log_dir) if f.startswith("set_cover_exp_res")]) + 1
    filename = f"{log_dir}/set_cover_exp_results_{count}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        
        def log(text):
            print(text)          
            f.write(text + "\n") 
            f.flush()            

        log(">>> LOADING DATA...")
        loader = DataLoader(EXP_PATH, MUT_PATH, CLIN_P_PATH, CLIN_S_PATH)
        datasets = loader.load_and_process_matrix() 

        case_descriptions = {
            1: "Case 1: Remove genes with lowest variance (< 25% IQR)                           ",
            2: "Case 2: Remove genes with highest variance (> 75% IQR)                          ",
            3: "Case 3: Remove genes with lowest and highest variance (< 25% IQR and > 75% IQR)",
            4: "Case 4: Keep all genes                                                          "
        }

        results_summary = []

        log("\n" + "="*70)
        log(">>> RUNNING GREEDY SET COVERING...")
        log("="*70)

        for case_id, (sub_exp, sub_mut) in datasets.items():
            desc = case_descriptions.get(case_id, f"Case {case_id}")
            n_genes = sub_exp.shape[1]
            n_patients = sub_exp.shape[0]
            
            log(f"\n[{desc}]")
            log(f"   - Input: {n_genes} genes, {n_patients} patients.")
            
            result_df, total_universe, total_covered, run_time = weighted_greedy_set_cover(sub_mut, sub_exp)
            
            n_selected = len(result_df)
            coverage_percent = (total_covered / total_universe * 100) if total_universe > 0 else 0
            
            log(f"   - Run time: {run_time:.4f} giây")
            log(f"   - Number of selected genes: {n_selected}")
            log(f"   - Coverage percent: {coverage_percent:.2f}% ({total_covered}/{total_universe} bệnh nhân)")
             
            if n_selected > 0:
                log("\n   > GENE LIST:")
                header = f"     {'STT':<4} | {'Gene':<15} | {'Score':<10} | {'Covered':<8}"
                log(header)
                log("     " + "-" * 45)
                
                for idx, row in result_df.iterrows():
                    stt = idx + 1
                    gene_name = str(row['Gene'])
                    score = row['Composite_Score']
                    n_cover = int(row['New_Patients_Covered'])
                    
                    line = f"     {stt:<4} | {gene_name:<15} | {score:<10.2f} | {n_cover:<8}"
                    log(line)
                log("     " + "-" * 45)
            else:
                log("\n   > NO GENES SELECTED.")

            results_summary.append({
                'Case': desc,
                'Input_Genes': n_genes,
                'Selected_Genes': n_selected,
                'Coverage_(%)': round(coverage_percent, 2),
                'Time_(s)': round(run_time, 4),
            })
            
            # Xuất file CSV chi tiết cho từng case
            # result_filename = f"results_case_{case_id}.csv"
            # result_df.to_csv(result_filename, index=False)
            # log(f"   -> Đã lưu danh sách chi tiết vào file: {result_filename}")

        log("\n" + "="*70)
        log("RESULTS SUMMARY")
        log("="*70)
        
        summary_df = pd.DataFrame(results_summary)
        
        summary_string = summary_df.to_string(index=False, col_space=15, justify='left')
        log(summary_string)
        log("="*70)
        log(f"DONE!")

if __name__ == "__main__":
    main()