import numpy as np
import pandas as pd
from scipy import stats
import time
import os
from data_loader import DataLoader


class GREDCOND:
    def __init__(self, I, genes_name):
        """
        I: numpy array (m samples x n genes), boolean matrix.
        """
        self.I = I.astype(bool)
        self.n_rows, self.n_cols = I.shape
        self.factors = []
        self.U = np.zeros_like(self.I, dtype=bool)
        self.genes_name = genes_name
        self.run_time = 0
        self.factor_coverages = []
        self.total_ones = np.sum(self.I)

    def diversity(self, current_factors, candidate_D):
        """
        Computes the average diversity score between a candidate gene set
        and the gene sets of all currently selected factors.

        div(A, B) = 1 - max(|A ∩ B| / |A|, |A ∩ B| / |B|)
        """
        if not current_factors:
            return 1.0

        total_score = 0
        for _, D_i in current_factors:
            set_Di = set(D_i)
            set_D = set(candidate_D)

            intersection = len(set_Di.intersection(set_D))
            len_Di = len(set_Di)
            len_D = len(set_D)

            if len_Di == 0 or len_D == 0:
                s_i = 1.0
            else:
                s_i = 1.0 - max(intersection / len_Di, intersection / len_D)

            total_score += s_i

        return total_score / len(current_factors)

    def fit(self, max_factors=50):
        start_time = time.time()

        while len(self.factors) < max_factors:
            R = self.I & (~self.U)
            if not np.any(R):
                break

            C = np.ones(self.n_rows, dtype=bool)
            D = []

            while True:
                best_col = -1
                best_score = -1

                if D:
                    rows_indices_curr = np.where(C)[0]
                    current_covered_count = np.sum(R[np.ix_(rows_indices_curr, D)])
                else:
                    current_covered_count = 0

                potential_cols = np.where(R.any(axis=0))[0]
                potential_cols = [c for c in potential_cols if c not in D]

                if not potential_cols:
                    break

                for col_idx in potential_cols:
                    D_prime = D + [col_idx]

                    cols_matrix = self.I[:, D_prime]
                    C_prime = np.all(cols_matrix, axis=1)

                    if not np.any(C_prime):
                        continue

                    rows_indices = np.where(C_prime)[0]
                    col_indices = D_prime
                    covered_entries = np.sum(R[np.ix_(rows_indices, col_indices)])
                    div_score = self.diversity(self.factors, D_prime)
                    score = covered_entries * div_score

                    if score > best_score:
                        best_score = score
                        best_col = col_idx
                        best_C = C_prime

                current_score = current_covered_count * self.diversity(self.factors, D)

                if best_score > current_score:
                    D.append(best_col)
                    C = best_C
                else:
                    break

            if not D:
                break

            self.factors.append((C, D))

            factor_matrix = np.zeros_like(self.I, dtype=bool)
            factor_matrix[np.ix_(C, D)] = True
            self.U = self.U | factor_matrix

            current_covered_ones = np.sum(self.U)
            coverage_percent = (current_covered_ones / self.total_ones) * 100
            self.factor_coverages.append(coverage_percent)

        self.run_time = time.time() - start_time

        return self.factors

    def run_analyze(self, max_factors=50):
        if not self.factors:
            _ = self.fit(max_factors=max_factors)

        gene_scores = {}

        start_time = time.time()

        for idx, (C_mask, D_indices) in enumerate(self.factors):
            struct_score = np.sum(C_mask) / self.n_rows

            for gene_idx in D_indices:
                gene_name = self.genes_name[gene_idx]

                if gene_name not in gene_scores:
                    gene_scores[gene_name] = {
                        'Struct_score': struct_score
                    }
                else:
                    gene_scores[gene_name] = {
                        'Struct_score': max(struct_score, gene_scores[gene_name].get('Struct_score', struct_score)),
                    }
                res = dict(sorted(gene_scores.items(), key=lambda item: item[1]['Struct_score'], reverse=True))

        elapsed_time = time.time() - start_time
        self.run_time += elapsed_time

        return pd.DataFrame.from_dict(res, orient='index'), self.run_time


def main():
    EXP_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_expression_zscore_all.csv'
    MUT_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_mutation_matched_expression_zscore_all.csv'
    CLIN_P_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_clinical_patient.csv'
    CLIN_S_PATH = 'data/TCGA Cancer/Glioblastoma Multiforme (GBM)/GBM_clinical_sample.csv'

    log_dir = 'results/gredcond'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    count = len([f for f in os.listdir(log_dir) if f.startswith("gredcond_res")]) + 1
    filename = f"{log_dir}/gredcond_results_{count}.txt"

    with open(filename, "w", encoding="utf-8") as f:

        def log(text):
            print(text)
            f.write(text + "\n")
            f.flush()

        log(">>> LOADING DATA...")
        loader = DataLoader(EXP_PATH, MUT_PATH, CLIN_P_PATH, CLIN_S_PATH)
        _, _, _, genes_name = loader.load()
        datasets = loader.load_and_process_matrix()

        case_descriptions = {
            1: "Case 1: Remove genes with lowest variance (< 25% IQR)                           ",
            2: "Case 2: Remove genes with highest variance (> 75% IQR)                          ",
            3: "Case 3: Remove genes with lowest and highest variance (< 25% IQR and > 75% IQR)",
            4: "Case 4: Keep all genes                                                          "
        }

        results_summary = []

        log("\n" + "=" * 70)
        log(">>> RUNNING GREDCOND ALGORITHM...")
        log("=" * 70)

        for case_id, (sub_exp, sub_mut) in datasets.items():
            desc = case_descriptions.get(case_id, f"Case {case_id}")
            n_genes = sub_exp.shape[1]
            n_patients = sub_exp.shape[0]

            I = sub_mut.values
            exp_mat = sub_exp.values

            log(f"\n[{desc}]")
            log(f"   - Input: {n_genes} genes, {n_patients} patients.")

            model = GREDCOND(I, exp_mat, genes_name)
            result_df, run_time = model.run_analyze(max_factors=75)

            log(f"   - Run time: {run_time:.4f} seconds")

            final_coverage = model.factor_coverages[-1] if model.factor_coverages else 0.0
            log(f"   - Final Data Coverage (Explained Variance): {final_coverage:.2f}%")

            log("\n   > GENE LIST:")
            header = f"     {'Gene':<15} | {'Struct_score':<10} | {'Func_score':<10} | {'Total_score':<10}"
            log(header)
            log("     " + "-" * 45)

            for idx, row in result_df.iterrows():
                gene_name = idx
                struct_score = row['Struct_score']
                func_score = row['Func_score']
                total_score = row['Total_score']

                line = f"     {gene_name:<15} | {struct_score:<10.2f} | {func_score:<10.2f} | {total_score:<10.2f}"
                log(line)
            log("     " + "-" * 45)

            results_summary.append({
                'Case': desc,
                'Input_Genes': n_genes,
                'Time_(s)': round(run_time, 4),
                'Coverage_(%)': round(final_coverage, 2)
            })

            log("\nGENES IN EACH FACTORS:")
            selected_genes = set()
            for idx, (patients_mask, genes_indices) in enumerate(model.factors):
                current_gene_names = [model.genes_name[i] for i in genes_indices]
                cov = model.factor_coverages[idx]
                log(f"Factor {idx+1} (Cov: {cov:.2f}%): {current_gene_names}")

        log("\n" + "=" * 70)
        log("RESULTS SUMMARY")
        log("=" * 70)

        summary_df = pd.DataFrame(results_summary)

        summary_string = summary_df.to_string(index=False, col_space=15, justify='left')
        log(summary_string)
        log("=" * 70)
        log(f"DONE!")


if __name__ == "__main__":
    main()
