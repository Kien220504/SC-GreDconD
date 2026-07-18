import pandas as pd


class DataLoader:
    def __init__(self, exp_path, mut_path, clin_p_path, clin_s_path):
        self.exp_path = exp_path
        self.mut_path = mut_path
        self.clin_p_path = clin_p_path
        self.clin_s_path = clin_s_path
        self.exp = None
        self.mut = None
        self.clin = None
        self.genes_name = None

    def load(self, get_values=False):
        try:
            df_exp = pd.read_csv(self.exp_path)
            df_mut = pd.read_csv(self.mut_path)
            df_clin_p = pd.read_csv(self.clin_p_path)
            df_clin_s = pd.read_csv(self.clin_s_path)
            clin = pd.merge(df_clin_s, df_clin_p, on='PATIENT_ID')

            df_exp['Sample_ID'] = df_exp['Sample_ID'].str[:15]
            df_mut['Sample_ID'] = df_mut['Sample_ID'].str[:15]
            clin['SAMPLE_ID'] = clin['SAMPLE_ID'].str[:15]

            common_ids = list(set(df_exp['Sample_ID']) & set(df_mut['Sample_ID']) & set(clin['SAMPLE_ID']))
            common_ids.sort()

            self.exp = df_exp.set_index('Sample_ID').loc[common_ids]
            self.mut = df_mut.set_index('Sample_ID').loc[common_ids]
            self.clin = clin.set_index('SAMPLE_ID').loc[common_ids]
            self.genes_name = self.mut.columns.tolist()

            print("===" * 20)
            print(f"DATA LOADED: {len(common_ids)} SAMPLES.")
            print(f"EXPRESSION MATRIX: {self.exp.shape}")
            print(f"MUTATION MATRIX: {self.mut.shape}")
            print(f"CLINICAL LABEL: {self.clin.shape}")

        except Exception as e:
            print(e)

        if get_values:
            y_clin = self.clin['OS_STATUS'].values
            y_mut = self.mut.values
            X = self.exp.values
            return X, y_mut, y_clin, self.genes_name
        else:
            return self.exp, self.mut, self.clin, self.genes_name

    def load_and_process_matrix(self):
        if self.exp is None:
            _, _, _, _ = self.load()

        gene_variances = self.exp.var()

        q1_val = gene_variances.quantile(0.25)
        q3_val = gene_variances.quantile(0.75)

        genes_case1 = gene_variances[gene_variances >= q1_val].index.tolist()
        genes_case2 = gene_variances[gene_variances <= q3_val].index.tolist()
        genes_case3 = gene_variances[(gene_variances >= q1_val) & (gene_variances <= q3_val)].index.tolist()
        genes_case4 = self.exp.columns.tolist()

        datasets = {
            1: genes_case1,
            2: genes_case2,
            3: genes_case3,
            4: genes_case4
        }

        results = {}
        for case_name, valid_genes in datasets.items():
            valid_genes = [g for g in valid_genes if g in self.mut.columns]

            sub_exp = self.exp[valid_genes]
            sub_mut = self.mut[valid_genes]

            results[case_name] = (sub_exp, sub_mut)

        return results
