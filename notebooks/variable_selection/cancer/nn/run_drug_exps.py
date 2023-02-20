#!/home/abdu/miniconda3/bin/python3
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from exp_utils import cross_val_runs
import pathlib
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.sparse import csgraph
import warnings
warnings.filterwarnings("ignore")

data_dir = "/home/abdu/bio_ai/moses-incons-pen-xp/data"
gdsc_dir = f"{data_dir}/cell_line/gdsc2"

exp_dir = f"{data_dir}/exp_data_5/cancer/gdsc"
drug_ids = [1502, 1814, 1007, 1558, 1031,
            1037, 1039, 1199, 1191, 1089] # Bicalutamide, Nelarabine, Docetaxel,
# Lapatinib, elesclomol, bx795, sl0101, Tamoxifen, Bortezomib, Oxaliplatin
VERSION = "1c"

seeds = [422,261,968,282,739,573,220,413,745,775,482,442,210,423,760,57,769,920,226,196]
curr_seeds = seeds
print(len(curr_seeds))

def run_single_drug(drug_id):
    gdsc_exp_data = pd.read_csv(f"{gdsc_dir}/gdsc_gene_expr.csv", index_col="model_id")
    cols = gdsc_exp_data.columns.to_list()
    cancer_driver_genes_df = pd.read_csv(f"{data_dir}/cell_line/driver_genes_20221018.csv")
    driver_syms = cancer_driver_genes_df["symbol"].to_list()
    driver_sym_list = [sym.strip() for sym in cols if sym in driver_syms]
    gdsc_response_data = pd.read_csv(f"{gdsc_dir}/GDSC2_fitted_dose_response_24Jul22.csv", index_col="SANGER_MODEL_ID")

    J = np.load(f"{data_dir}/cell_line/cancer_genes_net.npy")
    L = csgraph.laplacian(J, normed=True)
    hp_configs = {"epochs": 500, "act_fn": "swish",
                  "beta": 0.25, "hidden_sizes": [500],
                  "num_models": 1, "lr_0": 1e-3}
    drug_response_data = gdsc_response_data[gdsc_response_data["DRUG_ID"] == drug_id]
    drug_name = drug_response_data["DRUG_NAME"].iloc[0].lower()
    drug_exp_response = pd.merge(gdsc_exp_data, drug_response_data["LN_IC50"], left_index=True, right_index=True)
    print(f"Starting exp for Drug id: {drug_id}/{drug_name}")
    print(f"Total samples for drug {drug_id}/{drug_name}: {drug_exp_response.shape[0]}")

    save_dir = f"{exp_dir}/{drug_name}"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/configs").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/checkpoints").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{save_dir}/dropout").mkdir(parents=True, exist_ok=True)

    X, target = drug_exp_response.iloc[:,:-1], drug_exp_response.iloc[:,-1]
    target = -np.log10(np.exp(target))
    X_selected = X[driver_sym_list]
    cross_val_runs(curr_seeds, X_selected, target, L, VERSION, save_dir, saved_config=False,
                   timeout=900,
                   **hp_configs)
    print(f"Done for drug: {drug_id}/{drug_name}")

if __name__ == "__main__":
    pool = Pool(len(drug_ids))
    pool.map(run_single_drug, drug_ids)
    pool.close()
    pool.join()