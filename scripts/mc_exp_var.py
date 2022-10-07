import os
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro as npyro
import multiprocessing
import sacred
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from scripts.generate_samples import *
from notebooks.variable_selection.MosesEstimator import *
from notebooks.variable_selection.util import load_bmm_files
import warnings

warnings.filterwarnings('ignore')
# numpyro.set_platform("cpu")
# numpyro.set_host_device_count(multiprocessing.cpu_count())


ex = sacred.Experiment("mcmc_variable_selection_var")
# ex.observers.append(MongoObserver())
ex.captured_out_filter = apply_backspaces_and_linefeeds
seeds, data_dfs, net_dfs, feats = load_bmm_files("/home/xabush/code/snet/moses-incons-pen-xp/data/bmm_data_thr_2_new/")

rs_dir = os.path.abspath("/home/xabush/code/snet/moses-incons-pen-xp/data/mcmc_variable_selection_var_2")
ex.observers.append(FileStorageObserver(rs_dir))

if not os.path.exists(rs_dir):
    os.mkdir(rs_dir)

@ex.config
def config():
    sigma = 25
    n_chains = 5
    n_warmup = int(1e3)
    n_samples = int(1e4)
    seed = 13
    eta = 1.0
    mu = 1.0
    vars = np.linspace(0.5, 10, 10)
    rw = False

@ex.main
def run(sigma, n_chains, n_warmup, n_samples, seed,
        eta, mu, vars, rw, _seed):

    seed_idx = seeds.index(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    J = net_dfs[seed_idx].to_numpy()
    df = data_dfs[seed_idx]
    X, y = df[df.columns.difference(["y"])].to_numpy().astype(np.float_), df["y"].to_numpy().astype(np.float_)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seeds[seed_idx], shuffle=True, stratify=y)

    print(f"Running exp with seed {seed}, causal feats - {np.array(feats[seed_idx]) - 1}")

    kernel = MixedHMC(HMC(model), random_walk=rw)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples, num_chains=n_chains)
    mcmc.run(key, X, y, sigma*np.identity(X.shape[1]), J, eta, mu)

    beta_samples = jax.device_get(mcmc.get_samples()["beta"])

    for i in range(vars.shape[0]):
        var = vars[i]
        idx = np.var(beta_samples, axis=0) < var
        ar = np.arange(100)
        feats_idx = ar[idx]
        res_fs_name = os.path.join(rs_dir, f"res_auc_s_{seed}_var_{var:.2f}.csv")
        feats_fs_name = os.path.join(rs_dir, f"feats_idx_s_{seed}_var_{var:.2f}.csv")
        print(f"Feats with var thr {var} - {feats_idx}")

        res_hmc_auc = run_fs_moses(X_train, X_test, y_train, y_test, seed, [feats_idx], ex)
        res_hmc_auc.to_csv(res_fs_name, index=False)

        with open(feats_fs_name, "w") as fp:
            fp.write(",".join(map(str, feats_idx)) + "\n")


        ex.add_artifact(res_fs_name, "result_file")
        ex.add_artifact(feats_fs_name, "feat_idx_file")



rng_values = np.linspace(0.5, 10, 10)
seeds  = [10, 55, 96, 97]

if __name__ == "__main__":

    for seed in seeds:
        ex.run(config_updates={
            "seed": seed
        })