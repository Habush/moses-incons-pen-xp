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

numpyro.set_host_device_count(multiprocessing.cpu_count())

ex = sacred.Experiment("mcmc_variable_selection")
# ex.observers.append(MongoObserver())
ex.captured_out_filter = apply_backspaces_and_linefeeds
seeds, data_dfs, net_dfs, feats = load_bmm_files("/home/xabush/code/snet/moses-incons-pen-xp/data/bmm_data_thr_2_f100/")

rs_dir = os.path.abspath("/home/xabush/code/snet/moses-incons-pen-xp/data/mcmc_variable_selection_2")
ex.observers.append(FileStorageObserver(rs_dir))

if not os.path.exists(rs_dir):
    os.mkdir(rs_dir)

@ex.config
def config():
    sigma = 25
    method = "mh_mi"
    n_chains = 3
    n_warmup = int(1e3)
    n_samples = int(1e3)
    seed_idx = 2
    eta = 1.0
    mu = 1.0
    B = 1.0
    rw = False


@ex.main
def run(sigma, method, n_chains, n_warmup, n_samples, seed_idx,
        eta, mu, B, rw, _seed):

    seed = seeds[seed_idx]
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    J = net_dfs[seed_idx].to_numpy()
    df = data_dfs[seed_idx]
    X, y = df[df.columns.difference(["y"])].to_numpy().astype(np.float_), df["y"].to_numpy().astype(np.float_)
    p = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seeds[seed_idx], shuffle=True, stratify=y)

    print(f"Running {method} exp with seed {seed}, causal feats - {np.array(feats[seed_idx]) - 1}")

    res_fs_name = os.path.join(rs_dir, f"res_auc_{method}_s_{seed}_e_{eta:.2f}_m_{mu:.2f}_b_{B:.2f}.csv")

    if method == "mh_mi":
        feat_samples = draw_samples_mh_mi(X_train, y_train, J, n_chains, n_warmup+n_samples,
                                          n_samples, eta, mu, B)

        ex.log_scalar("avg_length", average_length(feat_samples))
        ex.log_scalar("av_hamm_dist", average_hamm_dist(feats, seed_idx, feat_samples, p))

        res_mh_mi_auc = run_fs_moses(X_train, X_test, y_train, y_test, seed, feat_samples[:100], ex)

        res_mh_mi_auc.to_csv(res_fs_name, index=False)


    elif method == "hmc_rand":

        feat_samples = draw_samples_hmc(key, X_train, y_train, J, sigma, n_chains, n_samples,
                                        n_warmup, eta, mu, rw)


        ex.log_scalar("avg_length", average_length(feat_samples))
        # ex.log_scalar("av_hamm_dist", average_hamm_dist(feats, seed_idx, feat_samples))

        res_hmc_rand_auc = run_fs_moses(X_train, X_test, y_train, y_test, seed,
                                        feat_samples[:100], ex)

        res_hmc_rand_auc.to_csv(res_fs_name, index=False)


    elif method == "hmc_mi":

        feat_samples = draw_samples_hmc_mi(key, X_train, y_train, J, sigma, n_chains, n_warmup, n_samples, eta, mu, B, rw)

        ex.log_scalar("avg_length", average_length(feat_samples))
        # ex.log_scalar("av_hamm_dist", average_hamm_dist(feats, seed_idx, feat_samples))

        res_hmc_mi_auc = run_fs_moses(X_train, X_test, y_train, y_test, seed, feat_samples[:100], ex)

        res_hmc_mi_auc.to_csv(res_fs_name, index=False)

    elif method == "rand":
        feat_samples = get_rand_feats(p, 100)

        ex.log_scalar("avg_length", average_length(feat_samples))
        # ex.log_scalar("av_hamm_dist", average_hamm_dist(feats, seed_idx, feat_samples))

        res_rand_auc = run_fs_moses(X_train, X_test, y_train, y_test, seed, feat_samples, ex)

        res_rand_auc.to_csv(res_fs_name, index=False)

    else:
        raise ValueError(f'Unsupported method {method}')


    ex.add_artifact(res_fs_name, "result_file")


# eta_rng = [0.1, 1.0, 5.0, 10.0, 100.0]
rng_values = [0.01, 0.1, 1.0, 5.0, 10.0, 100.0]



if __name__ == "__main__":
    # ex.run_commandline()
    # ex.run(config_updates={"method": "mh_mi", "n_warmup": int(98e3), "n_samples": int(1e3)})

    # ex.run(config_updates={"method": "mh_mi", "n_warmup": int(98e3), "n_samples": int(1e3)})

    #HMC_Rand experiments
    for e in eta_rng:
        for m in rng_values:
            ex.run(config_updates={
                "method": "hmc_rand",
                "eta": e, "mu": m
            })

    #HMC_MI experiments
    for e in rng_values:
        for m in rng_values:
            ex.run(config_updates={
                "method": "hmc_mi",
                "eta": e, "mu": m
            })


    #MI Experiments
    for e in rng_values:
        for m in rng_values:
            ex.run(config_updates={
                "method": "mh_mi",
                "n_warmup":  int(98e3),
                "n_samples": int(1e3),
                "eta": e, "mu": m
            })


    #Rand experiment
    ex.run(config_updates={"method": "rand"})