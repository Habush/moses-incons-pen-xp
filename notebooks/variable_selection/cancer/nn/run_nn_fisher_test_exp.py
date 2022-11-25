#!/usr/bin/env python
# Author Abdulrahman S. Omar<hsamireh@gmail>

import os
import subprocess

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn_genetic.space import Continuous
from sklearn_genetic import GASearchCV
from sklearn_genetic.callbacks import DeltaThreshold, TimerStopping
import pickle
import time
import datetime
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sgmcmc import *


def parse_args():

    parser = argparse.ArgumentParser(description="Run Neural Net experiments on Cancer Data.")

    parser.add_argument("--dataset", type=str, default=None, required=True, help="path to the dataset")
    parser.add_argument("--dir", type=str, default=None, required=True, help="directory to save results")
    parser.add_argument("--cosmic-path", type=str, default=None, required=True, help="Path to cosmic genes list")
    parser.add_argument("--net", type=str, default=None, required=True, help="path to the gene network data")
    parser.add_argument("--seed", type=str, default=None, required=True, help="path to the seeds file")
    parser.add_argument("--num-feats", type=int, default=70, help="Number of features to select")
    parser.add_argument("--out-label", type=str, default="posOutcome", help="The column name of the output label")
    return parser.parse_args()

def run_svm(X_train, X_test, y_train, y_test, cv, logger):
    svc_param_grid = {"C": np.logspace(-2, 1, 10), "kernel": ["rbf", "linear", "poly"], "degree": [2, 3]}
    svc_grid_cv = GridSearchCV(estimator=SVC(probability=True), param_grid=svc_param_grid, verbose=1,
                               scoring="roc_auc", cv=cv).fit(X_train, y_train)

    logger.info(f"SVM Best Params: {svc_grid_cv.best_params_}")
    clf = SVC(probability=True,  **svc_grid_cv.best_params_)
    svc_cv_score = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=cv))
    clf.fit(X_train, y_train)
    svc_test_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    logger.info(f"SVC scores - cv score: {svc_cv_score: .4f}, test_score: {svc_test_score: .4f}")
    return svc_grid_cv.best_params_, svc_cv_score, svc_test_score

def run_logistic_regression(X_train, X_test, y_train, y_test, cv, logger, feats=None):
    log_param_grid = {"C": np.logspace(-2, 1, 10)}

    if feats is None:
        X_train_sel, X_test_sel = X_train, X_test

    else:
        X_train_sel, X_test_sel = X_train[:, feats], X_test[:, feats]

    log_grid_cv = GridSearchCV(estimator=LogisticRegression(max_iter=10000), param_grid=log_param_grid, verbose=1,
                               scoring="roc_auc", cv=cv).fit(X_train_sel, y_train)
    logger.info(f"LR best params {log_grid_cv.best_params_}")
    clf = LogisticRegression(max_iter=10000,  **log_grid_cv.best_params_)
    log_cv_score = np.mean(cross_val_score(clf, X_train_sel, y_train, scoring="roc_auc", cv=cv))
    clf.fit(X_train_sel, y_train)
    log_test_score = roc_auc_score(y_test, clf.predict_proba(X_test_sel)[:, 1])
    logger.info(f"LR scores - cv score: {log_cv_score: .4f}, test_score: {log_test_score: .4f}")
    return log_grid_cv.best_params_, log_cv_score, log_test_score

def run_bnn(seed, X_train, X_test, y_train, y_test, J, cv, logger):
    delta_callback = DeltaThreshold(threshold=0.001, generations=2, metric='fitness')
    timer_callback = TimerStopping(total_seconds=1800)
    mixed_sgmcmc = MixedSGMCMC(seed=seed, n_samples=2000, n_chains=1, disc_lr=1e-1, layer_dims=[350],
                               lr_schedule="cyclical", batch_size=20, cycle_len=10)

    gamma_params = Continuous(0, 10., distribution="uniform")
    sigmas = Continuous(0., 10, distribution="uniform")
    temps = Continuous(0., 10, distribution="uniform")
    param_grid = {"eta": gamma_params, "mu": gamma_params, "temp": temps,
                  "sigma": sigmas}

    sgmcmc_grid_cv = GASearchCV(estimator=mixed_sgmcmc, cv=cv,
                                 param_grid=param_grid, verbose=True, population_size=10, generations=4).fit(X_train, y_train, callbacks=[delta_callback, timer_callback],  J=J, activation_fns=["tanh"])

    logger.info(f"SGMCMC Best params: {sgmcmc_grid_cv.best_params_}")
    mixed_sgmcmc = MixedSGMCMC(seed=seed, lr_schedule="cyclical", n_samples=2000, n_chains=1, disc_lr=1e-1, contin_lr=1e-5,
                               batch_size=20, layer_dims=[350], cycle_len=10, **sgmcmc_grid_cv.best_params_)

    sgmcmc_cv_score = np.mean(cross_val_score(mixed_sgmcmc, X_train, y_train, cv=cv,
                                              fit_params={"J": J, "activation_fns": ["tanh"]}))
    mixed_sgmcmc.fit(X_train, y_train, J=J, activation_fns=["tanh"])
    sgmcmc_test_score = mixed_sgmcmc.score(X_test, y_test)
    logger.info(f"SGMCMC scores: cv score: {sgmcmc_cv_score: .4f}, test score: {sgmcmc_test_score: .4f}")

    bnn_disc_mean = jnp.mean(mixed_sgmcmc.states_.discrete_position, axis=0)
    return sgmcmc_grid_cv.best_params_, bnn_disc_mean , sgmcmc_cv_score, sgmcmc_test_score


def run_seed(seed, X_df, y_df, cosmic_genes, J, save_dir, nfeats):
    np.random.seed(seed)
    start_time = time.time()
    logger = setup_logger(save_dir, seed)
    result_summary_dict = {"seed": [seed, seed, seed, seed], "classifier":
                    ["LR", "SVM", "BNN", "BNN + LR"], "num_feats": [len(cosmic_genes), len(cosmic_genes), len(cosmic_genes), nfeats], "cv_score": [], "test_score": []}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, random_state=seed,
                                                                    shuffle=True, stratify=y_df, test_size=0.3)

    X_train_sig, X_test_sig = X_train_df.loc[:,cosmic_genes], X_test_df.loc[:,cosmic_genes]

    X_train, X_test, y_train, y_test = X_train_sig.to_numpy(), X_test_sig.to_numpy(), \
                                       y_train_df.to_numpy(), y_test_df.to_numpy()

    log_best_params_1, log_cv_score_1, log_test_score_1 = run_logistic_regression(X_train, X_test, y_train, y_test, cv, logger)

    svc_best_params, svc_cv_score, svc_test_score = run_svm(X_train, X_test, y_train, y_test, cv, logger)

    X_train, X_test, y_train, y_test = jax.device_put(X_train), jax.device_put(X_test), \
                                       jax.device_put(y_train), jax.device_put(y_test)

    bnn_best_params, bnn_disc_mean, bnn_cv_score, bnn_test_score = run_bnn(seed, X_train, X_test, y_train, y_test, J, cv, logger)

    result_summary_dict["cv_score"].append(log_cv_score_1)
    result_summary_dict["cv_score"].append(svc_cv_score)
    result_summary_dict["cv_score"].append(bnn_cv_score)

    result_summary_dict["test_score"].append(log_test_score_1)
    result_summary_dict["test_score"].append(svc_test_score)
    result_summary_dict["test_score"].append(bnn_test_score)

    bnn_disc_mean_sorted = jnp.argsort(bnn_disc_mean)[::-1]
    sel_feat_idx = bnn_disc_mean_sorted[:nfeats]

    log_best_params_2, log_cv_score_2, log_test_score_2 = run_logistic_regression(X_train, X_test, y_train, y_test, cv, logger, feats=sel_feat_idx)

    result_summary_dict["cv_score"].append(log_cv_score_2)
    result_summary_dict["test_score"].append(log_test_score_2)

    # Save everything
    result_summary_df = pd.DataFrame(result_summary_dict)
    result_summary_df.to_csv(f"{save_dir}/res_summary_cosmic_genes_s_{seed}.csv", index_label=False)

    pickle.dump(log_best_params_1, open(f"{save_dir}/lr_best_params_s_{seed}.pickle", "wb"))
    pickle.dump(svc_best_params, open(f"{save_dir}/svm_best_params_s_{seed}.pickle", "wb"))
    pickle.dump(bnn_best_params, open(f"{save_dir}/bnn_best_params_s_{seed}.pickle", "wb"))
    pickle.dump(log_best_params_2, open(f"{save_dir}/bnn_lr_best_params_s_{seed}.pickle", "wb"))

    np.save(f"{save_dir}/bnn_disc_mean_s_{seed}.npy", bnn_disc_mean)
    np.save(f"{save_dir}/bnn_sel_idx_s_{seed}_n_{nfeats}.npy", sel_feat_idx)

    end_time = time.time()
    elapsed = datetime.timedelta(seconds=(end_time - start_time))
    logger.info(f"Done for seed {seed}. Time elapsed - {elapsed}")

def install_packages():
    subprocess.call(["pip", "install", "-U", "jax[cuda112]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"])
    subprocess.call(["pip", "install", "-U", "jaxlib[cuda112]", "-f", "https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"])
    subprocess.call(["pip", "install", "tensorflow-probability"])
    subprocess.call(["pip", "install", "optax"])
    subprocess.call(["pip", "install", "-U", "git+https://github.com/Habush/Sklearn-genetic-opt.git"])
    subprocess.call(["pip", "install", "git+https://github.com/blackjax-devs/blackjax.git"])

def main():

    args = parse_args()
    dataset_path = args.dataset
    dir_path = args.dir
    cosmic_path = args.cosmic_path
    seed_path = args.seed
    net_path = args.net
    nfeats = args.num_feats
    out_col = args.out_label

    if jax.default_device() != "gpu":
        RuntimeError("No Gpu found. Install the correct jax version")


    df = pd.read_csv(dataset_path)
    X_df, y_df = df[df.columns.difference([out_col])], df[out_col]

    # Get cosmic genes and match them with the input genes
    cosmic_genes_df = pd.read_csv(cosmic_path)
    cosmic_genes_df = cosmic_genes_df[~cosmic_genes_df["Entrez GeneId"].isnull()]
    cosmic_genes_ids = cosmic_genes_df["Entrez GeneId"].astype(int)
    cols = X_df.columns.to_list()
    cols = [int(c) for c in cols]
    cosmic_intr = list(set(set(cosmic_genes_ids) & set(cols)))
    cosmic_genes = [str(c) for c in cosmic_intr]
    X_cosmic = X_df.loc[:,cosmic_genes]

    # Construct gene network using the matched cosmic genes
    regnet_df = pd.read_table(net_path, sep="\t", header=None, names= ["REGULATOR SYMBOL", "REGULATOR ID", "TARGET SYMBOL", "TARGET ID"])
    net_intr = pd.Series(regnet_df["REGULATOR ID"].values, index=regnet_df["TARGET ID"])
    net_intr_rev = pd.Series(regnet_df["TARGET ID"].values, index=regnet_df["REGULATOR ID"])
    J = build_network(X_cosmic, net_intr, net_intr_rev)
    np.fill_diagonal(J, 0.0)

    exp_seeds = []
    with open(seed_path, "r") as fp:
        for line in fp.readlines():
            exp_seeds.append(int(line.strip()))

    for seed in exp_seeds:
        run_seed(seed, X_df, y_df, cosmic_genes, J, dir_path, nfeats)

    print("Done!")


if __name__ == "__main__":
    main()