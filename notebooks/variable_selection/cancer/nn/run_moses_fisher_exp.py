#!/usr/bin/env python
# Author Abdulrahman S. Omar<hsamireh@gmail>
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import time
import datetime
import argparse
from moses_estimator import *
from nn_util import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Run Neural Net experiments on Cancer Data.")

    parser.add_argument("--dataset", type=str, default=None, required=True, help="path to the dataset")
    parser.add_argument("--exp-dir", type=str, default=None, required=True, help="path to the experiment result dir")
    parser.add_argument("--dir", type=str, default=None, required=True, help="directory to save results")
    parser.add_argument("--cosmic-path", type=str, default=None, required=True, help="Path to cosmic genes list")
    parser.add_argument("--seed", type=str, default=None, required=True, help="path to the seeds file")
    parser.add_argument("--num-feats", type=int, default=70, help="Number of features to select")
    parser.add_argument("--out-label", type=str, default="posOutcome", help="The column name of the output label")
    return parser.parse_args()



def run_logistc_regression(X_train, X_test, y_train, y_test, cv, logger):

    log_param_grid = {"C": np.logspace(-2, 1, 10)}
    log_grid_cv = GridSearchCV(estimator=LogisticRegression(max_iter=10000), param_grid=log_param_grid, verbose=1,
                               scoring="roc_auc", cv=cv).fit(X_train, y_train)
    logger.info(f"LR best params {log_grid_cv.best_params_}")
    clf = LogisticRegression(max_iter=10000,  **log_grid_cv.best_params_)
    log_cv_score = np.mean(cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=cv))
    clf.fit(X_train, y_train)
    log_test_score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    logger.info(f"LR scores - cv score: {log_cv_score: .4f}, test_score: {log_test_score: .4f}")
    return log_grid_cv.best_params_, log_cv_score, log_test_score

def run_moses(seed, X_train, X_test, y_train, y_test, out_label, cv, logger):

    moses_grid_params = {"complexity_ratio": np.logspace(0, 3, 5), "complexity_temp": np.logspace(1, 3, 5),
                         "diversity_pressure": [0.0, 0.3, 0.6, 0.9], "hc_fraction_nn": [0.01, 0.1],
                         }

    moses_est = MosesEstimator(seed=seed, num_models=100, num_evals=1000, prob="it")

    moses_grid_cv = GridSearchCV(estimator=moses_est, param_grid=moses_grid_params,
                                 cv=cv, n_jobs=15).fit(X_train, y_train, output_label=out_label)

    moses_cv_score = moses_grid_cv.best_score_
    moses_est.fit(X_train, y_train, output_label=out_label)
    moses_test_score = moses_est.score(X_test, y_test)
    logger.info(f"MOSES scores - cv score: {moses_cv_score: .4f}, test_score: {moses_test_score: .4f}")

    train_eval_out = moses_est._eval_models(moses_est.models_, assign_cols(X_train)).T
    test_eval_out = moses_est._eval_models(moses_est.models_, assign_cols(X_test)).T

    return moses_grid_cv.best_params_, moses_cv_score, moses_test_score, train_eval_out, test_eval_out


def run_seed(seed, X_df, y_df, exp_path, save_dir, num_feats, out_label):

    np.random.seed(seed)
    start_time = time.time()
    logger = setup_logger(save_dir, seed)
    result_summary_dict = {"seed": [seed, seed], "classifier":
        ["BNN + MOSES", "BNN + MOSES + LR"], "num_feats": [num_feats, num_feats], "cv_score": [], "test_score": []}

    cv = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

    idx_sig = np.load(f"{exp_path}/fisher_idx_sig_{seed}.npy")
    selected_idx = np.load(f"{exp_path}/bnn_sel_idx_s_{seed}_n_{num_feats}.npy")

    # X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=seed, shuffle=True, stratify=y_df)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=seed, shuffle=True,
                                                                stratify=y_df, test_size=0.3)

    X_train_sig, X_test_sig = X_train.iloc[:, idx_sig], X_test.iloc[:,idx_sig]
    X_train_sel, X_test_sel = X_train_sig.iloc[:,selected_idx], X_test_sig.iloc[:,selected_idx]

    moses_best_params, moses_cv_score, moses_test_score, train_eval_out, test_eval_out = \
                                            run_moses(seed, X_train_sel, X_test_sel, y_train, y_test, out_label, cv, logger)


    X_train_sel_moses = np.concatenate([X_train_sel.to_numpy(), train_eval_out], axis=1)
    X_test_sel_moses = np.concatenate([X_test_sel.to_numpy(), test_eval_out], axis=1)

    log_best_params, log_cv_score, log_test_score = run_logistc_regression(X_train_sel_moses, X_test_sel_moses,
                                                                                y_train, y_test, cv, logger)

    result_summary_dict["cv_score"].append(moses_cv_score)
    result_summary_dict["test_score"].append(moses_test_score)

    result_summary_dict["cv_score"].append(log_cv_score)
    result_summary_dict["test_score"].append(log_test_score)

    # Save everything
    result_summary_df = pd.DataFrame(result_summary_dict)
    result_summary_df.to_csv(f"{save_dir}/res_summary_fisher_genes_moses_s_{seed}.csv", index_label=False)
    pickle.dump(moses_best_params, open(f"{save_dir}/moses_best_params_s_{seed}.pickle", "wb"))
    pickle.dump(log_best_params, open(f"{save_dir}/moses_lr_best_params_s_{seed}.pickle", "wb"))

    np.save(f"{save_dir}/moses_train_eval_out_s_{seed}.npy", train_eval_out)
    np.save(f"{save_dir}/moses_test_eval_out_s_{seed}.npy", test_eval_out)

    end_time = time.time()

    elapsed = datetime.timedelta(seconds=(end_time - start_time))
    logger.info(f"Done for seed {seed}. Time elapsed - {elapsed}")


def main():

    args = parse_args()
    dataset_path = args.dataset
    save_dir_path = args.dir
    exp_dir_path = args.exp_dir
    seed_path = args.seed
    nfeats = args.num_feats
    out_col = args.out_label


    df = pd.read_csv(dataset_path)
    X_df, y_df = df[df.columns.difference([out_col])], df[out_col]

    exp_seeds = []
    with open(seed_path, "r") as fp:
        for line in fp.readlines():
            exp_seeds.append(int(line.strip()))

    print(f"Total {len(exp_seeds)} runs")

    for seed in exp_seeds:
        print(f"Running seed {seed}")
        try:
            run_seed(seed, X_df, y_df, exp_dir_path, save_dir_path, nfeats, out_col)

        except Exception as e:
            print(f"Ran into an error {e} while running seed {seed}. Skipping it..")

    print("Done!")

if __name__ == "__main__":
    main()