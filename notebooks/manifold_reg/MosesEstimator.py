__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
import os
import subprocess as subp
import tempfile as tmp
import pandas as pd
import numpy as np
import scipy
import re
from feature_parser import *
from notebooks.manifold_reg.util import assign_cols
from notebooks.manifold_reg.log_util import log_msg
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

patt = re.compile(r"([+-]?[0-9]*[.]?[0-9]+) (.+) \[(.+)\]")

class MosesEstimator(BaseEstimator):
    """
    A wrapper sklearn api based classifier that uses [MOSES](https://github.com/opencog/moses) under the hood
    """

    def __init__(self, scorers=None, alpha=1.0, assoc_mat=None, num_evals=10000,
                    complexity_ratio=10, fs_algo="smd", fs_size=10, num_models=100, threshold=1e-3,
                    n_draws=30, ensemble=True, random_state=42, cols=None, prob="auc"):

        """
        Initialize Moses estimator
        :param scorers: The feature selection scorers to use - moses parameter
        :param alpha: Specific parameter for the weighting the background feature selection penalty - moses parameter
        :param assoc_mat: The network that represents feature relationships
        :param num_evals: Num of evaluation before stopping MOSES run - moses parameter
        :param complexity_ratio: the ratio of the score to complexity, to be used as a penalty, when ranking the metapopulation for fitness - moses parameter
        :param fs_algo: Feature selection algorithm to use - moses parameter
        :param fs_size: Num of features to select at a time - moses parameter
        :param num_models: Num of models to return from a MOSES run - moses parameter
        :param threshold: Improvement threshold for feature selection algorithms - moses parameter
        :param n_draws: Num of times to randomly draw features during feature selection step; only makes sense if fs_algo=random is being used - moses parameter
        :param ensemble: Whether to do majority voting on the model outputs or return average output
        :param random_state: Random seed value - moses parameter
        :param cols: Filter the dataset to only include these columns
        :param prob: What score is MOSES maximizing - moses parameter
        """
        self.scorers = scorers
        self.alpha = alpha
        self.assoc_mat = assoc_mat
        self.num_evals = num_evals
        self.complexity_ratio = complexity_ratio
        self.fs_algo = fs_algo
        self.fs_size = fs_size
        self.num_models = num_models
        self.threshold = threshold
        self.n_draws = n_draws
        self.ensemble = ensemble
        self.random_state = random_state
        self.cols = cols
        self.prob = prob

    def get_params(self, deep=True):
        return {"scorers": self.scorers, "alpha": self.alpha, "assoc_mat": self.assoc_mat,
                "num_evals": self.num_evals, "complexity_ratio": self.complexity_ratio, "fs_algo": self.fs_algo, "n_draws": self.n_draws,
                "ensemble": self.ensemble, "fs_size": self.fs_size, "num_models": self.num_models,
                "threshold": self.threshold, "random_state": self.random_state, "cols": self.cols, "prob": self.prob}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)

        return self

    def fit(self, X, y, output_label="y", moses_params=None):

        data = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        data_df = pd.DataFrame(data)
        if self.cols is None:
            assign_cols(data_df)
        else:
            data_df.columns = self.cols
        self.train_file_ = tmp.NamedTemporaryFile().name
        data_df.to_csv(self.train_file_, index=False)
        self.out_file_ = tmp.NamedTemporaryFile().name
        # print(self.out_file_)
        self.out_label_ = output_label
        return self._fit(moses_params)


    def predict(self, X):
        X_df = pd.DataFrame(X)
        if self.cols is None:
            assign_cols(X_df, append_y=False)
        else:
            if len(X_df.columns) > len(self.cols[:-1]):
                assign_cols(X_df, append_y=False)
                X_df = X_df[self.cols[:-1]]
            else:
                X_df.columns = self.cols[:-1]  # assumes the target column is the last in cols

        res = self._eval_models(self.models_, X_df)
        # return res
        if self.ensemble:
            return (scipy.stats.mode(res, axis=0).mode)[0]

        return np.sum(res, axis=0) / res.shape[0]

    def score(self, X, y):
        X_df = pd.DataFrame(X)
        if self.cols is None:
            assign_cols(X_df, append_y=False)
        else:
            if len(X_df.columns) > len(self.cols[:-1]):
                assign_cols(X_df, append_y=False)
                X_df = X_df[self.cols[:-1]] # assumes the target column is the last in cols
            else:
                X_df.columns = self.cols[:-1]  # assumes the target column is the last in cols

        res = self._eval_models(self.models_, X_df)
        if self.ensemble:
            y_pred = (scipy.stats.mode(res, axis=0).mode)[0]
            return roc_auc_score(y, y_pred)
        s = 0.0
        for i in range(res.shape[0]):
            s += roc_auc_score(y, res[i])

        return s/res.shape[0]

    def _fit(self, moses_params=None):
        # log_file = tmp.NamedTemporaryFile().name
        # print(log_file)
        opt = f"-i {self.train_file_} -o {self.out_file_} -u {self.out_label_} " \
              f"-m {self.num_evals} -W 1 --output-cscore 1 --output-deme-id 1 --result-count {self.num_models} " \
              f"--complexity-ratio={self.complexity_ratio} --random-seed={self.random_state} -H {self.prob} "

        if self.fs_algo is not None:
            opt += f" --enable-fs=1 --fs-target-size={self.fs_size} --fs-algo={self.fs_algo} " \
              f" --fs-n-draws={self.n_draws} --fs-threshold={self.threshold} --fs-smd-top-size={self.fs_size} "

            if self.assoc_mat is not None:
                opt += f" --assoc-mat={self.assoc_mat} --inconsistency-alpha={self.alpha} "

            for i in self.scorers:
                opt += f"--fs-scorer={i} "

        if moses_params is not None:
            opt += f" {moses_params} "

        rcode, stdout, stderr = MosesEstimator.run_moses(opt)

        if rcode != 0:
            raise RuntimeError(f"MOSES ran into an error with return code {rcode}. Here is the stderr output:\n{stderr.decode('utf-8')}")

        self.models_ = MosesEstimator.parse_models(self.out_file_)
        # print(f"{len(self.models_)} parsed!")

        return self

    @staticmethod
    def run_moses(mose_opts):
        cmd = ["asmoses"]

        for opt in mose_opts.split():
            cmd.append(opt)

        process = subp.Popen(args=cmd, stdout=subp.PIPE, stderr=subp.PIPE)

        stdout, stderr = process.communicate()

        return process.returncode, stdout, stderr

    def _eval_models(self, models, test_df):
        """
        Evaluate a list of model objects against an input file
        :param: models: list of model objects
        :param input_file: the location of the input file
        :return: matrix:
        nxm matrix where n is the number of models and m is the number of samples. the matrix contains the predicted
        output of each model on the sample
        """

        test_tmp_file = tmp.NamedTemporaryFile().name
        model_outs = np.zeros((len(models), test_df.shape[0]))
        # print(test_tmp_file)

        test_df.to_csv(test_tmp_file, index=False)
        for i, moses_model in enumerate(models):
            # print(moses_model.model)
            temp_eval_file = tmp.NamedTemporaryFile().name
            # print(temp_eval_file)
            cmd = ['aseval-table', "-i", test_tmp_file, "-c", moses_model.model, "-o", temp_eval_file]
            process = subp.Popen(args=cmd, stdout=subp.PIPE)

            stdout, stderr = process.communicate()
            try:
                if process.returncode == 0:
                    y_pred = np.genfromtxt(temp_eval_file, skip_header=1)
                    model_outs[i] = y_pred
                else:
                    print("Error: The following error raised by eval-table %s" % stderr.decode("utf-8"))
                    raise ChildProcessError(stderr.decode("utf-8"))
            except ValueError as e:
                print(f"Error occured while evaluating: {moses_model.model}")
                print(f"Eval file - {temp_eval_file}, Test file - {test_tmp_file}")
                raise e

        return model_outs
    @staticmethod
    def _mae(scores, labels):
        return np.sum(np.abs(scores - labels)) / float(np.size(labels))

    @staticmethod
    def parse_models(combo_file):
        models = []
        fp = open(combo_file, "r")
        for line in fp:
            match = patt.match(line.strip())
            if match is not None:
                model = match.group(2).strip()
                if model == "true" or model == "false":
                    continue
                # complexity = match.group(3).split(",")[2].split("=")[1]
                # incons_pen = match.group(3).split(",")[5].split("=")[1]
                models.append(MosesModel(model))

        return models


def feature_count(models, causal_feats, assoc_mat):
    feat_count = {}
    for m in models:
        feats = m.get_features()
        for f in feats:
            if not f in feat_count:
                feat_count[f] = {"count": 1, "causal": "No"}
                feat_count[f]["dist"] = np.sum(assoc_mat[f-1][np.array(causal_feats)])
            else:
                feat_count[f]["count"] += 1

            if f in causal_feats:
                feat_count[f]["causal"] = "Yes"

    return feat_count

class MosesModel:
    """
    Specifies a parsed MOSES Combo model - used for analyzing the features in the model
    """
    def __init__(self, txt):
        self.model = txt
        self.features = None

    def get_features(self):
        if self.features is None:
            tree = combo_parser.parse(self.model)
            transformer = ComboTreeTransform()
            transformer.transform(tree)
            self.features = [int(f) for f in transformer.features]
            return self.features

        return self.features

import traceback

def run_moses_exps(seeds, data_dfs, net_dfs, alphas=None, cmplx_ratios=None, moses_params=None, fs_algo="smd", n_draws=30):
    res = {}

    kcv = StratifiedKFold(n_splits=5)

    for i, s in enumerate(seeds):
        try:
            log_msg(f"Now running seed - {s}")
            data_df = data_dfs[i]
            net_df = net_dfs[i]

            # split the data frame into train/test
            X_df, y_df = data_df.iloc[:, :-1], data_df.iloc[:, -1:]
            X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df, y_df, shuffle=True, stratify=y_df,
                                                                            test_size=0.2, random_state=s)

            X_train, X_test, y_train, y_test = X_train_df.to_numpy(), X_test_df.to_numpy(), y_train_df.to_numpy(), y_test_df.to_numpy()
            y_train = np.ravel(y_train)
            y_test = np.ravel(y_test)

            # save the network on temp file
            assoc_mat = net_df.to_numpy()
            assoc_mat = np.abs(assoc_mat)

            net_file = tmp.NamedTemporaryFile(suffix=".npy").name
            np.save(net_file, assoc_mat)

            if alphas is None and cmplx_ratios is None:
                cv = GridSearchCV(MosesEstimator(scorers=["mi"], assoc_mat=None, fs_algo=fs_algo, threshold=1e-5, random_state=s, n_draws=n_draws),
                    {"alpha": [0.0], "complexity_ratio": [10]}, cv=kcv, verbose=1, n_jobs=-1).fit(X_train, y_train,
                                                                                                  moses_params=moses_params)

            elif alphas is None and cmplx_ratios is not None:
                cv = GridSearchCV(MosesEstimator(scorers=["mi"], assoc_mat=None, fs_algo=fs_algo, threshold=1e-5, random_state=s, n_draws=n_draws),
                    {"alpha": [0.0], "complexity_ratio": cmplx_ratios}, cv=kcv, verbose=1, n_jobs=-1).fit(X_train, y_train,
                                                            moses_params=moses_params)

            elif alphas is not None and cmplx_ratios is None:
                cv = GridSearchCV(MosesEstimator(scorers=["mi"], assoc_mat=net_file, fs_algo=fs_algo, threshold=1e-5, random_state=s,
                                   n_draws=n_draws), {"alpha": alphas, "complexity_ratio":[10]}, cv=kcv, verbose=1, n_jobs=-1).fit(X_train, y_train,
                                                                          moses_params=moses_params)
            else:
                cv = GridSearchCV(MosesEstimator(scorers=["mi"], assoc_mat=net_file, fs_algo=fs_algo, threshold=1e-5, random_state=s,
                                   n_draws=n_draws), {"alpha": alphas, "complexity_ratio":cmplx_ratios}, cv=kcv, verbose=1,
                    n_jobs=-1).fit(X_train, y_train, moses_params=moses_params)

            train_sc = cv.best_score_
            cv.best_estimator_.fit(X_train, y_train)
            test_sc = cv.best_estimator_.score(X_test, y_test)
            print(f"Best params: {cv.best_params_}, cv score: {train_sc}, test score: {test_sc}")

            res[s] = {"params": cv.best_params_, "cv_score": train_sc, "test_score": test_sc}

        except Exception as e:
            log_msg(f"Oh no! An error occurred\n{traceback.format_exc()}")

    log_msg("Done")

    return res

def log_roc_auc(est, X, y):
    y_pred = est.predict(X)
    return roc_auc_score(y, y_pred)


def run_fs_moses(df, seed, feats):
    results = {"moses_cv_score": [], "moses_test_score": [], "log_cv_score": [], "log_test_score": []}
    for j, fts in enumerate(feats):
        cols = [f"f{i+1}" for i in fts]
        X_s, y_s = df[cols], df["y"]
        X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, y_s, stratify=y_s, shuffle=True, random_state=seed)
        cv_score = np.mean(cross_val_score(MosesEstimator(fs_algo=None, random_state=seed), X_s_train.to_numpy(), y_s_train.to_numpy(), n_jobs=12))
        moses_est_r = MosesEstimator(fs_algo=None, random_state=seed).fit(X_s_train.to_numpy(), y_s_train.to_numpy())
        test_score = moses_est_r.score(X_s_test.to_numpy(), y_s_test.to_numpy())

        cv_score_2 = np.mean(cross_val_score(LogisticRegression(C=1e9), X_s_train.to_numpy(), y_s_train.to_numpy(), scoring=log_roc_auc))
        log_est = LogisticRegression(C=1e9).fit(X_s_train.to_numpy(), y_s_train.to_numpy())
        test_score_2 = roc_auc_score(y_s_test, log_est.predict(X_s_test.to_numpy()))

        print({"moses_cv_score": cv_score, "moses_test_score": test_score, "log_cv_score": cv_score_2, "log_test_score": test_score_2})
        results["moses_cv_score"].append(cv_score)
        results["moses_test_score"].append(test_score)
        results["log_cv_score"].append(cv_score_2)
        results["log_test_score"].append(test_score_2)

    return pd.DataFrame(results)

def result_df(res, target_count, base_auc):
    df_dict = {"seed": [], "alpha": [], "complexity_ratio": [], "cv_score": [], "test_score": [], "0/1": [],
                 "base_auc": []}

    i = 0
    for s, v in res.items():
        df_dict["seed"].append(s)
        df_dict["alpha"].append(v["params"]["alpha"])
        df_dict["complexity_ratio"].append(v["params"]["complexity_ratio"])
        df_dict["cv_score"].append(v["cv_score"])
        df_dict["test_score"].append(v["test_score"])
        df_dict["0/1"].append(f"{target_count[s][0]}/{target_count[s][1]}")
        df_dict["base_auc"].append(base_auc[i])
        i += 1

    res_df = pd.DataFrame(df_dict)

    return res_df